// Data scoreboard. Sequences post what the hardware must produce (expect_*);
// the scoreboard compares it with what the monitors observe:
//   - bytes of AXI4-Lite reads in the buffer windows,
//   - words of AXI4-Lite register reads (RESULT, LOGITS, ...),
//   - bytes the DMA writes to external memory.
// Each expectation is consumed by the first observation that covers it; any
// left at the end of the test was never checked and is an error. Register
// reads of non-volatile fields are checked separately by the register model
// (check_on_read).

`uvm_analysis_imp_decl(_axil)
`uvm_analysis_imp_decl(_axi)

class npu_scoreboard extends uvm_component;
  `uvm_component_utils(npu_scoreboard)

  uvm_analysis_imp_axil #(axil_item, npu_scoreboard)      axil_imp;
  uvm_analysis_imp_axi  #(axi_burst_item, npu_scoreboard) axi_imp;

  local bit [7:0]  exp_host[bit [31:0]];   // AXI4-Lite byte address -> byte
  local bit [31:0] exp_word[bit [31:0]];   // AXI4-Lite register address -> word
  local bit [7:0]  exp_ext[bit [31:0]];    // external byte address -> byte

  int unsigned n_bytes_ok, n_words_ok, n_ext_ok, n_errors;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    axil_imp = new("axil_imp", this);
    axi_imp  = new("axi_imp", this);
  endfunction

  // ------------------------------------------------------------ expectations
  function void expect_host_bytes(bit [31:0] addr, const ref byte unsigned b[$]);
    foreach (b[i]) exp_host[addr + 32'(i)] = b[i];
  endfunction

  function void expect_word(bit [31:0] addr, bit [31:0] value);
    exp_word[addr] = value;
  endfunction

  function void expect_ext_bytes(bit [31:0] addr, const ref byte unsigned b[$]);
    foreach (b[i]) exp_ext[addr + 32'(i)] = b[i];
  endfunction

  function int unsigned pending();
    return exp_host.num() + exp_word.num() + exp_ext.num();
  endfunction

  // ------------------------------------------------------------ observations
  function void write_axil(axil_item t);
    if (t.write || t.resp != 2'b00) return;
    if (exp_word.exists(t.addr)) begin
      if (t.data !== exp_word[t.addr]) error($sformatf("read %h = %h, expected %h", t.addr, t.data, exp_word[t.addr]));
      else n_words_ok++;
      exp_word.delete(t.addr);
    end
    for (int i = 0; i < 4; i++) begin
      bit [31:0] a = {t.addr[31:2], 2'b00} + 32'(i);
      if (exp_host.exists(a)) begin
        if (t.data[8 * i +: 8] !== exp_host[a])
          error($sformatf("read byte %h = %h, expected %h", a, t.data[8 * i +: 8], exp_host[a]));
        else n_bytes_ok++;
        exp_host.delete(a);
      end
    end
  endfunction

  function void write_axi(axi_burst_item t);
    if (!t.write || t.resp != 2'b00) return;
    foreach (t.data[b])
      for (int i = 0; i < 8; i++) begin
        bit [31:0] a = t.addr + 32'(8 * b + i);
        if (t.strb[b][i] && exp_ext.exists(a)) begin
          if (t.data[b][8 * i +: 8] !== exp_ext[a])
            error($sformatf("DMA wrote %h to %h, expected %h", t.data[b][8 * i +: 8], a, exp_ext[a]));
          else n_ext_ok++;
          exp_ext.delete(a);
        end
      end
  endfunction

  local function void error(string msg);
    n_errors++;
    if (n_errors <= 20) `uvm_error("SB/MISMATCH", msg)
  endfunction

  // ------------------------------------------------------------ end of test
  function void check_phase(uvm_phase phase);
    if (pending() != 0)
      `uvm_error("SB/UNCHECKED", $sformatf("%0d expectations never observed (%0d buffer bytes, %0d words, %0d external bytes)",
                 pending(), exp_host.num(), exp_word.num(), exp_ext.num()))
  endfunction

  function void report_phase(uvm_phase phase);
    `uvm_info("SB", $sformatf("checked %0d buffer bytes, %0d register words, %0d external bytes; %0d mismatches",
              n_bytes_ok, n_words_ok, n_ext_ok, n_errors), UVM_LOW)
  endfunction
endclass
