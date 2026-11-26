// Reactive AXI4 slave memory for the DMA master: INCR bursts of 64-bit beats,
// one transaction per direction at a time, random ready / valid stalls, and an
// address window that answers SLVERR. Every completed burst is published, with
// its data, on ap.

class axi_burst_item extends uvm_sequence_item;
  bit         write;
  bit [31:0]  addr;
  bit [7:0]   len;                  // beats - 1
  bit [1:0]   resp;
  bit [63:0]  data[$];
  bit [7:0]   strb[$];

  `uvm_object_utils(axi_burst_item)

  function new(string name = "axi_burst_item");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("%s addr=%h beats=%0d resp=%0d", write ? "WR" : "RD", addr, len + 1, resp);
  endfunction
endclass

class axi_slave_mem extends uvm_component;
  `uvm_component_utils(axi_slave_mem)

  virtual axi_if vif;
  uvm_analysis_port #(axi_burst_item) ap;

  bit [7:0]    mem[bit [31:0]];     // sparse; unwritten bytes read as default_byte()
  int unsigned max_stall = 3;
  int unsigned handshake_limit = 10_000;  // watchdog inside a burst (idle waits for AW / AR are unlimited)
  bit [31:0]   err_lo = '1, err_hi = '1;   // [err_lo, err_hi): bursts starting here get SLVERR

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db #(virtual axi_if)::get(this, "", "vif", vif))
      `uvm_fatal("AXI/NOVIF", "no virtual axi_if")
  endfunction

  // ---------------------------------------------------------------- backdoor
  static function bit [7:0] default_byte(bit [31:0] a);
    return a[7:0] ^ a[15:8] ^ a[23:16] ^ 8'h5A;
  endfunction

  function bit [7:0] read_byte(bit [31:0] a);
    return mem.exists(a) ? mem[a] : default_byte(a);
  endfunction

  function void write_byte(bit [31:0] a, bit [7:0] v);
    mem[a] = v;
  endfunction

  function bit in_err(bit [31:0] a);
    return a >= err_lo && a < err_hi;
  endfunction

  // AXI4 rules the DMA must keep (docs/programming_model.md section 7): INCR
  // bursts of 64-bit beats, at most 16 beats, aligned, never across 4 KB.
  function void check_burst(string ch, bit [31:0] addr, bit [7:0] len, bit [2:0] size, bit [1:0] burst);
    if (size != 3'd3 || burst != 2'b01)
      `uvm_error("AXI/SHAPE", $sformatf("%s: size %0d burst %0d, expected 64-bit INCR", ch, size, burst))
    if (len > 8'd15)
      `uvm_error("AXI/LEN", $sformatf("%s: %0d beats at %h, at most 16 allowed", ch, len + 1, addr))
    if (addr[2:0] != 3'b000)
      `uvm_error("AXI/ALIGN", $sformatf("%s: address %h not 8-byte aligned", ch, addr))
    if (addr[31:12] != (addr + 32'(8 * (int'(len) + 1)) - 32'd1) >> 12)
      `uvm_error("AXI/4K", $sformatf("%s: %0d beats at %h cross a 4 KB boundary", ch, len + 1, addr))
  endfunction

  // ---------------------------------------------------------------- run
  task run_phase(uvm_phase phase);
    vif.slave_idle();
    @(posedge vif.rst_n);
    fork
      write_loop();
      read_loop();
    join
  endtask

  // Random idle cycles. A ready / valid that was just dropped is only
  // dropped when the stall is not empty, so a signal never gets two drives
  // in the same time step.
  function automatic int unsigned stall_cycles();
    return $urandom_range(max_stall);
  endfunction

  task automatic tick(ref int unsigned n, input string what, input axi_burst_item t);
    @(vif.slv_cb);
    if (++n > handshake_limit)
      `uvm_fatal("AXI/HANG", $sformatf("no %s after %0d cycles (%s)", what, handshake_limit, t.convert2string()))
  endtask

  task automatic write_loop();
    forever begin
      axi_burst_item t = axi_burst_item::type_id::create("wr");
      bit err;
      repeat (stall_cycles()) @(vif.slv_cb);
      vif.slv_cb.awready <= 1'b1;
      do @(vif.slv_cb); while (!vif.slv_cb.awvalid);
      vif.slv_cb.awready <= 1'b0;
      t.write = 1; t.addr = vif.slv_cb.awaddr; t.len = vif.slv_cb.awlen;
      check_burst("AW", t.addr, t.len, vif.slv_cb.awsize, vif.slv_cb.awburst);
      err = in_err(t.addr);
      for (int b = 0; b <= int'(t.len); b++) begin
        bit [31:0] a = t.addr + 32'(8 * b);
        int unsigned n = stall_cycles();
        if (n > 0) begin
          vif.slv_cb.wready <= 1'b0;
          repeat (n) @(vif.slv_cb);
        end
        vif.slv_cb.wready <= 1'b1;
        begin
          int unsigned w = 0;
          do tick(w, "WVALID", t); while (!vif.slv_cb.wvalid);
        end
        t.data.push_back(vif.slv_cb.wdata);
        t.strb.push_back(vif.slv_cb.wstrb);
        if (vif.slv_cb.wlast != (b == int'(t.len)))
          `uvm_error("AXI/WLAST", $sformatf("WLAST=%0b on beat %0d of %0d", vif.slv_cb.wlast, b, t.len + 1))
        if (!err)
          for (int i = 0; i < 8; i++)
            if (vif.slv_cb.wstrb[i]) mem[a + 32'(i)] = vif.slv_cb.wdata[8 * i +: 8];
      end
      vif.slv_cb.wready <= 1'b0;
      t.resp = err ? 2'b10 : 2'b00;
      repeat (stall_cycles()) @(vif.slv_cb);
      vif.slv_cb.bresp  <= t.resp;
      vif.slv_cb.bvalid <= 1'b1;
      begin
        int unsigned w = 0;
        do tick(w, "BREADY", t); while (!vif.slv_cb.bready);
      end
      vif.slv_cb.bvalid <= 1'b0;
      ap.write(t);
    end
  endtask

  task automatic read_loop();
    forever begin
      axi_burst_item t = axi_burst_item::type_id::create("rd");
      bit err;
      repeat (stall_cycles()) @(vif.slv_cb);
      vif.slv_cb.arready <= 1'b1;
      do @(vif.slv_cb); while (!vif.slv_cb.arvalid);
      vif.slv_cb.arready <= 1'b0;
      t.write = 0; t.addr = vif.slv_cb.araddr; t.len = vif.slv_cb.arlen;
      check_burst("AR", t.addr, t.len, vif.slv_cb.arsize, vif.slv_cb.arburst);
      err = in_err(t.addr);
      for (int b = 0; b <= int'(t.len); b++) begin
        bit [31:0] a = t.addr + 32'(8 * b);
        bit [63:0] d;
        int unsigned n = stall_cycles();
        for (int i = 0; i < 8; i++) d[8 * i +: 8] = read_byte(a + 32'(i));
        if (n > 0) begin
          vif.slv_cb.rvalid <= 1'b0;
          vif.slv_cb.rlast  <= 1'b0;
          repeat (n) @(vif.slv_cb);
        end
        vif.slv_cb.rdata  <= d;
        vif.slv_cb.rresp  <= err ? 2'b10 : 2'b00;
        vif.slv_cb.rlast  <= (b == int'(t.len));
        vif.slv_cb.rvalid <= 1'b1;
        begin
          int unsigned w = 0;
          do tick(w, "RREADY", t); while (!vif.slv_cb.rready);
        end
        t.data.push_back(d);
      end
      vif.slv_cb.rvalid <= 1'b0;
      vif.slv_cb.rlast  <= 1'b0;
      t.resp = err ? 2'b10 : 2'b00;
      ap.write(t);
    end
  endtask
endclass
