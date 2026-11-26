// Register model of npu_top (docs/programming_model.md, sections 1-3 and 7).
//
// Volatile fields (status, results, performance counters) are excluded from
// mirror checks; everything else is compared on every observed read, because
// the map runs with check_on_read and an explicit predictor (npu_env).
// CTRL.START and DMA_CTRL.START are write-1-to-pulse and read as 0; they are
// modelled as WO, which the compare also skips.

// Base of every register here: uvm_reg itself has no build(), so the model
// declares one that creates and configures the fields.
virtual class npu_reg extends uvm_reg;
  function new(string name);
    super.new(name, 32, UVM_NO_COVERAGE);
  endfunction

  pure virtual function void build();
endclass

typedef npu_reg npu_reg_q_t[$];

// A register that is one field, `value`, starting at bit 0.
class npu_reg_word extends npu_reg;
  `uvm_object_utils(npu_reg_word)

  rand uvm_reg_field value;

  local int unsigned   m_bits = 32;
  local string         m_access = "RW";
  local bit            m_volatile;
  local uvm_reg_data_t m_reset;

  function new(string name = "npu_reg_word");
    super.new(name);
  endfunction

  // Call before build().
  function void setup(int unsigned bits, string access, bit is_volatile = 0, uvm_reg_data_t reset = 0);
    m_bits = bits; m_access = access; m_volatile = is_volatile; m_reset = reset;
  endfunction

  virtual function void build();
    value = uvm_reg_field::type_id::create("value");
    value.configure(this, m_bits, 0, m_access, m_volatile, m_reset, 1, m_access == "RW", 0);
  endfunction
endclass

class npu_reg_ctrl extends npu_reg;
  `uvm_object_utils(npu_reg_ctrl)
  uvm_reg_field      start;
  rand uvm_reg_field irq_en;

  function new(string name = "npu_reg_ctrl");
    super.new(name);
  endfunction

  virtual function void build();
    start  = uvm_reg_field::type_id::create("start");
    irq_en = uvm_reg_field::type_id::create("irq_en");
    start.configure (this, 1, 0, "WO", 0, 0, 1, 0, 0);
    irq_en.configure(this, 1, 1, "RW", 0, 0, 1, 1, 0);
  endfunction
endclass

class npu_reg_status extends npu_reg;
  `uvm_object_utils(npu_reg_status)
  uvm_reg_field busy, done, err, host_conflict, cur_layer, err_code;

  function new(string name = "npu_reg_status");
    super.new(name);
  endfunction

  virtual function void build();
    busy          = uvm_reg_field::type_id::create("busy");
    done          = uvm_reg_field::type_id::create("done");
    err           = uvm_reg_field::type_id::create("err");
    host_conflict = uvm_reg_field::type_id::create("host_conflict");
    cur_layer     = uvm_reg_field::type_id::create("cur_layer");
    err_code      = uvm_reg_field::type_id::create("err_code");
    busy.configure         (this, 1,  0, "RO",  1, 0, 1, 0, 0);
    done.configure         (this, 1,  1, "W1C", 1, 0, 1, 0, 0);
    err.configure          (this, 1,  2, "W1C", 1, 0, 1, 0, 0);
    host_conflict.configure(this, 1,  3, "W1C", 1, 0, 1, 0, 0);
    cur_layer.configure    (this, 4,  8, "RO",  1, 0, 1, 0, 0);
    err_code.configure     (this, 8, 16, "RO",  1, 0, 1, 0, 0);
  endfunction
endclass

class npu_reg_result extends npu_reg;
  `uvm_object_utils(npu_reg_result)
  uvm_reg_field argmax, valid;

  function new(string name = "npu_reg_result");
    super.new(name);
  endfunction

  virtual function void build();
    argmax = uvm_reg_field::type_id::create("argmax");
    valid  = uvm_reg_field::type_id::create("valid");
    argmax.configure(this, 4, 0, "RO", 1, 0, 1, 0, 0);
    valid.configure (this, 1, 8, "RO", 1, 0, 1, 0, 0);
  endfunction
endclass

class npu_reg_dma_ctrl extends npu_reg;
  `uvm_object_utils(npu_reg_dma_ctrl)
  uvm_reg_field      start;
  rand uvm_reg_field dir, dst_sel;

  function new(string name = "npu_reg_dma_ctrl");
    super.new(name);
  endfunction

  virtual function void build();
    start   = uvm_reg_field::type_id::create("start");
    dir     = uvm_reg_field::type_id::create("dir");
    dst_sel = uvm_reg_field::type_id::create("dst_sel");
    start.configure  (this, 1, 0, "WO", 0, 0, 1, 0, 0);
    dir.configure    (this, 1, 1, "RW", 0, 0, 1, 1, 0);
    dst_sel.configure(this, 1, 2, "RW", 0, 0, 1, 1, 0);
  endfunction
endclass

class npu_reg_dma_status extends npu_reg;
  `uvm_object_utils(npu_reg_dma_status)
  uvm_reg_field busy, done, err, err_code;

  function new(string name = "npu_reg_dma_status");
    super.new(name);
  endfunction

  virtual function void build();
    busy     = uvm_reg_field::type_id::create("busy");
    done     = uvm_reg_field::type_id::create("done");
    err      = uvm_reg_field::type_id::create("err");
    err_code = uvm_reg_field::type_id::create("err_code");
    busy.configure    (this, 1, 0, "RO",  1, 0, 1, 0, 0);
    done.configure    (this, 1, 1, "W1C", 1, 0, 1, 0, 0);
    err.configure     (this, 1, 2, "W1C", 1, 0, 1, 0, 0);
    err_code.configure(this, 8, 8, "RO",  1, 0, 1, 0, 0);
  endfunction
endclass

// ---------------------------------------------------------------- descriptor words
class npu_reg_desc_w0 extends npu_reg;
  `uvm_object_utils(npu_reg_desc_w0)
  rand uvm_reg_field op, k, pad, stride_m1;

  function new(string name = "npu_reg_desc_w0");
    super.new(name);
  endfunction

  virtual function void build();
    op        = uvm_reg_field::type_id::create("op");
    k         = uvm_reg_field::type_id::create("k");
    pad       = uvm_reg_field::type_id::create("pad");
    stride_m1 = uvm_reg_field::type_id::create("stride_m1");
    op.configure       (this, 4,  0, "RW", 0, 0, 1, 1, 0);
    k.configure        (this, 4,  4, "RW", 0, 0, 1, 1, 0);
    pad.configure      (this, 4,  8, "RW", 0, 0, 1, 1, 0);
    stride_m1.configure(this, 2, 12, "RW", 0, 0, 1, 1, 0);
  endfunction
endclass

// Two 16-bit fields: W1 (in_h, in_w) and W2 (c_in, c_out).
class npu_reg_desc_pair extends npu_reg;
  `uvm_object_utils(npu_reg_desc_pair)
  rand uvm_reg_field lo, hi;

  function new(string name = "npu_reg_desc_pair");
    super.new(name);
  endfunction

  virtual function void build();
    lo = uvm_reg_field::type_id::create("lo");
    hi = uvm_reg_field::type_id::create("hi");
    lo.configure(this, 16,  0, "RW", 0, 0, 1, 1, 0);
    hi.configure(this, 16, 16, "RW", 0, 0, 1, 1, 0);
  endfunction
endclass

class npu_reg_desc_w9 extends npu_reg;
  `uvm_object_utils(npu_reg_desc_w9)
  rand uvm_reg_field z_in, z_out, act_min, act_max;

  function new(string name = "npu_reg_desc_w9");
    super.new(name);
  endfunction

  virtual function void build();
    z_in    = uvm_reg_field::type_id::create("z_in");
    z_out   = uvm_reg_field::type_id::create("z_out");
    act_min = uvm_reg_field::type_id::create("act_min");
    act_max = uvm_reg_field::type_id::create("act_max");
    z_in.configure   (this, 8,  0, "RW", 0, 0, 1, 1, 0);
    z_out.configure  (this, 8,  8, "RW", 0, 0, 1, 1, 0);
    act_min.configure(this, 8, 16, "RW", 0, 0, 1, 1, 0);
    act_max.configure(this, 8, 24, "RW", 0, 0, 1, 1, 0);
  endfunction
endclass

// One descriptor: the ten stored words W0..W9 (W10..W15 are reserved and read as 0).
class npu_desc_rf extends uvm_reg_file;
  `uvm_object_utils(npu_desc_rf)

  rand npu_reg_desc_w0   w0;
  rand npu_reg_desc_pair shape;      // W1: lo = in_h, hi = in_w
  rand npu_reg_desc_pair chans;      // W2: lo = c_in, hi = c_out
  rand npu_reg_word      in_base, out_base, w_base, b_base, rq_mult;   // W3..W7
  rand npu_reg_word      rq_shift;   // W8, 6 bits
  rand npu_reg_desc_w9   w9;

  function new(string name = "npu_desc_rf");
    super.new(name);
  endfunction

  function void build(uvm_reg_block blk);
    w0       = npu_reg_desc_w0::type_id::create("w0");
    shape    = npu_reg_desc_pair::type_id::create("shape");
    chans    = npu_reg_desc_pair::type_id::create("chans");
    in_base  = npu_reg_word::type_id::create("in_base");
    out_base = npu_reg_word::type_id::create("out_base");
    w_base   = npu_reg_word::type_id::create("w_base");
    b_base   = npu_reg_word::type_id::create("b_base");
    rq_mult  = npu_reg_word::type_id::create("rq_mult");
    rq_shift = npu_reg_word::type_id::create("rq_shift");
    w9       = npu_reg_desc_w9::type_id::create("w9");
    rq_shift.setup(6, "RW");
    begin
      npu_reg_q_t ws = get_words();
      foreach (ws[i]) begin
        ws[i].configure(blk, this, "");
        ws[i].build();
      end
    end
  endfunction

  function void add_to_map(uvm_reg_map mp, uvm_reg_addr_t base);
    npu_reg_q_t ws = get_words();
    foreach (ws[i]) mp.add_reg(ws[i], base + uvm_reg_addr_t'(4 * i), "RW");
  endfunction

  // Desired values of W0..W9 from the stored words; update() then writes the
  // words whose desired value differs from the mirror.
  function void set_words(const ref bit [31:0] w[10]);
    npu_reg_q_t ws = get_words();
    foreach (ws[i]) ws[i].set(w[i]);
  endfunction

  task update_words(output uvm_status_e status, input uvm_sequence_base parent);
    npu_reg_q_t ws = get_words();
    status = UVM_IS_OK;
    foreach (ws[i]) begin
      uvm_status_e st;
      ws[i].update(st, UVM_FRONTDOOR, null, parent);
      if (st != UVM_IS_OK) status = st;
    end
  endtask

  // W0..W9 in address order.
  function npu_reg_q_t get_words();
    return '{w0, shape, chans, in_base, out_base, w_base, b_base, rq_mult, rq_shift, w9};
  endfunction
endclass

class npu_reg_block extends uvm_reg_block;
  `uvm_object_utils(npu_reg_block)

  localparam int unsigned NUM_DESC = 16;
  localparam bit [31:0]   NPU_ID   = 32'h4E50_5501;

  npu_reg_word            id;
  rand npu_reg_ctrl       ctrl;
  npu_reg_status          status;
  rand npu_reg_word       layer_cnt;
  npu_reg_result          result;
  npu_reg_word            logits[16];
  npu_reg_word            perf_total, perf_array_busy;
  npu_reg_word            perf_layer[16];
  rand npu_reg_dma_ctrl   dma_ctrl;
  rand npu_reg_word       dma_src, dma_dst, dma_len;
  npu_reg_dma_status      dma_status;
  rand npu_desc_rf        desc[NUM_DESC];
  uvm_mem                 abuf, wbuf;

  function new(string name = "npu_reg_block");
    super.new(name, UVM_NO_COVERAGE);
  endfunction

  local function npu_reg_word word(string name, uvm_reg_addr_t offset, int unsigned bits, string access,
                                   bit is_volatile = 0, uvm_reg_data_t reset = 0);
    npu_reg_word r = npu_reg_word::type_id::create(name, , get_full_name());
    r.setup(bits, access, is_volatile, reset);
    r.configure(this, null, "");
    r.build();
    default_map.add_reg(r, offset, access == "RO" ? "RO" : "RW");
    return r;
  endfunction

  local function void add(npu_reg r, uvm_reg_addr_t offset, string rights = "RW");
    r.configure(this, null, "");
    r.build();
    default_map.add_reg(r, offset, rights);
  endfunction

  virtual function void build();
    default_map = create_map("axil_map", 0, 4, UVM_LITTLE_ENDIAN, 1);

    id         = word("id", 'h000, 32, "RO", 0, NPU_ID);
    ctrl       = npu_reg_ctrl::type_id::create("ctrl", , get_full_name());
    status     = npu_reg_status::type_id::create("status", , get_full_name());
    result     = npu_reg_result::type_id::create("result", , get_full_name());
    dma_ctrl   = npu_reg_dma_ctrl::type_id::create("dma_ctrl", , get_full_name());
    dma_status = npu_reg_dma_status::type_id::create("dma_status", , get_full_name());
    add(ctrl, 'h004);
    add(status, 'h008);
    layer_cnt  = word("layer_cnt", 'h00C, 5, "RW");
    add(result, 'h010, "RO");
    foreach (logits[i])     logits[i]     = word($sformatf("logits_%0d", i), 'h020 + 4 * i, 32, "RO", 1);
    perf_total      = word("perf_total", 'h060, 32, "RO", 1);
    perf_array_busy = word("perf_array_busy", 'h064, 32, "RO", 1);
    foreach (perf_layer[i]) perf_layer[i] = word($sformatf("perf_layer_%0d", i), 'h080 + 4 * i, 32, "RO", 1);
    add(dma_ctrl, 'h0C0);
    dma_src    = word("dma_src", 'h0C4, 32, "RW");
    dma_dst    = word("dma_dst", 'h0C8, 20, "RW");
    dma_len    = word("dma_len", 'h0CC, 20, "RW");
    add(dma_status, 'h0D0);
    foreach (desc[i]) begin
      desc[i] = npu_desc_rf::type_id::create($sformatf("desc_%0d", i), , get_full_name());
      desc[i].configure(this, null, "");
      desc[i].build(this);
      desc[i].add_to_map(default_map, 'h100 + 64 * i);
    end

    abuf = new("abuf", 32768 / 4, 32, "RW", UVM_NO_COVERAGE);
    wbuf = new("wbuf", 131072 / 4, 32, "RW", UVM_NO_COVERAGE);
    abuf.configure(this);
    wbuf.configure(this);
    default_map.add_mem(abuf, 'h10000, "RW");
    default_map.add_mem(wbuf, 'h20000, "RW");
  endfunction
endclass
