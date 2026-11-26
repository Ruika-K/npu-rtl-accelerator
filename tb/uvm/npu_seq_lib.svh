// Sequences. All run on the AXI4-Lite sequencer: registers go through the
// register model, buffer contents through byte-strobed AXI4-Lite accesses or
// the DMA. Data checks are posted to the scoreboard before the read that
// returns the data; status checks that only the sequence can interpret
// (which error, which layer) are made here.

class npu_base_seq extends uvm_sequence #(axil_item);
  `uvm_object_utils(npu_base_seq)

  localparam bit [31:0] ABUF = 32'h1_0000, WBUF = 32'h2_0000;
  localparam bit [31:0] A_RESULT = 32'h010, A_LOGITS = 32'h020;
  localparam int unsigned POLL_LIMIT = 200_000;

  npu_env_cfg    cfg;
  npu_reg_block  regs;
  npu_scoreboard sb;

  function new(string name = "npu_base_seq");
    super.new(name);
  endfunction

  virtual task pre_start();
    if (!uvm_config_db #(npu_env_cfg)::get(m_sequencer, "", "cfg", cfg))
      `uvm_fatal("SEQ/NOCFG", "no npu_env_cfg")
    regs = cfg.regs;
    sb   = cfg.sb;
  endtask

  // ------------------------------------------------------------ raw AXI4-Lite
  task wr_resp(bit [31:0] addr, bit [31:0] data, bit [3:0] strb, output bit [1:0] resp);
    axil_item t = axil_item::type_id::create("wr");
    start_item(t);
    if (!t.randomize() with { write == 1; addr == local::addr; data == local::data; strb == local::strb; })
      `uvm_fatal("SEQ/RAND", "randomize failed")
    finish_item(t);
    resp = t.resp;
  endtask

  task rd_resp(bit [31:0] addr, output bit [31:0] data, output bit [1:0] resp);
    axil_item t = axil_item::type_id::create("rd");
    start_item(t);
    if (!t.randomize() with { write == 0; addr == local::addr; })
      `uvm_fatal("SEQ/RAND", "randomize failed")
    finish_item(t);
    data = t.data;
    resp = t.resp;
  endtask

  task wr(bit [31:0] addr, bit [31:0] data, bit [3:0] strb = 4'hF);
    bit [1:0] resp;
    wr_resp(addr, data, strb, resp);
    if (resp != 2'b00) `uvm_error("SEQ/RESP", $sformatf("write %h: response %0d", addr, resp))
  endtask

  task rd(bit [31:0] addr, output bit [31:0] data);
    bit [1:0] resp;
    rd_resp(addr, data, resp);
    if (resp != 2'b00) `uvm_error("SEQ/RESP", $sformatf("read %h: response %0d", addr, resp))
  endtask

  // Bytes at any byte address, as 32-bit aligned writes with byte strobes.
  task write_bytes(bit [31:0] addr, const ref npu_bytes_t b);
    bit [31:0] a = {addr[31:2], 2'b00};
    while (a < addr + 32'(b.size())) begin
      bit [31:0] data = '0;
      bit [3:0]  strb = '0;
      for (int i = 0; i < 4; i++) begin
        longint unsigned j = longint'(a) + i - longint'(addr);
        if (a + 32'(i) >= addr && j < b.size()) begin
          data[8 * i +: 8] = b[j];
          strb[i] = 1'b1;
        end
      end
      wr(a, data, strb);
      a += 4;
    end
  endtask

  // Reads every word that covers [addr, addr + n); the scoreboard compares.
  task read_region(bit [31:0] addr, int unsigned n);
    bit [31:0] d;
    for (bit [31:0] a = {addr[31:2], 2'b00}; a < addr + n; a += 4) rd(a, d);
  endtask

  // ------------------------------------------------------------ register model
  task reg_wr(uvm_reg r, uvm_reg_data_t v);
    uvm_status_e st;
    r.write(st, v, UVM_FRONTDOOR, null, this);
    if (st != UVM_IS_OK) `uvm_error("SEQ/REG", {"write failed: ", r.get_name()})
  endtask

  task reg_rd(uvm_reg r, output uvm_reg_data_t v);
    uvm_status_e st;
    r.read(st, v, UVM_FRONTDOOR, null, this);
    if (st != UVM_IS_OK) `uvm_error("SEQ/REG", {"read failed: ", r.get_name()})
  endtask

  task load_desc(int idx, bit [31:0] w[10]);
    uvm_status_e st;
    regs.desc[idx].set_words(w);
    regs.desc[idx].update_words(st, this);
    if (st != UVM_IS_OK) `uvm_error("SEQ/REG", $sformatf("descriptor %0d update failed", idx))
  endtask

  task start_run(bit irq_en = 0);
    reg_wr(regs.ctrl, {irq_en, 1'b1});
  endtask

  // Polls STATUS until BUSY clears; returns the STATUS word.
  task wait_idle(output bit [31:0] status);
    uvm_reg_data_t v;
    int unsigned n = 0;
    do begin
      reg_rd(regs.status, v);
      n++;
    end while (v[0] && n < POLL_LIMIT);
    if (v[0]) `uvm_fatal("SEQ/HANG", "STATUS.BUSY never cleared")
    status = 32'(v);
  endtask

  task clear_status();
    reg_wr(regs.status, 32'hE);
  endtask

  // One DMA transfer; returns the DMA_STATUS word once BUSY clears.
  task dma(bit store, bit sel, bit [31:0] ext, bit [31:0] off, bit [31:0] len, output bit [31:0] status);
    uvm_reg_data_t v;
    int unsigned n = 0;
    reg_wr(regs.dma_src, ext);
    reg_wr(regs.dma_dst, off);
    reg_wr(regs.dma_len, len);
    reg_wr(regs.dma_ctrl, {sel, store, 1'b1});
    do begin
      reg_rd(regs.dma_status, v);
      n++;
    end while (v[0] && n < POLL_LIMIT);
    if (v[0]) `uvm_fatal("SEQ/HANG", "DMA_STATUS.BUSY never cleared")
    status = 32'(v);
    reg_wr(regs.dma_status, 32'h6);
  endtask

  function void ext_write(bit [31:0] addr, const ref npu_bytes_t b);
    foreach (b[i]) cfg.ext_mem.write_byte(addr + 32'(i), b[i]);
  endfunction

  static function void rand_bytes(int unsigned n, ref npu_bytes_t b);
    b.delete();
    repeat (n) b.push_back(8'($urandom));
  endfunction

  function void expect_status(bit [31:0] st, bit done, int err_code, int cur_layer, string what,
                             bit host_conflict = 0);
    bit [31:0] exp = {8'h0, 8'(err_code), 4'h0, 4'(cur_layer), 4'h0, host_conflict, err_code != 0, done, 1'b0};
    if (st !== exp) `uvm_error("SEQ/STATUS", $sformatf("%s: STATUS %h, expected %h", what, st, exp))
  endfunction
endclass

// ---------------------------------------------------------------------------
// Single layers from gen_layer_vectors.py. Each case runs from a random
// descriptor slot, behind NOP descriptors, so programs of 1..16 entries and
// NOPs are exercised too.
class npu_layer_seq extends npu_base_seq;
  `uvm_object_utils(npu_layer_seq)

  string path;
  int unsigned n_run;

  function new(string name = "npu_layer_seq");
    super.new(name);
  endfunction

  task body();
    npu_layer_case cases[$];
    bit [31:0] nop[10] = '{default: 0};
    if (path == "" && !$value$plusargs("layer_vectors=%s", path)) `uvm_fatal("SEQ/ARG", "+layer_vectors=<file> missing")
    npu_vectors::read_layer_cases(path, cases);
    foreach (cases[i]) begin
      npu_layer_case c = cases[i];
      bit [31:0] w[10], st;
      int slot = $urandom_range(15);
      write_bytes(ABUF + 32'(c.in_base), c.in_b);
      if (c.w_b.size() > 0) write_bytes(WBUF + 32'(c.w_base), c.w_b);
      if (c.b_b.size() > 0) write_bytes(WBUF + 32'(c.b_base), c.b_b);
      for (int s = 0; s < slot; s++) load_desc(s, nop);
      c.pack(w);
      load_desc(slot, w);
      reg_wr(regs.layer_cnt, slot + 1);
      start_run();
      wait_idle(st);
      expect_status(st, 1, 0, slot, c.convert2string());
      if (c.op == 4) begin                          // ARGMAX: the result and the logits it read
        uvm_reg_data_t v;
        sb.expect_word(A_RESULT, 32'h100 | 32'(c.out_b[0]));
        foreach (c.in_b[j]) sb.expect_word(A_LOGITS + 32'(4 * j), 32'(signed'(c.in_b[j])));
        reg_rd(regs.result, v);
        foreach (c.in_b[j]) reg_rd(regs.logits[j], v);
      end else begin
        sb.expect_host_bytes(ABUF + 32'(c.out_base), c.out_b);
        read_region(ABUF + 32'(c.out_base), c.out_b.size());
      end
      clear_status();
      n_run++;
    end
    `uvm_info("SEQ/LAYER", $sformatf("%0d layer cases", n_run), UVM_LOW)
  endtask
endclass

// ---------------------------------------------------------------------------
// Descriptor checks from gen_desc_vectors.py: every vector with an error code
// must stop the run with that code at that layer; NOP vectors must finish.
// Then LAYER_CNT out of range.
class npu_desc_err_seq extends npu_base_seq;
  `uvm_object_utils(npu_desc_err_seq)

  string path;

  function new(string name = "npu_desc_err_seq");
    super.new(name);
  endfunction

  task body();
    npu_desc_vector_t vecs[$];
    int unsigned n_err, n_ok;
    bit [31:0] st;
    if (path == "" && !$value$plusargs("desc_vectors=%s", path)) `uvm_fatal("SEQ/ARG", "+desc_vectors=<file> missing")
    npu_vectors::read_desc_vectors(path, vecs);
    foreach (vecs[i]) begin
      bit [3:0] op = vecs[i].w[0][3:0];
      if (vecs[i].err_code == 0 && op != 0) continue;     // valid compute layers: npu_layer_seq
      load_desc(0, vecs[i].w);
      reg_wr(regs.layer_cnt, 1);
      start_run();
      wait_idle(st);
      expect_status(st, vecs[i].err_code == 0, vecs[i].err_code, 0, $sformatf("descriptor vector %0d", i));
      clear_status();
      if (vecs[i].err_code != 0) n_err++; else n_ok++;
    end
    for (int cnt = 0; cnt < 32; cnt += 17) begin            // 0 and 17: ERR_CODE 4 before any descriptor
      reg_wr(regs.layer_cnt, cnt);
      start_run();
      wait_idle(st);
      expect_status(st, 0, 4, 0, $sformatf("LAYER_CNT %0d", cnt));
      clear_status();
    end
    host_errors();
    `uvm_info("SEQ/DESC", $sformatf("%0d error vectors, %0d NOP vectors", n_err, n_ok), UVM_LOW)
  endtask

  // Unmapped and misaligned accesses answer SLVERR and change nothing (the
  // register model checks the registers they would have hit on later reads).
  task host_errors();
    bit [31:0] addrs[$] = '{32'h0_8000, 32'h1_8000, 32'h4_0000, 32'hF_FFFC, 32'h0_0006, 32'h0_0101, 32'h1_0002,
                            32'h2_0003};
    bit [31:0] d;
    bit [1:0]  resp;
    foreach (addrs[i]) begin
      rd_resp(addrs[i], d, resp);
      if (resp != 2'b10) `uvm_error("SEQ/SLVERR", $sformatf("read %h: response %0d, expected SLVERR", addrs[i], resp))
      wr_resp(addrs[i], 32'hFFFF_FFFF, 4'hF, resp);
      if (resp != 2'b10) `uvm_error("SEQ/SLVERR", $sformatf("write %h: response %0d, expected SLVERR", addrs[i], resp))
    end
  endtask
endclass

// ---------------------------------------------------------------------------
// DMA: random loads into both buffers (read back over AXI4-Lite), stores of
// the same data (checked on the AXI4 write channel), bursts that end on and
// cross 4 KB boundaries, parameter errors, bus errors, and host accesses that
// collide with a running transfer.
class npu_dma_seq extends npu_base_seq;
  `uvm_object_utils(npu_dma_seq)

  int unsigned n_transfers = 24;
  localparam bit [31:0] BUFS[2] = '{ABUF, WBUF};

  function new(string name = "npu_dma_seq");
    super.new(name);
  endfunction

  task body();
    bit [31:0] st, d;
    bit [1:0]  resp;
    uvm_reg_data_t v;
    void'($value$plusargs("dma_transfers=%d", n_transfers));

    for (int i = 0; i < n_transfers; i++) begin
      npu_bytes_t b;
      bit        sel  = $urandom_range(1);
      bit [31:0] size = sel ? 32'h2_0000 : 32'h8000;
      bit [31:0] len  = 32'(8 * $urandom_range(1, 300));
      bit [31:0] off  = 32'(8 * $urandom_range(0, (size - len) / 8));
      // External addresses: anywhere, or just below a 4 KB boundary so bursts split there.
      bit [31:0] src  = (i % 3 == 0) ? 32'h4000 * $urandom_range(1, 30) - 32'(8 * $urandom_range(1, 20))
                                     : 32'(8 * $urandom_range(0, 32'h1_0000));
      bit [31:0] dst  = 32'h8_0000 + 32'h1000 * i + 32'h1000 - 32'(8 * $urandom_range(1, 4));
      rand_bytes(len, b);
      ext_write(src, b);
      dma(0, sel, src, off, len, st);
      if (st !== 32'h2) `uvm_error("SEQ/DMA", $sformatf("load %0d: DMA_STATUS %h", i, st))
      sb.expect_host_bytes((sel ? WBUF : ABUF) + off, b);
      read_region((sel ? WBUF : ABUF) + off, len);
      sb.expect_ext_bytes(dst, b);
      dma(1, sel, dst, off, len, st);
      if (st !== 32'h2) `uvm_error("SEQ/DMA", $sformatf("store %0d: DMA_STATUS %h", i, st))
    end

    // Parameter errors: misaligned, zero length, past the end of the buffer.
    dma(0, 0, 32'h4, 0, 8, st);                if (st !== 32'h204) `uvm_error("SEQ/DMA", $sformatf("misaligned src: %h", st))
    dma(0, 0, 0, 32'h4, 8, st);                if (st !== 32'h204) `uvm_error("SEQ/DMA", $sformatf("misaligned dst: %h", st))
    dma(0, 1, 0, 0, 0, st);                    if (st !== 32'h204) `uvm_error("SEQ/DMA", $sformatf("zero length: %h", st))
    dma(1, 0, 0, 32'h7FF8, 16, st);            if (st !== 32'h204) `uvm_error("SEQ/DMA", $sformatf("past the end: %h", st))

    // Bus errors in both directions.
    cfg.ext_mem.err_lo = 32'h7_0000;
    cfg.ext_mem.err_hi = 32'h7_1000;
    dma(0, 0, 32'h7_0000, 0, 256, st);         if (st !== 32'h304) `uvm_error("SEQ/DMA", $sformatf("load SLVERR: %h", st))
    dma(1, 1, 32'h7_0800, 0, 64, st);          if (st !== 32'h304) `uvm_error("SEQ/DMA", $sformatf("store SLVERR: %h", st))
    cfg.ext_mem.err_lo = '1;
    cfg.ext_mem.err_hi = '1;

    // Host accesses to either buffer during a long transfer are rejected and set HOST_CONFLICT.
    reg_wr(regs.dma_src, 0);
    reg_wr(regs.dma_dst, 0);
    reg_wr(regs.dma_len, 32'h1_0000);
    reg_wr(regs.dma_ctrl, 32'h5);
    foreach (BUFS[i]) begin
      rd_resp(BUFS[i] + 32'h100, d, resp);
      if (resp != 2'b10) `uvm_error("SEQ/CONFLICT", $sformatf("read %h during DMA was not rejected", BUFS[i]))
      wr_resp(BUFS[i] + 32'h100, 32'hDEAD_BEEF, 4'hF, resp);
      if (resp != 2'b10) `uvm_error("SEQ/CONFLICT", $sformatf("write %h during DMA was not rejected", BUFS[i]))
    end
    do reg_rd(regs.dma_status, v); while (v[0]);
    reg_rd(regs.status, v);
    if (!v[3]) `uvm_error("SEQ/CONFLICT", "HOST_CONFLICT not set")
    clear_status();
    reg_wr(regs.dma_status, 32'h6);
  endtask
endclass

// ---------------------------------------------------------------------------
// Whole network from a quantize.py program: weights and images through the
// DMA, descriptors through the register model, completion by interrupt.
// RESULT must match the integer model's prediction for every image; for
// images with TRACE lines every layer output is checked as well (stored to
// external memory by the DMA when aligned, else read over AXI4-Lite), and
// the logits ARGMAX read.
class npu_lenet_seq extends npu_base_seq;
  `uvm_object_utils(npu_lenet_seq)

  localparam bit [31:0] EXT_WEIGHTS = 32'h0_0000, EXT_IMAGE = 32'h4_0000, EXT_TRACE = 32'h8_0000;

  string       path;
  int unsigned max_images = 10;
  int unsigned n_images, n_agree, n_correct, n_traces;

  function new(string name = "npu_lenet_seq");
    super.new(name);
  endfunction

  task body();
    npu_program p;
    npu_bytes_t wb;
    bit [31:0]  st;
    int         argmax_in = -1, argmax_n = 0;
    if (path == "" && !$value$plusargs("program=%s", path)) `uvm_fatal("SEQ/ARG", "+program=<file> missing")
    void'($value$plusargs("images=%d", max_images));
    p = npu_vectors::read_program(path);

    // Weights: one DMA of the whole blob (padded to 8 bytes).
    foreach (p.weights[i]) for (int k = 0; k < 4; k++) wb.push_back(p.weights[i][8 * k +: 8]);
    while (wb.size() % 8 != 0) wb.push_back(8'h00);
    ext_write(EXT_WEIGHTS, wb);
    dma(0, 1, EXT_WEIGHTS, 0, wb.size(), st);
    if (st !== 32'h2) `uvm_fatal("SEQ/DMA", $sformatf("weight load: DMA_STATUS %h", st))

    for (int l = 0; l < p.n_layers; l++) begin
      bit [31:0] w[10];
      foreach (w[k]) w[k] = p.desc[l][k];
      load_desc(l, w);
      if (w[0][3:0] == 4'd4) begin argmax_in = int'(w[3]); argmax_n = int'(w[2][15:0]); end
    end
    reg_wr(regs.layer_cnt, p.n_layers);

    foreach (p.images[n]) begin
      npu_image im = p.images[n];
      npu_bytes_t px = im.pixels;
      uvm_reg_data_t v;
      if (n_images >= max_images) break;
      while (px.size() % 8 != 0) px.push_back(8'h00);
      ext_write(EXT_IMAGE, px);
      if (im.in_base % 8 == 0) begin
        dma(0, 0, EXT_IMAGE, im.in_base, px.size(), st);
        if (st !== 32'h2) `uvm_error("SEQ/DMA", $sformatf("image %0d load: DMA_STATUS %h", im.idx, st))
      end else write_bytes(ABUF + 32'(im.in_base), im.pixels);

      start_run(1);
      if (n == 0) begin                       // the buffers belong to the NPU while it runs
        bit [31:0] d;
        bit [1:0]  resp;
        rd_resp(ABUF, d, resp);
        if (resp != 2'b10) `uvm_error("SEQ/CONFLICT", "buffer read while busy was not rejected")
      end
      wait_irq();
      reg_rd(regs.status, v);
      expect_status(32'(v), 1, 0, p.n_layers - 1, $sformatf("image %0d", im.idx), n == 0);
      sb.expect_word(A_RESULT, 32'h100 | 32'(im.expected));
      reg_rd(regs.result, v);
      n_images++;
      if (int'(v[3:0]) == im.expected) n_agree++;
      if (int'(v[3:0]) == im.label) n_correct++;

      foreach (im.trace_bytes[t]) check_trace(im, t, argmax_in, argmax_n);
      reg_wr(regs.status, 32'hA);             // DONE and HOST_CONFLICT; irq_o drops
    end
    `uvm_info("SEQ/LENET", $sformatf("%0d images: %0d agree with the integer model, %0d match the label, %0d traces",
              n_images, n_agree, n_correct, n_traces), UVM_LOW)
  endtask

  task check_trace(npu_image im, int t, int argmax_in, int argmax_n);
    int         base = im.trace_base[t];
    npu_bytes_t b    = im.trace_bytes[t];
    bit [31:0]  st;
    uvm_reg_data_t v;
    if (base % 8 == 0) begin
      bit [31:0] ext = EXT_TRACE + 32'(base);
      bit [31:0] len = 32'((b.size() + 7) / 8 * 8);
      sb.expect_ext_bytes(ext, b);
      dma(1, 0, ext, base, len, st);
      if (st !== 32'h2) `uvm_error("SEQ/DMA", $sformatf("trace store: DMA_STATUS %h", st))
    end else begin
      sb.expect_host_bytes(ABUF + 32'(base), b);
      read_region(ABUF + 32'(base), b.size());
    end
    if (base == argmax_in) begin
      for (int j = 0; j < argmax_n && j < b.size(); j++) begin
        sb.expect_word(A_LOGITS + 32'(4 * j), 32'(signed'(b[j])));
        reg_rd(regs.logits[j], v);
      end
    end
    n_traces++;
  endtask

  task wait_irq();
    fork : wait_or_timeout
      wait (cfg.irq_vif.irq === 1'b1);
      begin
        #(50ms);
        `uvm_fatal("SEQ/IRQ", "no interrupt")
      end
    join_any
    disable fork;
  endtask
endclass
