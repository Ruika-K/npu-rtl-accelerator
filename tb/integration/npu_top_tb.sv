`timescale 1ns/1ps
`default_nettype none

// Integration testbench for npu_top: everything goes through AXI4-Lite.
//   - Loads the input tensor, weights and biases into the memory windows with
//     32-bit writes, fills a guard band around the output region, writes the
//     descriptor and LAYER_CNT, starts, polls STATUS, reads the output region
//     back and compares it byte for byte with the Python model
//     (+layer_vectors=<path> from model/gen_layer_vectors.py).
//   - ARGMAX cases check RESULT and the LOGITS registers.
//   - Also: ID, an error descriptor (ERR, ERR_CODE, CUR_LAYER) followed by a
//     normal run, host memory access rejected while busy, perf counters,
//     DONE W1C and irq.
module npu_top_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;
  localparam int  MAX_BYTES  = 65536;
  localparam int  GUARD      = 16;

  logic clk_i, rst_ni;
  logic [31:0] awaddr; logic awvalid, awready;
  logic [31:0] wdata;  logic [3:0] wstrb; logic wvalid, wready;
  logic [1:0]  bresp;  logic bvalid, bready;
  logic [31:0] araddr; logic arvalid, arready;
  logic [31:0] rdata;  logic [1:0] rresp; logic rvalid, rready;
  logic irq_o;
  logic [31:0] m_awaddr, m_araddr; logic [7:0] m_awlen, m_arlen; logic [2:0] m_awsize, m_arsize; logic [1:0] m_awburst, m_arburst;
  logic m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready, m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [63:0] m_wdata, m_rdata; logic [7:0] m_wstrb; logic [1:0] m_bresp, m_rresp;

  npu_top dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .s_axil_awaddr_i (awaddr), .s_axil_awprot_i (3'b000), .s_axil_awvalid_i (awvalid), .s_axil_awready_o (awready),
      .s_axil_wdata_i (wdata), .s_axil_wstrb_i (wstrb), .s_axil_wvalid_i (wvalid), .s_axil_wready_o (wready),
      .s_axil_bresp_o (bresp), .s_axil_bvalid_o (bvalid), .s_axil_bready_i (bready),
      .s_axil_araddr_i (araddr), .s_axil_arprot_i (3'b000), .s_axil_arvalid_i (arvalid), .s_axil_arready_o (arready),
      .s_axil_rdata_o (rdata), .s_axil_rresp_o (rresp), .s_axil_rvalid_o (rvalid), .s_axil_rready_i (rready),
      .pix_wr_en_i (1'b0), .pix_wr_addr_i ('0), .pix_wr_data_i ('0), .pix_wr_be_i ('0), .pix_wr_ready_o (),
      .m_axi_awaddr_o (m_awaddr), .m_axi_awlen_o (m_awlen), .m_axi_awsize_o (m_awsize), .m_axi_awburst_o (m_awburst),
      .m_axi_awvalid_o (m_awvalid), .m_axi_awready_i (m_awready),
      .m_axi_wdata_o (m_wdata), .m_axi_wstrb_o (m_wstrb), .m_axi_wlast_o (m_wlast), .m_axi_wvalid_o (m_wvalid), .m_axi_wready_i (m_wready),
      .m_axi_bresp_i (m_bresp), .m_axi_bvalid_i (m_bvalid), .m_axi_bready_o (m_bready),
      .m_axi_araddr_o (m_araddr), .m_axi_arlen_o (m_arlen), .m_axi_arsize_o (m_arsize), .m_axi_arburst_o (m_arburst),
      .m_axi_arvalid_o (m_arvalid), .m_axi_arready_i (m_arready),
      .m_axi_rdata_i (m_rdata), .m_axi_rresp_i (m_rresp), .m_axi_rlast_i (m_rlast), .m_axi_rvalid_i (m_rvalid), .m_axi_rready_o (m_rready),
      .irq_o (irq_o)
  );

  // External memory behind the DMA's AXI4 master
  axi4_slave_mem #(.ADDR_BITS(20)) u_mem (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .awaddr_i (m_awaddr), .awlen_i (m_awlen), .awsize_i (m_awsize), .awburst_i (m_awburst), .awvalid_i (m_awvalid), .awready_o (m_awready),
      .wdata_i (m_wdata), .wstrb_i (m_wstrb), .wlast_i (m_wlast), .wvalid_i (m_wvalid), .wready_o (m_wready),
      .bresp_o (m_bresp), .bvalid_o (m_bvalid), .bready_i (m_bready),
      .araddr_i (m_araddr), .arlen_i (m_arlen), .arsize_i (m_arsize), .arburst_i (m_arburst), .arvalid_i (m_arvalid), .arready_o (m_arready),
      .rdata_o (m_rdata), .rresp_o (m_rresp), .rlast_o (m_rlast), .rvalid_o (m_rvalid), .rready_i (m_rready)
  );

  axil_master_bfm u_bfm (
      .clk_i (clk_i),
      .awaddr_o (awaddr), .awvalid_o (awvalid), .awready_i (awready),
      .wdata_o (wdata), .wstrb_o (wstrb), .wvalid_o (wvalid), .wready_i (wready),
      .bresp_i (bresp), .bvalid_i (bvalid), .bready_o (bready),
      .araddr_o (araddr), .arvalid_o (arvalid), .arready_i (arready),
      .rdata_i (rdata), .rresp_i (rresp), .rvalid_i (rvalid), .rready_o (rready)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_cases = 0, n_checked = 0, n_errors = 0;

  localparam logic [31:0] A_ID = 32'h000, A_CTRL = 32'h004, A_STATUS = 32'h008, A_LCNT = 32'h00C, A_RESULT = 32'h010;
  localparam logic [31:0] A_LOGITS = 32'h020, A_PTOTAL = 32'h060, A_PARRAY = 32'h064, A_PLAYER = 32'h080, A_DESC = 32'h100;

  // ---------------------------------------------------------------------------
  // AXI helpers
  // ---------------------------------------------------------------------------
  task automatic wr32(input logic [31:0] a, input logic [31:0] d, input logic [3:0] strb = 4'hF, input logic [1:0] exp_resp = 2'b00);
    logic [1:0] resp;
    u_bfm.write(a, d, strb, resp);
    if (resp !== exp_resp) begin n_errors++; $display("ERROR: write %h resp %b, expected %b", a, resp, exp_resp); end
  endtask

  task automatic rd32(input logic [31:0] a, output logic [31:0] d, input logic [1:0] exp_resp = 2'b00);
    logic [1:0] resp;
    u_bfm.read(a, d, resp);
    if (resp !== exp_resp) begin n_errors++; $display("ERROR: read %h resp %b, expected %b", a, resp, exp_resp); end
  endtask

  task automatic check(input bit cond, input string what);
    n_checked++;
    if (!cond) begin n_errors++; $display("ERROR @%0t: %s", $time, what); end
  endtask

  // Write n bytes to window base + offset with aligned 32-bit writes and strobes.
  task automatic write_bytes(input logic [31:0] base, input int offset, input int n, ref logic [7:0] src [MAX_BYTES]);
    int a, i;
    logic [31:0] d; logic [3:0] strb;
    a = (offset / 4) * 4;
    while (a < offset + n) begin
      d = '0; strb = '0;
      for (int b = 0; b < 4; b++) begin
        i = a + b - offset;
        if (i >= 0 && i < n) begin d[8*b +: 8] = src[i]; strb[b] = 1'b1; end
      end
      wr32(base + 32'(a), d, strb);
      a += 4;
    end
  endtask

  task automatic fill_bytes(input logic [31:0] base, input int offset, input int n, input logic [7:0] v);
    int a, i;
    logic [31:0] d; logic [3:0] strb;
    a = (offset / 4) * 4;
    while (a < offset + n) begin
      d = {4{v}}; strb = '0;
      for (int b = 0; b < 4; b++) begin
        i = a + b - offset;
        if (i >= 0 && i < n) strb[b] = 1'b1;
      end
      wr32(base + 32'(a), d, strb);
      a += 4;
    end
  endtask

  task automatic read_bytes(input logic [31:0] base, input int offset, input int n, ref logic [7:0] dst [MAX_BYTES]);
    int a, i;
    logic [31:0] d;
    a = (offset / 4) * 4;
    while (a < offset + n) begin
      rd32(base + 32'(a), d);
      for (int b = 0; b < 4; b++) begin
        i = a + b - offset;
        if (i >= 0 && i < n) dst[i] = d[8*b +: 8];
      end
      a += 4;
    end
  endtask

  task automatic write_desc(input int idx, input desc_t d);
    desc_words_t w;
    w = desc_to_words(d);
    for (int k = 0; k < DESC_USED_WORDS; k++) wr32(A_DESC + 32'(idx * 64 + k * 4), w[k]);
  endtask

  // Start and wait for DONE or ERR. Returns the STATUS word.
  task automatic run_and_wait(output logic [31:0] status);
    int guard;
    wr32(A_CTRL, 32'h3);                 // START + IRQ_EN
    guard = 0;
    do begin
      rd32(A_STATUS, status);
      guard++;
    end while (status[2:1] == 2'b00 && guard < 200000);
    if (guard >= 200000) begin n_errors++; $display("ERROR: run did not finish"); end
  endtask

  // ---------------------------------------------------------------------------
  // Case storage
  // ---------------------------------------------------------------------------
  int c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, c_in_base, c_out_base, c_w_base, c_b_base;
  int c_mult, c_shift, c_zin, c_zout, c_amin, c_amax;
  int n_in, n_w, n_b, n_out;
  logic [7:0] in_bytes [MAX_BYTES], w_bytes [MAX_BYTES], b_bytes [MAX_BYTES], out_bytes [MAX_BYTES], got [MAX_BYTES];

  function automatic desc_t case_desc();
    desc_t d;
    d = '0;
    d.op = op_e'(c_op); d.k = 4'(c_k); d.pad = 4'(c_pad); d.stride_m1 = 2'd0;
    d.in_h = 16'(c_in_h); d.in_w = 16'(c_in_w); d.c_in = 16'(c_cin); d.c_out = 16'(c_cout);
    d.in_base = 32'(c_in_base); d.out_base = 32'(c_out_base); d.w_base = 32'(c_w_base); d.b_base = 32'(c_b_base);
    d.rq_mult = 32'(c_mult); d.rq_shift = 6'(c_shift);
    d.z_in = 8'(c_zin); d.z_out = 8'(c_zout); d.act_min = 8'(c_amin); d.act_max = 8'(c_amax);
    return d;
  endfunction

  // The case with the most weight bytes is replayed through the DMA afterwards.
  int d_op, d_k, d_pad, d_in_h, d_in_w, d_cin, d_cout, d_in_base, d_out_base, d_w_base, d_b_base;
  int d_mult, d_shift, d_zin, d_zout, d_amin, d_amax, d_n_in = 0, d_n_w = 0, d_n_b = 0, d_n_out = 0;
  logic [7:0] d_in [MAX_BYTES], d_w [MAX_BYTES], d_b [MAX_BYTES], d_out [MAX_BYTES];

  task automatic remember_case();
    d_op = c_op; d_k = c_k; d_pad = c_pad; d_in_h = c_in_h; d_in_w = c_in_w; d_cin = c_cin; d_cout = c_cout;
    d_in_base = c_in_base; d_out_base = c_out_base; d_w_base = c_w_base; d_b_base = c_b_base;
    d_mult = c_mult; d_shift = c_shift; d_zin = c_zin; d_zout = c_zout; d_amin = c_amin; d_amax = c_amax;
    d_n_in = n_in; d_n_w = n_w; d_n_b = n_b; d_n_out = n_out;
    for (int i = 0; i < n_in; i++) d_in[i] = in_bytes[i];
    for (int i = 0; i < n_w; i++) d_w[i] = w_bytes[i];
    for (int i = 0; i < n_b; i++) d_b[i] = b_bytes[i];
    for (int i = 0; i < n_out; i++) d_out[i] = out_bytes[i];
  endtask

  task automatic run_case();
    logic [31:0] status, v, ptotal, player;
    int mism;
    if ((c_op == int'(OP_CONV2D) || c_op == int'(OP_FC)) && n_w > d_n_w) remember_case();
    // memory images
    write_bytes(32'(ABUF_BASE), c_in_base, n_in, in_bytes);
    if (n_w > 0) write_bytes(32'(WBUF_BASE), c_w_base, n_w, w_bytes);
    if (n_b > 0) write_bytes(32'(WBUF_BASE), c_b_base, n_b, b_bytes);
    if (c_op != int'(OP_ARGMAX)) fill_bytes(32'(ABUF_BASE), c_out_base - GUARD, n_out + 2 * GUARD, 8'hA5);
    // program
    write_desc(0, case_desc());
    wr32(A_LCNT, 32'h1);
    run_and_wait(status);
    check(status[1] == 1'b1 && status[2] == 1'b0, $sformatf("case %0d: DONE without ERR (status %h)", n_cases, status));
    check(status[0] == 1'b0, "BUSY low after DONE");
    check(irq_o == 1'b1, "irq with DONE and IRQ_EN");
    rd32(A_PTOTAL, ptotal); rd32(A_PLAYER, player);
    check(ptotal > 0 && player > 0 && player <= ptotal, $sformatf("perf counters plausible (%0d / %0d)", player, ptotal));
    // result
    mism = 0;
    if (c_op == int'(OP_ARGMAX)) begin
      rd32(A_RESULT, v);
      n_checked++;
      if (v !== {23'h0, 1'b1, 4'h0, 4'(out_bytes[0])}) begin mism++; $display("ERROR: case %0d: RESULT %h, expected index %0d", n_cases, v, out_bytes[0]); end
      for (int i = 0; i < NUM_LOGITS; i++) begin
        logic [31:0] exp;
        exp = (i < n_in) ? {{24{in_bytes[i][7]}}, in_bytes[i]} : 32'h0;
        rd32(A_LOGITS + 32'(4 * i), v);
        n_checked++;
        if (v !== exp) begin mism++; if (mism < 6) $display("ERROR: case %0d: LOGITS[%0d] = %h, expected %h", n_cases, i, v, exp); end
      end
    end else begin
      read_bytes(32'(ABUF_BASE), c_out_base - GUARD, n_out + 2 * GUARD, got);
      for (int i = 0; i < n_out + 2 * GUARD; i++) begin
        logic [7:0] exp;
        exp = (i < GUARD || i >= GUARD + n_out) ? 8'hA5 : out_bytes[i - GUARD];
        n_checked++;
        if (got[i] !== exp) begin mism++; if (mism < 6) $display("ERROR: case %0d: byte %0d = %0d, expected %0d", n_cases, i - GUARD, got[i], exp); end
      end
    end
    if (mism != 0) n_errors++;
    $display("case %0d: op=%0d k=%0d pad=%0d in=%0dx%0d cin=%0d cout=%0d: %0d cycles, %s",
             n_cases, c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, ptotal, (mism == 0) ? "ok" : "MISMATCH");
    wr32(A_STATUS, 32'h2);               // clear DONE
    check(irq_o == 1'b0, "irq cleared with DONE");
    n_cases++;
  endtask

  // ---------------------------------------------------------------------------
  // DMA
  // ---------------------------------------------------------------------------
  localparam logic [31:0] A_DMA_CTRL = 32'h0C0, A_DMA_SRC = 32'h0C4, A_DMA_DST = 32'h0C8, A_DMA_LEN = 32'h0CC, A_DMA_STATUS = 32'h0D0;

  // Place bytes in external memory so that a DMA from ext (8-aligned) covers
  // buffer offset `off` (any alignment): the bytes start at ext + (off & 7).
  task automatic ext_place(input int ext, input int off, input int n, ref logic [7:0] src [MAX_BYTES]);
    for (int i = 0; i < n; i++) u_mem.mem[ext + (off & 7) + i] = src[i];
  endtask

  function automatic int dma_len_for(input int off, input int n);
    return ((n + (off & 7) + 7) / 8) * 8;
  endfunction

  // Start a DMA and wait; returns the STATUS word.
  task automatic dma_run(input bit store, input bit wbuf, input int ext, input int off, input int len, output logic [31:0] status);
    int guard = 0;
    wr32(A_DMA_SRC, 32'(ext));
    wr32(A_DMA_DST, 32'(off));
    wr32(A_DMA_LEN, 32'(len));
    wr32(A_DMA_CTRL, {29'h0, wbuf, store, 1'b1});
    do begin rd32(A_DMA_STATUS, status); guard++; end while (status[2:1] == 2'b00 && guard < 200000);
    if (guard >= 200000) begin n_errors++; $display("ERROR: DMA did not finish"); end
    wr32(A_DMA_STATUS, 32'h6);
  endtask

  task automatic dma_phase();
    logic [31:0] status, v;
    int mism, ext_img, ext_w, ext_b, ext_out, len;
    desc_t d;
    ext_img = 32'h40000; ext_w = 32'h10000; ext_b = 32'h30000; ext_out = 32'h50000;
    $display("dma: replaying op=%0d %0dx%0d cin=%0d cout=%0d (%0d weight bytes)", d_op, d_in_h, d_in_w, d_cin, d_cout, d_n_w);
    for (int i = 0; i < (1 << 20); i++) u_mem.mem[i] = 8'h00;

    // loads: image, weights, biases (the weights are the long one: check a host access is rejected meanwhile)
    ext_place(ext_img, d_in_base, d_n_in, d_in);
    dma_run(1'b0, 1'b0, ext_img, d_in_base & ~7, dma_len_for(d_in_base, d_n_in), status);
    check(status[1] == 1'b1 && status[2] == 1'b0, $sformatf("DMA load image (status %h)", status));
    ext_place(ext_w, d_w_base, d_n_w, d_w);
    wr32(A_DMA_SRC, 32'(ext_w)); wr32(A_DMA_DST, 32'(d_w_base)); wr32(A_DMA_LEN, 32'(dma_len_for(d_w_base, d_n_w)));
    wr32(A_DMA_CTRL, 32'h5);                                  // wbuf, load, start
    rd32(32'(ABUF_BASE), v, 2'b10);                           // host memory access rejected while the DMA runs
    rd32(A_STATUS, v); check(v[3] == 1'b1, "HOST_CONFLICT during DMA"); wr32(A_STATUS, 32'h8);
    wr32(A_CTRL, 32'h1);                                      // START ignored while the DMA runs
    begin int guard = 0; do begin rd32(A_DMA_STATUS, status); guard++; end while (status[2:1] == 2'b00 && guard < 200000); end
    check(status[1] == 1'b1 && status[2] == 1'b0, $sformatf("DMA load weights (status %h)", status));
    rd32(A_STATUS, v); check(v[0] == 1'b0 && v[1] == 1'b0, "START during DMA was ignored");
    wr32(A_DMA_STATUS, 32'h6);
    ext_place(ext_b, d_b_base, d_n_b, d_b);
    dma_run(1'b0, 1'b1, ext_b, d_b_base, dma_len_for(d_b_base, d_n_b), status);
    check(status[1] == 1'b1 && status[2] == 1'b0, "DMA load biases");

    // run the layer
    fill_bytes(32'(ABUF_BASE), d_out_base - GUARD, d_n_out + 2 * GUARD, 8'hA5);
    d = '0;
    d.op = op_e'(d_op); d.k = 4'(d_k); d.pad = 4'(d_pad);
    d.in_h = 16'(d_in_h); d.in_w = 16'(d_in_w); d.c_in = 16'(d_cin); d.c_out = 16'(d_cout);
    d.in_base = 32'(d_in_base); d.out_base = 32'(d_out_base); d.w_base = 32'(d_w_base); d.b_base = 32'(d_b_base);
    d.rq_mult = 32'(d_mult); d.rq_shift = 6'(d_shift); d.z_in = 8'(d_zin); d.z_out = 8'(d_zout); d.act_min = 8'(d_amin); d.act_max = 8'(d_amax);
    write_desc(0, d);
    wr32(A_LCNT, 32'h1);
    run_and_wait(status);
    check(status[1] == 1'b1 && status[2] == 1'b0, "layer after DMA loads");
    wr32(A_STATUS, 32'h2);

    // store the output region (aligned window) to external memory and compare
    len = dma_len_for(d_out_base, d_n_out);
    dma_run(1'b1, 1'b0, ext_out, d_out_base & ~7, len, status);
    check(status[1] == 1'b1 && status[2] == 1'b0, "DMA store output");
    mism = 0;
    for (int i = 0; i < d_n_out; i++) begin
      n_checked++;
      if (u_mem.mem[ext_out + (d_out_base & 7) + i] !== d_out[i]) begin mism++; if (mism < 4) $display("ERROR: stored byte %0d = %0d, expected %0d", i, u_mem.mem[ext_out + (d_out_base & 7) + i], d_out[i]); end
    end
    if (mism != 0) n_errors++;
    // store a slice of the weight buffer back and compare
    dma_run(1'b1, 1'b1, 32'h60000, d_w_base, 64, status);
    check(status[1] == 1'b1 && status[2] == 1'b0, "DMA store weights");
    for (int i = 0; i < 64; i++) begin n_checked++; if (u_mem.mem[32'h60000 + i] !== d_w[i]) begin n_errors++; $display("ERROR: stored weight byte %0d", i); break; end end
    check(u_mem.n_proto_errors == 0, "no AXI protocol violations seen by the memory model");

    // errors
    dma_run(1'b0, 1'b0, ext_img, 32'h100, 12, status);
    check(status[2] == 1'b1 && status[15:8] == 8'h02, $sformatf("DMA misaligned length -> ERR code 2 (status %h)", status));
    dma_run(1'b0, 1'b0, ext_img, 32'h7F00, 32'h200, status);
    check(status[2] == 1'b1 && status[15:8] == 8'h02, "DMA past the activation buffer -> ERR code 2");
    u_mem.err_lo = 32'h70000; u_mem.err_hi = 32'h70100;
    dma_run(1'b0, 1'b0, 32'h70000, 32'h100, 64, status);
    check(status[2] == 1'b1 && status[15:8] == 8'h03, $sformatf("DMA read from an erroring region -> ERR code 3 (status %h)", status));
    dma_run(1'b1, 1'b0, 32'h70080, 32'h100, 64, status);
    check(status[2] == 1'b1 && status[15:8] == 8'h03, "DMA write to an erroring region -> ERR code 3");
    u_mem.err_lo = 32'hFFFF_FFFF; u_mem.err_hi = 32'hFFFF_FFFF;
    dma_run(1'b0, 1'b0, ext_img, 32'h100, 64, status);
    check(status[1] == 1'b1 && status[2] == 1'b0, "DMA works again after errors");
    $display("dma: %0d AW, %0d AR, %0d write beats, %0d read beats", u_mem.n_aw, u_mem.n_ar, u_mem.n_wbeats, u_mem.n_rbeats);
  endtask

  task automatic read_section(input int fd, output int n, ref logic [7:0] dst [MAX_BYTES]);
    int code, v;
    code = $fscanf(fd, "%d", n);
    if (n > MAX_BYTES) begin n_errors++; $display("ERROR: section too large"); n = MAX_BYTES; end
    for (int i = 0; i < n; i++) begin code = $fscanf(fd, "%d", v); dst[i] = 8'(v); end
  endtask

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  string vec_path = "";

  initial begin
    int fd, code;
    string tag;
    logic [31:0] v, status;
    desc_t d;

    void'($value$plusargs("layer_vectors=%s", vec_path));
    $display("npu_top_tb: vectors='%s'", vec_path);
    u_bfm.aw_w_order = 3;

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);

    rd32(A_ID, v);
    check(v == NPU_ID, "ID register");
    rd32(A_STATUS, v);
    check(v == 32'h0, "STATUS after reset");

    // ---- error descriptor: K = 0, then recovery is proven by the cases below ----
    d = '0; d.op = OP_CONV2D; d.k = 4'd0; d.in_h = 16'd4; d.in_w = 16'd4; d.c_in = 16'd1; d.c_out = 16'd1;
    d.rq_mult = 32'sh4000_0000; d.rq_shift = 6'd31; d.act_min = -8'sd128; d.act_max = 8'sd127;
    write_desc(0, d);
    write_desc(1, d);
    wr32(A_LCNT, 32'h2);
    d.k = 4'd1; write_desc(0, d);        // layer 0 valid, layer 1 bad
    run_and_wait(status);
    check(status[2] == 1'b1 && status[1] == 1'b0, $sformatf("ERR without DONE (status %h)", status));
    check(status[23:16] == 8'h02 && status[11:8] == 4'd1, $sformatf("ERR_CODE 2 at layer 1 (status %h)", status));
    wr32(A_STATUS, 32'h4);
    rd32(A_STATUS, v);
    check(v[2] == 1'b0, "ERR cleared");

    // ---- layer cases from the Python model ----
    if (vec_path != "") begin
      fd = $fopen(vec_path, "r");
      if (fd == 0) begin n_errors++; $display("ERROR: cannot open '%s'", vec_path); end
      else begin
        n_in = 0; n_w = 0; n_b = 0; n_out = 0;
        while (!$feof(fd)) begin
          code = $fscanf(fd, "%s", tag);
          if (code != 1) break;
          case (tag)
            "CASE": begin
              code = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                             c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, c_in_base, c_out_base, c_w_base, c_b_base,
                             c_mult, c_shift, c_zin, c_zout, c_amin, c_amax);
              n_in = 0; n_w = 0; n_b = 0; n_out = 0;
            end
            "IN":  read_section(fd, n_in, in_bytes);
            "W":   read_section(fd, n_w, w_bytes);
            "B":   read_section(fd, n_b, b_bytes);
            "OUT": begin read_section(fd, n_out, out_bytes); run_case(); end
            default: begin n_errors++; $display("ERROR: unknown tag '%s'", tag); end
          endcase
        end
        $fclose(fd);
      end
    end

    // ---- DMA: replay the biggest case with loads and a store through the AXI4 master ----
    if (d_n_w > 0) dma_phase();

    // ---- host memory access while busy: use the last case's descriptor again ----
    if (n_cases > 0 && c_op != int'(OP_ARGMAX)) begin
      write_desc(0, case_desc());
      wr32(A_LCNT, 32'h1);
      wr32(A_CTRL, 32'h1);
      rd32(32'(ABUF_BASE), v, 2'b10);                       // rejected while busy
      rd32(A_STATUS, v);
      check(v[3] == 1'b1, "HOST_CONFLICT set");
      do rd32(A_STATUS, status); while (status[2:1] == 2'b00);
      check(status[1] == 1'b1, "run after the rejected access still completes");
      wr32(A_STATUS, 32'hA);
      rd32(A_STATUS, v);
      check(v == 32'h0, "DONE and HOST_CONFLICT cleared");
    end

    if (n_errors == 0 && n_cases > 0) begin
      $display("PASS: npu_top_tb cases=%0d checked=%0d", n_cases, n_checked);
      $finish;
    end else begin
      $display("FAIL: npu_top_tb errors=%0d cases=%0d", n_errors, n_cases);
      $fatal(1, "npu_top_tb failed");
    end
  end

  initial begin
    #5s;
    $fatal(1, "TIMEOUT: npu_top_tb did not finish");
  end

endmodule

`default_nettype wire
