`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for npu_regs behind axil_slave.
//   - The testbench plays the sequencer (busy / done / err / counters), the
//     argmax engine and the two on-chip memories behind the host memory port.
//   - Checks register reset values, RW / RO / W1P / W1C semantics, descriptor
//     storage with byte strobes and reserved-bit masking, the sequencer's
//     descriptor read port (host priority on the shared mux, one-cycle data),
//     memory-window routing, SLVERR for unmapped / misaligned / busy accesses,
//     HOST_CONFLICT, START side effects and irq_o.
// Plusargs: +seed=<int> (default 1), +n=<random accesses> (default 500).
module npu_regs_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i, rst_ni;

  // AXI
  logic [31:0] awaddr; logic awvalid, awready;
  logic [31:0] wdata;  logic [3:0] wstrb; logic wvalid, wready;
  logic [1:0]  bresp;  logic bvalid, bready;
  logic [31:0] araddr; logic arvalid, arready;
  logic [31:0] rdata;  logic [1:0] rresp; logic rvalid, rready;
  // internal bus
  logic        wr_req, wr_ack, wr_err, rd_req, rd_ack, rd_err;
  logic [19:0] wr_addr, rd_addr;
  logic [31:0] wr_data, rd_data;
  logic [3:0]  wr_strb;
  // host memory port
  logic        hm_req, hm_we, hm_sel, hm_ack;
  logic [19:0] hm_addr;
  logic [31:0] hm_wdata, hm_rdata;
  logic [3:0]  hm_be;
  // sequencer side (driven by the TB)
  logic        start_o;
  logic [4:0]  layer_cnt_o;
  logic        desc_rd_en_i = 1'b0;
  logic [3:0]  desc_rd_idx_i = '0, desc_rd_word_i = '0;
  logic        desc_rd_gnt_o, desc_rd_valid_o;
  logic [31:0] desc_rd_data_o;
  logic        busy_i = 1'b0, done_i = 1'b0, err_i = 1'b0;
  err_e        err_code_i = ERR_NONE;
  logic [3:0]  cur_layer_i = '0;
  logic [NUM_DESC-1:0][31:0] layer_cycles_i = '0;
  logic [31:0] total_cycles_i = '0, array_busy_cycles_i = '0;
  logic        argmax_valid_i = 1'b0;
  logic [3:0]  argmax_idx_i = '0;
  logic [NUM_LOGITS-1:0][7:0] argmax_logits_i = '0;
  logic        irq_o;

  axil_slave u_axil (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .s_axil_awaddr_i (awaddr), .s_axil_awprot_i (3'b000), .s_axil_awvalid_i (awvalid), .s_axil_awready_o (awready),
      .s_axil_wdata_i (wdata), .s_axil_wstrb_i (wstrb), .s_axil_wvalid_i (wvalid), .s_axil_wready_o (wready),
      .s_axil_bresp_o (bresp), .s_axil_bvalid_o (bvalid), .s_axil_bready_i (bready),
      .s_axil_araddr_i (araddr), .s_axil_arprot_i (3'b000), .s_axil_arvalid_i (arvalid), .s_axil_arready_o (arready),
      .s_axil_rdata_o (rdata), .s_axil_rresp_o (rresp), .s_axil_rvalid_o (rvalid), .s_axil_rready_i (rready),
      .wr_req_o (wr_req), .wr_addr_o (wr_addr), .wr_data_o (wr_data), .wr_strb_o (wr_strb),
      .wr_ack_i (wr_ack), .wr_err_i (wr_err),
      .rd_req_o (rd_req), .rd_addr_o (rd_addr), .rd_ack_i (rd_ack), .rd_data_i (rd_data), .rd_err_i (rd_err)
  );

  npu_regs dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .wr_req_i (wr_req), .wr_addr_i (wr_addr), .wr_data_i (wr_data), .wr_strb_i (wr_strb),
      .wr_ack_o (wr_ack), .wr_err_o (wr_err),
      .rd_req_i (rd_req), .rd_addr_i (rd_addr), .rd_ack_o (rd_ack), .rd_data_o (rd_data), .rd_err_o (rd_err),
      .hm_req_o (hm_req), .hm_we_o (hm_we), .hm_sel_o (hm_sel), .hm_addr_o (hm_addr),
      .hm_wdata_o (hm_wdata), .hm_be_o (hm_be), .hm_ack_i (hm_ack), .hm_rdata_i (hm_rdata),
      .start_o (start_o), .layer_cnt_o (layer_cnt_o),
      .desc_rd_en_i (desc_rd_en_i), .desc_rd_idx_i (desc_rd_idx_i), .desc_rd_word_i (desc_rd_word_i),
      .desc_rd_gnt_o (desc_rd_gnt_o), .desc_rd_valid_o (desc_rd_valid_o), .desc_rd_data_o (desc_rd_data_o),
      .busy_i (busy_i), .done_i (done_i), .err_i (err_i), .err_code_i (err_code_i), .cur_layer_i (cur_layer_i),
      .layer_cycles_i (layer_cycles_i), .total_cycles_i (total_cycles_i), .array_busy_cycles_i (array_busy_cycles_i),
      .argmax_valid_i (argmax_valid_i), .argmax_idx_i (argmax_idx_i), .argmax_logits_i (argmax_logits_i),
      .irq_o (irq_o)
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

  int n_checked = 0, n_errors = 0;

  // ---------------------------------------------------------------------------
  // Memory model behind the host memory port (two spaces, word granular)
  // ---------------------------------------------------------------------------
  logic [31:0] abuf_mem [logic [19:0]];
  logic [31:0] wbuf_mem [logic [19:0]];
  logic        hm_pend_q; int hm_cnt_q;
  logic        hm_we_q, hm_sel_q; logic [19:0] hm_addr_q; logic [31:0] hm_wdata_q; logic [3:0] hm_be_q;
  int          hm_req_count = 0;

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      hm_ack <= 1'b0; hm_rdata <= '0; hm_pend_q <= 1'b0; hm_cnt_q <= 0;
      hm_we_q <= 1'b0; hm_sel_q <= 1'b0; hm_addr_q <= '0; hm_wdata_q <= '0; hm_be_q <= '0;
    end else begin
      hm_ack <= 1'b0;
      if (hm_req) begin
        hm_pend_q <= 1'b1; hm_cnt_q <= $urandom_range(2);
        hm_we_q <= hm_we; hm_sel_q <= hm_sel; hm_addr_q <= hm_addr; hm_wdata_q <= hm_wdata; hm_be_q <= hm_be;
        hm_req_count++;
      end else if (hm_pend_q) begin
        if (hm_cnt_q == 0) begin
          logic [31:0] old;
          if (hm_sel_q) old = wbuf_mem.exists(hm_addr_q) ? wbuf_mem[hm_addr_q] : 32'h0;
          else          old = abuf_mem.exists(hm_addr_q) ? abuf_mem[hm_addr_q] : 32'h0;
          if (hm_we_q) begin
            for (int b = 0; b < 4; b++) if (hm_be_q[b]) old[8*b +: 8] = hm_wdata_q[8*b +: 8];
            if (hm_sel_q) wbuf_mem[hm_addr_q] = old; else abuf_mem[hm_addr_q] = old;
          end
          hm_rdata <= old;
          hm_ack <= 1'b1; hm_pend_q <= 1'b0;
        end else hm_cnt_q <= hm_cnt_q - 1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // start_o must be a single-cycle pulse; count them.
  int start_pulses = 0;
  logic start_prev = 1'b0;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (rst_ni) begin
      if (start_o && !start_prev) start_pulses++;
      if (start_o && start_prev) begin n_errors++; $display("ERROR @%0t: start_o longer than one cycle", $time); end
      start_prev = start_o;
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  task automatic wr(input logic [19:0] a, input logic [31:0] d, input logic [3:0] strb = 4'hF, input logic [1:0] exp_resp = 2'b00);
    logic [1:0] resp;
    u_bfm.write({12'h0, a}, d, strb, resp);
    n_checked++;
    if (resp !== exp_resp) begin n_errors++; $display("ERROR: write %h <= %h resp %b, expected %b", a, d, resp, exp_resp); end
  endtask

  task automatic rd_expect(input logic [19:0] a, input logic [31:0] exp, input logic [1:0] exp_resp = 2'b00, input string what = "");
    logic [1:0] resp; logic [31:0] data;
    u_bfm.read({12'h0, a}, data, resp);
    n_checked++;
    if (resp !== exp_resp) begin n_errors++; $display("ERROR: read %h (%s) resp %b, expected %b", a, what, resp, exp_resp); end
    if (exp_resp == 2'b00 && data !== exp) begin n_errors++; $display("ERROR: read %h (%s) = %h, expected %h", a, what, data, exp); end
  endtask

  /* verilator lint_off BLKSEQ */   // also called from the clocked descriptor-port monitor
  task automatic check(input bit cond, input string what);
    n_checked++;
    if (!cond) begin n_errors++; $display("ERROR @%0t: %s", $time, what); end
  endtask
  /* verilator lint_on BLKSEQ */

  task automatic pulse_done();
    @(negedge clk_i); done_i = 1'b1; @(negedge clk_i); done_i = 1'b0;
  endtask
  task automatic pulse_err(input err_e code);
    @(negedge clk_i); err_i = 1'b1; err_code_i = code; @(negedge clk_i); err_i = 1'b0;
  endtask

  localparam logic [19:0] A_ID = 20'h000, A_CTRL = 20'h004, A_STATUS = 20'h008, A_LCNT = 20'h00C, A_RESULT = 20'h010;
  localparam logic [19:0] A_LOGITS = 20'h020, A_PTOTAL = 20'h060, A_PARRAY = 20'h064, A_PLAYER = 20'h080, A_DESC = 20'h100;

  function automatic logic [19:0] desc_addr(input int idx, input int word);
    return A_DESC + 20'(idx * 64 + word * 4);
  endfunction

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  int seed = 1;
  int n_random = 500;
  logic [31:0] exp_words [NUM_DESC][DESC_USED_WORDS];
  logic [31:0] exp_abuf [logic [19:0]];
  logic [31:0] exp_wbuf [logic [19:0]];

  // Sequencer descriptor read port, played by the TB while seq_walk_en is set:
  // one request per cycle walking every (descriptor, word) pair, words 10 .. 15
  // included (they read as 0). The grant must drop exactly in the cycles of a
  // host descriptor read, and every grant must return its word a cycle later.
  bit          seq_walk_en = 0, seq_pend = 0;
  logic [31:0] seq_pend_exp = '0;
  int          seq_pos = 0, seq_grants = 0, seq_conflicts = 0;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    bit host_desc, en_now;
    en_now = seq_walk_en;   // sampled once: the test sequence may change it meanwhile
    if (en_now || seq_pend) begin
      if (seq_pend) begin
        check(desc_rd_valid_o && desc_rd_data_o == seq_pend_exp,
              $sformatf("descriptor read port data %h, expected %h", desc_rd_data_o, seq_pend_exp));
      end else begin
        check(!desc_rd_valid_o, "descriptor read port: no data without a grant");
      end
      desc_rd_en_i   = en_now;
      desc_rd_idx_i  = 4'(seq_pos / 16);
      desc_rd_word_i = 4'(seq_pos % 16);
      #1;
      host_desc = rd_req && (rd_addr >= A_DESC) && (rd_addr < A_DESC + 20'h400);
      check(desc_rd_gnt_o == (en_now && !host_desc), "descriptor read port grant follows host priority");
      if (en_now && host_desc) seq_conflicts++;
      seq_pend     = desc_rd_gnt_o;
      seq_pend_exp = (seq_pos % 16 < DESC_USED_WORDS) ? exp_words[seq_pos / 16][seq_pos % 16] : 32'h0;
      if (desc_rd_gnt_o) begin
        seq_grants++;
        seq_pos = (seq_pos + 1) % (NUM_DESC * 16);
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  initial begin
    logic [31:0] v, old;
    int pulses_before, req_before;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 500;
    void'($urandom(seed));
    $display("npu_regs_tb: seed=%0d n_random=%0d", seed, n_random);
    u_bfm.aw_w_order = 3;

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);

    // ---- reset values ----
    rd_expect(A_ID, NPU_ID, 2'b00, "ID");
    rd_expect(A_CTRL, 32'h0, 2'b00, "CTRL reset");
    rd_expect(A_STATUS, 32'h0, 2'b00, "STATUS reset");
    rd_expect(A_LCNT, 32'h0, 2'b00, "LAYER_CNT reset");
    rd_expect(A_RESULT, 32'h0, 2'b00, "RESULT reset");
    rd_expect(A_LOGITS + 20'h3C, 32'h0, 2'b00, "LOGITS reset");
    rd_expect(A_PTOTAL, 32'h0, 2'b00, "PERF_TOTAL reset");
    rd_expect(desc_addr(15, 9), 32'h0, 2'b00, "DESC reset");
    rd_expect(20'h0FFC, 32'h0, 2'b00, "undefined register reads 0");
    check(irq_o == 1'b0, "irq low after reset");

    // ---- LAYER_CNT ----
    wr(A_LCNT, 32'hFFFF_FFFF);           rd_expect(A_LCNT, 32'h1F, 2'b00, "LAYER_CNT 5 bits");
    check(layer_cnt_o == 5'h1F, "layer_cnt_o follows");
    wr(A_LCNT, 32'h0000_0008, 4'b1110);  rd_expect(A_LCNT, 32'h1F, 2'b00, "LAYER_CNT ignores strobe without byte 0");
    wr(A_LCNT, 32'h0000_0008);           rd_expect(A_LCNT, 32'h08, 2'b00, "LAYER_CNT = 8");
    wr(A_ID, 32'h1234_5678);             rd_expect(A_ID, NPU_ID, 2'b00, "ID is read-only");

    // ---- CTRL: START pulse, IRQ_EN ----
    pulses_before = start_pulses;
    wr(A_CTRL, 32'h1);
    check(start_pulses == pulses_before + 1, "START produced one pulse");
    rd_expect(A_CTRL, 32'h0, 2'b00, "START reads 0");
    wr(A_CTRL, 32'h3);
    check(start_pulses == pulses_before + 2, "START with IRQ_EN produced a pulse");
    rd_expect(A_CTRL, 32'h2, 2'b00, "IRQ_EN set");
    wr(A_CTRL, 32'h1, 4'b1110);
    check(start_pulses == pulses_before + 2, "START ignored without byte strobe 0");
    busy_i = 1'b1;
    wr(A_CTRL, 32'h1);
    check(start_pulses == pulses_before + 2, "START ignored while busy");
    rd_expect(A_STATUS, 32'h1, 2'b00, "STATUS.BUSY mirrors busy_i");
    busy_i = 1'b0;
    wr(A_CTRL, 32'h2);                   // IRQ_EN only, no start
    check(start_pulses == pulses_before + 2, "IRQ_EN write without START");

    // ---- STATUS: DONE / ERR / HOST_CONFLICT / CUR_LAYER / ERR_CODE ----
    pulse_done();
    rd_expect(A_STATUS, 32'h2, 2'b00, "DONE set");
    check(irq_o == 1'b1, "irq follows DONE & IRQ_EN");
    wr(A_CTRL, 32'h0);                   // IRQ_EN = 0
    check(irq_o == 1'b0, "irq drops with IRQ_EN");
    wr(A_CTRL, 32'h2);
    check(irq_o == 1'b1, "irq back");
    wr(A_STATUS, 32'h1);                 // bit 0 is RO, DONE must stay
    rd_expect(A_STATUS, 32'h2, 2'b00, "DONE survives a write of 1 to bit 0");
    wr(A_STATUS, 32'h2);
    rd_expect(A_STATUS, 32'h0, 2'b00, "DONE W1C");
    check(irq_o == 1'b0, "irq cleared with DONE");
    cur_layer_i = 4'd5;
    pulse_err(ERR_BAD_ADDR);
    rd_expect(A_STATUS, 32'h0003_0504, 2'b00, "ERR, ERR_CODE, CUR_LAYER");
    wr(A_STATUS, 32'h4);
    rd_expect(A_STATUS, 32'h0003_0500, 2'b00, "ERR W1C keeps ERR_CODE");
    pulse_done(); pulse_err(ERR_BAD_OUT_DIM);
    rd_expect(A_STATUS, 32'h0005_0506, 2'b00, "DONE and ERR together");
    wr(A_CTRL, 32'h1);                   // START clears DONE, ERR, ERR_CODE
    rd_expect(A_STATUS, 32'h0000_0500, 2'b00, "START clears DONE / ERR / ERR_CODE");
    cur_layer_i = 4'd0;

    // ---- RESULT / LOGITS ----
    @(negedge clk_i);
    argmax_valid_i = 1'b1; argmax_idx_i = 4'd7;
    for (int i = 0; i < NUM_LOGITS; i++) argmax_logits_i[i] = 8'(i * 17 - 100);
    @(negedge clk_i); argmax_valid_i = 1'b0;
    rd_expect(A_RESULT, 32'h0000_0107, 2'b00, "RESULT");
    for (int i = 0; i < NUM_LOGITS; i++)
      rd_expect(A_LOGITS + 20'(4 * i), 32'($signed(8'(i * 17 - 100))), 2'b00, "LOGITS sign-extended");
    wr(A_CTRL, 32'h1);
    rd_expect(A_RESULT, 32'h0000_0007, 2'b00, "START clears RESULT.VALID");

    // ---- PERF ----
    total_cycles_i = 32'h0001_2345; array_busy_cycles_i = 32'h55;
    for (int i = 0; i < NUM_DESC; i++) layer_cycles_i[i] = 32'd100 + 32'(i);
    rd_expect(A_PTOTAL, 32'h0001_2345, 2'b00, "PERF_TOTAL");
    rd_expect(A_PARRAY, 32'h55, 2'b00, "PERF_ARRAY_BUSY");
    for (int i = 0; i < NUM_DESC; i++) rd_expect(A_PLAYER + 20'(4 * i), 32'd100 + 32'(i), 2'b00, "PERF_LAYER");

    // ---- descriptor table ----
    for (int i = 0; i < NUM_DESC; i++)
      for (int k = 0; k < 16; k++) begin
        v = $urandom();
        wr(desc_addr(i, k), v);
        if (k < DESC_USED_WORDS) exp_words[i][k] = v & desc_word_mask(k);
      end
    for (int i = 0; i < NUM_DESC; i++)
      for (int k = 0; k < 16; k++)
        rd_expect(desc_addr(i, k), (k < DESC_USED_WORDS) ? exp_words[i][k] : 32'h0, 2'b00, "DESC word");
    // byte strobe on W3 of descriptor 2
    old = exp_words[2][3];
    wr(desc_addr(2, 3), 32'hA5A5_A5A5, 4'b0100);
    exp_words[2][3] = {old[31:24], 8'hA5, old[15:0]};
    rd_expect(desc_addr(2, 3), exp_words[2][3], 2'b00, "DESC byte strobe");
    // sequencer read port: walk the whole table (twice) while the host keeps
    // reading descriptor words, so that both share the read mux
    seq_walk_en = 1;
    while (seq_grants < 2 * NUM_DESC * 16) begin
      int i, k;
      i = $urandom_range(NUM_DESC - 1);
      k = $urandom_range(15);
      rd_expect(desc_addr(i, k), (k < DESC_USED_WORDS) ? exp_words[i][k] : 32'h0, 2'b00, "DESC word during fetch");
    end
    seq_walk_en = 0;
    @(negedge clk_i); @(negedge clk_i);
    check(seq_conflicts > 0, "host and sequencer collided on the descriptor mux at least once");

    // ---- memory windows ----
    wr(ABUF_BASE + 20'h0100, 32'h1122_3344);
    rd_expect(ABUF_BASE + 20'h0100, 32'h1122_3344, 2'b00, "act_buffer word");
    check(abuf_mem[20'h0100] == 32'h1122_3344, "host write reached act_buffer offset 0x100");
    wr(ABUF_BASE + 20'h0100, 32'hFFEE_DDCC, 4'b0011);
    rd_expect(ABUF_BASE + 20'h0100, 32'h1122_DDCC, 2'b00, "act_buffer byte enables");
    wr(ABUF_BASE + 20'h7FFC, 32'h0A0B_0C0D);
    rd_expect(ABUF_BASE + 20'h7FFC, 32'h0A0B_0C0D, 2'b00, "act_buffer last word");
    wr(WBUF_BASE + 20'h0000, 32'h9999_0000);
    wr(WBUF_BASE + 20'h1FFFC, 32'h9999_FFFC);
    rd_expect(WBUF_BASE + 20'h0000, 32'h9999_0000, 2'b00, "weight_buffer first word");
    rd_expect(WBUF_BASE + 20'h1FFFC, 32'h9999_FFFC, 2'b00, "weight_buffer last word");
    check(wbuf_mem[20'h1FFFC] == 32'h9999_FFFC && !abuf_mem.exists(20'h1FFFC), "weight writes do not land in act_buffer");
    rd_expect(ABUF_BASE + 20'h0200, 32'h0, 2'b00, "unwritten memory reads 0");
    // unmapped and misaligned
    wr(20'h18000, 32'h1, 4'hF, 2'b10);
    rd_expect(20'h18000, 32'h0, 2'b10, "gap between windows");
    wr(20'h40000, 32'h1, 4'hF, 2'b10);
    rd_expect(20'hFFFFC, 32'h0, 2'b10, "above weight_buffer");
    rd_expect(20'h00FFC, 32'h0, 2'b00, "top of the 4 KB register window");
    rd_expect(20'h0FFFC, 32'h0, 2'b10, "gap between registers and act_buffer");
    wr(20'h01000, 32'h1, 4'hF, 2'b10);
    wr(20'h10002, 32'h1, 4'hF, 2'b10);
    rd_expect(20'h00005, 32'h0, 2'b10, "misaligned register address");
    rd_expect(20'h20001, 32'h0, 2'b10, "misaligned memory address");
    // busy: memory rejected, registers fine
    busy_i = 1'b1;
    req_before = hm_req_count;
    wr(ABUF_BASE + 20'h0100, 32'h0, 4'hF, 2'b10);
    rd_expect(WBUF_BASE + 20'h0000, 32'h0, 2'b10, "memory read while busy");
    check(hm_req_count == req_before, "no memory request while busy");
    rd_expect(A_STATUS, 32'h0000_0009, 2'b00, "HOST_CONFLICT set, BUSY");
    rd_expect(A_ID, NPU_ID, 2'b00, "registers readable while busy");
    wr(A_LCNT, 32'h3); rd_expect(A_LCNT, 32'h3, 2'b00, "registers writable while busy");
    busy_i = 1'b0;
    wr(A_STATUS, 32'h8);
    rd_expect(A_STATUS, 32'h0, 2'b00, "HOST_CONFLICT W1C");
    rd_expect(ABUF_BASE + 20'h0100, 32'h1122_DDCC, 2'b00, "memory intact after rejected write");

    // ---- random mixed traffic against a reference ----
    exp_abuf[20'h0100] = 32'h1122_DDCC; exp_abuf[20'h7FFC] = 32'h0A0B_0C0D;
    exp_wbuf[20'h0000] = 32'h9999_0000; exp_wbuf[20'h1FFFC] = 32'h9999_FFFC;
    for (int i = 0; i < n_random; i++) begin
      logic [19:0] a; logic [3:0] strb; int kind;
      u_bfm.max_resp_delay = $urandom_range(2);
      kind = $urandom_range(5);
      v = $urandom(); strb = 4'($urandom_range(15, 1));
      case (kind)
        0, 1: begin   // act_buffer
          a = ABUF_BASE + 20'($urandom_range(63)) * 4;
          if ($urandom_range(1) == 0) begin
            old = exp_abuf.exists(a - ABUF_BASE) ? exp_abuf[a - ABUF_BASE] : 32'h0;
            for (int b = 0; b < 4; b++) if (strb[b]) old[8*b +: 8] = v[8*b +: 8];
            exp_abuf[a - ABUF_BASE] = old;
            wr(a, v, strb);
          end else rd_expect(a, exp_abuf.exists(a - ABUF_BASE) ? exp_abuf[a - ABUF_BASE] : 32'h0, 2'b00, "random abuf");
        end
        2: begin      // weight_buffer
          a = WBUF_BASE + 20'($urandom_range(63)) * 4;
          if ($urandom_range(1) == 0) begin
            old = exp_wbuf.exists(a - WBUF_BASE) ? exp_wbuf[a - WBUF_BASE] : 32'h0;
            for (int b = 0; b < 4; b++) if (strb[b]) old[8*b +: 8] = v[8*b +: 8];
            exp_wbuf[a - WBUF_BASE] = old;
            wr(a, v, strb);
          end else rd_expect(a, exp_wbuf.exists(a - WBUF_BASE) ? exp_wbuf[a - WBUF_BASE] : 32'h0, 2'b00, "random wbuf");
        end
        3, 4: begin   // descriptor words
          int di, dk;
          di = $urandom_range(NUM_DESC - 1); dk = $urandom_range(DESC_USED_WORDS - 1);
          a = desc_addr(di, dk);
          if ($urandom_range(1) == 0) begin
            old = exp_words[di][dk];
            for (int b = 0; b < 4; b++) if (strb[b]) old[8*b +: 8] = v[8*b +: 8];
            exp_words[di][dk] = old & desc_word_mask(dk);
            wr(a, v, strb);
          end else rd_expect(a, exp_words[di][dk], 2'b00, "random desc");
        end
        default: begin // unmapped
          a = ($urandom_range(1) == 0) ? 20'h18000 + 20'($urandom_range(255)) * 4 : 20'h40000 + 20'($urandom_range(255)) * 4;
          if ($urandom_range(1) == 0) wr(a, v, strb, 2'b10); else rd_expect(a, 32'h0, 2'b10, "random unmapped");
        end
      endcase
    end

    if (n_errors == 0) begin
      $display("PASS: npu_regs_tb checked=%0d", n_checked);
      $finish;
    end else begin
      $display("FAIL: npu_regs_tb errors=%0d checked=%0d", n_errors, n_checked);
      $fatal(1, "npu_regs_tb failed");
    end
  end

  initial begin
    #200ms;
    $fatal(1, "TIMEOUT: npu_regs_tb did not finish");
  end

endmodule

`default_nettype wire
