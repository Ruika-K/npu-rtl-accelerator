`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for npu_sequencer.
//   - Stub engines complete a layer after a random delay, or report an error
//     when told to. Every start is logged with the forwarded descriptor.
//   - Checks the layer loop (engine order, FC view, NOP layers, done / busy /
//     cur_layer timing), start-while-busy, every error code, engine-reported
//     errors, and the three performance counters against a cycle-exact model.
//   - Phase 3 replays descriptors from model/gen_desc_vectors.py and compares
//     the error code with the Python model (+desc_vectors=<path>).
// Plusargs: +seed=<int> (default 1).
module npu_sequencer_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i, rst_ni;
  logic start_i = 1'b0;
  logic [4:0] layer_cnt_i = '0;
  desc_t [NUM_DESC-1:0] desc_i = '0;
  logic busy_o, done_o, err_o;
  err_e err_code_o;
  logic [3:0] cur_layer_o;
  logic [NUM_DESC-1:0][31:0] layer_cycles_o;
  logic [31:0] total_cycles_o, array_busy_cycles_o;
  logic array_valid_i = 1'b0;
  desc_t eng_desc_o;
  logic conv_start_o, pool_start_o, argmax_start_o;
  logic conv_done_i = 1'b0, conv_err_i = 1'b0, pool_done_i = 1'b0, pool_err_i = 1'b0, argmax_done_i = 1'b0, argmax_err_i = 1'b0;
  err_e conv_err_code_i = ERR_NONE, pool_err_code_i = ERR_NONE, argmax_err_code_i = ERR_NONE;
  // descriptor read port (modeled below like npu_regs)
  logic        desc_rd_en_o;
  logic [3:0]  desc_rd_idx_o, desc_rd_word_o;
  logic        desc_rd_gnt_i = 1'b0, desc_rd_valid_i = 1'b0;
  logic [31:0] desc_rd_data_i = '0;

  npu_sequencer dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (start_i), .layer_cnt_i (layer_cnt_i),
      .desc_rd_en_o (desc_rd_en_o), .desc_rd_idx_o (desc_rd_idx_o), .desc_rd_word_o (desc_rd_word_o),
      .desc_rd_gnt_i (desc_rd_gnt_i), .desc_rd_valid_i (desc_rd_valid_i), .desc_rd_data_i (desc_rd_data_i),
      .busy_o (busy_o), .done_o (done_o), .err_o (err_o), .err_code_o (err_code_o), .cur_layer_o (cur_layer_o),
      .layer_cycles_o (layer_cycles_o), .total_cycles_o (total_cycles_o), .array_busy_cycles_o (array_busy_cycles_o),
      .array_valid_i (array_valid_i),
      .eng_desc_o (eng_desc_o),
      .conv_start_o (conv_start_o), .conv_done_i (conv_done_i), .conv_err_i (conv_err_i), .conv_err_code_i (conv_err_code_i),
      .pool_start_o (pool_start_o), .pool_done_i (pool_done_i), .pool_err_i (pool_err_i), .pool_err_code_i (pool_err_code_i),
      .argmax_start_o (argmax_start_o), .argmax_done_i (argmax_done_i), .argmax_err_i (argmax_err_i), .argmax_err_code_i (argmax_err_code_i)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_checked = 0, n_errors = 0;

  // ---------------------------------------------------------------------------
  // Stub engines and event logging (falling edge)
  // ---------------------------------------------------------------------------
  int    fail_layer = -1;          // layer whose engine reports an error
  err_e  fail_code  = ERR_NONE;
  int    max_delay  = 20;          // engine latency 1 .. max_delay cycles
  int    start_log[$];             // 0 conv, 1 pool, 2 argmax
  desc_t desc_log[$];
  int    conv_cnt = 0, pool_cnt = 0, argmax_cnt = 0;
  int    done_pulses = 0, err_pulses = 0;
  err_e  last_err_code;
  int    last_err_layer;
  int    busy_at_done_or_err = 0;

  // cycle-exact model of the counters
  int unsigned m_total = 0, m_array = 0;
  int unsigned m_layer [NUM_DESC];
  int          eng_timer = 0, eng_kind = -1;

  logic tb_en = 1'b0;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (tb_en) begin
      // Drive this cycle's array_valid first: the DUT samples it at the next
      // rising edge together with the busy state visible now.
      array_valid_i = 1'($urandom_range(1));
      // counters model: every cycle in which busy_o is high
      if (busy_o) begin
        m_total++;
        m_layer[cur_layer_o]++;
        if (array_valid_i) m_array++;
      end
      // event capture
      if (done_o) begin done_pulses++; if (busy_o) busy_at_done_or_err++; end
      if (err_o)  begin err_pulses++;  last_err_code = err_code_o; last_err_layer = int'(cur_layer_o); if (busy_o) busy_at_done_or_err++; end
      // stub engines
      conv_done_i = 1'b0; pool_done_i = 1'b0; argmax_done_i = 1'b0;
      conv_err_i  = 1'b0; pool_err_i  = 1'b0; argmax_err_i  = 1'b0;
      if (conv_start_o || pool_start_o || argmax_start_o) begin
        if (eng_kind != -1) begin n_errors++; $display("ERROR @%0t: engine started while another is running", $time); end
        eng_kind = conv_start_o ? 0 : (pool_start_o ? 1 : 2);
        if (conv_start_o) conv_cnt++; if (pool_start_o) pool_cnt++; if (argmax_start_o) argmax_cnt++;
        start_log.push_back(eng_kind);
        desc_log.push_back(eng_desc_o);
        eng_timer = $urandom_range(max_delay, 1);
      end else if (eng_kind != -1) begin
        eng_timer--;
        if (eng_timer == 0) begin
          if (int'(cur_layer_o) == fail_layer) begin
            case (eng_kind) 0: begin conv_err_i = 1'b1; conv_err_code_i = fail_code; end
                            1: begin pool_err_i = 1'b1; pool_err_code_i = fail_code; end
                            default: begin argmax_err_i = 1'b1; argmax_err_code_i = fail_code; end endcase
          end else begin
            case (eng_kind) 0: conv_done_i = 1'b1; 1: pool_done_i = 1'b1; default: argmax_done_i = 1'b1; endcase
          end
          eng_kind = -1;
        end
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Descriptor read port model, as npu_regs behaves: a request is granted in
  // its own cycle unless a host read takes the mux (random, desc_busy_pct of
  // the cycles), and the word of the table entry follows one cycle later.
  // Requests must be for the current layer and arrive in word order.
  // ---------------------------------------------------------------------------
  int          desc_busy_pct = 25;
  int          desc_req_words = 0, desc_stalls = 0;
  logic        desc_pend = 1'b0;
  logic [31:0] desc_pend_word = '0;
  int          desc_next_word = 0;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    desc_words_t words;
    desc_rd_valid_i = desc_pend;
    desc_rd_data_i  = desc_pend_word;
    desc_rd_gnt_i   = desc_rd_en_o && ($urandom_range(99) >= desc_busy_pct);
    if (desc_rd_en_o && !desc_rd_gnt_i) desc_stalls++;
    if (desc_rd_en_o) begin
      if (desc_rd_idx_o != cur_layer_o || int'(desc_rd_word_o) >= DESC_USED_WORDS) begin
        n_errors++;
        $display("ERROR @%0t: descriptor request idx %0d word %0d during layer %0d", $time,
                 desc_rd_idx_o, desc_rd_word_o, cur_layer_o);
      end
    end
    if (desc_rd_gnt_i) begin
      if (int'(desc_rd_word_o) != desc_next_word) begin
        n_errors++;
        $display("ERROR @%0t: descriptor word %0d requested, expected %0d", $time, desc_rd_word_o, desc_next_word);
      end
      desc_next_word = (desc_next_word + 1) % DESC_USED_WORDS;
      words          = desc_to_words(desc_i[desc_rd_idx_o]);
      desc_pend_word = words[desc_rd_word_o];
      desc_req_words++;
    end
    desc_pend = desc_rd_gnt_i;
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  task automatic check(input bit cond, input string what);
    n_checked++;
    if (!cond) begin n_errors++; $display("ERROR @%0t: %s", $time, what); end
  endtask

  function automatic desc_t mk(input op_e op, input int k, input int pad, input int in_h, input int in_w,
                               input int c_in, input int c_out, input int in_base, input int out_base,
                               input int w_base, input int b_base, input int relu);
    desc_t d;
    /* verilator lint_off UNUSEDSIGNAL */
    int unused_relu;
    /* verilator lint_on UNUSEDSIGNAL */
    unused_relu = relu;
    d = '0;
    d.op = op; d.k = 4'(k); d.pad = 4'(pad); d.stride_m1 = 2'd0;
    d.in_h = 16'(in_h); d.in_w = 16'(in_w); d.c_in = 16'(c_in); d.c_out = 16'(c_out);
    d.in_base = 32'(in_base); d.out_base = 32'(out_base); d.w_base = 32'(w_base); d.b_base = 32'(b_base);
    d.rq_mult = 32'sh4000_0000; d.rq_shift = 6'd31;
    d.z_in = -8'sd128; d.z_out = -8'sd128; d.act_min = -8'sd128; d.act_max = 8'sd127;
    return d;
  endfunction

  task automatic load_lenet();
    desc_i = '0;
    desc_i[0] = mk(OP_CONV2D,     5, 2, 28, 28,   1,   6, 32'h0000, 32'h0400, 32'h00000, 32'h00100, 1);
    desc_i[1] = mk(OP_MAXPOOL2X2, 0, 0, 28, 28,   6,   0, 32'h0400, 32'h1800, 0, 0, 0);
    desc_i[2] = mk(OP_CONV2D,     5, 0, 14, 14,   6,  16, 32'h1800, 32'h1D00, 32'h00200, 32'h00C00, 1);
    desc_i[3] = mk(OP_MAXPOOL2X2, 0, 0, 10, 10,  16,   0, 32'h1D00, 32'h2400, 0, 0, 0);
    desc_i[4] = mk(OP_FC,         1, 0,  1,  1, 400, 120, 32'h2400, 32'h2600, 32'h01000, 32'h0CC00, 1);
    desc_i[5] = mk(OP_FC,         1, 0,  1,  1, 120,  84, 32'h2600, 32'h2700, 32'h0D000, 32'h0FA00, 1);
    desc_i[6] = mk(OP_FC,         1, 0,  1,  1,  84,  10, 32'h2700, 32'h2780, 32'h10000, 32'h10600, 0);
    desc_i[7] = mk(OP_ARGMAX,     0, 0,  0,  0,  10,   0, 32'h2780, 0, 0, 0, 0);
  endtask

  // Start a run and wait until done or err. Resets the models first.
  task automatic run(input int layers, output bit got_done, output bit got_err);
    int d0, e0, guard;
    @(negedge clk_i);
    layer_cnt_i = 5'(layers);
    m_total = 0; m_array = 0;
    for (int i = 0; i < NUM_DESC; i++) m_layer[i] = 0;
    start_log.delete(); desc_log.delete();
    conv_cnt = 0; pool_cnt = 0; argmax_cnt = 0;
    d0 = done_pulses; e0 = err_pulses;
    start_i = 1'b1;
    @(negedge clk_i);
    start_i = 1'b0;
    guard = 0;
    while (done_pulses == d0 && err_pulses == e0 && guard < 100000) begin @(negedge clk_i); guard++; end
    got_done = (done_pulses != d0);
    got_err  = (err_pulses != e0);
    if (guard >= 100000) begin n_errors++; $display("ERROR: run did not finish"); end
    @(negedge clk_i);
  endtask

  task automatic check_counters(input string what);
    check(total_cycles_o == m_total, $sformatf("%s: total_cycles %0d vs model %0d", what, total_cycles_o, m_total));
    check(array_busy_cycles_o == m_array, $sformatf("%s: array_busy_cycles %0d vs model %0d", what, array_busy_cycles_o, m_array));
    for (int i = 0; i < NUM_DESC; i++)
      check(layer_cycles_o[i] == m_layer[i], $sformatf("%s: layer_cycles[%0d] %0d vs model %0d", what, i, layer_cycles_o[i], m_layer[i]));
  endtask

  // Expect a single-layer program to fail with `code` at layer `layer`.
  task automatic expect_err(input int layers, input err_e code, input int layer, input string what);
    bit gd, ge;
    run(layers, gd, ge);
    check(ge && !gd, $sformatf("%s: expected err pulse only (done=%0d err=%0d)", what, gd, ge));
    check(last_err_code == code, $sformatf("%s: err_code %0d, expected %0d", what, last_err_code, code));
    check(last_err_layer == layer, $sformatf("%s: cur_layer %0d, expected %0d", what, last_err_layer, layer));
    check(busy_o == 1'b0, $sformatf("%s: busy after error", what));
  endtask

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  int seed = 1;
  string vec_path = "";

  initial begin
    bit gd, ge;
    int fd, code, v_code, n_vec = 0, hist [6];
    desc_words_t w;
    int wv;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    void'($value$plusargs("desc_vectors=%s", vec_path));
    void'($urandom(seed));
    $display("npu_sequencer_tb: seed=%0d vectors='%s'", seed, vec_path);

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    tb_en = 1'b1;

    // ---- layer count errors ----
    load_lenet();
    expect_err(0, ERR_BAD_LAYER_CNT, 0, "LAYER_CNT = 0");
    expect_err(17, ERR_BAD_LAYER_CNT, 0, "LAYER_CNT = 17");
    check(start_log.size() == 0, "no engine started on a bad layer count");

    // ---- LeNet program ----
    run(8, gd, ge);
    check(gd && !ge, "LeNet: done without error");
    check(start_log.size() == 8, $sformatf("LeNet: 8 engine starts (got %0d)", start_log.size()));
    if (start_log.size() == 8) begin
      check(start_log[0] == 0 && start_log[1] == 1 && start_log[2] == 0 && start_log[3] == 1 &&
            start_log[4] == 0 && start_log[5] == 0 && start_log[6] == 0 && start_log[7] == 2, "LeNet: engine order");
      check(desc_log[0] == desc_i[0] && desc_log[2] == desc_i[2] && desc_log[7] == desc_i[7], "LeNet: descriptors forwarded unchanged");
      check(desc_log[4].op == OP_FC && desc_log[4].k == 4'd1 && desc_log[4].pad == 4'd0 &&
            desc_log[4].in_h == 16'd1 && desc_log[4].in_w == 16'd1 && desc_log[4].c_in == 16'd400 &&
            desc_log[4].c_out == 16'd120 && desc_log[4].w_base == 32'h01000, "LeNet: FC forwarded as 1x1 view");
    end
    check(conv_cnt == 5 && pool_cnt == 2 && argmax_cnt == 1, "LeNet: engine counts");
    check(cur_layer_o == 4'd7, "LeNet: cur_layer stays at the last layer");
    check(busy_at_done_or_err == 0, "busy is low in the done / err cycle");
    check_counters("LeNet");
    check(total_cycles_o > 8, "LeNet: counters ran");

    // ---- start while busy is ignored ----
    begin
      int d0;
      @(negedge clk_i);
      d0 = done_pulses;
      start_i = 1'b1; @(negedge clk_i); start_i = 1'b0;
      repeat (3) @(negedge clk_i);
      check(busy_o, "second run started");
      start_i = 1'b1; @(negedge clk_i); start_i = 1'b0;      // while busy
      while (done_pulses == d0) @(negedge clk_i);
      @(negedge clk_i);
      check(done_pulses == d0 + 1 && conv_cnt == 10 && pool_cnt == 4 && argmax_cnt == 2, "start while busy ignored");
    end

    // ---- NOP layers ----
    desc_i = '0;
    desc_i[1] = mk(OP_CONV2D, 3, 1, 8, 8, 2, 4, 32'h0000, 32'h1000, 32'h0000, 32'h1000, 1);
    run(4, gd, ge);
    check(gd && !ge && start_log.size() == 1 && start_log[0] == 0, "NOP layers skipped, one conv start");
    check_counters("NOP");

    // ---- descriptor errors, single layer ----
    desc_i = '0;
    desc_i[0] = mk(OP_CONV2D, 5, 2, 28, 28, 1, 6, 32'h0000, 32'h0400, 32'h00000, 32'h00100, 1);
    run(1, gd, ge); check(gd && !ge, "baseline conv1 descriptor is valid");
    desc_i[0].op = op_e'(4'd7);                                   expect_err(1, ERR_BAD_OP, 0, "op 7");
    desc_i[0].op = op_e'(4'd15);                                  expect_err(1, ERR_BAD_OP, 0, "op 15");
    desc_i[0].op = OP_CONV2D;
    desc_i[0].k = 4'd0;                                           expect_err(1, ERR_BAD_PARAM, 0, "k = 0");
    desc_i[0].k = 4'd8;                                           expect_err(1, ERR_BAD_PARAM, 0, "k = 8");
    desc_i[0].k = 4'd5;
    desc_i[0].pad = 4'd8;                                         expect_err(1, ERR_BAD_PARAM, 0, "pad = 8");
    desc_i[0].pad = 4'd2;
    desc_i[0].stride_m1 = 2'd1;                                   expect_err(1, ERR_BAD_PARAM, 0, "stride 2");
    desc_i[0].stride_m1 = 2'd0;
    desc_i[0].c_in = 16'd0;                                       expect_err(1, ERR_BAD_PARAM, 0, "c_in = 0");
    desc_i[0].c_in = 16'd1;
    desc_i[0].c_out = 16'd0;                                      expect_err(1, ERR_BAD_PARAM, 0, "c_out = 0");
    desc_i[0].c_out = 16'd6;
    desc_i[0].rq_shift = 6'd0;                                    expect_err(1, ERR_BAD_PARAM, 0, "rq_shift = 0");
    desc_i[0].rq_shift = 6'd63;                                   expect_err(1, ERR_BAD_PARAM, 0, "rq_shift = 63");
    desc_i[0].rq_shift = 6'd31;
    desc_i[0].rq_mult = 32'sh3FFF_FFFF;                           expect_err(1, ERR_BAD_PARAM, 0, "rq_mult below 2^30");
    desc_i[0].rq_mult = 32'sh8000_0000;                           expect_err(1, ERR_BAD_PARAM, 0, "rq_mult negative");
    desc_i[0].rq_mult = 32'sh7FFF_FFFF;
    run(1, gd, ge); check(gd && !ge, "rq_mult = 2^31 - 1 is valid");
    desc_i[0].rq_mult = 32'sh4000_0000;
    desc_i[0].act_min = 8'sd10; desc_i[0].act_max = 8'sd9;        expect_err(1, ERR_BAD_PARAM, 0, "act_min > act_max");
    desc_i[0].act_min = -8'sd128; desc_i[0].act_max = 8'sd127;
    desc_i[0].in_h = 16'd1; desc_i[0].pad = 4'd0;                 expect_err(1, ERR_BAD_OUT_DIM, 0, "k > in_h + 2 pad");
    desc_i[0].in_h = 16'd28; desc_i[0].pad = 4'd2;
    desc_i[0].w_base = 32'h4;                                     expect_err(1, ERR_BAD_ADDR, 0, "w_base misaligned");
    desc_i[0].w_base = 32'h0;
    desc_i[0].b_base = 32'h102;                                   expect_err(1, ERR_BAD_ADDR, 0, "b_base misaligned");
    desc_i[0].b_base = 32'h104;                                   expect_err(1, ERR_BAD_ADDR, 0, "b_base only 4-byte aligned");
    desc_i[0].b_base = 32'h100;
    desc_i[0].out_base = 32'(ABUF_BYTES - 4703);                  expect_err(1, ERR_BAD_ADDR, 0, "output one byte past the buffer");
    desc_i[0].out_base = 32'(ABUF_BYTES - 4704);
    run(1, gd, ge); check(gd && !ge, "output exactly at the end of the buffer is valid");
    desc_i[0].out_base = 32'h0400;
    desc_i[0].in_base = 32'hFFFF_FF00;                            expect_err(1, ERR_BAD_ADDR, 0, "in_base huge");
    desc_i[0].in_base = 32'h0;
    desc_i[0].w_base = 32'(WBUF_BYTES - 192);                     expect_err(1, ERR_BAD_ADDR, 0, "weights past the buffer");
    desc_i[0].w_base = 32'(WBUF_BYTES - 200);
    run(1, gd, ge); check(gd && !ge, "weights exactly at the end are valid");
    desc_i[0].w_base = 32'h0;
    // k_total limit: c_in 4097 with k = 1 is one too many; 4096 is fine
    desc_i[0].k = 4'd1; desc_i[0].pad = 4'd0; desc_i[0].c_in = 16'd4097; desc_i[0].in_h = 16'd1; desc_i[0].in_w = 16'd1;
    desc_i[0].c_out = 16'd1;                                      expect_err(1, ERR_BAD_PARAM, 0, "k_total 4097");
    desc_i[0].c_in = 16'd4096;
    run(1, gd, ge); check(gd && !ge, "k_total 4096 is valid");
    // FC ignores k / pad / in_h
    desc_i[0] = mk(OP_FC, 9, 15, 7, 7, 400, 120, 32'h2400, 32'h2600, 32'h01000, 32'h0CC00, 1);
    run(1, gd, ge); check(gd && !ge, "FC ignores k, pad and in_h/in_w");
    check(desc_log[0].k == 4'd1 && desc_log[0].pad == 4'd0 && desc_log[0].in_h == 16'd1, "FC view in forwarded descriptor");
    desc_i[0].stride_m1 = 2'd3;
    run(1, gd, ge); check(gd && !ge, "FC ignores stride");
    // maxpool and argmax errors
    desc_i[0] = mk(OP_MAXPOOL2X2, 0, 0, 1, 8, 6, 0, 32'h0, 32'h1000, 0, 0, 0);  expect_err(1, ERR_BAD_OUT_DIM, 0, "pool in_h = 1");
    desc_i[0] = mk(OP_MAXPOOL2X2, 0, 0, 8, 8, 0, 0, 32'h0, 32'h1000, 0, 0, 0);  expect_err(1, ERR_BAD_PARAM, 0, "pool c_in = 0");
    desc_i[0] = mk(OP_MAXPOOL2X2, 0, 0, 8, 8, 6, 0, 32'h7F00, 32'h1000, 0, 0, 0); expect_err(1, ERR_BAD_ADDR, 0, "pool input past the buffer");
    desc_i[0] = mk(OP_ARGMAX, 0, 0, 0, 0, 17, 0, 32'h0, 0, 0, 0, 0);            expect_err(1, ERR_BAD_PARAM, 0, "argmax c_in = 17");
    desc_i[0] = mk(OP_ARGMAX, 0, 0, 0, 0, 0, 0, 32'h0, 0, 0, 0, 0);             expect_err(1, ERR_BAD_PARAM, 0, "argmax c_in = 0");
    desc_i[0] = mk(OP_ARGMAX, 0, 0, 0, 0, 16, 0, 32'(ABUF_BYTES - 15), 0, 0, 0, 0); expect_err(1, ERR_BAD_ADDR, 0, "argmax past the buffer");
    desc_i[0] = mk(OP_ARGMAX, 0, 0, 0, 0, 16, 0, 32'(ABUF_BYTES - 16), 0, 0, 0, 0);
    run(1, gd, ge); check(gd && !ge, "argmax exactly at the end is valid");

    // ---- error in the middle of a program points at that layer ----
    load_lenet();
    desc_i[3].c_in = 16'd0;
    expect_err(8, ERR_BAD_PARAM, 3, "bad layer 3 of 8");
    check(start_log.size() == 3, "three layers ran before the bad one");
    check_counters("bad layer 3");

    // ---- engine-reported error ----
    load_lenet();
    fail_layer = 5; fail_code = ERR_BAD_ADDR;
    expect_err(8, ERR_BAD_ADDR, 5, "engine error at layer 5");
    check(start_log.size() == 6, "six starts before the engine error");
    fail_layer = -1;

    // ---- Python descriptor vectors ----
    if (vec_path != "") begin
      fd = $fopen(vec_path, "r");
      if (fd == 0) begin n_errors++; $display("ERROR: cannot open '%s'", vec_path); end
      else begin
        max_delay = 2;
        for (int i = 0; i < 6; i++) hist[i] = 0;
        while (!$feof(fd)) begin
          code = 0;
          for (int k = 0; k < DESC_USED_WORDS; k++) begin code += $fscanf(fd, "%d", wv); w[k] = wv; end
          code += $fscanf(fd, "%d\n", v_code);
          if (code != DESC_USED_WORDS + 1) break;
          desc_i = '0;
          desc_i[0] = desc_from_words(w);
          run(1, gd, ge);
          n_vec++;
          hist[v_code]++;
          if (v_code == 0) check(gd && !ge, $sformatf("vector %0d: Python says valid", n_vec));
          else begin
            check(ge && !gd, $sformatf("vector %0d: Python expects error %0d", n_vec, v_code));
            check(int'(last_err_code) == v_code, $sformatf("vector %0d: code %0d, Python %0d", n_vec, last_err_code, v_code));
          end
        end
        $fclose(fd);
        $display("python vectors: %0d (none=%0d op=%0d param=%0d addr=%0d outdim=%0d)", n_vec, hist[0], hist[1], hist[2], hist[3], hist[5]);
      end
    end

    check(desc_req_words > 0 && desc_stalls > 0, "descriptor fetch exercised with grant stalls");
    if (n_errors == 0) begin
      $display("PASS: npu_sequencer_tb checked=%0d python=%0d desc_words=%0d stalls=%0d",
               n_checked, n_vec, desc_req_words, desc_stalls);
      $finish;
    end else begin
      $display("FAIL: npu_sequencer_tb errors=%0d checked=%0d", n_errors, n_checked);
      $fatal(1, "npu_sequencer_tb failed");
    end
  end

  initial begin
    #500ms;
    $fatal(1, "TIMEOUT: npu_sequencer_tb did not finish");
  end

endmodule

`default_nettype wire
