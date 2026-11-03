`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for requant_unit.
//   Phase 1: directed cases with hand-computed expectations (also validates
//            the reference function itself).
//   Phase 2: random cases checked against npu_tb_pkg::ref_requant.
//   Phase 3: optional vector file from model/gen_requant_vectors.py
//            (+vectors=<path>): checks RTL and ref_requant against Python.
// Plusargs: +seed=<int> (default 1), +n=<random cases> (default 20000).
module requant_unit_tb
  import npu_pkg::*;
  import npu_tb_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i;
  logic rst_ni;

  logic                     valid_i;
  logic signed [ACC_W-1:0]  acc_i, bias_i, mult_i;
  logic [RQ_SHIFT_W-1:0]    shift_i;
  logic signed [ACT_W-1:0]  z_out_i, act_min_i, act_max_i;
  logic                     valid_o;
  logic signed [ACT_W-1:0]  q_o;

  requant_unit dut (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .valid_i   (valid_i),
      .acc_i     (acc_i),
      .bias_i    (bias_i),
      .mult_i    (mult_i),
      .shift_i   (shift_i),
      .z_out_i   (z_out_i),
      .act_min_i (act_min_i),
      .act_max_i (act_max_i),
      .valid_o   (valid_o),
      .q_o       (q_o)
  );

  // ---------------------------------------------------------------------------
  // Clock, reset, bookkeeping
  // ---------------------------------------------------------------------------
  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_checked = 0;
  int n_errors  = 0;
  int n_hand    = 0;
  int n_vec     = 0;

  logic signed [ACT_W-1:0] exp_q[$];

  // valid_i delayed by RQ_PIPE_LAT cycles, to prove the fixed latency.
  logic [RQ_PIPE_LAT-1:0] valid_pipe_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) valid_pipe_q <= '0;
    else         valid_pipe_q <= {valid_pipe_q[RQ_PIPE_LAT-2:0], valid_i};
  end

  // Monitor on the falling edge: all signals are stable.
  logic mon_en = 1'b0;
  logic signed [ACT_W-1:0] exp_val;
  // Blocking assignments are intended here: this is a checker, not a flop.
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      if (valid_o !== valid_pipe_q[RQ_PIPE_LAT-1]) begin
        n_errors++;
        $display("ERROR @%0t: valid_o=%b but expected %b (latency is not %0d)",
                 $time, valid_o, valid_pipe_q[RQ_PIPE_LAT-1], RQ_PIPE_LAT);
      end
      if (valid_o) begin
        if (exp_q.size() == 0) begin
          n_errors++;
          $display("ERROR @%0t: valid_o with no expected value queued", $time);
        end else begin
          exp_val = exp_q.pop_front();
          n_checked++;
          if (q_o !== exp_val) begin
            n_errors++;
            if (n_errors < 20)
              $display("ERROR @%0t: q_o=%0d expected %0d", $time, q_o, exp_val);
          end
        end
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Drivers
  // ---------------------------------------------------------------------------
  // Inputs are driven on the falling edge with blocking assignments, so the
  // DUT samples them on the next rising edge in every simulator alike.
  task automatic idle_cycle();
    @(negedge clk_i);
    valid_i   = 1'b0;
    // Garbage on the data inputs while idle must be ignored.
    acc_i     = $urandom();
    bias_i    = $urandom();
    mult_i    = $urandom();
    shift_i   = 6'($urandom());
    z_out_i   = 8'($urandom());
    act_min_i = 8'($urandom());
    act_max_i = 8'($urandom());
  endtask

  task automatic drive(
      input logic signed [ACC_W-1:0] acc, bias, mult,
      input logic [RQ_SHIFT_W-1:0]   shift,
      input logic signed [ACT_W-1:0] z_out, act_min, act_max,
      input logic signed [ACT_W-1:0] expected);
    @(negedge clk_i);
    valid_i   = 1'b1;
    acc_i     = acc;
    bias_i    = bias;
    mult_i    = mult;
    shift_i   = shift;
    z_out_i   = z_out;
    act_min_i = act_min;
    act_max_i = act_max;
    exp_q.push_back(expected);
  endtask

  // Reference-checked drive, with random bubbles.
  task automatic drive_ref(
      input logic signed [ACC_W-1:0] acc, bias, mult,
      input logic [RQ_SHIFT_W-1:0]   shift,
      input logic signed [ACT_W-1:0] z_out, act_min, act_max);
    if ($urandom_range(3) == 0) idle_cycle();
    drive(acc, bias, mult, shift, z_out, act_min, act_max,
          ref_requant(acc, bias, mult, shift, z_out, act_min, act_max));
  endtask

  // Hand-computed case: first validates ref_requant, then drives the DUT.
  task automatic hand(
      input logic signed [ACC_W-1:0] acc, bias, mult,
      input logic [RQ_SHIFT_W-1:0]   shift,
      input logic signed [ACT_W-1:0] z_out, act_min, act_max,
      input logic signed [ACT_W-1:0] expected);
    logic signed [ACT_W-1:0] r;
    r = ref_requant(acc, bias, mult, shift, z_out, act_min, act_max);
    if (r !== expected) begin
      n_errors++;
      $display("ERROR: ref_requant(%0d,%0d,%0d,%0d,%0d,%0d,%0d)=%0d, hand value %0d",
               acc, bias, mult, shift, z_out, act_min, act_max, r, expected);
    end
    n_hand++;
    drive(acc, bias, mult, shift, z_out, act_min, act_max, expected);
  endtask

  // Random operand generators that stress the interesting regions.
  function automatic logic signed [ACC_W-1:0] rand_acc();
    case ($urandom_range(3))
      0:       return $signed($urandom());                              // full range
      1:       return $signed($urandom_range(32'h000F_FFFF)) - 32'sd524288;  // +-2^19
      2:       return $signed($urandom_range(32'h0000_0FFF)) - 32'sd2048;    // small
      default: return ($urandom_range(1) == 0) ? 32'sh7FFF_FFFF : 32'sh8000_0000;
    endcase
  endfunction

  function automatic logic signed [ACC_W-1:0] rand_mult();
    case ($urandom_range(7))
      0:       return 32'sh4000_0000;
      1:       return 32'sh7FFF_FFFF;
      default: return $signed($urandom_range(32'h7FFF_FFFF, 32'h4000_0000));
    endcase
  endfunction

  function automatic logic [RQ_SHIFT_W-1:0] rand_shift();
    case ($urandom_range(3))
      0:       return 6'($urandom_range(62, 1));
      1:       return 6'($urandom_range(62, 56));   // large shifts
      default: return 6'($urandom_range(44, 24));   // realistic for LeNet
    endcase
  endfunction

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  int seed = 1;
  int n_random = 20000;
  string vec_path = "";

  initial begin
    logic signed [ACC_W-1:0] acc, bias, mult;
    logic [RQ_SHIFT_W-1:0]   shift;
    logic signed [ACT_W-1:0] z, mn, mx;
    int fd, code;
    int v_acc, v_bias, v_mult, v_shift, v_z, v_min, v_max, v_exp;
    logic signed [ACT_W-1:0] r;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 20000;
    void'($value$plusargs("vectors=%s", vec_path));
    void'($urandom(seed));
    $display("requant_unit_tb: seed=%0d n_random=%0d vectors='%s'", seed, n_random, vec_path);

    valid_i = 1'b0; acc_i = '0; bias_i = '0; mult_i = '0; shift_i = '0;
    z_out_i = '0; act_min_i = -8'sd128; act_max_i = 8'sd127;
    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    // Phase 1: hand-computed. mult = 2^30 and shift = 31 give an exact x0.5
    // multiplier, which makes rounding ties easy to construct.
    hand(32'sd0,  32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd0);
    hand(32'sd3,  32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd2);    //  1.5 ->  2
    hand(-32'sd3, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, -8'sd1);   // -1.5 -> -1
    hand(32'sd1,  32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd1);    //  0.5 ->  1
    hand(-32'sd1, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd0);    // -0.5 ->  0
    hand(32'sd5,  32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd3);    //  2.5 ->  3
    hand(32'sd4,  32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd2);    //  2.0 ->  2
    hand(32'sd200, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd100); // 100 exact
    // zero point and clamp
    hand(32'sd0, 32'sd0, 32'sh4000_0000, 6'd31, -8'sd128, -8'sd128, 8'sd127, -8'sd128);
    hand(32'sd0, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd100, -8'sd128, 8'sd127, 8'sd100);
    hand(32'sd60, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd100, -8'sd128, 8'sd127, 8'sd127);  // 30+100 -> clamp
    hand(-32'sd20, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd3, 8'sd3, 8'sd127, 8'sd3);      // ReLU: -10+3 -> z_out
    hand(32'sd20, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd3, 8'sd3, 8'sd50, 8'sd13);       // 10+3 inside
    hand(32'sd200, 32'sd0, 32'sh4000_0000, 6'd31, 8'sd3, 8'sd3, 8'sd50, 8'sd50);      // act_max
    // bias and 33-bit sum
    hand(32'sd10, 32'sd12, 32'sh4000_0000, 6'd31, 8'sd0, -8'sd128, 8'sd127, 8'sd11);  // 22*0.5
    hand(32'sh7FFF_FFFF, 32'sh7FFF_FFFF, 32'sh7FFF_FFFF, 6'd1, 8'sd0, -8'sd128, 8'sd127, 8'sd127);
    hand(32'sh8000_0000, 32'sh8000_0000, 32'sh7FFF_FFFF, 6'd1, 8'sd0, -8'sd128, 8'sd127, -8'sd128);
    hand(32'sh7FFF_FFFF, 32'sh8000_0000, 32'sh7FFF_FFFF, 6'd1, 8'sd0, -8'sd128, 8'sd127, -8'sd128); // sum = -1
    // large shift and shift = 0
    hand(32'sh7FFF_FFFF, 32'sd0, 32'sh7FFF_FFFF, 6'd62, 8'sd0, -8'sd128, 8'sd127, 8'sd1);
    hand(32'sh7FFF_FFFF, 32'sd0, 32'sh4000_0000, 6'd62, 8'sd0, -8'sd128, 8'sd127, 8'sd0);  // 2^61-2^30 + 2^61 < 2^62
    hand(32'sd0, 32'sd0, 32'sh7FFF_FFFF, 6'd0, 8'sd7, -8'sd128, 8'sd127, 8'sd7);
    hand(32'sd1, 32'sd0, 32'sh4000_0000, 6'd0, 8'sd0, -8'sd128, 8'sd127, 8'sd127);

    // Phase 2: random against the reference function.
    for (int i = 0; i < n_random; i++) begin
      acc   = rand_acc();
      bias  = ($urandom_range(1) == 0) ? rand_acc() : 32'sd0;
      mult  = rand_mult();
      shift = rand_shift();
      z     = 8'($urandom_range(255)) ;
      case ($urandom_range(2))
        0: begin mn = -8'sd128; mx = 8'sd127; end
        1: begin mn = z;        mx = 8'sd127; end      // ReLU
        default: begin
          mn = 8'($urandom_range(255));
          mx = 8'($urandom_range(255));
          if (mn > mx) begin logic signed [7:0] t; t = mn; mn = mx; mx = t; end
        end
      endcase
      drive_ref(acc, bias, mult, shift, z, mn, mx);
    end

    // Phase 3: Python-generated vectors.
    if (vec_path != "") begin
      fd = $fopen(vec_path, "r");
      if (fd == 0) begin
        n_errors++;
        $display("ERROR: cannot open vector file '%s'", vec_path);
      end else begin
        while (!$feof(fd)) begin
          code = $fscanf(fd, "%d %d %d %d %d %d %d %d\n",
                         v_acc, v_bias, v_mult, v_shift, v_z, v_min, v_max, v_exp);
          if (code != 8) break;
          acc = v_acc; bias = v_bias; mult = v_mult; shift = 6'(v_shift);
          z = 8'(v_z); mn = 8'(v_min); mx = 8'(v_max);
          r = ref_requant(acc, bias, mult, shift, z, mn, mx);
          if (r !== 8'(v_exp)) begin
            n_errors++;
            if (n_errors < 20)
              $display("ERROR: python expects %0d but ref_requant gives %0d for %0d %0d %0d %0d %0d %0d %0d",
                       v_exp, r, v_acc, v_bias, v_mult, v_shift, v_z, v_min, v_max);
          end
          n_vec++;
          if ($urandom_range(7) == 0) idle_cycle();
          drive(acc, bias, mult, shift, z, mn, mx, 8'(v_exp));
        end
        $fclose(fd);
      end
    end

    // Drain the pipeline.
    idle_cycle();
    repeat (RQ_PIPE_LAT + 2) @(posedge clk_i);
    @(negedge clk_i);

    if (exp_q.size() != 0) begin
      n_errors++;
      $display("ERROR: %0d expected values never appeared on the output", exp_q.size());
    end

    if (n_errors == 0) begin
      $display("PASS: requant_unit_tb checked=%0d (hand=%0d random=%0d python=%0d)",
               n_checked, n_hand, n_random, n_vec);
      $finish;
    end else begin
      $display("FAIL: requant_unit_tb errors=%0d checked=%0d", n_errors, n_checked);
      $fatal(1, "requant_unit_tb failed");
    end
  end

  initial begin
    #500ms;
    $fatal(1, "TIMEOUT: requant_unit_tb did not finish");
  end

endmodule

`default_nettype wire
