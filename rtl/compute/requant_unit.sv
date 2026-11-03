`default_nettype none

// Requantization unit: one int32 accumulator value -> one int8 activation.
//
// Implements docs/architecture.md section 3.3 as a fixed 3-stage pipeline:
//   S1  sum     = acc + bias                                   (33-bit signed)
//   S2  prod    = sum * mult                                   (65-bit signed)
//   S3  rounded = (prod + 2^(shift-1)) >>> shift               (round half up)
//       q       = clamp(rounded + z_out, act_min, act_max)     (int8)
//
// S3 is computed without the 65-bit rounding add and with a narrow tail:
//   - for shift >= 1, (prod + 2^(shift-1)) >>> shift == ((prod >>> (shift-1)) + 1) >>> 1
//     (floor division: write prod >>> (shift-1) as k, then both sides are floor((k+1)/2));
//   - prod >>> (shift-1) is saturated to SAT_W bits first. Any value outside that
//     range is at least 4096 away from zero after rounding, so with an int8 zero
//     point it compares the same way against every int8 act_min / act_max, and the
//     clamp result is unchanged. The zero-point add and the compares are SAT_W+1 bits.
//
// Contract:
//   - mult is a positive int32 in [2^30, 2^31); shift is in 1..62.
//     shift == 0 is outside the documented range but kept well defined:
//     no rounding term and no shift.
//   - act_min <= act_max. If violated the result is act_max when the value is
//     above act_max, else act_min when below act_min, else the value.
//   - valid_o is valid_i delayed by exactly RQ_PIPE_LAT cycles; q_o is only
//     meaningful when valid_o is high.
module requant_unit
  import npu_pkg::*;
(
  input  logic                        clk_i,
  input  logic                        rst_ni,

  input  logic                        valid_i,
  input  logic signed [ACC_W-1:0]     acc_i,
  input  logic signed [ACC_W-1:0]     bias_i,
  input  logic signed [ACC_W-1:0]     mult_i,
  input  logic [RQ_SHIFT_W-1:0]       shift_i,
  input  logic signed [ACT_W-1:0]     z_out_i,
  input  logic signed [ACT_W-1:0]     act_min_i,
  input  logic signed [ACT_W-1:0]     act_max_i,

  output logic                        valid_o,
  output logic signed [ACT_W-1:0]     q_o
);

  localparam int unsigned SUM_W  = ACC_W + 1;          // 33
  localparam int unsigned MUL_W  = SUM_W + ACC_W;      // 65
  localparam int unsigned SAT_W  = 14;                 // saturated width of the shifted product

  // ---------------------------------------------------------------------------
  // Stage 1: bias add
  // ---------------------------------------------------------------------------
  logic signed [SUM_W-1:0] sum_c;
  assign sum_c = $signed({acc_i[ACC_W-1], acc_i}) + $signed({bias_i[ACC_W-1], bias_i});

  logic                     s1_valid_q;
  logic signed [SUM_W-1:0]  s1_sum_q;
  logic signed [ACC_W-1:0]  s1_mult_q;
  logic [RQ_SHIFT_W-1:0]    s1_shift_q;
  logic signed [ACT_W-1:0]  s1_z_q, s1_min_q, s1_max_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1_valid_q <= 1'b0;
      s1_sum_q   <= '0;
      s1_mult_q  <= '0;
      s1_shift_q <= '0;
      s1_z_q     <= '0;
      s1_min_q   <= '0;
      s1_max_q   <= '0;
    end else begin
      s1_valid_q <= valid_i;
      if (valid_i) begin
        s1_sum_q   <= sum_c;
        s1_mult_q  <= mult_i;
        s1_shift_q <= shift_i;
        s1_z_q     <= z_out_i;
        s1_min_q   <= act_min_i;
        s1_max_q   <= act_max_i;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 2: fixed-point multiply
  // ---------------------------------------------------------------------------
  logic signed [MUL_W-1:0] prod_c;
  assign prod_c = $signed({{ACC_W{s1_sum_q[SUM_W-1]}}, s1_sum_q})
                * $signed({{SUM_W{s1_mult_q[ACC_W-1]}}, s1_mult_q});

  logic                      s2_valid_q;
  logic signed [MUL_W-1:0]  s2_prod_q;
  logic [RQ_SHIFT_W-1:0]     s2_shift_q;
  logic signed [ACT_W-1:0]   s2_z_q, s2_min_q, s2_max_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s2_valid_q <= 1'b0;
      s2_prod_q  <= '0;
      s2_shift_q <= '0;
      s2_z_q     <= '0;
      s2_min_q   <= '0;
      s2_max_q   <= '0;
    end else begin
      s2_valid_q <= s1_valid_q;
      if (s1_valid_q) begin
        s2_prod_q  <= prod_c;
        s2_shift_q <= s1_shift_q;
        s2_z_q     <= s1_z_q;
        s2_min_q   <= s1_min_q;
        s2_max_q   <= s1_max_q;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 3: round, shift, zero point, clamp
  // ---------------------------------------------------------------------------
  logic [RQ_SHIFT_W-1:0]    sh_m1_c;      // shift - 1, or 0 when shift is 0
  logic signed [MUL_W-1:0]  shr_c;        // prod >>> (shift - 1)
  logic                     in_range_c;   // shr_c fits in SAT_W bits
  logic signed [SAT_W-1:0]  sat_c;
  /* verilator lint_off UNUSEDSIGNAL */
  logic signed [SAT_W:0]    inc_c;        // bit 0 is dropped by the >>> 1
  /* verilator lint_on UNUSEDSIGNAL */
  logic signed [SAT_W-1:0]  rounded_c;
  logic signed [SAT_W:0]    with_zp_c;
  logic signed [SAT_W:0]    min_x_c, max_x_c;
  logic signed [ACT_W-1:0]  q_c;

  always_comb begin
    sh_m1_c    = (s2_shift_q == '0) ? '0 : s2_shift_q - RQ_SHIFT_W'(1);
    shr_c      = s2_prod_q >>> sh_m1_c;
    in_range_c = (shr_c[MUL_W-1:SAT_W-1] == '0) || (shr_c[MUL_W-1:SAT_W-1] == '1);
    if (in_range_c) begin
      sat_c = shr_c[SAT_W-1:0];
    end else begin
      sat_c = shr_c[MUL_W-1] ? {1'b1, {(SAT_W-1){1'b0}}} : {1'b0, {(SAT_W-1){1'b1}}};
    end
    inc_c = $signed({sat_c[SAT_W-1], sat_c}) + $signed((SAT_W+1)'(1));
    if (s2_shift_q == '0) begin
      rounded_c = sat_c;                       // shift 0: no rounding term, no shift
    end else begin
      rounded_c = inc_c[SAT_W:1];              // (k + 1) >>> 1
    end
    with_zp_c = $signed({rounded_c[SAT_W-1], rounded_c})
              + $signed({{(SAT_W+1-ACT_W){s2_z_q[ACT_W-1]}}, s2_z_q});
    min_x_c   = $signed({{(SAT_W+1-ACT_W){s2_min_q[ACT_W-1]}}, s2_min_q});
    max_x_c   = $signed({{(SAT_W+1-ACT_W){s2_max_q[ACT_W-1]}}, s2_max_q});

    if (with_zp_c > max_x_c) begin
      q_c = s2_max_q;
    end else if (with_zp_c < min_x_c) begin
      q_c = s2_min_q;
    end else begin
      q_c = with_zp_c[ACT_W-1:0];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_o <= 1'b0;
      q_o     <= '0;
    end else begin
      valid_o <= s2_valid_q;
      if (s2_valid_q) begin
        q_o <= q_c;
      end
    end
  end

endmodule

`default_nettype wire
