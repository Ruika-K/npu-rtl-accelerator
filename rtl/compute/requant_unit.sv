`default_nettype none

// Requantization unit: one int32 accumulator value -> one int8 activation.
//
// Implements docs/architecture.md section 3.3 as a fixed 3-stage pipeline:
//   S1  sum     = acc + bias                                   (33-bit signed)
//   S2  prod    = sum * mult                                   (65-bit signed)
//   S3  rounded = (prod + 2^(shift-1)) >>> shift               (round half up)
//       q       = clamp(rounded + z_out, act_min, act_max)     (int8)
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
  localparam int unsigned ZP_W   = MUL_W + 1;          // 66

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
  logic signed [MUL_W-1:0] rnd_term_c;
  logic signed [MUL_W-1:0] rounded_c;
  logic signed [MUL_W-1:0] shifted_c;
  logic signed [ZP_W-1:0]   with_zp_c;
  logic signed [ZP_W-1:0]   min_x_c, max_x_c;
  logic signed [ACT_W-1:0]  q_c;

  always_comb begin
    if (s2_shift_q == '0) begin
      rnd_term_c = '0;
    end else begin
      rnd_term_c = MUL_W'(1) <<< (s2_shift_q - RQ_SHIFT_W'(1));
    end
    rounded_c = s2_prod_q + rnd_term_c;
    shifted_c = rounded_c >>> s2_shift_q;
    with_zp_c = $signed({shifted_c[MUL_W-1], shifted_c})
              + $signed({{(ZP_W-ACT_W){s2_z_q[ACT_W-1]}}, s2_z_q});
    min_x_c   = $signed({{(ZP_W-ACT_W){s2_min_q[ACT_W-1]}}, s2_min_q});
    max_x_c   = $signed({{(ZP_W-ACT_W){s2_max_q[ACT_W-1]}}, s2_max_q});

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
