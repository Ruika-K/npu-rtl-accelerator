`default_nettype none

// Reference functions shared by the testbenches. They mirror the Python golden
// model in model/npu_ref.py; the unit testbenches check both against each other.
package npu_tb_pkg;
  import npu_pkg::*;

  // docs/architecture.md section 3.3, same four steps as requant_unit.
  function automatic logic signed [ACT_W-1:0] ref_requant(
      input logic signed [ACC_W-1:0]  acc,
      input logic signed [ACC_W-1:0]  bias,
      input logic signed [ACC_W-1:0]  mult,
      input logic [RQ_SHIFT_W-1:0]    shift,
      input logic signed [ACT_W-1:0]  z_out,
      input logic signed [ACT_W-1:0]  act_min,
      input logic signed [ACT_W-1:0]  act_max);
    logic signed [32:0] sum;
    logic signed [64:0] prod, rnd, rounded, shifted;
    logic signed [65:0] with_zp, min_x, max_x;
    sum     = $signed({acc[31], acc}) + $signed({bias[31], bias});
    prod    = $signed({{32{sum[32]}}, sum}) * $signed({{33{mult[31]}}, mult});
    rnd     = (shift == 6'd0) ? 65'sd0 : (65'sd1 <<< (shift - 6'd1));
    rounded = prod + rnd;
    shifted = rounded >>> shift;
    with_zp = $signed({shifted[64], shifted}) + $signed({{58{z_out[7]}}, z_out});
    min_x   = $signed({{58{act_min[7]}}, act_min});
    max_x   = $signed({{58{act_max[7]}}, act_max});
    if (with_zp > max_x) return act_max;
    if (with_zp < min_x) return act_min;
    return with_zp[7:0];
  endfunction

endpackage

`default_nettype wire
