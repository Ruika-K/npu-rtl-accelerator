`default_nettype none

// Processing element of the output-stationary systolic array
// (docs/architecture.md section 5.2).
//
// Each PE owns one output element. Per cycle, when valid_i is high:
//   prod = a_i * w_i                      (int8 x int8 -> int16)
//   acc  = first_i ? prod : acc + prod    (int32)
//   if last_i: shadow <= new acc          (result parked for the column drain)
// a_i and the row flags are forwarded one cycle later to the right neighbour,
// w_i one cycle later to the neighbour below. The forwarding registers run
// every cycle so that the array skew is a fixed delay line.
// first_i and last_i are only interpreted when valid_i is high; they are
// forwarded unmodified.
module pe
  import npu_pkg::*;
(
  input  logic                     clk_i,
  input  logic                     rst_ni,

  input  logic                     valid_i,
  input  logic                     first_i,
  input  logic                     last_i,
  input  logic signed [ACT_W-1:0]  a_i,
  input  logic signed [WGT_W-1:0]  w_i,

  output logic                     valid_o,
  output logic                     first_o,
  output logic                     last_o,
  output logic signed [ACT_W-1:0]  a_o,
  output logic signed [WGT_W-1:0]  w_o,
  output logic signed [ACC_W-1:0]  shadow_o
);

  logic signed [PROD_W-1:0] prod_c;
  logic signed [ACC_W-1:0]  prod_x_c;
  logic signed [ACC_W-1:0]  acc_q, acc_next_c;

  assign prod_c   = $signed({{(PROD_W-ACT_W){a_i[ACT_W-1]}}, a_i})
                  * $signed({{(PROD_W-WGT_W){w_i[WGT_W-1]}}, w_i});
  assign prod_x_c = $signed({{(ACC_W-PROD_W){prod_c[PROD_W-1]}}, prod_c});
  assign acc_next_c = first_i ? prod_x_c : (acc_q + prod_x_c);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_q    <= '0;
      shadow_o <= '0;
    end else if (valid_i) begin
      acc_q <= acc_next_c;
      if (last_i) begin
        shadow_o <= acc_next_c;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_o <= 1'b0;
      first_o <= 1'b0;
      last_o  <= 1'b0;
      a_o     <= '0;
      w_o     <= '0;
    end else begin
      valid_o <= valid_i;
      first_o <= first_i;
      last_o  <= last_i;
      a_o     <= a_i;
      w_o     <= w_i;
    end
  end

endmodule

`default_nettype wire
