`default_nettype none

// Assertions for requant_unit, bound into every instance: the fixed latency
// and the output range. Bit-exactness is the testbenches' job (they compare
// with the Python model); these hold for any input.
module requant_unit_sva
  import npu_pkg::*;
(
  input logic                    clk_i,
  input logic                    rst_ni,
  input logic                    valid_i,
  input logic [RQ_SHIFT_W-1:0]   shift_i,
  input logic signed [ACT_W-1:0] act_min_i,
  input logic signed [ACT_W-1:0] act_max_i,
  input logic                    valid_o,
  input logic signed [ACT_W-1:0] q_o
);

  // valid and the clamp bounds, delayed by the documented latency
  logic [RQ_PIPE_LAT-1:0]                  v_q;
  logic [RQ_PIPE_LAT-1:0][ACT_W-1:0]       min_q, max_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v_q <= '0; min_q <= '0; max_q <= '0;
    end else begin
      v_q   <= {v_q[RQ_PIPE_LAT-2:0], valid_i};
      min_q <= {min_q[RQ_PIPE_LAT-2:0], act_min_i};
      max_q <= {max_q[RQ_PIPE_LAT-2:0], act_max_i};
    end
  end
  logic signed [ACT_W-1:0] min_c, max_c;
  assign min_c = min_q[RQ_PIPE_LAT-1];
  assign max_c = max_q[RQ_PIPE_LAT-1];

  a_latency: assert property (@(posedge clk_i) disable iff (!rst_ni) valid_o == v_q[RQ_PIPE_LAT-1])
    else $error("valid_o is not valid_i delayed by RQ_PIPE_LAT");
  a_range: assert property (@(posedge clk_i) disable iff (!rst_ni)
      valid_o && min_c <= max_c |-> q_o >= min_c && q_o <= max_c)
    else $error("q_o outside [act_min, act_max]");

  c_clamp_min:   cover property (@(posedge clk_i) disable iff (!rst_ni) valid_o && min_c < max_c && q_o == min_c);
  c_clamp_max:   cover property (@(posedge clk_i) disable iff (!rst_ni) valid_o && min_c < max_c && q_o == max_c);
  c_inside:      cover property (@(posedge clk_i) disable iff (!rst_ni) valid_o && q_o > min_c && q_o < max_c);
  c_shift_min:   cover property (@(posedge clk_i) disable iff (!rst_ni) valid_i && shift_i == RQ_SHIFT_W'(1));
  c_shift_large: cover property (@(posedge clk_i) disable iff (!rst_ni) valid_i && shift_i >= RQ_SHIFT_W'(32));
  c_back_to_back: cover property (@(posedge clk_i) disable iff (!rst_ni) valid_i && v_q[0]);

endmodule

bind requant_unit requant_unit_sva u_sva (.*);

`default_nettype wire
