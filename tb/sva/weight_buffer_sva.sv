`default_nettype none

// Assertions for weight_buffer, bound into every instance: the read latency
// conv_engine's weight hold register relies on.
module weight_buffer_sva
  import npu_pkg::*;
(
  input logic       clk_i,
  input logic       rst_ni,
  input logic       rd_en_i,
  input logic       rd_valid_o,
  input logic       wr_en_i,
  input logic [7:0] wr_be_i
);

  logic [WBUF_RD_LAT-1:0] v_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) v_q <= '0;
    else         v_q <= WBUF_RD_LAT'({v_q, rd_en_i});
  end

  a_rd_latency: assert property (@(posedge clk_i) disable iff (!rst_ni) rd_valid_o == v_q[WBUF_RD_LAT-1])
    else $error("rd_valid_o is not rd_en_i delayed by WBUF_RD_LAT");

  c_wr_partial: cover property (@(posedge clk_i) disable iff (!rst_ni) wr_en_i && wr_be_i != '1 && wr_be_i != '0);
  c_rd_stream:  cover property (@(posedge clk_i) disable iff (!rst_ni) rd_en_i && v_q[0]);

endmodule

bind weight_buffer weight_buffer_sva u_sva (.*);

`default_nettype wire
