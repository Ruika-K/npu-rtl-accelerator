`default_nettype none

// Assertions for act_buffer, bound into every instance: the read latency the
// engines' delay lines rely on, and the access scenarios.
module act_buffer_sva
  import npu_pkg::*;
#(
  parameter int unsigned BYTES = ABUF_BYTES,
  parameter int unsigned N     = ARR_N,
  localparam int unsigned AW   = $clog2(BYTES)
) (
  input logic          clk_i,
  input logic          rst_ni,
  input logic          rd_en_i,
  /* verilator lint_off UNUSEDSIGNAL */   // only the lane offset is covered
  input logic [AW-1:0] rd_addr_i,
  /* verilator lint_on UNUSEDSIGNAL */
  input logic          rd_valid_o,
  input logic          wr_en_i,
  /* verilator lint_off UNUSEDSIGNAL */   // only the lane offset is covered
  input logic [AW-1:0] wr_addr_i,
  /* verilator lint_on UNUSEDSIGNAL */
  input logic [N-1:0]  wr_be_i
);

  localparam int unsigned LANE_W = $clog2(N);

  logic [ABUF_RD_LAT-1:0] v_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) v_q <= '0;
    else         v_q <= {v_q[ABUF_RD_LAT-2:0], rd_en_i};
  end

  a_rd_latency: assert property (@(posedge clk_i) disable iff (!rst_ni) rd_valid_o == v_q[ABUF_RD_LAT-1])
    else $error("rd_valid_o is not rd_en_i delayed by ABUF_RD_LAT");

  c_rd_aligned:   cover property (@(posedge clk_i) disable iff (!rst_ni) rd_en_i && rd_addr_i[LANE_W-1:0] == '0);
  c_rd_unaligned: cover property (@(posedge clk_i) disable iff (!rst_ni) rd_en_i && rd_addr_i[LANE_W-1:0] != '0);
  c_wr_unaligned: cover property (@(posedge clk_i) disable iff (!rst_ni) wr_en_i && wr_addr_i[LANE_W-1:0] != '0);
  c_wr_partial:   cover property (@(posedge clk_i) disable iff (!rst_ni) wr_en_i && wr_be_i != '1 && wr_be_i != '0);
  c_rd_wr_same:   cover property (@(posedge clk_i) disable iff (!rst_ni) rd_en_i && wr_en_i);
  c_rd_stream:    cover property (@(posedge clk_i) disable iff (!rst_ni) rd_en_i && v_q[0]);

endmodule

bind act_buffer act_buffer_sva #(.BYTES(BYTES), .N(N)) u_sva (.*);

`default_nettype wire
