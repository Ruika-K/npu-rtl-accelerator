`default_nettype none

// Assertions for conv_engine, bound into every instance: every tile enters the
// array with exactly K_total beats, the tile queue never overflows or
// underflows, the shared weight-buffer port is never claimed by a
// beat and a bias prefetch together, writes happen only while busy and line up
// with the requant pipeline, and done is a single pulse.
module conv_engine_sva
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input logic         clk_i,
  input logic         rst_ni,
  input logic         start_i,
  input logic         busy_o,
  input logic         done_o,
  input logic         abuf_rd_en_o,
  input logic         abuf_wr_en_o,
  input logic [N-1:0] abuf_wr_be_o,
  input logic         wbuf_rd_en_o,
  input logic         g_active_q,
  input logic         g_tile_start_q,
  input logic [N-1:0] g_ow_mask_q,
  input logic [N-1:0] g_cmask_q,
  input logic         pf_issue_c,
  input logic         drain_start_c,
  input logic [N-1:0] rq_valid,
  input logic         arr_valid_q,
  input logic         arr_first_q,
  input logic         arr_last_q,
  input logic [21:0]  k_total_q,
  input logic [21:0]  period_q
);

  localparam int unsigned TQ_DEPTH = 4;    // conv_engine's tile queue depth

  // Tiles pushed and not yet popped
  logic [2:0] tq_level_q, tq_level_max_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tq_level_q <= '0; tq_level_max_q <= '0;
    end else begin
      tq_level_q <= tq_level_q + 3'(g_tile_start_q) - 3'(drain_start_c);
      if (tq_level_q > tq_level_max_q) tq_level_max_q <= tq_level_q;
    end
  end

  // Beats of the tile entering the array, counted from its first beat
  logic [21:0] beats_q, beats_c;
  assign beats_c = arr_first_q ? 22'd1 : beats_q + 22'd1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)          beats_q <= '0;
    else if (arr_valid_q) beats_q <= beats_c;
  end

  a_tile_beats: assert property (@(posedge clk_i) disable iff (!rst_ni)
      arr_valid_q && arr_last_q |-> beats_c == k_total_q)
    else $error("a tile entered the array with a beat count other than K_total");

  a_tq_no_overflow: assert property (@(posedge clk_i) disable iff (!rst_ni)
      g_tile_start_q |-> tq_level_q < 3'(TQ_DEPTH) || drain_start_c)
    else $error("tile queue overflow");
  a_tq_no_underflow: assert property (@(posedge clk_i) disable iff (!rst_ni) drain_start_c |-> tq_level_q != '0)
    else $error("drain started with an empty tile queue");
  a_tq_empty_at_done: assert property (@(posedge clk_i) disable iff (!rst_ni) done_o |-> tq_level_q == '0)
    else $error("done with tiles still queued");

  a_wbuf_port: assert property (@(posedge clk_i) disable iff (!rst_ni) !(g_active_q && pf_issue_c))
    else $error("weight read and bias prefetch in the same cycle");
  a_rd_pair: assert property (@(posedge clk_i) disable iff (!rst_ni) abuf_rd_en_o |-> wbuf_rd_en_o && g_active_q)
    else $error("activation read without its weight read");

  a_active_only_busy: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (abuf_rd_en_o || abuf_wr_en_o || wbuf_rd_en_o) |-> busy_o)
    else $error("memory access while idle");
  a_wr_with_requant: assert property (@(posedge clk_i) disable iff (!rst_ni) abuf_wr_en_o |-> &rq_valid)
    else $error("write without a requant result");
  a_rq_lanes_together: assert property (@(posedge clk_i) disable iff (!rst_ni) (|rq_valid) == (&rq_valid))
    else $error("requant lanes out of step");

  a_done_pulse: assert property (@(posedge clk_i) disable iff (!rst_ni) done_o |=> !done_o)
    else $error("done longer than one cycle");
  a_done_ends_busy: assert property (@(posedge clk_i) disable iff (!rst_ni) done_o |-> !busy_o)
    else $error("done while still busy");
  m_start_idle: assume property (@(posedge clk_i) disable iff (!rst_ni) start_i |-> !busy_o)
    else $error("start while busy");

  c_tq_two:        cover property (@(posedge clk_i) disable iff (!rst_ni) tq_level_q >= 3'd2);
  c_padded_tile:   cover property (@(posedge clk_i) disable iff (!rst_ni) g_tile_start_q && k_total_q < 22'(N));
  c_k_equals_n:    cover property (@(posedge clk_i) disable iff (!rst_ni) g_tile_start_q && k_total_q == 22'(N));
  c_long_tile:     cover property (@(posedge clk_i) disable iff (!rst_ni) g_tile_start_q && period_q > 22'd256);
  c_partial_row:   cover property (@(posedge clk_i) disable iff (!rst_ni) g_tile_start_q && g_ow_mask_q != '1);
  c_partial_cout:  cover property (@(posedge clk_i) disable iff (!rst_ni) g_tile_start_q && g_cmask_q != '1);
  c_wr_partial:    cover property (@(posedge clk_i) disable iff (!rst_ni) abuf_wr_en_o && abuf_wr_be_o != '1);
  c_wr_during_rd:  cover property (@(posedge clk_i) disable iff (!rst_ni) abuf_wr_en_o && abuf_rd_en_o);

endmodule

bind conv_engine conv_engine_sva #(.N(N)) u_sva (.*);

`default_nettype wire
