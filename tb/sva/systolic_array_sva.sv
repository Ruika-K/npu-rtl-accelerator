`default_nettype none

// Assertions for systolic_array, bound into every instance: the producer's
// tile-period contract (m_*), the drain protocol and its fixed position (a_*).
module systolic_array_sva
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input logic                 clk_i,
  input logic                 rst_ni,
  input logic                 in_valid_i,
  input logic                 in_first_i,
  input logic                 in_last_i,
  input logic                 drain_valid_o,
  input logic [$clog2(N)-1:0] drain_col_idx_o
);

  localparam int unsigned IDX_W = $clog2(N);
  localparam int unsigned CW    = $clog2(N + 1) + 1;

  logic first_c, last_c;
  assign first_c = in_valid_i && in_first_i;
  assign last_c  = in_valid_i && in_last_i;

  // Cycles since the previous tile's first beat (saturating), and whether a
  // tile is open (first seen, last not yet)
  logic [CW-1:0] since_first_q;
  logic          seen_first_q, open_q;
  // last beats delayed to where their drain must start
  logic [N-1:0]  last_q;
  logic          drain_prev_q;
  logic [IDX_W-1:0] idx_prev_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      since_first_q <= '0; seen_first_q <= 1'b0; open_q <= 1'b0; last_q <= '0;
      drain_prev_q <= 1'b0; idx_prev_q <= '0;
    end else begin
      if (first_c) begin
        since_first_q <= CW'(1); seen_first_q <= 1'b1;
      end else if (since_first_q != '1) begin
        since_first_q <= since_first_q + CW'(1);
      end
      if (last_c)       open_q <= 1'b0;
      else if (first_c) open_q <= 1'b1;
      last_q       <= {last_q[N-2:0], last_c};
      drain_prev_q <= drain_valid_o;
      idx_prev_q   <= drain_col_idx_o;
    end
  end

  // ---------------------------------------------------------------------------
  // Environment: the tile contract (docs/architecture.md section 5.3)
  // ---------------------------------------------------------------------------
  m_period: assume property (@(posedge clk_i) disable iff (!rst_ni)
      first_c && seen_first_q |-> since_first_q >= CW'(N))
    else $error("tile period below N: the previous tile's drain would be overwritten");
  m_first_closes: assume property (@(posedge clk_i) disable iff (!rst_ni) first_c |-> !open_q)
    else $error("first beat while the previous tile has not seen its last beat");
  m_beat_in_tile: assume property (@(posedge clk_i) disable iff (!rst_ni) in_valid_i && !in_first_i |-> open_q)
    else $error("beat outside a tile");

  // ---------------------------------------------------------------------------
  // Guarantees: the drain is N columns in order, starting N cycles after the
  // last beat was presented
  // ---------------------------------------------------------------------------
  a_drain_start: assert property (@(posedge clk_i) disable iff (!rst_ni)
      last_q[N-1] |-> drain_valid_o && drain_col_idx_o == '0)
    else $error("drain did not start N cycles after the last beat");
  a_drain_only_after_last: assert property (@(posedge clk_i) disable iff (!rst_ni)
      drain_valid_o && drain_col_idx_o == '0 |-> last_q[N-1])
    else $error("drain started without a last beat N cycles earlier");
  a_drain_order: assert property (@(posedge clk_i) disable iff (!rst_ni)
      drain_valid_o && drain_col_idx_o != '0 |-> drain_prev_q && drain_col_idx_o == idx_prev_q + IDX_W'(1))
    else $error("drain columns out of order");
  a_drain_complete: assert property (@(posedge clk_i) disable iff (!rst_ni)
      drain_prev_q && idx_prev_q != IDX_W'(N - 1) |-> drain_valid_o)
    else $error("drain stopped before column N-1");

  c_single_beat_tile: cover property (@(posedge clk_i) disable iff (!rst_ni) first_c && last_c);
  c_idle_in_tile:     cover property (@(posedge clk_i) disable iff (!rst_ni) open_q && !in_valid_i);
  c_back_to_back:     cover property (@(posedge clk_i) disable iff (!rst_ni) first_c && last_q[0]);
  c_period_min:       cover property (@(posedge clk_i) disable iff (!rst_ni) first_c && seen_first_q && since_first_q == CW'(N));
  c_drain_overlap:    cover property (@(posedge clk_i) disable iff (!rst_ni) drain_valid_o && in_valid_i);

endmodule

bind systolic_array systolic_array_sva #(.N(N)) u_sva (.*);

`default_nettype wire
