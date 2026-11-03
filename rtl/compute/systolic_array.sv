`default_nettype none

// N x N output-stationary systolic array (docs/architecture.md sections 5.2, 5.3, 7.7).
//
// Dataflow
//   - act_i[i] enters row i after a skew of i cycles and flows left to right.
//   - wgt_i[j] enters column j after a skew of j cycles and flows top to bottom.
//   - The row flags (valid / first / last) travel with the activation.
//   - PE(i, j) accumulates Out[m0+i][n0+j] over one tile (first .. last) and parks
//     the result in its shadow register at `last`.
//
// Drain
//   - drain_valid_o goes high in the cycle after the clock edge at which
//     PE(N-1, 0) consumes the `last` beat, i.e. N-1 edges after the edge that
//     consumed that beat at the array input. It stays high for N cycles while
//     drain_col_idx_o counts 0 .. N-1 and drain_data_o presents column
//     drain_col_idx_o of the shadows (lane i = row i). drain_data_o is driven by
//     the shadow registers through a mux; the consumer registers it.
//   - Correctness requires at least N cycles between consecutive `last` beats at
//     the input (tile period P >= N). The producer inserts idle cycles when
//     K_total < N. A `last` arriving while a drain is in progress restarts the
//     drain and corrupts the previous tile; that is a producer bug.
module systolic_array
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,

  input  logic                         in_valid_i,
  input  logic                         in_first_i,
  input  logic                         in_last_i,
  input  logic [N-1:0][ACT_W-1:0]      act_i,
  input  logic [N-1:0][WGT_W-1:0]      wgt_i,

  output logic                         drain_valid_o,
  output logic [$clog2(N)-1:0]         drain_col_idx_o,
  output logic [N-1:0][ACC_W-1:0]      drain_data_o
);

  localparam int unsigned IDX_W = $clog2(N);

  typedef struct packed {
    logic             valid;
    logic             first;
    logic             last;
    logic [ACT_W-1:0] a;
  } row_beat_t;

  // ---------------------------------------------------------------------------
  // Input skew: row i delayed by i cycles, column j delayed by j cycles
  // ---------------------------------------------------------------------------
  row_beat_t        row_in [N];   // beat entering PE(i, 0)
  logic [WGT_W-1:0] col_in [N];   // weight entering PE(0, j)

  for (genvar i = 0; i < N; i++) begin : g_row_skew
    if (i == 0) begin : g_direct
      assign row_in[0] = '{valid: in_valid_i, first: in_first_i, last: in_last_i, a: act_i[0]};
    end else begin : g_delay
      row_beat_t pipe_q [i];
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          for (int s = 0; s < i; s++) pipe_q[s] <= '0;
        end else begin
          pipe_q[0] <= '{valid: in_valid_i, first: in_first_i, last: in_last_i, a: act_i[i]};
          for (int s = 1; s < i; s++) pipe_q[s] <= pipe_q[s-1];
        end
      end
      assign row_in[i] = pipe_q[i-1];
    end
  end

  for (genvar j = 0; j < N; j++) begin : g_col_skew
    if (j == 0) begin : g_direct
      assign col_in[0] = wgt_i[0];
    end else begin : g_delay
      logic [WGT_W-1:0] pipe_q [j];
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          for (int s = 0; s < j; s++) pipe_q[s] <= '0;
        end else begin
          pipe_q[0] <= wgt_i[j];
          for (int s = 1; s < j; s++) pipe_q[s] <= pipe_q[s-1];
        end
      end
      assign col_in[j] = pipe_q[j-1];
    end
  end

  // ---------------------------------------------------------------------------
  // PE grid. h_* [i][j] is the signal entering PE(i, j) from the left
  // (index N is the unused right edge); v_w[i][j] enters PE(i, j) from above
  // (index N is the unused bottom edge).
  // ---------------------------------------------------------------------------
  logic [N-1:0][N:0]               h_valid, h_first, h_last;
  logic [N-1:0][N:0][ACT_W-1:0]    h_a;
  logic [N:0][N-1:0][WGT_W-1:0]    v_w;
  logic [N-1:0][N-1:0][ACC_W-1:0]  shadow;

  for (genvar i = 0; i < N; i++) begin : g_row
    assign h_valid[i][0] = row_in[i].valid;
    assign h_first[i][0] = row_in[i].first;
    assign h_last[i][0]  = row_in[i].last;
    assign h_a[i][0]     = row_in[i].a;
    for (genvar j = 0; j < N; j++) begin : g_col
      if (i == 0) begin : g_top
        assign v_w[0][j] = col_in[j];
      end
      pe u_pe (
          .clk_i    (clk_i),
          .rst_ni   (rst_ni),
          .valid_i  (h_valid[i][j]),
          .first_i  (h_first[i][j]),
          .last_i   (h_last[i][j]),
          .a_i      (h_a[i][j]),
          .w_i      (v_w[i][j]),
          .valid_o  (h_valid[i][j+1]),
          .first_o  (h_first[i][j+1]),
          .last_o   (h_last[i][j+1]),
          .a_o      (h_a[i][j+1]),
          .w_o      (v_w[i+1][j]),
          .shadow_o (shadow[i][j])
      );
    end
  end

  // The right and bottom edges of the grid leave the array; nothing consumes them.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [N-1:0]            unused_edge_flags;
  logic [N-1:0][ACT_W-1:0] unused_edge_a;
  logic [N-1:0][WGT_W-1:0] unused_edge_w;
  for (genvar i = 0; i < N; i++) begin : g_edge
    assign unused_edge_flags[i] = h_valid[i][N] | h_first[i][N] | h_last[i][N];
    assign unused_edge_a[i]     = h_a[i][N];
    assign unused_edge_w[i]     = v_w[N][i];
  end
  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------------------
  // Column drain
  // ---------------------------------------------------------------------------
  logic drain_trig_c;
  assign drain_trig_c = row_in[N-1].valid & row_in[N-1].last;  // PE(N-1, 0) consumes `last` this edge

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      drain_valid_o   <= 1'b0;
      drain_col_idx_o <= '0;
    end else if (drain_trig_c) begin
      drain_valid_o   <= 1'b1;
      drain_col_idx_o <= '0;
    end else if (drain_valid_o) begin
      if (drain_col_idx_o == IDX_W'(N - 1)) begin
        drain_valid_o   <= 1'b0;
        drain_col_idx_o <= '0;
      end else begin
        drain_col_idx_o <= drain_col_idx_o + IDX_W'(1);
      end
    end
  end

  always_comb begin
    for (int i = 0; i < N; i++) begin
      drain_data_o[i] = shadow[i][drain_col_idx_o];
    end
  end

endmodule

`default_nettype wire
