`default_nettype none

// 2x2 stride-2 max-pooling engine (docs/architecture.md section 6).
//
// For every channel, output row and group of N output columns it reads the
// two input rows 2r and 2r+1 at column 2*ow0, 2N bytes each (two N-byte
// reads per row), takes the maximum of each 2x2 window with signed compares
// and writes N output bytes with a lane mask for the end of the row. Output
// size rounds down. Descriptor fields are trusted (validated by the
// sequencer). Reads past the end of a row only feed masked lanes.
module pool_engine
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                start_i,
  input  desc_t               desc_i,
  output logic                busy_o,
  output logic                done_o,

  output logic                abuf_rd_en_o,
  output logic [ABUF_AW-1:0]  abuf_rd_addr_o,
  input  logic [N-1:0][7:0]   abuf_rd_data_i,
  input  logic                abuf_rd_valid_i,
  output logic                abuf_wr_en_o,
  output logic [ABUF_AW-1:0]  abuf_wr_addr_o,
  output logic [N-1:0][7:0]   abuf_wr_data_o,
  output logic [N-1:0]        abuf_wr_be_o
);

  typedef enum logic [2:0] {S_IDLE, S_SETUP, S_ISSUE, S_WAIT, S_WRITE, S_DONE} state_e;
  state_e state_q;

  logic [15:0]        h_q, w_q, c_q;
  logic [15:0]        oh_q, ow_q;
  logic [ABUF_AW-1:0] hw_q, ohw_q;      // channel strides, bounded by the buffer
  logic [ABUF_AW-1:0] in_base_q, out_base_q;

  logic [15:0]        c_i_q, r_q;        // channel, output row
  logic [15:0]        ow0_q;             // first output column of the group
  logic [ABUF_AW-1:0] in_ch_q, out_ch_q; // channel bases
  logic [ABUF_AW-1:0] in_row_q, out_row_q;
  logic [2:0]         issue_q, got_q;    // 4 reads per group
  logic [2*N-1:0][7:0] row0_q, row1_q;

  logic last_grp_c, last_row_c, last_ch_c;
  assign last_grp_c = (ow0_q + 16'(N)) >= ow_q;
  assign last_row_c = (r_q == oh_q - 16'd1);
  assign last_ch_c  = (c_i_q == c_q - 16'd1);

  assign busy_o = (state_q != S_IDLE);

  // Read address for issue step 0..3: row0 lo, row0 hi, row1 lo, row1 hi
  logic [ABUF_AW-1:0] rd_addr_c;
  always_comb begin
    rd_addr_c = in_row_q + ABUF_AW'({ow0_q, 1'b0});
    if (issue_q[1]) rd_addr_c = rd_addr_c + w_q[ABUF_AW-1:0];
    if (issue_q[0]) rd_addr_c = rd_addr_c + ABUF_AW'(N);
  end
  assign abuf_rd_en_o   = (state_q == S_ISSUE);
  assign abuf_rd_addr_o = rd_addr_c;

  // Max of the 2x2 window for each output lane
  function automatic logic signed [7:0] max2(input logic signed [7:0] a, input logic signed [7:0] b);
    return (a > b) ? a : b;
  endfunction

  logic [N-1:0][7:0] out_c;
  always_comb begin
    for (int i = 0; i < N; i++) begin
      out_c[i] = max2(max2(row0_q[2*i], row0_q[2*i+1]), max2(row1_q[2*i], row1_q[2*i+1]));
    end
  end

  logic [N-1:0] be_c;
  always_comb begin
    for (int i = 0; i < N; i++) be_c[i] = (ow0_q + 16'(i)) < ow_q;
  end

  assign abuf_wr_en_o   = (state_q == S_WRITE);
  assign abuf_wr_addr_o = out_row_q + ow0_q[ABUF_AW-1:0];
  assign abuf_wr_data_o = out_c;
  assign abuf_wr_be_o   = be_c;

  // The sequencer consumed the other descriptor fields.
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_ok;
  assign unused_ok = ^{desc_i.op, desc_i.k, desc_i.pad, desc_i.stride_m1, desc_i.c_out, desc_i.w_base, desc_i.b_base,
                       desc_i.rq_mult, desc_i.rq_shift, desc_i.z_in, desc_i.z_out, desc_i.act_min, desc_i.act_max,
                       desc_i.in_base[31:ABUF_AW], desc_i.out_base[31:ABUF_AW]};
  /* verilator lint_on UNUSEDSIGNAL */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_IDLE;
      h_q <= '0; w_q <= '0; c_q <= '0; oh_q <= '0; ow_q <= '0; hw_q <= '0; ohw_q <= '0;
      in_base_q <= '0; out_base_q <= '0;
      c_i_q <= '0; r_q <= '0; ow0_q <= '0; in_ch_q <= '0; out_ch_q <= '0; in_row_q <= '0; out_row_q <= '0;
      issue_q <= '0; got_q <= '0; row0_q <= '0; row1_q <= '0;
      done_o <= 1'b0;
    end else begin
      done_o <= 1'b0;

      // collect read data in order
      if (abuf_rd_valid_i && (state_q == S_ISSUE || state_q == S_WAIT)) begin
        case (got_q)
          3'd0: row0_q[N-1:0]     <= abuf_rd_data_i;
          3'd1: row0_q[2*N-1:N]   <= abuf_rd_data_i;
          3'd2: row1_q[N-1:0]     <= abuf_rd_data_i;
          default: row1_q[2*N-1:N] <= abuf_rd_data_i;
        endcase
        got_q <= got_q + 3'd1;
      end

      case (state_q)
        S_IDLE: begin
          if (start_i) begin
            h_q        <= desc_i.in_h;
            w_q        <= desc_i.in_w;
            c_q        <= desc_i.c_in;
            in_base_q  <= desc_i.in_base[ABUF_AW-1:0];
            out_base_q <= desc_i.out_base[ABUF_AW-1:0];
            state_q    <= S_SETUP;
          end
        end
        S_SETUP: begin
          oh_q     <= {1'b0, h_q[15:1]};
          ow_q     <= {1'b0, w_q[15:1]};
          hw_q     <= ABUF_AW'(32'(h_q) * 32'(w_q));
          ohw_q    <= ABUF_AW'(32'(h_q[15:1]) * 32'(w_q[15:1]));
          c_i_q    <= '0; r_q <= '0; ow0_q <= '0;
          in_ch_q  <= in_base_q;  out_ch_q  <= out_base_q;
          in_row_q <= in_base_q;  out_row_q <= out_base_q;
          issue_q  <= '0; got_q <= '0;
          state_q  <= S_ISSUE;
        end
        S_ISSUE: begin
          issue_q <= issue_q + 3'd1;
          if (issue_q == 3'd3) state_q <= S_WAIT;
        end
        S_WAIT: begin
          if (got_q == 3'd4) state_q <= S_WRITE;
        end
        S_WRITE: begin
          issue_q <= '0; got_q <= '0;
          if (last_grp_c) begin
            ow0_q <= '0;
            if (last_row_c) begin
              if (last_ch_c) begin
                state_q <= S_DONE;
              end else begin
                c_i_q    <= c_i_q + 16'd1;
                r_q      <= '0;
                in_ch_q  <= in_ch_q + hw_q;
                out_ch_q <= out_ch_q + ohw_q;
                in_row_q <= in_ch_q + hw_q;
                out_row_q <= out_ch_q + ohw_q;
                state_q  <= S_ISSUE;
              end
            end else begin
              r_q       <= r_q + 16'd1;
              in_row_q  <= in_row_q + ABUF_AW'({w_q, 1'b0});   // two input rows
              out_row_q <= out_row_q + ow_q[ABUF_AW-1:0];
              state_q   <= S_ISSUE;
            end
          end else begin
            ow0_q   <= ow0_q + 16'(N);
            state_q <= S_ISSUE;
          end
        end
        S_DONE: begin
          done_o  <= 1'b1;
          state_q <= S_IDLE;
        end
        default: state_q <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
