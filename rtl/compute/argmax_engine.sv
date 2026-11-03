`default_nettype none

// Argmax engine (docs/architecture.md section 6).
// Reads c_in (1..NUM_LOGITS) int8 logits from in_base with two N-byte reads,
// scans them with signed compares (ties go to the lowest index) and reports
// the index and the logits to npu_regs with a single-cycle result_valid_o.
// Lanes beyond c_in are reported as 0.
module argmax_engine
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,

  input  logic                        start_i,
  input  desc_t                       desc_i,
  output logic                        busy_o,
  output logic                        done_o,

  output logic                        abuf_rd_en_o,
  output logic [ABUF_AW-1:0]          abuf_rd_addr_o,
  input  logic [N-1:0][7:0]           abuf_rd_data_i,
  input  logic                        abuf_rd_valid_i,

  output logic                        result_valid_o,
  output logic [3:0]                  result_idx_o,
  output logic [NUM_LOGITS-1:0][7:0]  logits_o
);

  localparam int unsigned READS = NUM_LOGITS / N;   // 2 for N = 8

  typedef enum logic [2:0] {S_IDLE, S_ISSUE, S_WAIT, S_SCAN, S_DONE} state_e;
  state_e state_q;

  logic [15:0]        c_q;
  logic [ABUF_AW-1:0] in_base_q;
  logic [2:0]         issue_q, got_q;
  logic [NUM_LOGITS-1:0][7:0] raw_q;
  logic [4:0]         scan_q;
  logic signed [7:0]  best_q;
  logic [3:0]         best_idx_q;

  assign busy_o         = (state_q != S_IDLE);
  assign abuf_rd_en_o   = (state_q == S_ISSUE);
  assign abuf_rd_addr_o = in_base_q + ABUF_AW'(32'(issue_q) << $clog2(N));

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_ok;
  assign unused_ok = ^{desc_i.op, desc_i.k, desc_i.pad, desc_i.stride_m1, desc_i.in_h, desc_i.in_w, desc_i.c_out,
                       desc_i.out_base, desc_i.w_base, desc_i.b_base, desc_i.rq_mult, desc_i.rq_shift,
                       desc_i.z_in, desc_i.z_out, desc_i.act_min, desc_i.act_max, desc_i.in_base[31:ABUF_AW]};
  /* verilator lint_on UNUSEDSIGNAL */

  logic signed [7:0] cur_c;
  assign cur_c = raw_q[scan_q[3:0]];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_IDLE;
      c_q <= '0; in_base_q <= '0; issue_q <= '0; got_q <= '0; raw_q <= '0;
      scan_q <= '0; best_q <= '0; best_idx_q <= '0;
      done_o <= 1'b0; result_valid_o <= 1'b0; result_idx_o <= '0; logits_o <= '0;
    end else begin
      done_o         <= 1'b0;
      result_valid_o <= 1'b0;

      if (abuf_rd_valid_i && (state_q == S_ISSUE || state_q == S_WAIT)) begin
        for (int i = 0; i < N; i++) raw_q[got_q[0] * N + i] <= abuf_rd_data_i[i];
        got_q <= got_q + 3'd1;
      end

      case (state_q)
        S_IDLE: begin
          if (start_i) begin
            c_q       <= desc_i.c_in;
            in_base_q <= desc_i.in_base[ABUF_AW-1:0];
            issue_q   <= '0;
            got_q     <= '0;
            state_q   <= S_ISSUE;
          end
        end
        S_ISSUE: begin
          issue_q <= issue_q + 3'd1;
          if (issue_q == 3'(READS - 1)) state_q <= S_WAIT;
        end
        S_WAIT: begin
          if (got_q == 3'(READS)) begin
            // lanes beyond c_in read as 0
            for (int i = 0; i < NUM_LOGITS; i++) if (16'(i) >= c_q) raw_q[i] <= 8'h0;
            scan_q     <= 5'd1;
            best_q     <= raw_q[0];
            best_idx_q <= 4'd0;
            state_q    <= S_SCAN;
          end
        end
        S_SCAN: begin
          if (scan_q >= 5'(c_q[4:0]) || scan_q >= 5'(NUM_LOGITS)) begin
            state_q <= S_DONE;
          end else begin
            if (cur_c > best_q) begin
              best_q     <= cur_c;
              best_idx_q <= scan_q[3:0];
            end
            scan_q <= scan_q + 5'd1;
          end
        end
        S_DONE: begin
          result_valid_o <= 1'b1;
          result_idx_o   <= best_idx_q;
          logits_o       <= raw_q;
          done_o         <= 1'b1;
          state_q        <= S_IDLE;
        end
        default: state_q <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
