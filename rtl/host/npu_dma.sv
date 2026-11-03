`default_nettype none

// AXI4 master DMA between external memory and the on-chip buffers
// (docs/architecture.md section 12, registers in programming_model.md section 7).
//
// One transfer per start: len_i bytes from src_i (external) to dst (buffer
// dst_sel_i at byte offset dst_off_i) when dir_i = 0 (load), or from the
// buffer to src_i when dir_i = 1 (store). Addresses and the length are
// 8-byte aligned; the external address is split into INCR bursts of at most
// MAX_BURST 64-bit beats that never cross a 4 KB boundary. Load writes one
// beat per cycle; store reads one buffer word per beat without prefetch.
// err_o is pulsed with ERR_BAD_PARAM for misaligned or out-of-range
// parameters and ERR_BAD_ADDR for an AXI error response; the transfer stops.
module npu_dma
  import npu_pkg::*;
#(
  parameter int unsigned MAX_BURST = 16,
  parameter int unsigned N = ARR_N
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,

  input  logic                      start_i,
  input  logic                      dir_i,
  input  logic                      dst_sel_i,
  input  logic [31:0]               src_i,
  input  logic [19:0]               dst_off_i,
  input  logic [19:0]               len_i,
  output logic                      busy_o,
  output logic                      done_o,
  output logic                      err_o,
  output err_e                      err_code_o,

  // activation buffer
  output logic                      abuf_wr_en_o,
  output logic [ABUF_AW-1:0]        abuf_wr_addr_o,
  output logic [N-1:0][7:0]         abuf_wr_data_o,
  output logic [N-1:0]              abuf_wr_be_o,
  output logic                      abuf_rd_en_o,
  output logic [ABUF_AW-1:0]        abuf_rd_addr_o,
  input  logic [N-1:0][7:0]         abuf_rd_data_i,
  input  logic                      abuf_rd_valid_i,
  // weight buffer
  output logic                      wbuf_wr_en_o,
  output logic [WBUF_WAW-1:0]       wbuf_wr_addr_o,
  output logic [63:0]               wbuf_wr_data_o,
  output logic [7:0]                wbuf_wr_be_o,
  output logic                      wbuf_rd_en_o,
  output logic [WBUF_WAW-1:0]       wbuf_rd_addr_o,
  input  logic [63:0]               wbuf_rd_data_i,
  input  logic                      wbuf_rd_valid_i,

  // AXI4 master, 64-bit data, single ID
  output logic [31:0]               m_axi_awaddr_o,
  output logic [7:0]                m_axi_awlen_o,
  output logic [2:0]                m_axi_awsize_o,
  output logic [1:0]                m_axi_awburst_o,
  output logic                      m_axi_awvalid_o,
  input  logic                      m_axi_awready_i,
  output logic [63:0]               m_axi_wdata_o,
  output logic [7:0]                m_axi_wstrb_o,
  output logic                      m_axi_wlast_o,
  output logic                      m_axi_wvalid_o,
  input  logic                      m_axi_wready_i,
  input  logic [1:0]                m_axi_bresp_i,
  input  logic                      m_axi_bvalid_i,
  output logic                      m_axi_bready_o,
  output logic [31:0]               m_axi_araddr_o,
  output logic [7:0]                m_axi_arlen_o,
  output logic [2:0]                m_axi_arsize_o,
  output logic [1:0]                m_axi_arburst_o,
  output logic                      m_axi_arvalid_o,
  input  logic                      m_axi_arready_i,
  input  logic [63:0]               m_axi_rdata_i,
  input  logic [1:0]                m_axi_rresp_i,
  input  logic                      m_axi_rlast_i,
  input  logic                      m_axi_rvalid_i,
  output logic                      m_axi_rready_o
);

  typedef enum logic [3:0] {
    S_IDLE, S_CHECK,
    S_LD_AR, S_LD_R,
    S_ST_AW, S_ST_RD, S_ST_WAIT, S_ST_W, S_ST_B,
    S_DONE, S_ERR
  } state_e;
  state_e state_q;

  logic        dir_q, dst_sel_q;
  logic [31:0] ext_addr_q;      // next external address
  logic [19:0] int_off_q;       // next buffer byte offset
  logic [19:0] remaining_q;     // bytes left
  logic [7:0]  burst_beats_q;   // beats in the current burst
  logic [7:0]  beat_q;          // beats issued / received in the current burst
  logic [63:0] st_data_q;       // buffer word waiting to go out
  err_e        err_code_q;

  // Beats of the next burst: at most MAX_BURST, at most what is left, never
  // past the next 4 KB boundary.
  logic [19:0] left_words_c, to_boundary_words_c;
  logic [8:0]  next_beats_c;
  assign left_words_c        = remaining_q >> 3;
  assign to_boundary_words_c = 20'((32'h1000 - 32'(ext_addr_q[11:0])) >> 3);
  always_comb begin
    next_beats_c = 9'(MAX_BURST);
    if (left_words_c < 20'(next_beats_c))        next_beats_c = 9'(left_words_c);
    if (to_boundary_words_c < 20'(next_beats_c)) next_beats_c = 9'(to_boundary_words_c);
  end

  logic params_ok_c;
  always_comb begin
    params_ok_c = (src_i[2:0] == 3'b000) && (dst_off_i[2:0] == 3'b000) && (len_i[2:0] == 3'b000) && (len_i != 20'd0);
    if (dst_sel_i) params_ok_c = params_ok_c && (32'(dst_off_i) + 32'(len_i) <= 32'(WBUF_BYTES));
    else           params_ok_c = params_ok_c && (32'(dst_off_i) + 32'(len_i) <= 32'(ABUF_BYTES));
  end

  // ---------------------------------------------------------------------------
  // Internal memory ports
  // ---------------------------------------------------------------------------
  logic ld_beat_c, st_rd_c;
  assign ld_beat_c = (state_q == S_LD_R) && m_axi_rvalid_i;          // a load beat lands this cycle
  assign st_rd_c   = (state_q == S_ST_RD);                           // issue a buffer read

  assign abuf_wr_en_o   = ld_beat_c && !dst_sel_q;
  assign abuf_wr_addr_o = int_off_q[ABUF_AW-1:0];
  assign abuf_wr_be_o   = '1;
  always_comb begin
    for (int i = 0; i < N; i++) abuf_wr_data_o[i] = m_axi_rdata_i[8*i +: 8];
  end
  assign wbuf_wr_en_o   = ld_beat_c && dst_sel_q;
  assign wbuf_wr_addr_o = int_off_q[WBUF_WAW+2:3];
  assign wbuf_wr_data_o = m_axi_rdata_i;
  assign wbuf_wr_be_o   = '1;

  assign abuf_rd_en_o   = st_rd_c && !dst_sel_q;
  assign abuf_rd_addr_o = int_off_q[ABUF_AW-1:0];
  assign wbuf_rd_en_o   = st_rd_c && dst_sel_q;
  assign wbuf_rd_addr_o = int_off_q[WBUF_WAW+2:3];

  logic        rd_data_valid_c;
  logic [63:0] rd_data_c;
  always_comb begin
    rd_data_valid_c = dst_sel_q ? wbuf_rd_valid_i : abuf_rd_valid_i;
    rd_data_c = wbuf_rd_data_i;
    if (!dst_sel_q) for (int i = 0; i < N; i++) rd_data_c[8*i +: 8] = abuf_rd_data_i[i];
  end

  // ---------------------------------------------------------------------------
  // AXI
  // ---------------------------------------------------------------------------
  assign m_axi_awaddr_o  = ext_addr_q;
  assign m_axi_awlen_o   = burst_beats_q - 8'd1;
  assign m_axi_awsize_o  = 3'b011;        // 8 bytes
  assign m_axi_awburst_o = 2'b01;         // INCR
  assign m_axi_awvalid_o = (state_q == S_ST_AW);
  assign m_axi_wdata_o   = st_data_q;
  assign m_axi_wstrb_o   = 8'hFF;
  assign m_axi_wlast_o   = (beat_q == burst_beats_q - 8'd1);
  assign m_axi_wvalid_o  = (state_q == S_ST_W);
  assign m_axi_bready_o  = (state_q == S_ST_B);
  assign m_axi_araddr_o  = ext_addr_q;
  assign m_axi_arlen_o   = burst_beats_q - 8'd1;
  assign m_axi_arsize_o  = 3'b011;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = (state_q == S_LD_AR);
  assign m_axi_rready_o  = (state_q == S_LD_R);

  assign busy_o     = (state_q != S_IDLE);
  assign err_code_o = err_code_q;

  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0] unused_resp_lsb;
  assign unused_resp_lsb = {m_axi_bresp_i[0], m_axi_rresp_i[0]};
  logic [11:0] unused_off;
  assign unused_off = {int_off_q[19:ABUF_AW], int_off_q[2:0], 4'h0};
  /* verilator lint_on UNUSEDSIGNAL */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_IDLE;
      dir_q <= 1'b0; dst_sel_q <= 1'b0; ext_addr_q <= '0; int_off_q <= '0; remaining_q <= '0;
      burst_beats_q <= '0; beat_q <= '0; st_data_q <= '0;
      done_o <= 1'b0; err_o <= 1'b0; err_code_q <= ERR_NONE;
    end else begin
      done_o <= 1'b0;
      err_o  <= 1'b0;
      case (state_q)
        S_IDLE: begin
          if (start_i) begin
            dir_q       <= dir_i;
            dst_sel_q   <= dst_sel_i;
            ext_addr_q  <= src_i;
            int_off_q   <= dst_off_i;
            remaining_q <= len_i;
            err_code_q  <= ERR_NONE;
            state_q     <= params_ok_c ? S_CHECK : S_ERR;
            if (!params_ok_c) err_code_q <= ERR_BAD_PARAM;
          end
        end
        S_CHECK: begin
          // start the next burst, or finish
          if (remaining_q == 20'd0) begin
            state_q <= S_DONE;
          end else begin
            burst_beats_q <= next_beats_c[7:0];
            beat_q        <= '0;
            state_q       <= dir_q ? S_ST_AW : S_LD_AR;
          end
        end

        // ---- load: external -> buffer ----
        S_LD_AR: begin
          if (m_axi_arready_i) state_q <= S_LD_R;
        end
        S_LD_R: begin
          if (m_axi_rvalid_i) begin
            int_off_q   <= int_off_q + 20'd8;
            ext_addr_q  <= ext_addr_q + 32'd8;
            remaining_q <= remaining_q - 20'd8;
            beat_q      <= beat_q + 8'd1;
            if (m_axi_rresp_i[1]) begin
              err_code_q <= ERR_BAD_ADDR;
              state_q    <= m_axi_rlast_i ? S_ERR : S_LD_R;   // drain the burst, then stop
            end else if (m_axi_rlast_i) begin
              state_q <= (err_code_q != ERR_NONE) ? S_ERR : S_CHECK;
            end
          end
        end

        // ---- store: buffer -> external ----
        S_ST_AW: begin
          if (m_axi_awready_i) state_q <= S_ST_RD;
        end
        S_ST_RD: begin
          state_q <= S_ST_WAIT;
        end
        S_ST_WAIT: begin
          if (rd_data_valid_c) begin
            st_data_q <= rd_data_c;
            state_q   <= S_ST_W;
          end
        end
        S_ST_W: begin
          if (m_axi_wready_i) begin
            int_off_q   <= int_off_q + 20'd8;
            ext_addr_q  <= ext_addr_q + 32'd8;
            remaining_q <= remaining_q - 20'd8;
            beat_q      <= beat_q + 8'd1;
            state_q     <= m_axi_wlast_o ? S_ST_B : S_ST_RD;
          end
        end
        S_ST_B: begin
          if (m_axi_bvalid_i) begin
            if (m_axi_bresp_i[1]) begin
              err_code_q <= ERR_BAD_ADDR;
              state_q    <= S_ERR;
            end else begin
              state_q <= S_CHECK;
            end
          end
        end

        S_DONE: begin
          done_o  <= 1'b1;
          state_q <= S_IDLE;
        end
        S_ERR: begin
          err_o   <= 1'b1;
          state_q <= S_IDLE;
        end
        default: state_q <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
