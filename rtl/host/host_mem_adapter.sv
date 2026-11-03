`default_nettype none

// Adapts the 32-bit host memory port of npu_regs to the 64-bit ports of
// act_buffer (hm_sel_i = 0) and weight_buffer (hm_sel_i = 1).
// act_buffer: a 32-bit access is an unaligned 8-byte access whose lanes 0..3
//   carry the word; lanes 4..7 are disabled.
// weight_buffer: word address = byte offset >> 3, the half is selected by bit 2.
// One access at a time; hm_ack_o comes 2 cycles (writes, weight reads) or
// 3 cycles (activation reads) after the request.
module host_mem_adapter
  import npu_pkg::*;
(
  input  logic                 clk_i,
  input  logic                 rst_ni,

  input  logic                 hm_req_i,
  input  logic                 hm_we_i,
  input  logic                 hm_sel_i,
  input  logic [19:0]          hm_addr_i,
  input  logic [31:0]          hm_wdata_i,
  input  logic [3:0]           hm_be_i,
  output logic                 hm_ack_o,
  output logic [31:0]          hm_rdata_o,

  output logic                 abuf_rd_en_o,
  output logic [ABUF_AW-1:0]   abuf_rd_addr_o,
  input  logic [ARR_N-1:0][7:0] abuf_rd_data_i,
  input  logic                 abuf_rd_valid_i,
  output logic                 abuf_wr_en_o,
  output logic [ABUF_AW-1:0]   abuf_wr_addr_o,
  output logic [ARR_N-1:0][7:0] abuf_wr_data_o,
  output logic [ARR_N-1:0]     abuf_wr_be_o,

  output logic                 wbuf_rd_en_o,
  output logic [WBUF_WAW-1:0]  wbuf_rd_addr_o,
  input  logic [63:0]          wbuf_rd_data_i,
  input  logic                 wbuf_rd_valid_i,
  output logic                 wbuf_wr_en_o,
  output logic [WBUF_WAW-1:0]  wbuf_wr_addr_o,
  output logic [63:0]          wbuf_wr_data_o,
  output logic [7:0]           wbuf_wr_be_o
);

  typedef enum logic [1:0] {IDLE, WR_ACK, RD_WAIT} state_e;
  state_e state_q;
  logic   sel_q, half_q;

  // Request decode (combinational, used in the request cycle only)
  assign abuf_rd_en_o   = hm_req_i && !hm_we_i && !hm_sel_i;
  assign abuf_rd_addr_o = hm_addr_i[ABUF_AW-1:0];
  assign abuf_wr_en_o   = hm_req_i &&  hm_we_i && !hm_sel_i;
  assign abuf_wr_addr_o = hm_addr_i[ABUF_AW-1:0];
  assign wbuf_rd_en_o   = hm_req_i && !hm_we_i &&  hm_sel_i;
  assign wbuf_rd_addr_o = hm_addr_i[WBUF_WAW+2:3];
  assign wbuf_wr_en_o   = hm_req_i &&  hm_we_i &&  hm_sel_i;
  assign wbuf_wr_addr_o = hm_addr_i[WBUF_WAW+2:3];
  assign wbuf_wr_data_o = {hm_wdata_i, hm_wdata_i};
  assign wbuf_wr_be_o   = hm_addr_i[2] ? {hm_be_i, 4'h0} : {4'h0, hm_be_i};

  always_comb begin
    abuf_wr_data_o = '0;
    abuf_wr_be_o   = '0;
    for (int i = 0; i < 4; i++) begin
      abuf_wr_data_o[i] = hm_wdata_i[8*i +: 8];
      abuf_wr_be_o[i]   = hm_be_i[i];
    end
  end

  // The upper address bits are covered by npu_regs' window decode; only the
  // low four lanes of an activation read carry the requested word.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [19-ABUF_AW:0] unused_addr;
  logic [31:0]         unused_lanes;
  assign unused_addr  = hm_addr_i[19:ABUF_AW];
  assign unused_lanes = {abuf_rd_data_i[7], abuf_rd_data_i[6], abuf_rd_data_i[5], abuf_rd_data_i[4]};
  /* verilator lint_on UNUSEDSIGNAL */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q    <= IDLE;
      sel_q      <= 1'b0;
      half_q     <= 1'b0;
      hm_ack_o   <= 1'b0;
      hm_rdata_o <= '0;
    end else begin
      hm_ack_o <= 1'b0;
      case (state_q)
        IDLE: begin
          if (hm_req_i) begin
            sel_q  <= hm_sel_i;
            half_q <= hm_addr_i[2];
            state_q <= hm_we_i ? WR_ACK : RD_WAIT;
          end
        end
        WR_ACK: begin
          hm_ack_o <= 1'b1;
          state_q  <= IDLE;
        end
        RD_WAIT: begin
          if (sel_q && wbuf_rd_valid_i) begin
            hm_rdata_o <= half_q ? wbuf_rd_data_i[63:32] : wbuf_rd_data_i[31:0];
            hm_ack_o   <= 1'b1;
            state_q    <= IDLE;
          end else if (!sel_q && abuf_rd_valid_i) begin
            hm_rdata_o <= {abuf_rd_data_i[3], abuf_rd_data_i[2], abuf_rd_data_i[1], abuf_rd_data_i[0]};
            hm_ack_o   <= 1'b1;
            state_q    <= IDLE;
          end
        end
        default: state_q <= IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
