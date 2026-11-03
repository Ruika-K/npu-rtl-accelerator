`default_nettype none

// AXI4-Lite slave to the internal single-transaction request/acknowledge bus
// (docs/architecture.md section 7.2).
//
// - One transaction at a time; no outstanding requests. A write is accepted
//   only when AW and W are both valid (allowed by the AXI4-Lite protocol);
//   writes win over a simultaneously pending read.
// - wr_req_o / rd_req_o are single-cycle pulses. The responder must answer with
//   wr_ack_i / rd_ack_i at least one cycle after the request; rd_data_i and
//   the err flags are sampled in the ack cycle. err maps to SLVERR.
// - awprot / arprot are accepted and ignored.
module axil_slave
  import npu_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,

  // AXI4-Lite slave
  input  logic [31:0] s_axil_awaddr_i,
  input  logic [2:0]  s_axil_awprot_i,
  input  logic        s_axil_awvalid_i,
  output logic        s_axil_awready_o,
  input  logic [31:0] s_axil_wdata_i,
  input  logic [3:0]  s_axil_wstrb_i,
  input  logic        s_axil_wvalid_i,
  output logic        s_axil_wready_o,
  output logic [1:0]  s_axil_bresp_o,
  output logic        s_axil_bvalid_o,
  input  logic        s_axil_bready_i,
  input  logic [31:0] s_axil_araddr_i,
  input  logic [2:0]  s_axil_arprot_i,
  input  logic        s_axil_arvalid_i,
  output logic        s_axil_arready_o,
  output logic [31:0] s_axil_rdata_o,
  output logic [1:0]  s_axil_rresp_o,
  output logic        s_axil_rvalid_o,
  input  logic        s_axil_rready_i,

  // internal bus
  output logic        wr_req_o,
  output logic [19:0] wr_addr_o,
  output logic [31:0] wr_data_o,
  output logic [3:0]  wr_strb_o,
  input  logic        wr_ack_i,
  input  logic        wr_err_i,
  output logic        rd_req_o,
  output logic [19:0] rd_addr_o,
  input  logic        rd_ack_i,
  input  logic [31:0] rd_data_i,
  input  logic        rd_err_i
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_SLVERR = 2'b10;

  typedef enum logic [2:0] {
    IDLE,
    WR_REQ,
    WR_WAIT,
    WR_RESP,
    RD_REQ,
    RD_WAIT,
    RD_RESP
  } state_e;

  state_e state_q;

  // prot is ignored; only the low 20 address bits are decoded (1 MB window).
  /* verilator lint_off UNUSEDSIGNAL */
  logic [29:0] unused_in;
  assign unused_in = {s_axil_awprot_i, s_axil_arprot_i, s_axil_awaddr_i[31:20], s_axil_araddr_i[31:20]};
  /* verilator lint_on UNUSEDSIGNAL */

  // Handshakes: a write needs AW and W together; a read is taken when idle and
  // no write is being accepted in the same cycle.
  logic wr_accept_c, rd_accept_c;
  assign wr_accept_c = (state_q == IDLE) && s_axil_awvalid_i && s_axil_wvalid_i;
  assign rd_accept_c = (state_q == IDLE) && s_axil_arvalid_i && !wr_accept_c;

  assign s_axil_awready_o = wr_accept_c;
  assign s_axil_wready_o  = wr_accept_c;
  assign s_axil_arready_o = rd_accept_c;

  assign wr_req_o = (state_q == WR_REQ);
  assign rd_req_o = (state_q == RD_REQ);

  assign s_axil_bvalid_o = (state_q == WR_RESP);
  assign s_axil_rvalid_o = (state_q == RD_RESP);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q        <= IDLE;
      wr_addr_o      <= '0;
      wr_data_o      <= '0;
      wr_strb_o      <= '0;
      rd_addr_o      <= '0;
      s_axil_bresp_o <= RESP_OKAY;
      s_axil_rdata_o <= '0;
      s_axil_rresp_o <= RESP_OKAY;
    end else begin
      case (state_q)
        IDLE: begin
          if (wr_accept_c) begin
            wr_addr_o <= s_axil_awaddr_i[19:0];
            wr_data_o <= s_axil_wdata_i;
            wr_strb_o <= s_axil_wstrb_i;
            state_q   <= WR_REQ;
          end else if (rd_accept_c) begin
            rd_addr_o <= s_axil_araddr_i[19:0];
            state_q   <= RD_REQ;
          end
        end
        WR_REQ: begin
          state_q <= WR_WAIT;
        end
        WR_WAIT: begin
          if (wr_ack_i) begin
            s_axil_bresp_o <= wr_err_i ? RESP_SLVERR : RESP_OKAY;
            state_q        <= WR_RESP;
          end
        end
        WR_RESP: begin
          if (s_axil_bready_i) state_q <= IDLE;
        end
        RD_REQ: begin
          state_q <= RD_WAIT;
        end
        RD_WAIT: begin
          if (rd_ack_i) begin
            s_axil_rdata_o <= rd_data_i;
            s_axil_rresp_o <= rd_err_i ? RESP_SLVERR : RESP_OKAY;
            state_q        <= RD_RESP;
          end
        end
        RD_RESP: begin
          if (s_axil_rready_i) state_q <= IDLE;
        end
        default: state_q <= IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
