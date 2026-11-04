`default_nettype none

// APB (APB4 signals, APB3 compatible) slave to AXI4-Lite master bridge.
// One transfer at a time, same clock domain. A PSEL && !PENABLE setup phase is
// followed by the access phase, during which the AXI transaction runs;
// PREADY rises with the AXI response and PSLVERR mirrors a SLVERR.
// block_i forces an immediate PSLVERR without an AXI transaction (used while
// the core clock is gated).
module apb2axil (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic        block_i,

  input  logic        s_apb_psel_i,
  input  logic        s_apb_penable_i,
  input  logic        s_apb_pwrite_i,
  input  logic [31:0] s_apb_paddr_i,
  input  logic [31:0] s_apb_pwdata_i,
  input  logic [3:0]  s_apb_pstrb_i,
  output logic        s_apb_pready_o,
  output logic [31:0] s_apb_prdata_o,
  output logic        s_apb_pslverr_o,

  output logic [31:0] m_axil_awaddr_o,
  output logic [2:0]  m_axil_awprot_o,
  output logic        m_axil_awvalid_o,
  input  logic        m_axil_awready_i,
  output logic [31:0] m_axil_wdata_o,
  output logic [3:0]  m_axil_wstrb_o,
  output logic        m_axil_wvalid_o,
  input  logic        m_axil_wready_i,
  input  logic [1:0]  m_axil_bresp_i,
  input  logic        m_axil_bvalid_i,
  output logic        m_axil_bready_o,
  output logic [31:0] m_axil_araddr_o,
  output logic [2:0]  m_axil_arprot_o,
  output logic        m_axil_arvalid_o,
  input  logic        m_axil_arready_i,
  input  logic [31:0] m_axil_rdata_i,
  input  logic [1:0]  m_axil_rresp_i,
  input  logic        m_axil_rvalid_i,
  output logic        m_axil_rready_o
);

  typedef enum logic [2:0] {IDLE, WR_ADDR_DATA, WR_RESP, RD_ADDR, RD_RESP, DONE} state_e;
  state_e state_q;

  logic aw_done_q, w_done_q;
  logic err_q;
  logic [31:0] rdata_q;

  assign m_axil_awprot_o = 3'b000;
  assign m_axil_arprot_o = 3'b000;
  assign m_axil_awaddr_o = s_apb_paddr_i;
  assign m_axil_araddr_o = s_apb_paddr_i;
  assign m_axil_wdata_o  = s_apb_pwdata_i;
  assign m_axil_wstrb_o  = s_apb_pstrb_i;

  assign m_axil_awvalid_o = (state_q == WR_ADDR_DATA) && !aw_done_q;
  assign m_axil_wvalid_o  = (state_q == WR_ADDR_DATA) && !w_done_q;
  assign m_axil_bready_o  = (state_q == WR_RESP);
  assign m_axil_arvalid_o = (state_q == RD_ADDR);
  assign m_axil_rready_o = (state_q == RD_RESP);

  assign s_apb_pready_o  = (state_q == DONE);
  assign s_apb_pslverr_o = (state_q == DONE) && err_q;
  assign s_apb_prdata_o  = rdata_q;

  logic setup_c;
  assign setup_c = s_apb_psel_i && !s_apb_penable_i && (state_q == IDLE);

  // Only the error bit of a response matters here (OKAY / SLVERR).
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0] unused_resp_lsb;
  assign unused_resp_lsb = {m_axil_bresp_i[0], m_axil_rresp_i[0]};
  /* verilator lint_on UNUSEDSIGNAL */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q   <= IDLE;
      aw_done_q <= 1'b0;
      w_done_q  <= 1'b0;
      err_q     <= 1'b0;
      rdata_q   <= '0;
    end else begin
      case (state_q)
        IDLE: begin
          if (setup_c) begin
            aw_done_q <= 1'b0;
            w_done_q  <= 1'b0;
            err_q     <= 1'b0;
            if (block_i) begin
              err_q   <= 1'b1;
              rdata_q <= '0;
              state_q <= DONE;
            end else begin
              state_q <= s_apb_pwrite_i ? WR_ADDR_DATA : RD_ADDR;
            end
          end
        end
        WR_ADDR_DATA: begin
          if (m_axil_awvalid_o && m_axil_awready_i) aw_done_q <= 1'b1;
          if (m_axil_wvalid_o && m_axil_wready_i)   w_done_q  <= 1'b1;
          if ((aw_done_q || (m_axil_awvalid_o && m_axil_awready_i)) &&
              (w_done_q  || (m_axil_wvalid_o && m_axil_wready_i))) state_q <= WR_RESP;
        end
        WR_RESP: begin
          if (m_axil_bvalid_i) begin
            err_q   <= m_axil_bresp_i[1];
            state_q <= DONE;
          end
        end
        RD_ADDR: begin
          if (m_axil_arready_i) state_q <= RD_RESP;
        end
        RD_RESP: begin
          if (m_axil_rvalid_i) begin
            rdata_q <= m_axil_rdata_i;
            err_q   <= m_axil_rresp_i[1];
            state_q <= DONE;
          end
        end
        DONE: begin
          state_q <= IDLE;   // PREADY for exactly the one access-phase cycle
        end
        default: state_q <= IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
