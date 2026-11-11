`default_nettype none

// Assertions for axil_slave, bound into every instance.
//   m_*  what the AXI4-Lite master and the internal responder must do (assume)
//   a_*  what axil_slave guarantees (assert)
//   c_*  scenarios the tests must reach (cover)
// Sequences are written with |=> and $past only, the subset Verilator 5.020
// supports; multi-cycle facts use small auxiliary registers instead.
module axil_slave_sva (
  input logic        clk_i,
  input logic        rst_ni,
  input logic [31:0] s_axil_awaddr_i,
  input logic        s_axil_awvalid_i,
  input logic        s_axil_awready_o,
  input logic [31:0] s_axil_wdata_i,
  input logic [3:0]  s_axil_wstrb_i,
  input logic        s_axil_wvalid_i,
  input logic        s_axil_wready_o,
  input logic [1:0]  s_axil_bresp_o,
  input logic        s_axil_bvalid_o,
  input logic        s_axil_bready_i,
  input logic [31:0] s_axil_araddr_i,
  input logic        s_axil_arvalid_i,
  input logic        s_axil_arready_o,
  input logic [31:0] s_axil_rdata_o,
  input logic [1:0]  s_axil_rresp_o,
  input logic        s_axil_rvalid_o,
  input logic        s_axil_rready_i,
  input logic        wr_req_o,
  input logic [19:0] wr_addr_o,
  input logic [31:0] wr_data_o,
  input logic [3:0]  wr_strb_o,
  input logic        wr_ack_i,
  input logic        wr_err_i,
  input logic        rd_req_o,
  input logic [19:0] rd_addr_o,
  input logic        rd_ack_i,
  input logic [31:0] rd_data_i,
  input logic        rd_err_i
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_SLVERR = 2'b10;

  logic aw_hs_c, ar_hs_c, b_hs_c, r_hs_c;
  assign aw_hs_c = s_axil_awvalid_i && s_axil_awready_o;
  assign ar_hs_c = s_axil_arvalid_i && s_axil_arready_o;
  assign b_hs_c  = s_axil_bvalid_o && s_axil_bready_i;
  assign r_hs_c  = s_axil_rvalid_o && s_axil_rready_i;

  // Reference model of the one transaction in flight
  logic        wr_pend_q, rd_pend_q;   // accepted, response not yet taken
  logic        acked_q;                // the responder has answered it
  logic [19:0] exp_addr_q;
  logic [31:0] exp_wdata_q, exp_rdata_q;
  logic [3:0]  exp_strb_q;
  logic        exp_err_q;
  logic        resp_done_q;            // a response handshake happened last cycle

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_pend_q <= 1'b0; rd_pend_q <= 1'b0; acked_q <= 1'b0; resp_done_q <= 1'b0;
      exp_addr_q <= '0; exp_wdata_q <= '0; exp_rdata_q <= '0; exp_strb_q <= '0; exp_err_q <= 1'b0;
    end else begin
      resp_done_q <= b_hs_c || r_hs_c;
      if (aw_hs_c) begin
        wr_pend_q <= 1'b1; acked_q <= 1'b0;
        exp_addr_q <= s_axil_awaddr_i[19:0]; exp_wdata_q <= s_axil_wdata_i; exp_strb_q <= s_axil_wstrb_i;
      end
      if (ar_hs_c) begin
        rd_pend_q <= 1'b1; acked_q <= 1'b0;
        exp_addr_q <= s_axil_araddr_i[19:0];
      end
      if (wr_pend_q && !wr_req_o && wr_ack_i && !acked_q) begin
        acked_q <= 1'b1; exp_err_q <= wr_err_i;
      end
      if (rd_pend_q && !rd_req_o && rd_ack_i && !acked_q) begin
        acked_q <= 1'b1; exp_err_q <= rd_err_i; exp_rdata_q <= rd_data_i;
      end
      if (b_hs_c) wr_pend_q <= 1'b0;
      if (r_hs_c) rd_pend_q <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------
  // Environment: AXI4-Lite master
  // ---------------------------------------------------------------------------
  m_aw_hold: assume property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_awvalid_i && !s_axil_awready_o |=> s_axil_awvalid_i && $stable(s_axil_awaddr_i))
    else $error("AW dropped or changed before AWREADY");
  m_w_hold: assume property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_wvalid_i && !s_axil_wready_o |=> s_axil_wvalid_i && $stable({s_axil_wdata_i, s_axil_wstrb_i}))
    else $error("W dropped or changed before WREADY");
  m_ar_hold: assume property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_arvalid_i && !s_axil_arready_o |=> s_axil_arvalid_i && $stable(s_axil_araddr_i))
    else $error("AR dropped or changed before ARREADY");

  // Environment: internal responder. An acknowledge in the request cycle would
  // be lost (documented contract: at least one cycle after the request).
  m_wr_ack_late: assume property (@(posedge clk_i) disable iff (!rst_ni) wr_req_o |-> !wr_ack_i)
    else $error("wr_ack_i in the request cycle");
  m_rd_ack_late: assume property (@(posedge clk_i) disable iff (!rst_ni) rd_req_o |-> !rd_ack_i)
    else $error("rd_ack_i in the request cycle");

  // ---------------------------------------------------------------------------
  // Guarantees
  // ---------------------------------------------------------------------------
  a_b_hold: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_bvalid_o && !s_axil_bready_i |=> s_axil_bvalid_o && $stable(s_axil_bresp_o))
    else $error("B dropped or changed before BREADY");
  a_r_hold: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_rvalid_o && !s_axil_rready_i |=> s_axil_rvalid_o && $stable({s_axil_rdata_o, s_axil_rresp_o}))
    else $error("R dropped or changed before RREADY");

  // A write is accepted on AW and W together
  a_aw_w_together: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_awready_o == s_axil_wready_o && (!s_axil_awready_o || (s_axil_awvalid_i && s_axil_wvalid_i)))
    else $error("AWREADY / WREADY not a joint handshake");

  // One transaction at a time: nothing is accepted while one is in flight
  a_single: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (wr_pend_q || rd_pend_q) |-> !s_axil_awready_o && !s_axil_arready_o)
    else $error("request accepted while another is in flight");
  a_not_both: assert property (@(posedge clk_i) disable iff (!rst_ni)
      !(wr_pend_q && rd_pend_q) && !(s_axil_awready_o && s_axil_arready_o))
    else $error("read and write in flight together");

  // Responses belong to an accepted, acknowledged request and carry its result
  a_b_expected: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_bvalid_o |-> wr_pend_q && acked_q && s_axil_bresp_o == (exp_err_q ? RESP_SLVERR : RESP_OKAY))
    else $error("unexpected B response or wrong BRESP");
  a_r_expected: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_rvalid_o |-> rd_pend_q && acked_q && s_axil_rresp_o == (exp_err_q ? RESP_SLVERR : RESP_OKAY)
                          && s_axil_rdata_o == exp_rdata_q)
    else $error("unexpected R response, wrong RRESP or RDATA");

  // Internal requests: one pulse per accepted transaction, with its payload
  a_wr_req: assert property (@(posedge clk_i) disable iff (!rst_ni)
      wr_req_o |-> wr_pend_q && !acked_q && !rd_req_o && wr_addr_o == exp_addr_q
                   && wr_data_o == exp_wdata_q && wr_strb_o == exp_strb_q)
    else $error("wr_req_o without a pending write or with the wrong payload");
  a_rd_req: assert property (@(posedge clk_i) disable iff (!rst_ni)
      rd_req_o |-> rd_pend_q && !acked_q && !wr_req_o && rd_addr_o == exp_addr_q)
    else $error("rd_req_o without a pending read or with the wrong address");
  a_wr_req_after_accept: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_awvalid_i && s_axil_awready_o |=> wr_req_o)
    else $error("accepted write did not issue wr_req_o");
  a_rd_req_after_accept: assert property (@(posedge clk_i) disable iff (!rst_ni)
      s_axil_arvalid_i && s_axil_arready_o |=> rd_req_o)
    else $error("accepted read did not issue rd_req_o");
  a_req_pulse: assert property (@(posedge clk_i) disable iff (!rst_ni)
      wr_req_o || rd_req_o |=> !wr_req_o && !rd_req_o)
    else $error("request longer than one cycle");

  // ---------------------------------------------------------------------------
  // Scenarios
  // ---------------------------------------------------------------------------
  c_aw_first:      cover property (@(posedge clk_i) disable iff (!rst_ni) s_axil_awvalid_i && !s_axil_wvalid_i);
  c_w_first:       cover property (@(posedge clk_i) disable iff (!rst_ni) s_axil_wvalid_i && !s_axil_awvalid_i);
  c_wr_beats_rd:   cover property (@(posedge clk_i) disable iff (!rst_ni) aw_hs_c && s_axil_arvalid_i);
  c_b_wait:        cover property (@(posedge clk_i) disable iff (!rst_ni) s_axil_bvalid_o && !s_axil_bready_i);
  c_r_wait:        cover property (@(posedge clk_i) disable iff (!rst_ni) s_axil_rvalid_o && !s_axil_rready_i);
  c_wr_slverr:     cover property (@(posedge clk_i) disable iff (!rst_ni) b_hs_c && s_axil_bresp_o == RESP_SLVERR);
  c_rd_slverr:     cover property (@(posedge clk_i) disable iff (!rst_ni) r_hs_c && s_axil_rresp_o == RESP_SLVERR);
  c_ack_delayed:   cover property (@(posedge clk_i) disable iff (!rst_ni)
      (wr_pend_q || rd_pend_q) && !acked_q && !wr_req_o && !rd_req_o && !wr_ack_i && !rd_ack_i);
  c_back_to_back:  cover property (@(posedge clk_i) disable iff (!rst_ni) resp_done_q && (aw_hs_c || ar_hs_c));

endmodule

bind axil_slave axil_slave_sva u_sva (.*);

`default_nettype wire
