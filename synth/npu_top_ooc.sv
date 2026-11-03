`default_nettype none
// Out-of-context wrapper for FPGA place and route (synthesis only, not RTL).
//
// npu_top has 273 input and 211 output bits, more than any ECP5 package can
// bond. This wrapper keeps the core whole and every path register to register:
// all core inputs come from a shift register fed by one pin, all core outputs
// are registered and XOR-reduced onto one pin, so no logic is optimized away
// and the clock stays on a global network. Overhead: 484 + 1 flip-flops and an
// XOR tree of about 70 LUT4s, reported with the numbers in docs/synthesis.md.
module npu_top_ooc
  import npu_pkg::*;
(
  input  logic clk_i,
  input  logic rst_ni,
  input  logic si_i,
  output logic so_o
);

  localparam int unsigned IN_W  = 273;
  localparam int unsigned OUT_W = 211;

  logic [IN_W-1:0]  in_q;
  logic [OUT_W-1:0] out_c, out_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_q  <= '0;
      out_q <= '0;
      so_o  <= 1'b0;
    end else begin
      in_q  <= {in_q[IN_W-2:0], si_i};
      out_q <= out_c;
      so_o  <= ^out_q;
    end
  end

  // core inputs
  logic [31:0] s_axil_awaddr, s_axil_wdata, s_axil_araddr;
  logic [2:0]  s_axil_awprot, s_axil_arprot;
  logic [3:0]  s_axil_wstrb;
  logic        s_axil_awvalid, s_axil_wvalid, s_axil_bready, s_axil_arvalid, s_axil_rready;
  logic                  pix_wr_en;
  logic [ABUF_AW-1:0]    pix_wr_addr;
  logic [ARR_N-1:0][7:0] pix_wr_data;
  logic [ARR_N-1:0]      pix_wr_be;
  logic        m_axi_awready, m_axi_wready, m_axi_bvalid, m_axi_arready, m_axi_rlast, m_axi_rvalid;
  logic [1:0]  m_axi_bresp, m_axi_rresp;
  logic [63:0] m_axi_rdata;

  assign {s_axil_awaddr, s_axil_awprot, s_axil_awvalid, s_axil_wdata, s_axil_wstrb, s_axil_wvalid,
          s_axil_bready, s_axil_araddr, s_axil_arprot, s_axil_arvalid, s_axil_rready,
          pix_wr_en, pix_wr_addr, pix_wr_data, pix_wr_be,
          m_axi_awready, m_axi_wready, m_axi_bresp, m_axi_bvalid, m_axi_arready,
          m_axi_rdata, m_axi_rresp, m_axi_rlast, m_axi_rvalid} = in_q;

  // core outputs
  logic        s_axil_awready, s_axil_wready, s_axil_bvalid, s_axil_arready, s_axil_rvalid;
  logic [1:0]  s_axil_bresp, s_axil_rresp;
  logic [31:0] s_axil_rdata;
  logic        pix_wr_ready;
  logic [31:0] m_axi_awaddr, m_axi_araddr;
  logic [7:0]  m_axi_awlen, m_axi_arlen, m_axi_wstrb;
  logic [2:0]  m_axi_awsize, m_axi_arsize;
  logic [1:0]  m_axi_awburst, m_axi_arburst;
  logic        m_axi_awvalid, m_axi_wlast, m_axi_wvalid, m_axi_bready, m_axi_arvalid, m_axi_rready;
  logic [63:0] m_axi_wdata;
  logic        irq;

  assign out_c = {s_axil_awready, s_axil_wready, s_axil_bresp, s_axil_bvalid, s_axil_arready,
                  s_axil_rdata, s_axil_rresp, s_axil_rvalid, pix_wr_ready,
                  m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst, m_axi_awvalid,
                  m_axi_wdata, m_axi_wstrb, m_axi_wlast, m_axi_wvalid, m_axi_bready,
                  m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst, m_axi_arvalid,
                  m_axi_rready, irq};

  npu_top u_npu (
      .clk_i            (clk_i),
      .rst_ni           (rst_ni),
      .s_axil_awaddr_i  (s_axil_awaddr),
      .s_axil_awprot_i  (s_axil_awprot),
      .s_axil_awvalid_i (s_axil_awvalid),
      .s_axil_awready_o (s_axil_awready),
      .s_axil_wdata_i   (s_axil_wdata),
      .s_axil_wstrb_i   (s_axil_wstrb),
      .s_axil_wvalid_i  (s_axil_wvalid),
      .s_axil_wready_o  (s_axil_wready),
      .s_axil_bresp_o   (s_axil_bresp),
      .s_axil_bvalid_o  (s_axil_bvalid),
      .s_axil_bready_i  (s_axil_bready),
      .s_axil_araddr_i  (s_axil_araddr),
      .s_axil_arprot_i  (s_axil_arprot),
      .s_axil_arvalid_i (s_axil_arvalid),
      .s_axil_arready_o (s_axil_arready),
      .s_axil_rdata_o   (s_axil_rdata),
      .s_axil_rresp_o   (s_axil_rresp),
      .s_axil_rvalid_o  (s_axil_rvalid),
      .s_axil_rready_i  (s_axil_rready),
      .pix_wr_en_i      (pix_wr_en),
      .pix_wr_addr_i    (pix_wr_addr),
      .pix_wr_data_i    (pix_wr_data),
      .pix_wr_be_i      (pix_wr_be),
      .pix_wr_ready_o   (pix_wr_ready),
      .m_axi_awaddr_o   (m_axi_awaddr),
      .m_axi_awlen_o    (m_axi_awlen),
      .m_axi_awsize_o   (m_axi_awsize),
      .m_axi_awburst_o  (m_axi_awburst),
      .m_axi_awvalid_o  (m_axi_awvalid),
      .m_axi_awready_i  (m_axi_awready),
      .m_axi_wdata_o    (m_axi_wdata),
      .m_axi_wstrb_o    (m_axi_wstrb),
      .m_axi_wlast_o    (m_axi_wlast),
      .m_axi_wvalid_o   (m_axi_wvalid),
      .m_axi_wready_i   (m_axi_wready),
      .m_axi_bresp_i    (m_axi_bresp),
      .m_axi_bvalid_i   (m_axi_bvalid),
      .m_axi_bready_o   (m_axi_bready),
      .m_axi_araddr_o   (m_axi_araddr),
      .m_axi_arlen_o    (m_axi_arlen),
      .m_axi_arsize_o   (m_axi_arsize),
      .m_axi_arburst_o  (m_axi_arburst),
      .m_axi_arvalid_o  (m_axi_arvalid),
      .m_axi_arready_i  (m_axi_arready),
      .m_axi_rdata_i    (m_axi_rdata),
      .m_axi_rresp_i    (m_axi_rresp),
      .m_axi_rlast_i    (m_axi_rlast),
      .m_axi_rvalid_i   (m_axi_rvalid),
      .m_axi_rready_o   (m_axi_rready),
      .irq_o            (irq)
  );

endmodule

`default_nettype wire
