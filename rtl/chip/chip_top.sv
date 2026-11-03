`default_nettype none

// Chip-level wrapper around npu_top (docs/architecture.md section 11):
//   - one asynchronous reset, synchronized into the system and pixel domains;
//   - APB configuration port in the system clock domain: the NPU window
//     (0x00000 .. 0xFFFFF, the AXI4-Lite map) through apb2axil, and the chip
//     registers at 0x100000;
//   - a pixel stream in its own clock domain, crossed with a Gray-pointer
//     async FIFO and written into the activation buffer by pix_writer;
//   - a clock gate on the NPU core and the pixel writer (CLK_CTRL.CORE_CLK_EN);
//   - irq_o = NPU done interrupt or pixel frame-done interrupt.
module chip_top
  import npu_pkg::*;
#(
  parameter int unsigned PIX_FIFO_DEPTH = 16
) (
  input  logic        sys_clk_i,
  input  logic        pix_clk_i,
  input  logic        rst_ni,

  input  logic        s_apb_psel_i,
  input  logic        s_apb_penable_i,
  input  logic        s_apb_pwrite_i,
  input  logic [31:0] s_apb_paddr_i,
  input  logic [31:0] s_apb_pwdata_i,
  input  logic [3:0]  s_apb_pstrb_i,
  output logic        s_apb_pready_o,
  output logic [31:0] s_apb_prdata_o,
  output logic        s_apb_pslverr_o,

  input  logic        pix_valid_i,
  input  logic        pix_sof_i,
  input  logic [7:0]  pix_data_i,
  output logic        pix_ready_o,

  // AXI4 master of the NPU DMA (core clock domain)
  output logic [31:0] m_axi_awaddr_o,
  output logic [7:0]  m_axi_awlen_o,
  output logic [2:0]  m_axi_awsize_o,
  output logic [1:0]  m_axi_awburst_o,
  output logic        m_axi_awvalid_o,
  input  logic        m_axi_awready_i,
  output logic [63:0] m_axi_wdata_o,
  output logic [7:0]  m_axi_wstrb_o,
  output logic        m_axi_wlast_o,
  output logic        m_axi_wvalid_o,
  input  logic        m_axi_wready_i,
  input  logic [1:0]  m_axi_bresp_i,
  input  logic        m_axi_bvalid_i,
  output logic        m_axi_bready_o,
  output logic [31:0] m_axi_araddr_o,
  output logic [7:0]  m_axi_arlen_o,
  output logic [2:0]  m_axi_arsize_o,
  output logic [1:0]  m_axi_arburst_o,
  output logic        m_axi_arvalid_o,
  input  logic        m_axi_arready_i,
  input  logic [63:0] m_axi_rdata_i,
  input  logic [1:0]  m_axi_rresp_i,
  input  logic        m_axi_rlast_i,
  input  logic        m_axi_rvalid_i,
  output logic        m_axi_rready_o,

  output logic        irq_o
);

  localparam int unsigned N = ARR_N;

  // ---------------------------------------------------------------------------
  // Resets and clocks
  // ---------------------------------------------------------------------------
  logic sys_rst_n, pix_rst_n, core_clk, core_clk_en;

  rst_sync u_rst_sys (.clk_i (sys_clk_i), .rst_ni (rst_ni), .rst_sync_no (sys_rst_n));
  rst_sync u_rst_pix (.clk_i (pix_clk_i), .rst_ni (rst_ni), .rst_sync_no (pix_rst_n));
  clk_gate u_core_gate (.clk_i (sys_clk_i), .en_i (core_clk_en), .clk_o (core_clk));

  // ---------------------------------------------------------------------------
  // APB decode: bit 20 selects the chip registers
  // ---------------------------------------------------------------------------
  logic sel_chip_c, core_psel_c, chip_psel_c;
  logic core_pready, chip_pready, core_pslverr, chip_pslverr;
  logic [31:0] core_prdata, chip_prdata;

  assign sel_chip_c  = s_apb_paddr_i[20];
  assign core_psel_c = s_apb_psel_i && !sel_chip_c;
  assign chip_psel_c = s_apb_psel_i &&  sel_chip_c;
  assign s_apb_pready_o  = sel_chip_c ? chip_pready  : core_pready;
  assign s_apb_prdata_o  = sel_chip_c ? chip_prdata  : core_prdata;
  assign s_apb_pslverr_o = sel_chip_c ? chip_pslverr : core_pslverr;

  // ---------------------------------------------------------------------------
  // Chip registers and pixel path control
  // ---------------------------------------------------------------------------
  logic               pix_en;
  logic [ABUF_AW-1:0] pix_base, pix_len;
  logic               frame_done, sof_err, overrun_sys, pix_irq, npu_irq;

  chip_regs u_regs (
      .clk_i (sys_clk_i), .rst_ni (sys_rst_n),
      .psel_i (chip_psel_c), .penable_i (s_apb_penable_i), .pwrite_i (s_apb_pwrite_i), .paddr_i (s_apb_paddr_i[7:0]),
      .pwdata_i (s_apb_pwdata_i), .pstrb_i (s_apb_pstrb_i), .pready_o (chip_pready), .prdata_o (chip_prdata), .pslverr_o (chip_pslverr),
      .core_clk_en_o (core_clk_en), .pix_en_o (pix_en), .pix_base_o (pix_base), .pix_len_o (pix_len),
      .frame_done_i (frame_done), .sof_err_i (sof_err), .overrun_i (overrun_sys), .irq_o (pix_irq)
  );

  // ---------------------------------------------------------------------------
  // Pixel stream: pix_clk domain -> async FIFO -> core_clk domain
  // ---------------------------------------------------------------------------
  logic       fifo_rd_valid, fifo_rd_ready;
  logic [8:0] fifo_rd_data;

  async_fifo #(.WIDTH(9), .DEPTH(PIX_FIFO_DEPTH)) u_pix_fifo (
      .wr_clk_i (pix_clk_i), .wr_rst_ni (pix_rst_n), .wr_valid_i (pix_valid_i), .wr_ready_o (pix_ready_o),
      .wr_data_i ({pix_sof_i, pix_data_i}),
      .rd_clk_i (core_clk), .rd_rst_ni (sys_rst_n), .rd_valid_o (fifo_rd_valid), .rd_ready_i (fifo_rd_ready),
      .rd_data_o (fifo_rd_data)
  );

  // Overrun: the source pushed while the FIFO was full. Toggle in the pixel
  // domain, synchronize, edge-detect in the system domain.
  logic overrun_tgl_q, overrun_sync, overrun_sync_q;
  always_ff @(posedge pix_clk_i or negedge pix_rst_n) begin
    if (!pix_rst_n) overrun_tgl_q <= 1'b0;
    else if (pix_valid_i && !pix_ready_o) overrun_tgl_q <= ~overrun_tgl_q;
  end
  cdc_sync u_overrun_sync (.clk_i (sys_clk_i), .rst_ni (sys_rst_n), .d_i (overrun_tgl_q), .q_o (overrun_sync));
  always_ff @(posedge sys_clk_i or negedge sys_rst_n) begin
    if (!sys_rst_n) overrun_sync_q <= 1'b0;
    else            overrun_sync_q <= overrun_sync;
  end
  assign overrun_sys = overrun_sync ^ overrun_sync_q;

  logic              pix_wr_en, pix_wr_ready;
  logic [ABUF_AW-1:0] pix_wr_addr;
  logic [N-1:0][7:0] pix_wr_data;
  logic [N-1:0]      pix_wr_be;

  pix_writer #(.N(N)) u_pix_writer (
      .clk_i (core_clk), .rst_ni (sys_rst_n),
      .en_i (pix_en), .base_i (pix_base), .len_i (pix_len),
      .fifo_valid_i (fifo_rd_valid), .fifo_ready_o (fifo_rd_ready), .fifo_data_i (fifo_rd_data),
      .wr_en_o (pix_wr_en), .wr_addr_o (pix_wr_addr), .wr_data_o (pix_wr_data), .wr_be_o (pix_wr_be), .wr_ready_i (pix_wr_ready),
      .frame_done_o (frame_done), .sof_err_o (sof_err)
  );

  // ---------------------------------------------------------------------------
  // APB -> AXI4-Lite -> NPU core (gated clock)
  // ---------------------------------------------------------------------------
  logic [31:0] axil_awaddr, axil_wdata, axil_araddr, axil_rdata;
  logic [2:0]  axil_awprot, axil_arprot;
  logic [3:0]  axil_wstrb;
  logic [1:0]  axil_bresp, axil_rresp;
  logic axil_awvalid, axil_awready, axil_wvalid, axil_wready, axil_bvalid, axil_bready;
  logic axil_arvalid, axil_arready, axil_rvalid, axil_rready;

  apb2axil u_bridge (
      .clk_i (sys_clk_i), .rst_ni (sys_rst_n), .block_i (!core_clk_en),
      .s_apb_psel_i (core_psel_c), .s_apb_penable_i (s_apb_penable_i), .s_apb_pwrite_i (s_apb_pwrite_i),
      .s_apb_paddr_i (s_apb_paddr_i), .s_apb_pwdata_i (s_apb_pwdata_i), .s_apb_pstrb_i (s_apb_pstrb_i),
      .s_apb_pready_o (core_pready), .s_apb_prdata_o (core_prdata), .s_apb_pslverr_o (core_pslverr),
      .m_axil_awaddr_o (axil_awaddr), .m_axil_awprot_o (axil_awprot), .m_axil_awvalid_o (axil_awvalid), .m_axil_awready_i (axil_awready),
      .m_axil_wdata_o (axil_wdata), .m_axil_wstrb_o (axil_wstrb), .m_axil_wvalid_o (axil_wvalid), .m_axil_wready_i (axil_wready),
      .m_axil_bresp_i (axil_bresp), .m_axil_bvalid_i (axil_bvalid), .m_axil_bready_o (axil_bready),
      .m_axil_araddr_o (axil_araddr), .m_axil_arprot_o (axil_arprot), .m_axil_arvalid_o (axil_arvalid), .m_axil_arready_i (axil_arready),
      .m_axil_rdata_i (axil_rdata), .m_axil_rresp_i (axil_rresp), .m_axil_rvalid_i (axil_rvalid), .m_axil_rready_o (axil_rready)
  );

  npu_top u_npu (
      .clk_i (core_clk), .rst_ni (sys_rst_n),
      .s_axil_awaddr_i (axil_awaddr), .s_axil_awprot_i (axil_awprot), .s_axil_awvalid_i (axil_awvalid), .s_axil_awready_o (axil_awready),
      .s_axil_wdata_i (axil_wdata), .s_axil_wstrb_i (axil_wstrb), .s_axil_wvalid_i (axil_wvalid), .s_axil_wready_o (axil_wready),
      .s_axil_bresp_o (axil_bresp), .s_axil_bvalid_o (axil_bvalid), .s_axil_bready_i (axil_bready),
      .s_axil_araddr_i (axil_araddr), .s_axil_arprot_i (axil_arprot), .s_axil_arvalid_i (axil_arvalid), .s_axil_arready_o (axil_arready),
      .s_axil_rdata_o (axil_rdata), .s_axil_rresp_o (axil_rresp), .s_axil_rvalid_o (axil_rvalid), .s_axil_rready_i (axil_rready),
      .pix_wr_en_i (pix_wr_en), .pix_wr_addr_i (pix_wr_addr), .pix_wr_data_i (pix_wr_data), .pix_wr_be_i (pix_wr_be),
      .pix_wr_ready_o (pix_wr_ready),
      .m_axi_awaddr_o (m_axi_awaddr_o), .m_axi_awlen_o (m_axi_awlen_o), .m_axi_awsize_o (m_axi_awsize_o), .m_axi_awburst_o (m_axi_awburst_o),
      .m_axi_awvalid_o (m_axi_awvalid_o), .m_axi_awready_i (m_axi_awready_i),
      .m_axi_wdata_o (m_axi_wdata_o), .m_axi_wstrb_o (m_axi_wstrb_o), .m_axi_wlast_o (m_axi_wlast_o), .m_axi_wvalid_o (m_axi_wvalid_o), .m_axi_wready_i (m_axi_wready_i),
      .m_axi_bresp_i (m_axi_bresp_i), .m_axi_bvalid_i (m_axi_bvalid_i), .m_axi_bready_o (m_axi_bready_o),
      .m_axi_araddr_o (m_axi_araddr_o), .m_axi_arlen_o (m_axi_arlen_o), .m_axi_arsize_o (m_axi_arsize_o), .m_axi_arburst_o (m_axi_arburst_o),
      .m_axi_arvalid_o (m_axi_arvalid_o), .m_axi_arready_i (m_axi_arready_i),
      .m_axi_rdata_i (m_axi_rdata_i), .m_axi_rresp_i (m_axi_rresp_i), .m_axi_rlast_i (m_axi_rlast_i), .m_axi_rvalid_i (m_axi_rvalid_i), .m_axi_rready_o (m_axi_rready_o),
      .irq_o (npu_irq)
  );

  assign irq_o = npu_irq | pix_irq;

endmodule

`default_nettype wire
