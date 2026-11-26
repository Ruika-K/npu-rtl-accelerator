`timescale 1ns/1ps
`default_nettype none

// Top of the UVM testbench: clock, reset, interfaces, npu_top, run_test().
// The SVA checkers are bound into the design from tb/sva (sim/filelists/sva.f).
// +UVM_TESTNAME selects the test (see npu_tests.svh).
module npu_uvm_top;
  import uvm_pkg::*;
  import npu_uvm_pkg::*;
  import npu_pkg::*;

  logic clk = 1'b0;
  logic rst_n = 1'b0;

  always #5ns clk = ~clk;
  initial begin
    repeat (10) @(posedge clk);
    rst_n <= 1'b1;
  end

  axil_if axil (.clk(clk), .rst_n(rst_n));
  axi_if  axi  (.clk(clk), .rst_n(rst_n));
  irq_if  irq  (.clk(clk), .rst_n(rst_n));

  npu_top u_dut (
    .clk_i            (clk),
    .rst_ni           (rst_n),
    .s_axil_awaddr_i  (axil.awaddr),
    .s_axil_awprot_i  (axil.awprot),
    .s_axil_awvalid_i (axil.awvalid),
    .s_axil_awready_o (axil.awready),
    .s_axil_wdata_i   (axil.wdata),
    .s_axil_wstrb_i   (axil.wstrb),
    .s_axil_wvalid_i  (axil.wvalid),
    .s_axil_wready_o  (axil.wready),
    .s_axil_bresp_o   (axil.bresp),
    .s_axil_bvalid_o  (axil.bvalid),
    .s_axil_bready_i  (axil.bready),
    .s_axil_araddr_i  (axil.araddr),
    .s_axil_arprot_i  (axil.arprot),
    .s_axil_arvalid_i (axil.arvalid),
    .s_axil_arready_o (axil.arready),
    .s_axil_rdata_o   (axil.rdata),
    .s_axil_rresp_o   (axil.rresp),
    .s_axil_rvalid_o  (axil.rvalid),
    .s_axil_rready_i  (axil.rready),
    .pix_wr_en_i      (1'b0),
    .pix_wr_addr_i    ('0),
    .pix_wr_data_i    ('0),
    .pix_wr_be_i      ('0),
    .pix_wr_ready_o   (),
    .m_axi_awaddr_o   (axi.awaddr),
    .m_axi_awlen_o    (axi.awlen),
    .m_axi_awsize_o   (axi.awsize),
    .m_axi_awburst_o  (axi.awburst),
    .m_axi_awvalid_o  (axi.awvalid),
    .m_axi_awready_i  (axi.awready),
    .m_axi_wdata_o    (axi.wdata),
    .m_axi_wstrb_o    (axi.wstrb),
    .m_axi_wlast_o    (axi.wlast),
    .m_axi_wvalid_o   (axi.wvalid),
    .m_axi_wready_i   (axi.wready),
    .m_axi_bresp_i    (axi.bresp),
    .m_axi_bvalid_i   (axi.bvalid),
    .m_axi_bready_o   (axi.bready),
    .m_axi_araddr_o   (axi.araddr),
    .m_axi_arlen_o    (axi.arlen),
    .m_axi_arsize_o   (axi.arsize),
    .m_axi_arburst_o  (axi.arburst),
    .m_axi_arvalid_o  (axi.arvalid),
    .m_axi_arready_i  (axi.arready),
    .m_axi_rdata_i    (axi.rdata),
    .m_axi_rresp_i    (axi.rresp),
    .m_axi_rlast_i    (axi.rlast),
    .m_axi_rvalid_i   (axi.rvalid),
    .m_axi_rready_o   (axi.rready),
    .irq_o            (irq.irq)
  );

  initial begin
    uvm_config_db #(virtual axil_if)::set(null, "uvm_test_top.env.host.*", "vif", axil);
    uvm_config_db #(virtual axi_if)::set(null, "uvm_test_top.env.ext_mem", "vif", axi);
    uvm_config_db #(virtual irq_if)::set(null, "uvm_test_top.env", "irq_vif", irq);
    run_test();
  end
endmodule

`default_nettype wire
