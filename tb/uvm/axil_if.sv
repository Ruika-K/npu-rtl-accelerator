`timescale 1ns/1ps
`default_nettype none

// AXI4-Lite host port of npu_top. The driver uses drv_cb (master side), the
// monitor mon_cb (inputs only), so both see the same sampled values.
interface axil_if (input wire logic clk, input wire logic rst_n);
  logic [31:0] awaddr;
  logic [2:0]  awprot;
  logic        awvalid;
  logic        awready;
  logic [31:0] wdata;
  logic [3:0]  wstrb;
  logic        wvalid;
  logic        wready;
  logic [1:0]  bresp;
  logic        bvalid;
  logic        bready;
  logic [31:0] araddr;
  logic [2:0]  arprot;
  logic        arvalid;
  logic        arready;
  logic [31:0] rdata;
  logic [1:0]  rresp;
  logic        rvalid;
  logic        rready;

  clocking drv_cb @(posedge clk);
    default input #1step output #1;
    output awaddr, awprot, awvalid, wdata, wstrb, wvalid, bready, araddr, arprot, arvalid, rready;
    input  awready, wready, bresp, bvalid, arready, rdata, rresp, rvalid;
  endclocking

  clocking mon_cb @(posedge clk);
    default input #1step;
    input awaddr, awprot, awvalid, awready, wdata, wstrb, wvalid, wready, bresp, bvalid, bready,
          araddr, arprot, arvalid, arready, rdata, rresp, rvalid, rready;
  endclocking

  // Idle levels, applied during reset (outside the clocking block on purpose).
  function automatic void master_idle();
    awvalid = 1'b0; wvalid = 1'b0; bready = 1'b0; arvalid = 1'b0; rready = 1'b0;
    awaddr = '0; awprot = '0; wdata = '0; wstrb = '0; araddr = '0; arprot = '0;
  endfunction
endinterface

`default_nettype wire
