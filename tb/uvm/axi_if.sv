`timescale 1ns/1ps
`default_nettype none

// AXI4 master port of the DMA (64-bit data). The testbench is the slave: the
// memory model drives the ready / response side through slv_cb.
interface axi_if (input wire logic clk, input wire logic rst_n);
  logic [31:0] awaddr;
  logic [7:0]  awlen;
  logic [2:0]  awsize;
  logic [1:0]  awburst;
  logic        awvalid;
  logic        awready;
  logic [63:0] wdata;
  logic [7:0]  wstrb;
  logic        wlast;
  logic        wvalid;
  logic        wready;
  logic [1:0]  bresp;
  logic        bvalid;
  logic        bready;
  logic [31:0] araddr;
  logic [7:0]  arlen;
  logic [2:0]  arsize;
  logic [1:0]  arburst;
  logic        arvalid;
  logic        arready;
  logic [63:0] rdata;
  logic [1:0]  rresp;
  logic        rlast;
  logic        rvalid;
  logic        rready;

  clocking slv_cb @(posedge clk);
    default input #1step output #1;
    input  awaddr, awlen, awsize, awburst, awvalid, wdata, wstrb, wlast, wvalid, bready,
           araddr, arlen, arsize, arburst, arvalid, rready;
    output awready, wready, bresp, bvalid, arready, rdata, rresp, rlast, rvalid;
  endclocking

  function automatic void slave_idle();
    awready = 1'b0; wready = 1'b0; bvalid = 1'b0; bresp = '0;
    arready = 1'b0; rvalid = 1'b0; rdata = '0; rresp = '0; rlast = 1'b0;
  endfunction
endinterface

`default_nettype wire
