`timescale 1ns/1ps
`default_nettype none

// AXI4-Lite master bus-functional model for the testbenches.
//
// Usage: instantiate, connect to the DUT's slave port, then call the tasks
// hierarchically, e.g. `u_bfm.write(addr, data, 4'hF, resp)`.
// Signals are driven on the falling clock edge with blocking assignments and
// the ready inputs are sampled 1 ns after the falling edge, so a handshake is
// recognized exactly when the DUT will see it at the next rising edge.
//
// Knobs (set from the testbench):
//   aw_w_order      0: AW and W in the same cycle, 1: AW first, 2: W first,
//                   3: random per transaction
//   max_resp_delay  bready / rready are asserted after a random delay of
//                   0 .. max_resp_delay cycles
module axil_master_bfm (
  input  logic        clk_i,

  output logic [31:0] awaddr_o,
  output logic        awvalid_o,
  input  logic        awready_i,
  output logic [31:0] wdata_o,
  output logic [3:0]  wstrb_o,
  output logic        wvalid_o,
  input  logic        wready_i,
  input  logic [1:0]  bresp_i,
  input  logic        bvalid_i,
  output logic        bready_o,
  output logic [31:0] araddr_o,
  output logic        arvalid_o,
  input  logic        arready_i,
  input  logic [31:0] rdata_i,
  input  logic [1:0]  rresp_i,
  input  logic        rvalid_i,
  output logic        rready_o
);

  int aw_w_order     = 0;
  int max_resp_delay = 0;

  initial begin
    awaddr_o  = '0; awvalid_o = 1'b0;
    wdata_o   = '0; wstrb_o   = '0; wvalid_o = 1'b0;
    bready_o  = 1'b0;
    araddr_o  = '0; arvalid_o = 1'b0;
    rready_o  = 1'b0;
  end

  task automatic write(input logic [31:0] addr, input logic [31:0] data, input logic [3:0] strb,
                       output logic [1:0] resp);
    int  order, d_aw, d_w;
    bit  aw_done, w_done, aw_hs, w_hs;
    order   = (aw_w_order == 3) ? $urandom_range(2) : aw_w_order;
    d_aw    = (order == 2) ? $urandom_range(3, 1) : 0;
    d_w     = (order == 1) ? $urandom_range(3, 1) : 0;
    aw_done = 1'b0;
    w_done  = 1'b0;
    @(negedge clk_i);
    while (!(aw_done && w_done)) begin
      if (!aw_done && d_aw == 0) begin awaddr_o = addr; awvalid_o = 1'b1; end
      if (!w_done  && d_w  == 0) begin wdata_o = data; wstrb_o = strb; wvalid_o = 1'b1; end
      #1;
      aw_hs = awvalid_o && awready_i;
      w_hs  = wvalid_o  && wready_i;
      @(posedge clk_i);
      @(negedge clk_i);
      if (aw_hs) begin aw_done = 1'b1; awvalid_o = 1'b0; end
      if (w_hs)  begin w_done  = 1'b1; wvalid_o  = 1'b0; end
      if (d_aw > 0) d_aw--;
      if (d_w  > 0) d_w--;
    end
    repeat ($urandom_range(max_resp_delay)) @(negedge clk_i);
    bready_o = 1'b1;
    forever begin
      #1;
      if (bvalid_i) begin
        resp = bresp_i;
        @(posedge clk_i);
        @(negedge clk_i);
        bready_o = 1'b0;
        break;
      end
      @(negedge clk_i);
    end
  endtask

  task automatic read(input logic [31:0] addr, output logic [31:0] data, output logic [1:0] resp);
    bit ar_hs;
    @(negedge clk_i);
    araddr_o  = addr;
    arvalid_o = 1'b1;
    forever begin
      #1;
      ar_hs = arvalid_o && arready_i;
      @(posedge clk_i);
      @(negedge clk_i);
      if (ar_hs) begin arvalid_o = 1'b0; break; end
    end
    repeat ($urandom_range(max_resp_delay)) @(negedge clk_i);
    rready_o = 1'b1;
    forever begin
      #1;
      if (rvalid_i) begin
        data = rdata_i;
        resp = rresp_i;
        @(posedge clk_i);
        @(negedge clk_i);
        rready_o = 1'b0;
        break;
      end
      @(negedge clk_i);
    end
  endtask

  // Write with a read request raised in the same cycle (AW, W and AR all
  // valid together), to exercise the slave's arbitration. Returns after the
  // write response; the caller finishes the read with read_complete.
  // Note: a slave that serves one transaction at a time will take the read
  // first if AR is raised a cycle earlier, so the three must be simultaneous.
  task automatic write_with_ar(input logic [31:0] waddr, input logic [31:0] data, input logic [3:0] strb,
                               input logic [31:0] raddr, output logic [1:0] resp);
    bit aw_hs, w_hs, aw_done, w_done;
    aw_done = 1'b0;
    w_done  = 1'b0;
    @(negedge clk_i);
    araddr_o  = raddr;
    arvalid_o = 1'b1;
    awaddr_o  = waddr;
    awvalid_o = 1'b1;
    wdata_o   = data;
    wstrb_o   = strb;
    wvalid_o  = 1'b1;
    while (!(aw_done && w_done)) begin
      #1;
      aw_hs = awvalid_o && awready_i;
      w_hs  = wvalid_o  && wready_i;
      @(posedge clk_i);
      @(negedge clk_i);
      if (aw_hs) begin aw_done = 1'b1; awvalid_o = 1'b0; end
      if (w_hs)  begin w_done  = 1'b1; wvalid_o  = 1'b0; end
    end
    bready_o = 1'b1;
    forever begin
      #1;
      if (bvalid_i) begin
        resp = bresp_i;
        @(posedge clk_i);
        @(negedge clk_i);
        bready_o = 1'b0;
        break;
      end
      @(negedge clk_i);
    end
  endtask

  // Finish a read whose AR was raised by write_with_ar.
  task automatic read_complete(output logic [31:0] data, output logic [1:0] resp);
    bit ar_hs;
    forever begin
      #1;
      ar_hs = arvalid_o && arready_i;
      if (rvalid_i) begin arvalid_o = 1'b0; break; end      // accepted before we looked
      @(posedge clk_i);
      @(negedge clk_i);
      if (ar_hs) begin arvalid_o = 1'b0; break; end
    end
    rready_o = 1'b1;
    forever begin
      #1;
      if (rvalid_i) begin
        data = rdata_i;
        resp = rresp_i;
        @(posedge clk_i);
        @(negedge clk_i);
        rready_o = 1'b0;
        break;
      end
      @(negedge clk_i);
    end
  endtask

endmodule

`default_nettype wire
