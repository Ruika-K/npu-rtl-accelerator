`timescale 1ns/1ps
`default_nettype none

// APB master bus-functional model: one transfer per call, driven on the
// falling edge; PREADY is sampled 1 ns after the falling edge of each access
// cycle, which is what the DUT presents to the next rising edge.
module apb_master_bfm (
  input  logic        clk_i,
  output logic        psel_o,
  output logic        penable_o,
  output logic        pwrite_o,
  output logic [31:0] paddr_o,
  output logic [31:0] pwdata_o,
  output logic [3:0]  pstrb_o,
  input  logic        pready_i,
  input  logic [31:0] prdata_i,
  input  logic        pslverr_i
);

  int max_idle = 0;   // random idle cycles between transfers, 0 .. max_idle

  initial begin
    psel_o = 1'b0; penable_o = 1'b0; pwrite_o = 1'b0; paddr_o = '0; pwdata_o = '0; pstrb_o = '0;
  end

  task automatic transfer(input bit write, input logic [31:0] addr, input logic [31:0] wdata, input logic [3:0] strb,
                          output logic [31:0] rdata, output logic slverr);
    repeat ($urandom_range(max_idle)) @(negedge clk_i);
    @(negedge clk_i);
    psel_o = 1'b1; penable_o = 1'b0; pwrite_o = write; paddr_o = addr; pwdata_o = wdata; pstrb_o = write ? strb : 4'h0;
    @(negedge clk_i);
    penable_o = 1'b1;
    forever begin
      #1;
      if (pready_i) begin
        rdata  = prdata_i;
        slverr = pslverr_i;
        @(posedge clk_i);
        @(negedge clk_i);
        psel_o = 1'b0; penable_o = 1'b0;
        break;
      end
      @(negedge clk_i);
    end
  endtask

  task automatic write(input logic [31:0] addr, input logic [31:0] data, input logic [3:0] strb, output logic slverr);
    logic [31:0] unused;
    transfer(1'b1, addr, data, strb, unused, slverr);
  endtask

  task automatic read(input logic [31:0] addr, output logic [31:0] data, output logic slverr);
    transfer(1'b0, addr, '0, 4'h0, data, slverr);
  endtask

endmodule

`default_nettype wire
