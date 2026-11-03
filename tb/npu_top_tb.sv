`timescale 1ns/1ps
`default_nettype none

module npu_top_tb;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i;
  logic rst_ni;

  npu_top dut (
      .clk_i  (clk_i),
      .rst_ni (rst_ni)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  initial begin
    rst_ni = 1'b0;
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    repeat (4) @(posedge clk_i);
    $display("SMOKE PASS: reset sequence completed");
    $finish;
  end

  initial begin
    #1us;
    $fatal(1, "TIMEOUT: simulation did not finish");
  end

endmodule

`default_nettype wire
