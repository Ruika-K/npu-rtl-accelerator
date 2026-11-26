`timescale 1ns/1ps
`default_nettype none

// Interrupt output of npu_top, for sequences that wait on it.
interface irq_if (input wire logic clk, input wire logic rst_n);
  logic irq;
endinterface

`default_nettype wire
