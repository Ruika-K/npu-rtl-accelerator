`default_nettype none

// Integrated clock gating cell, behavioural: the enable is captured in a
// latch while the clock is low, so the gated clock has no glitches. A
// technology library provides the real cell; this module is the one to swap.
module clk_gate (
  input  logic clk_i,
  input  logic en_i,
  output logic clk_o
);

  logic en_latch;

  /* verilator lint_off LATCH */
  always_latch begin
    if (!clk_i) en_latch = en_i;
  end
  /* verilator lint_on LATCH */

  assign clk_o = clk_i & en_latch;

endmodule

`default_nettype wire
