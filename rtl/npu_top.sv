`default_nettype none

// Top-level placeholder for the NPU.
// Add architectural interfaces only after they are documented in
// docs/architecture.md.
module npu_top (
    input logic clk_i,
    input logic rst_ni
);

  // Placeholder until M4 wires up the host interface and engines; the ports
  // are intentionally unused for now.
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_clk, unused_rst_n;
  assign unused_clk   = clk_i;
  assign unused_rst_n = rst_ni;
  /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
