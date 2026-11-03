`default_nettype none

// Reset synchronizer: asynchronous assertion, release synchronized to clk_i
// through STAGES flops, so every flop in the domain leaves reset on a clock edge.
module rst_sync #(
  parameter int unsigned STAGES = 2
) (
  input  logic clk_i,
  input  logic rst_ni,      // asynchronous, active low
  output logic rst_sync_no  // asynchronous assert, synchronous release
);

  logic [STAGES-1:0] chain_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) chain_q <= '0;
    else         chain_q <= {chain_q[STAGES-2:0], 1'b1};
  end

  assign rst_sync_no = chain_q[STAGES-1];

endmodule

`default_nettype wire
