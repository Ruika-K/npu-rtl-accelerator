`default_nettype none

// Multi-flop synchronizer for a single bit or a Gray-coded vector crossing
// into clk_i's domain. No logic between the stages, so a synthesis tool can
// apply the usual CDC constraints to the first flop.
module cdc_sync #(
  parameter int unsigned WIDTH  = 1,
  parameter int unsigned STAGES = 2
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic [WIDTH-1:0] d_i,
  output logic [WIDTH-1:0] q_o
);

  logic [STAGES-1:0][WIDTH-1:0] sync_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sync_q <= '0;
    end else begin
      sync_q[0] <= d_i;
      for (int s = 1; s < STAGES; s++) sync_q[s] <= sync_q[s-1];
    end
  end

  assign q_o = sync_q[STAGES-1];

endmodule

`default_nettype wire
