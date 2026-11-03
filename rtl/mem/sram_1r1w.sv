`default_nettype none

// Behavioural 1-read 1-write SRAM with byte enables and a 1-cycle read latency.
// This is the only module to replace with a technology macro. A read and a
// write to the same address in the same cycle return the old data (read before
// write). No reset: the array powers up undefined, like a real macro.
module sram_1r1w #(
  parameter int unsigned WIDTH = 64,
  parameter int unsigned DEPTH = 1024,
  localparam int unsigned AW   = $clog2(DEPTH),
  localparam int unsigned BE_W = WIDTH / 8
) (
  input  logic             clk_i,

  input  logic             rd_en_i,
  input  logic [AW-1:0]    rd_addr_i,
  output logic [WIDTH-1:0] rd_data_o,

  input  logic             wr_en_i,
  input  logic [AW-1:0]    wr_addr_i,
  input  logic [WIDTH-1:0] wr_data_i,
  input  logic [BE_W-1:0]  wr_be_i
);

  logic [WIDTH-1:0] mem [DEPTH];

  always_ff @(posedge clk_i) begin
    if (rd_en_i) rd_data_o <= mem[rd_addr_i];
    if (wr_en_i) begin
      for (int b = 0; b < BE_W; b++) begin
        if (wr_be_i[b]) mem[wr_addr_i][8*b +: 8] <= wr_data_i[8*b +: 8];
      end
    end
  end

endmodule

`default_nettype wire
