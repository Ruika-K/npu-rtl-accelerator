`default_nettype none

// Weight buffer: 64-bit words, one read port with a fixed 1-cycle latency
// (WBUF_RD_LAT) and one write port with byte enables
// (docs/architecture.md section 4.3).
module weight_buffer
  import npu_pkg::*;
#(
  parameter int unsigned BYTES = WBUF_BYTES,
  localparam int unsigned WORDS = BYTES / 8,
  localparam int unsigned AW    = $clog2(WORDS)
) (
  input  logic          clk_i,
  input  logic          rst_ni,

  input  logic          rd_en_i,
  input  logic [AW-1:0] rd_addr_i,
  output logic [63:0]   rd_data_o,
  output logic          rd_valid_o,

  input  logic          wr_en_i,
  input  logic [AW-1:0] wr_addr_i,
  input  logic [63:0]   wr_data_i,
  input  logic [7:0]    wr_be_i
);

  sram_1r1w #(.WIDTH(64), .DEPTH(WORDS)) u_sram (
      .clk_i     (clk_i),
      .rd_en_i   (rd_en_i),
      .rd_addr_i (rd_addr_i),
      .rd_data_o (rd_data_o),
      .wr_en_i   (wr_en_i),
      .wr_addr_i (wr_addr_i),
      .wr_data_i (wr_data_i),
      .wr_be_i   (wr_be_i)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) rd_valid_o <= 1'b0;
    else         rd_valid_o <= rd_en_i;
  end

endmodule

`default_nettype wire
