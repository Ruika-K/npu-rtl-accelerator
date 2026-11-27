`default_nettype none

// Assertions for sram_1r1w, bound into every bank of both buffers. The
// behavioural model returns the old data when a read and a write hit the same
// word in one cycle; the SKY130 SRAM macros do not define that case, so the
// design must never rely on it.
module sram_1r1w_sva #(
  parameter int unsigned WIDTH = 64,
  parameter int unsigned DEPTH = 1024,
  localparam int unsigned AW   = $clog2(DEPTH),
  localparam int unsigned BE_W = WIDTH / 8
) (
  input logic            clk_i,
  input logic            rd_en_i,
  input logic [AW-1:0]   rd_addr_i,
  input logic            wr_en_i,
  input logic [AW-1:0]   wr_addr_i,
  input logic [BE_W-1:0] wr_be_i
);

  a_no_rw_same_word: assert property (@(posedge clk_i)
      !(rd_en_i && wr_en_i && |wr_be_i && rd_addr_i == wr_addr_i))
    else $error("read and write of word %0d in one cycle (undefined in the SRAM macros)", rd_addr_i);

  c_rw_other_word: cover property (@(posedge clk_i) rd_en_i && wr_en_i && |wr_be_i && rd_addr_i != wr_addr_i);

endmodule

bind sram_1r1w sram_1r1w_sva #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_sva (.*);

`default_nettype wire
