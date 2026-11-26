`default_nettype none

// Assertions for async_fifo, bound into every instance. The occupancy checks
// read the other domain's binary pointer directly, which only a simulation can
// do: whichever of its old or new value is seen, the bound must hold, because
// the flags are computed from an older (synchronized) copy.
module async_fifo_sva #(
  parameter int unsigned WIDTH = 8,
  parameter int unsigned DEPTH = 16,
  localparam int unsigned AW   = $clog2(DEPTH)
) (
  input logic             wr_clk_i,
  input logic             wr_rst_ni,
  input logic             wr_valid_i,
  input logic             wr_ready_o,
  input logic             rd_clk_i,
  input logic             rd_rst_ni,
  input logic             rd_valid_o,
  input logic             rd_ready_i,
  input logic [WIDTH-1:0] rd_data_o,
  input logic [AW:0]      wr_bin_q,
  input logic [AW:0]      wr_gray_q,
  input logic [AW:0]      rd_bin_q,
  input logic [AW:0]      rd_gray_q
);

  logic [AW:0] level_c;
  assign level_c = wr_bin_q - rd_bin_q;

  // Gray pointers: the code of the binary pointer, one bit per step
  logic [AW:0] wr_gray_prev_q, rd_gray_prev_q;
  always_ff @(posedge wr_clk_i or negedge wr_rst_ni) begin
    if (!wr_rst_ni) wr_gray_prev_q <= '0;
    else            wr_gray_prev_q <= wr_gray_q;
  end
  always_ff @(posedge rd_clk_i or negedge rd_rst_ni) begin
    if (!rd_rst_ni) rd_gray_prev_q <= '0;
    else            rd_gray_prev_q <= rd_gray_q;
  end

  a_wr_gray_code: assert property (@(posedge wr_clk_i) disable iff (!wr_rst_ni)
      wr_gray_q == (wr_bin_q ^ (wr_bin_q >> 1)))
    else $error("write Gray pointer is not the code of the binary pointer");
  a_wr_gray_step: assert property (@(posedge wr_clk_i) disable iff (!wr_rst_ni)
      $countones(wr_gray_q ^ wr_gray_prev_q) <= 1)
    else $error("write Gray pointer changed more than one bit");
  a_rd_gray_code: assert property (@(posedge rd_clk_i) disable iff (!rd_rst_ni)
      rd_gray_q == (rd_bin_q ^ (rd_bin_q >> 1)))
    else $error("read Gray pointer is not the code of the binary pointer");
  a_rd_gray_step: assert property (@(posedge rd_clk_i) disable iff (!rd_rst_ni)
      $countones(rd_gray_q ^ rd_gray_prev_q) <= 1)
    else $error("read Gray pointer changed more than one bit");

  // No overflow, no underflow
  a_no_overflow: assert property (@(posedge wr_clk_i) disable iff (!wr_rst_ni || !rd_rst_ni)
      level_c <= (AW+1)'(DEPTH))
    else $error("more than DEPTH words stored");
  a_no_underflow: assert property (@(posedge rd_clk_i) disable iff (!wr_rst_ni || !rd_rst_ni)
      rd_valid_o |-> level_c != '0 && level_c <= (AW+1)'(DEPTH))
    else $error("rd_valid_o with nothing stored");

  // The read side holds its word until it is taken
  a_rd_hold: assert property (@(posedge rd_clk_i) disable iff (!rd_rst_ni)
      rd_valid_o && !rd_ready_i |=> rd_valid_o && $stable(rd_data_o))
    else $error("read word dropped or changed before rd_ready_i");

  c_full:        cover property (@(posedge wr_clk_i) disable iff (!wr_rst_ni) wr_valid_i && !wr_ready_o);
  c_full_level:  cover property (@(posedge wr_clk_i) disable iff (!wr_rst_ni || !rd_rst_ni) level_c == (AW+1)'(DEPTH));
  c_empty_wait:  cover property (@(posedge rd_clk_i) disable iff (!rd_rst_ni) rd_ready_i && !rd_valid_o);
  c_wr_wrap:     cover property (@(posedge wr_clk_i) disable iff (!wr_rst_ni) wr_gray_prev_q[AW] && !wr_gray_q[AW]);
  c_rd_wrap:     cover property (@(posedge rd_clk_i) disable iff (!rd_rst_ni) rd_gray_prev_q[AW] && !rd_gray_q[AW]);

endmodule

bind async_fifo async_fifo_sva #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_sva (.*);

`default_nettype wire
