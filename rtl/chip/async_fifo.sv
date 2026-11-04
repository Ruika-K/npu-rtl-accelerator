`default_nettype none

// Asynchronous FIFO with Gray-coded pointers (Cummings style).
// DEPTH is a power of two. Write side: wr_valid_i is accepted when wr_ready_o
// is high (full is registered, so it is conservative by the synchronizer
// latency). Read side: rd_valid_o shows that rd_data_o is valid; rd_ready_i
// pops it. Each side has its own asynchronous active-low reset; both resets
// must be asserted together before use.
module async_fifo #(
  parameter int unsigned WIDTH = 8,
  parameter int unsigned DEPTH = 16,
  localparam int unsigned AW   = $clog2(DEPTH)
) (
  input  logic             wr_clk_i,
  input  logic             wr_rst_ni,
  input  logic             wr_valid_i,
  output logic             wr_ready_o,
  input  logic [WIDTH-1:0] wr_data_i,

  input  logic             rd_clk_i,
  input  logic             rd_rst_ni,
  output logic             rd_valid_o,
  input  logic             rd_ready_i,
  output logic [WIDTH-1:0] rd_data_o
);

  // Storage, written in the write domain, read asynchronously by address.
  logic [WIDTH-1:0] mem [DEPTH];

  // Pointers carry one extra bit to tell full from empty.
  logic [AW:0] wr_bin_q, wr_gray_q, rd_bin_q, rd_gray_q;
  logic [AW:0] wr_gray_in_rd, rd_gray_in_wr;   // synchronized Gray pointers

  function automatic logic [AW:0] bin2gray(input logic [AW:0] b);
    return b ^ (b >> 1);
  endfunction

  // ---------------------------------------------------------------------------
  // Write side
  // ---------------------------------------------------------------------------
  logic wr_fire_c;
  logic [AW:0] wr_bin_next_c;
  logic full_q;
  assign wr_fire_c    = wr_valid_i && wr_ready_o;
  assign wr_bin_next_c = wr_bin_q + (AW+1)'(wr_fire_c);
  assign wr_ready_o   = !full_q;

  always_ff @(posedge wr_clk_i or negedge wr_rst_ni) begin
    if (!wr_rst_ni) begin
      wr_bin_q  <= '0;
      wr_gray_q <= '0;
      full_q    <= 1'b0;
    end else begin
      wr_bin_q  <= wr_bin_next_c;
      wr_gray_q <= bin2gray(wr_bin_next_c);
      // full when the next write pointer equals the read pointer with the MSBs inverted
      full_q    <= (bin2gray(wr_bin_next_c) == {~rd_gray_in_wr[AW:AW-1], rd_gray_in_wr[AW-2:0]});
    end
  end

  always_ff @(posedge wr_clk_i) begin
    if (wr_fire_c) mem[wr_bin_q[AW-1:0]] <= wr_data_i;
  end

  cdc_sync #(.WIDTH(AW+1)) u_sync_rd2wr (
      .clk_i (wr_clk_i), .rst_ni (wr_rst_ni), .d_i (rd_gray_q), .q_o (rd_gray_in_wr)
  );

  // ---------------------------------------------------------------------------
  // Read side
  // ---------------------------------------------------------------------------
  logic rd_fire_c;
  logic [AW:0] rd_bin_next_c;
  logic empty_q;
  assign rd_fire_c     = rd_valid_o && rd_ready_i;
  assign rd_bin_next_c = rd_bin_q + (AW+1)'(rd_fire_c);
  assign rd_valid_o    = !empty_q;
  assign rd_data_o     = mem[rd_bin_q[AW-1:0]];

  always_ff @(posedge rd_clk_i or negedge rd_rst_ni) begin
    if (!rd_rst_ni) begin
      rd_bin_q  <= '0;
      rd_gray_q <= '0;
      empty_q   <= 1'b1;
    end else begin
      rd_bin_q  <= rd_bin_next_c;
      rd_gray_q <= bin2gray(rd_bin_next_c);
      empty_q   <= (bin2gray(rd_bin_next_c) == wr_gray_in_rd);
    end
  end

  cdc_sync #(.WIDTH(AW+1)) u_sync_wr2rd (
      .clk_i (rd_clk_i), .rst_ni (rd_rst_ni), .d_i (wr_gray_q), .q_o (wr_gray_in_rd)
  );

endmodule

`default_nettype wire
