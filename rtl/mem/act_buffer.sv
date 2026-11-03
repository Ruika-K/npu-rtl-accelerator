`default_nettype none

// Activation buffer: N byte-wide banks, bank = byte address mod N
// (docs/architecture.md section 4.2).
//
// Read:  N consecutive bytes starting at any byte address, fixed latency
//        ABUF_RD_LAT = 2 (one SRAM cycle plus one rotation register).
//        Lane i of rd_data_o is the byte at rd_addr_i + i.
// Write: N consecutive bytes starting at any byte address, byte enable per
//        lane, single cycle.
// Bank b serves lane (b - A[LANE_W-1:0]) mod N at row A >> LANE_W, plus one
// when the lane wrapped past the end of the row (b < A mod N).
// Addresses past the end of the buffer wrap; the callers never rely on that.
module act_buffer
  import npu_pkg::*;
#(
  parameter int unsigned BYTES = ABUF_BYTES,
  parameter int unsigned N     = ARR_N,
  localparam int unsigned AW     = $clog2(BYTES),
  localparam int unsigned LANE_W = $clog2(N),
  localparam int unsigned ROW_W  = AW - LANE_W,
  localparam int unsigned ROWS   = BYTES / N
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               rd_en_i,
  input  logic [AW-1:0]      rd_addr_i,
  output logic [N-1:0][7:0]  rd_data_o,
  output logic               rd_valid_o,

  input  logic               wr_en_i,
  input  logic [AW-1:0]      wr_addr_i,
  input  logic [N-1:0][7:0]  wr_data_i,
  input  logic [N-1:0]       wr_be_i
);

  // ---------------------------------------------------------------------------
  // Per-bank addressing
  // ---------------------------------------------------------------------------
  logic [LANE_W-1:0] rd_rot_c, wr_rot_c;
  logic [ROW_W-1:0]  rd_row_c, wr_row_c;
  assign rd_rot_c = rd_addr_i[LANE_W-1:0];
  assign rd_row_c = rd_addr_i[AW-1:LANE_W];
  assign wr_rot_c = wr_addr_i[LANE_W-1:0];
  assign wr_row_c = wr_addr_i[AW-1:LANE_W];

  logic [N-1:0][ROW_W-1:0] bank_rd_row_c, bank_wr_row_c;
  logic [N-1:0][7:0]       bank_wr_data_c, bank_rd_data;
  logic [N-1:0]            bank_wr_be_c;
  logic [N-1:0][LANE_W-1:0] wr_lane_c;

  always_comb begin
    for (int b = 0; b < N; b++) begin
      bank_rd_row_c[b]  = rd_row_c + ((LANE_W'(b) < rd_rot_c) ? ROW_W'(1) : ROW_W'(0));
      bank_wr_row_c[b]  = wr_row_c + ((LANE_W'(b) < wr_rot_c) ? ROW_W'(1) : ROW_W'(0));
      wr_lane_c[b]      = LANE_W'(b) - wr_rot_c;           // mod N by truncation
      bank_wr_data_c[b] = wr_data_i[wr_lane_c[b]];
      bank_wr_be_c[b]   = wr_be_i[wr_lane_c[b]];
    end
  end

  for (genvar b = 0; b < N; b++) begin : g_bank
    sram_1r1w #(.WIDTH(8), .DEPTH(ROWS)) u_sram (
        .clk_i     (clk_i),
        .rd_en_i   (rd_en_i),
        .rd_addr_i (bank_rd_row_c[b]),
        .rd_data_o (bank_rd_data[b]),
        .wr_en_i   (wr_en_i),
        .wr_addr_i (bank_wr_row_c[b]),
        .wr_data_i (bank_wr_data_c[b]),
        .wr_be_i   (bank_wr_be_c[b])
    );
  end

  // ---------------------------------------------------------------------------
  // Read pipeline: rotation after the SRAM cycle
  // ---------------------------------------------------------------------------
  logic              rd_valid_s1_q;
  logic [LANE_W-1:0] rd_rot_s1_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_valid_s1_q <= 1'b0;
      rd_rot_s1_q   <= '0;
      rd_valid_o    <= 1'b0;
      rd_data_o     <= '0;
    end else begin
      rd_valid_s1_q <= rd_en_i;
      rd_rot_s1_q   <= rd_rot_c;
      rd_valid_o    <= rd_valid_s1_q;
      if (rd_valid_s1_q) begin
        for (int i = 0; i < N; i++) begin
          rd_data_o[i] <= bank_rd_data[LANE_W'(i) + rd_rot_s1_q];   // mod N by truncation
        end
      end
    end
  end

endmodule

`default_nettype wire
