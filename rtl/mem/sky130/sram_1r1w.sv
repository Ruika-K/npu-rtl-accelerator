`default_nettype none

// sram_1r1w for SKY130: the same ports and behaviour as rtl/mem/sram_1r1w.sv,
// built from OpenRAM macros (github.com/VLSIDA/sky130_sram_macros). Selected
// by the ASIC filelists instead of the behavioural model.
//
//   WIDTH 8                          sky130_sram_1kbyte_1rw1r_8x1024_8  (8 bits x 1024)
//   WIDTH multiple of 32, DEPTH 256  sky130_sram_1kbyte_1rw1r_32x256_8  (32 bits x 256, byte mask)
//   WIDTH multiple of 32, otherwise  sky130_sram_2kbyte_1rw1r_32x512_8  (32 bits x 512, byte mask)
//
// Port 0 (read/write) of every macro is the write port, port 1 the read port.
// A macro's output is only valid in the cycle after the read (it goes unknown
// after the next clock edge), while the behavioural model holds the last read
// data until the next read; hold_q keeps that behaviour without adding a cycle.
// A read and a write of the same word in one cycle are undefined in the
// macros; tb/sva/sram_1r1w_sva.sv checks the design never does it.
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

  localparam bit          NARROW = (WIDTH == 8);
  localparam bit          SHORT  = !NARROW && (DEPTH == 256);
  localparam int unsigned MW     = NARROW ? 8 : 32;                       // macro word
  localparam int unsigned MD     = NARROW ? 1024 : (SHORT ? 256 : 512);  // macro depth
  localparam int unsigned MAW    = $clog2(MD);
  localparam int unsigned COLS   = WIDTH / MW;
  localparam int unsigned ROWS   = DEPTH / MD;
  localparam int unsigned RSW    = (ROWS > 1) ? $clog2(ROWS) : 1;

  if (WIDTH % MW != 0 || DEPTH % MD != 0 || (1 << AW) != DEPTH) begin : g_bad_geometry
    $error("sram_1r1w (sky130): WIDTH %0d x DEPTH %0d does not tile onto %0d x %0d macros", WIDTH, DEPTH, MW, MD);
    initial $fatal(1, "sram_1r1w (sky130): unsupported geometry");   // for tools that skip elaboration tasks
  end

  logic [RSW-1:0] rd_row_c, wr_row_c;
  if (ROWS > 1) begin : g_rows
    assign rd_row_c = rd_addr_i[AW-1:MAW];
    assign wr_row_c = wr_addr_i[AW-1:MAW];
  end else begin : g_one_row
    assign rd_row_c = '0;
    assign wr_row_c = '0;
  end

  logic [ROWS-1:0][WIDTH-1:0] dout;

  for (genvar r = 0; r < ROWS; r++) begin : g_row
    logic wr_csb, rd_csb;
    assign wr_csb = !(wr_en_i && (wr_row_c == RSW'(r)));
    assign rd_csb = !(rd_en_i && (rd_row_c == RSW'(r)));
    for (genvar c = 0; c < COLS; c++) begin : g_col
      if (NARROW) begin : g_8x1024
        sky130_sram_1kbyte_1rw1r_8x1024_8 #(.VERBOSE(0)) u_macro (
            .clk0   (clk_i),
            .csb0   (wr_csb),
            .web0   (1'b0),
            .wmask0 (wr_be_i[c]),
            .addr0  (wr_addr_i[MAW-1:0]),
            .din0   (wr_data_i[c*MW +: MW]),
            .dout0  (),
            .clk1   (clk_i),
            .csb1   (rd_csb),
            .addr1  (rd_addr_i[MAW-1:0]),
            .dout1  (dout[r][c*MW +: MW])
        );
      end else if (SHORT) begin : g_32x256
        sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) u_macro (
            .clk0   (clk_i),
            .csb0   (wr_csb),
            .web0   (1'b0),
            .wmask0 (wr_be_i[c*4 +: 4]),
            .addr0  (wr_addr_i[MAW-1:0]),
            .din0   (wr_data_i[c*MW +: MW]),
            .dout0  (),
            .clk1   (clk_i),
            .csb1   (rd_csb),
            .addr1  (rd_addr_i[MAW-1:0]),
            .dout1  (dout[r][c*MW +: MW])
        );
      end else begin : g_32x512
        sky130_sram_2kbyte_1rw1r_32x512_8 #(.VERBOSE(0)) u_macro (
            .clk0   (clk_i),
            .csb0   (wr_csb),
            .web0   (1'b0),
            .wmask0 (wr_be_i[c*4 +: 4]),
            .addr0  (wr_addr_i[MAW-1:0]),
            .din0   (wr_data_i[c*MW +: MW]),
            .dout0  (),
            .clk1   (clk_i),
            .csb1   (rd_csb),
            .addr1  (rd_addr_i[MAW-1:0]),
            .dout1  (dout[r][c*MW +: MW])
        );
      end
    end
  end

  // Read data: the macro outputs in the cycle after a read, the held value otherwise.
  logic             rd_pending_q;
  logic [RSW-1:0]   rd_row_q;
  logic [WIDTH-1:0] hold_q;

  always_ff @(posedge clk_i) begin
    rd_pending_q <= rd_en_i;
    if (rd_en_i)      rd_row_q <= rd_row_c;
    if (rd_pending_q) hold_q   <= dout[rd_row_q];
  end

  assign rd_data_o = rd_pending_q ? dout[rd_row_q] : hold_q;

endmodule

`default_nettype wire
