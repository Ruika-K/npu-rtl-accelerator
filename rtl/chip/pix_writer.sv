`default_nettype none

// Pixel stream writer (system clock domain). Pops {sof, byte} words from the
// pixel FIFO, packs up to N bytes and writes them to the activation buffer at
// base + frame offset. A frame is len bytes starting at the byte flagged sof.
// The writer only pops while it has room, so the FIFO back-pressures the
// source whenever the memory port is not granted (wr_ready_i low).
//   frame_done_o : one-cycle pulse after the last byte of a frame is written
//   sof_err_o    : one-cycle pulse when a byte arrives before the first sof
//                  (the byte is dropped) or a sof arrives inside a frame (the
//                  frame restarts from that byte)
module pix_writer
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                en_i,
  input  logic [ABUF_AW-1:0]  base_i,
  input  logic [ABUF_AW-1:0]  len_i,

  input  logic                fifo_valid_i,
  output logic                fifo_ready_o,
  input  logic [8:0]          fifo_data_i,     // {sof, byte}

  output logic                wr_en_o,
  output logic [ABUF_AW-1:0]  wr_addr_o,
  output logic [N-1:0][7:0]   wr_data_o,
  output logic [N-1:0]        wr_be_o,
  input  logic                wr_ready_i,

  output logic                frame_done_o,
  output logic                sof_err_o
);

  localparam int unsigned CNT_W = $clog2(N) + 1;

  typedef enum logic [1:0] {S_WAIT_SOF, S_COLLECT, S_FLUSH} state_e;
  state_e state_q;

  logic [N-1:0][7:0]  pack_q;
  logic [CNT_W-1:0]   fill_q;       // bytes in the pack, 0 .. N
  logic [ABUF_AW-1:0] pack_off_q;   // frame offset of the pack's first byte
  logic [ABUF_AW-1:0] count_q;      // bytes of the frame accepted so far

  logic byte_c, sof_c;
  assign sof_c  = fifo_data_i[8];
  assign byte_c = fifo_valid_i && fifo_ready_o;

  assign fifo_ready_o = en_i && (state_q != S_FLUSH);

  assign wr_en_o   = (state_q == S_FLUSH) && wr_ready_i;
  assign wr_addr_o = base_i + pack_off_q;
  assign wr_data_o = pack_q;
  always_comb begin
    for (int i = 0; i < N; i++) wr_be_o[i] = (CNT_W'(i) < fill_q);
  end

  logic frame_end_c, pack_full_c;
  assign frame_end_c = (count_q + ABUF_AW'(1) >= len_i);    // this byte completes the frame
  assign pack_full_c = (fill_q + CNT_W'(1) == CNT_W'(N));   // this byte fills the pack

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q      <= S_WAIT_SOF;
      pack_q       <= '0;
      fill_q       <= '0;
      pack_off_q   <= '0;
      count_q      <= '0;
      frame_done_o <= 1'b0;
      sof_err_o    <= 1'b0;
    end else begin
      frame_done_o <= 1'b0;
      sof_err_o    <= 1'b0;
      case (state_q)
        S_WAIT_SOF, S_COLLECT: begin
          if (byte_c) begin
            if (state_q == S_WAIT_SOF && !sof_c) begin
              sof_err_o <= 1'b1;                       // dropped
            end else begin
              if (sof_c && state_q == S_COLLECT) sof_err_o <= 1'b1;   // restart
              if (sof_c) begin
                pack_q     <= '0;
                pack_q[0]  <= fifo_data_i[7:0];
                fill_q     <= CNT_W'(1);
                pack_off_q <= '0;
                count_q    <= ABUF_AW'(1);
                state_q    <= (len_i <= ABUF_AW'(1)) ? S_FLUSH : ((CNT_W'(1) == CNT_W'(N)) ? S_FLUSH : S_COLLECT);
              end else begin
                pack_q[fill_q[CNT_W-2:0]] <= fifo_data_i[7:0];
                fill_q  <= fill_q + CNT_W'(1);
                count_q <= count_q + ABUF_AW'(1);
                if (frame_end_c || pack_full_c) state_q <= S_FLUSH;
              end
            end
          end
        end
        S_FLUSH: begin
          if (wr_ready_i) begin
            pack_off_q <= pack_off_q + ABUF_AW'(fill_q);
            fill_q     <= '0;
            pack_q     <= '0;
            if (count_q >= len_i) begin
              frame_done_o <= 1'b1;
              state_q      <= S_WAIT_SOF;
            end else begin
              state_q <= S_COLLECT;
            end
          end
        end
        default: state_q <= S_WAIT_SOF;
      endcase
    end
  end

endmodule

`default_nettype wire
