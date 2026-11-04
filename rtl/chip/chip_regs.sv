`default_nettype none

// Chip-level registers (APB, system clock, never gated). Window base
// 0x100000 relative to the APB address space; offsets:
//   0x00 CHIP_ID    RO  0x4E505543
//   0x04 CLK_CTRL   [0] CORE_CLK_EN (RW, reset 1): clock of the NPU core and the pixel writer
//   0x08 PIX_CTRL   [0] EN (RW), [1] IRQ_EN (RW)
//   0x0C PIX_BASE   RW  byte offset in the activation buffer
//   0x10 PIX_LEN    RW  bytes per frame
//   0x14 PIX_STATUS [0] FRAME_DONE (W1C), [1] SOF_ERR (W1C), [2] OVERRUN (W1C)
//   0x18 PIX_FRAMES RO  frames completed since reset
// Writes to CLK_CTRL take effect immediately; software must only gate the
// core while it is idle and must not touch the core window while gated
// (such accesses return PSLVERR).
module chip_regs
  import npu_pkg::*;
(
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               psel_i,
  input  logic               penable_i,
  input  logic               pwrite_i,
  input  logic [7:0]         paddr_i,
  input  logic [31:0]        pwdata_i,
  input  logic [3:0]         pstrb_i,
  output logic               pready_o,
  output logic [31:0]        prdata_o,
  output logic               pslverr_o,

  output logic               core_clk_en_o,
  output logic               pix_en_o,
  output logic [ABUF_AW-1:0] pix_base_o,
  output logic [ABUF_AW-1:0] pix_len_o,
  input  logic               frame_done_i,
  input  logic               sof_err_i,
  input  logic               overrun_i,
  output logic               irq_o
);

  localparam logic [31:0] CHIP_ID = 32'h4E50_5543;

  logic pix_irq_en_q, frame_done_q, sof_err_q, overrun_q;
  logic [31:0] frames_q;

  assign pready_o  = 1'b1;
  assign pslverr_o = 1'b0;
  assign irq_o     = frame_done_q & pix_irq_en_q;

  logic access_c, write_c;
  assign access_c = psel_i && penable_i;
  assign write_c  = access_c && pwrite_i && pstrb_i[0];

  always_comb begin
    case (paddr_i[7:2])
      6'h00:   prdata_o = CHIP_ID;
      6'h01:   prdata_o = {31'h0, core_clk_en_o};
      6'h02:   prdata_o = {30'h0, pix_irq_en_q, pix_en_o};
      6'h03:   prdata_o = {{(32-ABUF_AW){1'b0}}, pix_base_o};
      6'h04:   prdata_o = {{(32-ABUF_AW){1'b0}}, pix_len_o};
      6'h05:   prdata_o = {29'h0, overrun_q, sof_err_q, frame_done_q};
      6'h06:   prdata_o = frames_q;
      default: prdata_o = 32'h0;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      core_clk_en_o <= 1'b1;
      pix_en_o      <= 1'b0;
      pix_irq_en_q  <= 1'b0;
      pix_base_o    <= '0;
      pix_len_o     <= '0;
      frame_done_q  <= 1'b0;
      sof_err_q     <= 1'b0;
      overrun_q     <= 1'b0;
      frames_q      <= '0;
    end else begin
      if (frame_done_i) begin frame_done_q <= 1'b1; frames_q <= frames_q + 32'd1; end
      if (sof_err_i)    sof_err_q <= 1'b1;
      if (overrun_i)    overrun_q <= 1'b1;
      if (write_c) begin
        case (paddr_i[7:2])
          6'h01: core_clk_en_o <= pwdata_i[0];
          6'h02: begin pix_en_o <= pwdata_i[0]; pix_irq_en_q <= pwdata_i[1]; end
          6'h03: pix_base_o <= pwdata_i[ABUF_AW-1:0];
          6'h04: pix_len_o  <= pwdata_i[ABUF_AW-1:0];
          6'h05: begin
            if (pwdata_i[0] && !frame_done_i) frame_done_q <= 1'b0;
            if (pwdata_i[1] && !sof_err_i)    sof_err_q    <= 1'b0;
            if (pwdata_i[2] && !overrun_i)    overrun_q    <= 1'b0;
          end
          default: ;
        endcase
      end
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_ok;
  assign unused_ok = ^{paddr_i[1:0], pstrb_i[3:1], pwdata_i[31:ABUF_AW]};
  /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
