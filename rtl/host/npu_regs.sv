`default_nettype none

// Register file, descriptor table and host routing (docs/programming_model.md
// sections 1 to 3, docs/architecture.md section 7.3).
//
// Internal bus: wr_req_i / rd_req_i are single-cycle pulses from axil_slave.
// Register accesses are acknowledged one cycle later. Accesses to the
// act_buffer / weight_buffer windows are forwarded to the host memory port
// (hm_*) while the NPU is idle and acknowledged when hm_ack_i returns; while
// busy they are rejected with err and set STATUS.HOST_CONFLICT. Unmapped or
// misaligned addresses are acknowledged with err. Writes to read-only or
// undefined registers are ignored without error.
//
// CTRL.START (W1P) is ignored while busy. An accepted START pulses start_o
// and clears DONE, ERR, ERR_CODE and RESULT.VALID.
module npu_regs
  import npu_pkg::*;
(
  input  logic                          clk_i,
  input  logic                          rst_ni,

  // internal bus from axil_slave
  input  logic                          wr_req_i,
  input  logic [19:0]                   wr_addr_i,
  input  logic [31:0]                   wr_data_i,
  input  logic [3:0]                    wr_strb_i,
  output logic                          wr_ack_o,
  output logic                          wr_err_o,
  input  logic                          rd_req_i,
  input  logic [19:0]                   rd_addr_i,
  output logic                          rd_ack_o,
  output logic [31:0]                   rd_data_o,
  output logic                          rd_err_o,

  // host memory port: 32-bit accesses into act_buffer (hm_sel_o = 0) or
  // weight_buffer (hm_sel_o = 1); hm_addr_o is the byte offset inside the
  // selected buffer, 4-byte aligned. hm_ack_i returns at least one cycle later.
  output logic                          hm_req_o,
  output logic                          hm_we_o,
  output logic                          hm_sel_o,
  output logic [19:0]                   hm_addr_o,
  output logic [31:0]                   hm_wdata_o,
  output logic [3:0]                    hm_be_o,
  input  logic                          hm_ack_i,
  input  logic [31:0]                   hm_rdata_i,

  // sequencer
  output logic                          start_o,
  output logic [4:0]                    layer_cnt_o,
  output desc_t [NUM_DESC-1:0]          desc_o,
  input  logic                          busy_i,
  input  logic                          done_i,
  input  logic                          err_i,
  input  err_e                          err_code_i,
  input  logic [3:0]                    cur_layer_i,
  input  logic [NUM_DESC-1:0][31:0]     layer_cycles_i,
  input  logic [31:0]                   total_cycles_i,
  input  logic [31:0]                   array_busy_cycles_i,

  // argmax result
  input  logic                          argmax_valid_i,
  input  logic [3:0]                    argmax_idx_i,
  input  logic [NUM_LOGITS-1:0][7:0]    argmax_logits_i,

  output logic                          irq_o
);

  // ---------------------------------------------------------------------------
  // Address decode
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {TGT_REG, TGT_ABUF, TGT_WBUF, TGT_NONE} tgt_e;

  function automatic tgt_e decode(input logic [19:0] a);
    if (a[1:0] != 2'b00) return TGT_NONE;
    if (a < 20'h01000) return TGT_REG;
    if (a >= ABUF_BASE && a < ABUF_END) return TGT_ABUF;
    if (a >= WBUF_BASE && a < WBUF_END) return TGT_WBUF;
    return TGT_NONE;
  endfunction

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  logic        irq_en_q;
  logic        done_q, err_q, host_conflict_q;
  err_e        err_code_q;
  logic [4:0]  layer_cnt_q;
  logic        result_valid_q;
  logic [3:0]  result_idx_q;
  logic [NUM_LOGITS-1:0][7:0] logits_q;
  desc_words_t [NUM_DESC-1:0] desc_words_q;

  logic        hm_wait_q;   // a memory access is outstanding
  logic        hm_is_rd_q;

  assign layer_cnt_o = layer_cnt_q;
  assign irq_o       = done_q & irq_en_q;

  always_comb begin
    for (int i = 0; i < NUM_DESC; i++) desc_o[i] = desc_from_words(desc_words_q[i]);
  end

  // ---------------------------------------------------------------------------
  // Register read mux
  // ---------------------------------------------------------------------------
  function automatic logic [31:0] reg_read(input logic [11:0] a);
    logic [7:0]  woff;   // word offset inside a block
    logic [3:0]  idx;
    logic [3:0]  word;
    if (a >= REG_DESC_BASE && a < REG_DESC_BASE + 12'(NUM_DESC * DESC_WORDS * 4)) begin
      woff = 8'((a - REG_DESC_BASE) >> 2);
      idx  = woff[7:4];
      word = woff[3:0];
      return (word < 4'(DESC_USED_WORDS)) ? desc_words_q[idx][word] : 32'h0;
    end
    if (a >= REG_LOGITS_BASE && a < REG_LOGITS_BASE + 12'(NUM_LOGITS * 4)) begin
      idx = 4'((a - REG_LOGITS_BASE) >> 2);
      return {{24{logits_q[idx][7]}}, logits_q[idx]};
    end
    if (a >= REG_PERF_LAYER_BASE && a < REG_PERF_LAYER_BASE + 12'(NUM_DESC * 4)) begin
      idx = 4'((a - REG_PERF_LAYER_BASE) >> 2);
      return layer_cycles_i[idx];
    end
    case (a)
      REG_ID:              return NPU_ID;
      REG_CTRL:            return {30'h0, irq_en_q, 1'b0};
      REG_STATUS:          return {8'h0, err_code_q, 4'h0, cur_layer_i, 4'h0, host_conflict_q, err_q, done_q, busy_i};
      REG_LAYER_CNT:       return {27'h0, layer_cnt_q};
      REG_RESULT:          return {23'h0, result_valid_q, 4'h0, result_idx_q};
      REG_PERF_TOTAL:      return total_cycles_i;
      REG_PERF_ARRAY_BUSY: return array_busy_cycles_i;
      default:             return 32'h0;
    endcase
  endfunction

  // ---------------------------------------------------------------------------
  // Byte-strobed merge of a 32-bit write
  // ---------------------------------------------------------------------------
  function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] wdata, input logic [3:0] strb);
    logic [31:0] r;
    for (int b = 0; b < 4; b++) r[8*b +: 8] = strb[b] ? wdata[8*b +: 8] : old[8*b +: 8];
    return r;
  endfunction

  // ---------------------------------------------------------------------------
  // Main sequential logic
  // ---------------------------------------------------------------------------
  tgt_e wr_tgt_c, rd_tgt_c;
  assign wr_tgt_c = decode(wr_addr_i);
  assign rd_tgt_c = decode(rd_addr_i);

  logic        start_c;        // accepted START this cycle
  logic [11:0] wr_off_c;       // register offset
  logic [7:0]  desc_woff_c;    // word offset inside the descriptor block
  logic [3:0]  desc_idx_c, desc_word_c;

  assign wr_off_c    = wr_addr_i[11:0];
  assign desc_woff_c = 8'((wr_off_c - REG_DESC_BASE) >> 2);
  assign desc_idx_c  = desc_woff_c[7:4];
  assign desc_word_c = desc_woff_c[3:0];
  assign start_c     = wr_req_i && (wr_tgt_c == TGT_REG) && (wr_off_c == REG_CTRL) &&
                       wr_strb_i[0] && wr_data_i[0] && !busy_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ack_o        <= 1'b0;
      wr_err_o        <= 1'b0;
      rd_ack_o        <= 1'b0;
      rd_data_o       <= '0;
      rd_err_o        <= 1'b0;
      hm_req_o        <= 1'b0;
      hm_we_o         <= 1'b0;
      hm_sel_o        <= 1'b0;
      hm_addr_o       <= '0;
      hm_wdata_o      <= '0;
      hm_be_o         <= '0;
      hm_wait_q       <= 1'b0;
      hm_is_rd_q      <= 1'b0;
      start_o         <= 1'b0;
      irq_en_q        <= 1'b0;
      done_q          <= 1'b0;
      err_q           <= 1'b0;
      host_conflict_q <= 1'b0;
      err_code_q      <= ERR_NONE;
      layer_cnt_q     <= '0;
      result_valid_q  <= 1'b0;
      result_idx_q    <= '0;
      logits_q        <= '0;
      desc_words_q    <= '0;
    end else begin
      // defaults for the pulses
      wr_ack_o <= 1'b0;
      wr_err_o <= 1'b0;
      rd_ack_o <= 1'b0;
      rd_err_o <= 1'b0;
      hm_req_o <= 1'b0;
      start_o  <= 1'b0;

      // ---- sequencer and argmax status, highest priority ----
      if (done_i) done_q <= 1'b1;
      if (err_i) begin
        err_q      <= 1'b1;
        err_code_q <= err_code_i;
      end
      if (argmax_valid_i) begin
        result_valid_q <= 1'b1;
        result_idx_q   <= argmax_idx_i;
        logits_q       <= argmax_logits_i;
      end

      // ---- write ----
      if (wr_req_i) begin
        case (wr_tgt_c)
          TGT_REG: begin
            wr_ack_o <= 1'b1;
            if (wr_off_c == REG_CTRL) begin
              if (wr_strb_i[0]) begin
                irq_en_q <= wr_data_i[1];
                if (start_c) begin
                  start_o        <= 1'b1;
                  if (!done_i) done_q <= 1'b0;
                  if (!err_i) begin
                    err_q      <= 1'b0;
                    err_code_q <= ERR_NONE;
                  end
                  if (!argmax_valid_i) result_valid_q <= 1'b0;
                end
              end
            end else if (wr_off_c == REG_STATUS) begin
              if (wr_strb_i[0]) begin
                if (wr_data_i[1] && !done_i) done_q <= 1'b0;
                if (wr_data_i[2] && !err_i)  err_q  <= 1'b0;
                if (wr_data_i[3])            host_conflict_q <= 1'b0;
              end
            end else if (wr_off_c == REG_LAYER_CNT) begin
              if (wr_strb_i[0]) layer_cnt_q <= wr_data_i[4:0];
            end else if (wr_off_c >= REG_DESC_BASE && wr_off_c < REG_DESC_BASE + 12'(NUM_DESC * DESC_WORDS * 4)) begin
              if (desc_word_c < 4'(DESC_USED_WORDS)) begin
                desc_words_q[desc_idx_c][desc_word_c] <=
                    merge(desc_words_q[desc_idx_c][desc_word_c], wr_data_i, wr_strb_i) & desc_word_mask(32'(desc_word_c));
              end
            end
            // other offsets: read-only or undefined, write ignored
          end
          TGT_ABUF, TGT_WBUF: begin
            if (busy_i) begin
              wr_ack_o        <= 1'b1;
              wr_err_o        <= 1'b1;
              host_conflict_q <= 1'b1;
            end else begin
              hm_req_o   <= 1'b1;
              hm_we_o    <= 1'b1;
              hm_sel_o   <= (wr_tgt_c == TGT_WBUF);
              hm_addr_o  <= wr_addr_i - ((wr_tgt_c == TGT_WBUF) ? WBUF_BASE : ABUF_BASE);
              hm_wdata_o <= wr_data_i;
              hm_be_o    <= wr_strb_i;
              hm_wait_q  <= 1'b1;
              hm_is_rd_q <= 1'b0;
            end
          end
          default: begin
            wr_ack_o <= 1'b1;
            wr_err_o <= 1'b1;
          end
        endcase
      end

      // ---- read ----
      if (rd_req_i) begin
        case (rd_tgt_c)
          TGT_REG: begin
            rd_ack_o  <= 1'b1;
            rd_data_o <= reg_read(rd_addr_i[11:0]);
          end
          TGT_ABUF, TGT_WBUF: begin
            if (busy_i) begin
              rd_ack_o        <= 1'b1;
              rd_err_o        <= 1'b1;
              rd_data_o       <= '0;
              host_conflict_q <= 1'b1;
            end else begin
              hm_req_o   <= 1'b1;
              hm_we_o    <= 1'b0;
              hm_sel_o   <= (rd_tgt_c == TGT_WBUF);
              hm_addr_o  <= rd_addr_i - ((rd_tgt_c == TGT_WBUF) ? WBUF_BASE : ABUF_BASE);
              hm_be_o    <= 4'hF;
              hm_wait_q  <= 1'b1;
              hm_is_rd_q <= 1'b1;
            end
          end
          default: begin
            rd_ack_o  <= 1'b1;
            rd_err_o  <= 1'b1;
            rd_data_o <= '0;
          end
        endcase
      end

      // ---- memory completion ----
      if (hm_wait_q && hm_ack_i) begin
        hm_wait_q <= 1'b0;
        if (hm_is_rd_q) begin
          rd_ack_o  <= 1'b1;
          rd_data_o <= hm_rdata_i;
        end else begin
          wr_ack_o <= 1'b1;
        end
      end
    end
  end

endmodule

`default_nettype wire
