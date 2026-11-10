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
//
// The descriptor table is a 1R1W RAM. The sequencer reads descriptor words
// through desc_rd_*: a request is granted in the cycle it is made unless the
// host reads a descriptor word in the same cycle (the host has priority), and
// the word follows one cycle after the grant. Host reads of descriptor words
// also take the RAM's one cycle, so every register read is still acknowledged
// one cycle after the request.
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
  input  logic                          desc_rd_en_i,
  input  logic [3:0]                    desc_rd_idx_i,
  input  logic [3:0]                    desc_rd_word_i,
  output logic                          desc_rd_gnt_o,
  output logic                          desc_rd_valid_o,
  output logic [31:0]                   desc_rd_data_o,
  input  logic                          busy_i,
  input  logic                          done_i,
  input  logic                          err_i,
  input  err_e                          err_code_i,
  input  logic [3:0]                    cur_layer_i,
  input  logic [NUM_DESC-1:0][31:0]     layer_cycles_i,
  input  logic [31:0]                   total_cycles_i,
  input  logic [31:0]                   array_busy_cycles_i,

  // DMA
  output logic                          dma_start_o,
  output logic                          dma_dir_o,
  output logic                          dma_dst_sel_o,
  output logic [31:0]                   dma_src_o,
  output logic [19:0]                   dma_dst_o,
  output logic [19:0]                   dma_len_o,
  input  logic                          dma_busy_i,
  input  logic                          dma_done_i,
  input  logic                          dma_err_i,
  input  err_e                          dma_err_code_i,

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

  // Windows are decoded with equality tests only. A window [base, base + bytes)
  // is a run of aligned granules, the granule being the lowest set bit of
  // base | bytes, and each granule is one compare of the address bits above it.
  // This is exact for any base and size; a magnitude compare against the bounds
  // built a carry chain, which sat on the host read path into the descriptor
  // RAM on ECP5. With the default map every window is one or a few granules.
  localparam int unsigned REGS_BYTES   = 32'h1000;
  localparam int unsigned DESC_BYTES   = NUM_DESC * DESC_WORDS * 4;
  localparam int unsigned LOGITS_BYTES = NUM_LOGITS * 4;
  localparam int unsigned PERF_BYTES   = NUM_DESC * 4;

  localparam int unsigned ABUF_X    = 32'(ABUF_BASE) | ABUF_BYTES;
  localparam int unsigned ABUF_GW   = $clog2(ABUF_X & ~(ABUF_X - 1));
  localparam int unsigned ABUF_NG   = ABUF_BYTES >> ABUF_GW;
  localparam int unsigned WBUF_X    = 32'(WBUF_BASE) | WBUF_BYTES;
  localparam int unsigned WBUF_GW   = $clog2(WBUF_X & ~(WBUF_X - 1));
  localparam int unsigned WBUF_NG   = WBUF_BYTES >> WBUF_GW;
  localparam int unsigned DESC_X    = 32'(REG_DESC_BASE) | DESC_BYTES;
  localparam int unsigned DESC_GW   = $clog2(DESC_X & ~(DESC_X - 1));
  localparam int unsigned DESC_NG   = DESC_BYTES >> DESC_GW;
  localparam int unsigned LOGITS_X  = 32'(REG_LOGITS_BASE) | LOGITS_BYTES;
  localparam int unsigned LOGITS_GW = $clog2(LOGITS_X & ~(LOGITS_X - 1));
  localparam int unsigned LOGITS_NG = LOGITS_BYTES >> LOGITS_GW;
  localparam int unsigned PERF_X    = 32'(REG_PERF_LAYER_BASE) | PERF_BYTES;
  localparam int unsigned PERF_GW   = $clog2(PERF_X & ~(PERF_X - 1));
  localparam int unsigned PERF_NG   = PERF_BYTES >> PERF_GW;

  function automatic logic in_abuf(input logic [19:0] a);
    logic hit;
    hit = 1'b0;
    for (int k = 0; k < ABUF_NG; k++) hit |= ((a >> ABUF_GW) == 20'((ABUF_BASE >> ABUF_GW) + 20'(k)));
    return hit;
  endfunction

  function automatic logic in_wbuf(input logic [19:0] a);
    logic hit;
    hit = 1'b0;
    for (int k = 0; k < WBUF_NG; k++) hit |= ((a >> WBUF_GW) == 20'((WBUF_BASE >> WBUF_GW) + 20'(k)));
    return hit;
  endfunction

  function automatic logic in_desc(input logic [11:0] a);
    logic hit;
    hit = 1'b0;
    for (int k = 0; k < DESC_NG; k++) hit |= ((a >> DESC_GW) == 12'((REG_DESC_BASE >> DESC_GW) + 12'(k)));
    return hit;
  endfunction

  function automatic logic in_logits(input logic [11:0] a);
    logic hit;
    hit = 1'b0;
    for (int k = 0; k < LOGITS_NG; k++) hit |= ((a >> LOGITS_GW) == 12'((REG_LOGITS_BASE >> LOGITS_GW) + 12'(k)));
    return hit;
  endfunction

  function automatic logic in_perf_layer(input logic [11:0] a);
    logic hit;
    hit = 1'b0;
    for (int k = 0; k < PERF_NG; k++) hit |= ((a >> PERF_GW) == 12'((REG_PERF_LAYER_BASE >> PERF_GW) + 12'(k)));
    return hit;
  endfunction

  // Word offset inside the descriptor block, {idx, word}: the bits below the
  // window granule pass through and only the granule number is offset.
  function automatic logic [7:0] desc_woff(input logic [11:0] a);
    logic [11:0] d;
    d = 12'(((a >> DESC_GW) - 12'(REG_DESC_BASE >> DESC_GW)) << DESC_GW) | (a & 12'((1 << DESC_GW) - 1));
    return 8'(d >> 2);
  endfunction

  function automatic tgt_e decode(input logic [19:0] a);
    if (a[1:0] != 2'b00) return TGT_NONE;
    if ((a >> $clog2(REGS_BYTES)) == '0) return TGT_REG;
    if (in_abuf(a)) return TGT_ABUF;
    if (in_wbuf(a)) return TGT_WBUF;
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

  logic        hm_wait_q;   // a memory access is outstanding
  logic        hm_is_rd_q;
  logic        dma_done_q, dma_err_q;
  err_e        dma_err_code_q;
  logic        mem_blocked_c;   // on-chip memory not available to the host
  assign mem_blocked_c = busy_i || dma_busy_i;

  assign layer_cnt_o = layer_cnt_q;
  assign irq_o       = done_q & irq_en_q;

  // Host read of a descriptor word (address decode for the descriptor RAM)
  logic       host_desc_rd_c;
  logic [7:0] host_desc_woff_c;
  assign host_desc_rd_c   = rd_req_i && (decode(rd_addr_i) == TGT_REG) && in_desc(rd_addr_i[11:0]);
  assign host_desc_woff_c = desc_woff(rd_addr_i[11:0]);   // {idx, word}

  // ---------------------------------------------------------------------------
  // Register read mux
  // ---------------------------------------------------------------------------
  // Descriptor words come from the descriptor RAM (rd_data_o below), so they
  // read as 0 here. The layer counters are read through a one-hot AND-OR mux.
  function automatic logic [31:0] layer_cycles_read(input logic [3:0] idx);
    logic [31:0] r;
    r = '0;
    for (int i = 0; i < NUM_DESC; i++) begin
      r |= layer_cycles_i[i] & {32{idx == 4'(i)}};
    end
    return r;
  endfunction

  function automatic logic [31:0] reg_read(input logic [11:0] a);
    logic [3:0]  idx;
    if (in_desc(a)) begin
      return 32'h0;        // the word itself comes from the descriptor RAM
    end
    if (in_logits(a)) begin
      idx = 4'((a - REG_LOGITS_BASE) >> 2);
      return {{24{logits_q[idx][7]}}, logits_q[idx]};
    end
    if (in_perf_layer(a)) begin
      idx = 4'((a - REG_PERF_LAYER_BASE) >> 2);
      return layer_cycles_read(idx);
    end
    case (a)
      REG_ID:              return NPU_ID;
      REG_CTRL:            return {30'h0, irq_en_q, 1'b0};
      REG_STATUS:          return {8'h0, err_code_q, 4'h0, cur_layer_i, 4'h0, host_conflict_q, err_q, done_q, busy_i};
      REG_LAYER_CNT:       return {27'h0, layer_cnt_q};
      REG_RESULT:          return {23'h0, result_valid_q, 4'h0, result_idx_q};
      REG_PERF_TOTAL:      return total_cycles_i;
      REG_PERF_ARRAY_BUSY: return array_busy_cycles_i;
      REG_DMA_CTRL:        return {29'h0, dma_dst_sel_o, dma_dir_o, 1'b0};
      REG_DMA_SRC:         return dma_src_o;
      REG_DMA_DST:         return {12'h0, dma_dst_o};
      REG_DMA_LEN:         return {12'h0, dma_len_o};
      REG_DMA_STATUS:      return {16'h0, dma_err_code_q, 5'h0, dma_err_q, dma_done_q, dma_busy_i};
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
  assign desc_woff_c = desc_woff(wr_off_c);
  assign desc_idx_c  = desc_woff_c[7:4];
  assign desc_word_c = desc_woff_c[3:0];
  // Write data of a descriptor word with its reserved bits cleared. Stored words
  // always have them cleared, so writing only the strobed bytes is the same as
  // merging with the old word and masking.
  logic [31:0] desc_wdata_c;
  assign desc_wdata_c = wr_data_i & desc_word_mask(32'(desc_word_c));
  assign start_c     = wr_req_i && (wr_tgt_c == TGT_REG) && (wr_off_c == REG_CTRL) &&
                       wr_strb_i[0] && wr_data_i[0] && !busy_i && !dma_busy_i;
  logic dma_start_c;
  assign dma_start_c = wr_req_i && (wr_tgt_c == TGT_REG) && (wr_off_c == REG_DMA_CTRL) &&
                       wr_strb_i[0] && wr_data_i[0] && !busy_i && !dma_busy_i;

  // ---------------------------------------------------------------------------
  // Descriptor table: a 1R1W RAM of NUM_DESC x DESC_WORDS words at {idx, word}
  // ---------------------------------------------------------------------------
  // W10 .. W15 are never written. One valid bit per word makes the table read
  // as 0 after reset without a clearing sequence: a word that was never written
  // reads as 0, and the first write to a word writes all four bytes (the bytes
  // without a strobe as 0). The valid bit is read together with the RAM, so a
  // read and a write of the same word in one cycle both see the old word.
  // The read port serves host reads first, then sequencer fetches.
  localparam int unsigned DESC_RAM_DEPTH = NUM_DESC * DESC_WORDS;
  localparam int unsigned DESC_AW        = $clog2(DESC_RAM_DEPTH);

  logic desc_wr_c, desc_store_c;
  assign desc_wr_c    = wr_req_i && (wr_tgt_c == TGT_REG) && in_desc(wr_off_c);
  assign desc_store_c = desc_wr_c && (desc_word_c < 4'(DESC_USED_WORDS));

  logic [DESC_RAM_DEPTH-1:0] desc_valid_q;
  logic                      desc_ram_rd_en_c;
  logic [DESC_AW-1:0]        desc_ram_rd_addr_c, desc_ram_wr_addr_c;
  logic [31:0]               desc_ram_rdata, desc_ram_wdata_c;
  logic [3:0]                desc_ram_be_c;
  logic                      desc_first_wr_c;
  logic                      desc_rvalid_q;     // valid bit of the word on desc_ram_rdata
  logic [31:0]               desc_rword_c;      // that word, or 0 if it was never written

  assign desc_rd_gnt_o      = desc_rd_en_i && !host_desc_rd_c;
  assign desc_ram_rd_en_c   = host_desc_rd_c || desc_rd_gnt_o;
  assign desc_ram_rd_addr_c = host_desc_rd_c ? DESC_AW'(host_desc_woff_c) : DESC_AW'({desc_rd_idx_i, desc_rd_word_i});
  assign desc_ram_wr_addr_c = DESC_AW'({desc_idx_c, desc_word_c});
  assign desc_first_wr_c    = !desc_valid_q[desc_ram_wr_addr_c];
  assign desc_ram_be_c      = wr_strb_i | {4{desc_first_wr_c}};
  always_comb begin
    for (int b = 0; b < 4; b++) desc_ram_wdata_c[8*b +: 8] = wr_strb_i[b] ? desc_wdata_c[8*b +: 8] : 8'h00;
  end

  sram_1r1w #(.WIDTH(32), .DEPTH(DESC_RAM_DEPTH)) u_desc_ram (
      .clk_i     (clk_i),
      .rd_en_i   (desc_ram_rd_en_c),
      .rd_addr_i (desc_ram_rd_addr_c),
      .rd_data_o (desc_ram_rdata),
      .wr_en_i   (desc_store_c),
      .wr_addr_i (desc_ram_wr_addr_c),
      .wr_data_i (desc_ram_wdata_c),
      .wr_be_i   (desc_ram_be_c)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      desc_valid_q    <= '0;
      desc_rvalid_q   <= 1'b0;
      desc_rd_valid_o <= 1'b0;
    end else begin
      if (desc_store_c)     desc_valid_q[desc_ram_wr_addr_c] <= 1'b1;
      if (desc_ram_rd_en_c) desc_rvalid_q <= desc_valid_q[desc_ram_rd_addr_c];
      desc_rd_valid_o <= desc_rd_gnt_o;
    end
  end

  assign desc_rword_c   = desc_rvalid_q ? desc_ram_rdata : 32'h0;
  assign desc_rd_data_o = desc_rword_c;

  // Host read data: a descriptor word straight from the RAM in the cycle after
  // the request, everything else from the read register.
  logic [31:0] rd_data_q;
  logic        rd_from_desc_q;
  assign rd_data_o = rd_from_desc_q ? desc_rword_c : rd_data_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ack_o        <= 1'b0;
      wr_err_o        <= 1'b0;
      rd_ack_o        <= 1'b0;
      rd_data_q       <= '0;
      rd_from_desc_q  <= 1'b0;
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
      dma_start_o     <= 1'b0;
      dma_dir_o       <= 1'b0;
      dma_dst_sel_o   <= 1'b0;
      dma_src_o       <= '0;
      dma_dst_o       <= '0;
      dma_len_o       <= '0;
      dma_done_q      <= 1'b0;
      dma_err_q       <= 1'b0;
      dma_err_code_q  <= ERR_NONE;
    end else begin
      // defaults for the pulses
      rd_from_desc_q <= host_desc_rd_c;
      wr_ack_o <= 1'b0;
      wr_err_o <= 1'b0;
      rd_ack_o <= 1'b0;
      rd_err_o <= 1'b0;
      hm_req_o <= 1'b0;
      start_o  <= 1'b0;
      dma_start_o <= 1'b0;

      // ---- sequencer, DMA and argmax status, highest priority ----
      if (dma_done_i) dma_done_q <= 1'b1;
      if (dma_err_i) begin
        dma_err_q      <= 1'b1;
        dma_err_code_q <= dma_err_code_i;
      end
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
            end else if (wr_off_c == REG_DMA_CTRL) begin
              if (wr_strb_i[0]) begin
                dma_dir_o     <= wr_data_i[1];
                dma_dst_sel_o <= wr_data_i[2];
                if (dma_start_c) begin
                  dma_start_o <= 1'b1;
                  if (!dma_done_i) dma_done_q <= 1'b0;
                  if (!dma_err_i) begin dma_err_q <= 1'b0; dma_err_code_q <= ERR_NONE; end
                end
              end
            end else if (wr_off_c == REG_DMA_SRC) begin
              dma_src_o <= merge(dma_src_o, wr_data_i, wr_strb_i);
            end else if (wr_off_c == REG_DMA_DST) begin
              dma_dst_o <= 20'(merge({12'h0, dma_dst_o}, wr_data_i, wr_strb_i));
            end else if (wr_off_c == REG_DMA_LEN) begin
              dma_len_o <= 20'(merge({12'h0, dma_len_o}, wr_data_i, wr_strb_i));
            end else if (wr_off_c == REG_DMA_STATUS) begin
              if (wr_strb_i[0]) begin
                if (wr_data_i[1] && !dma_done_i) dma_done_q <= 1'b0;
                if (wr_data_i[2] && !dma_err_i)  dma_err_q  <= 1'b0;
              end
            end
            // descriptor words: written by the process above
            // other offsets: read-only or undefined, write ignored
          end
          TGT_ABUF, TGT_WBUF: begin
            if (mem_blocked_c) begin
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
            rd_data_q <= reg_read(rd_addr_i[11:0]);
          end
          TGT_ABUF, TGT_WBUF: begin
            if (mem_blocked_c) begin
              rd_ack_o        <= 1'b1;
              rd_err_o        <= 1'b1;
              rd_data_q       <= '0;
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
            rd_data_q <= '0;
          end
        endcase
      end

      // ---- memory completion ----
      if (hm_wait_q && hm_ack_i) begin
        hm_wait_q <= 1'b0;
        if (hm_is_rd_q) begin
          rd_ack_o  <= 1'b1;
          rd_data_q <= hm_rdata_i;
        end else begin
          wr_ack_o <= 1'b1;
        end
      end
    end
  end

endmodule

`default_nettype wire
