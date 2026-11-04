`default_nettype none

// Layer sequencer (docs/architecture.md sections 7.3, 7.4; error codes in
// docs/programming_model.md section 5).
//
// On start_i it walks descriptors 0 .. layer_cnt_i-1. For each layer it
// fetches the descriptor's DESC_USED_WORDS words from npu_regs (one request
// per cycle on desc_rd_*, the word arrives the cycle after the grant), then
// validates it (all static checks live here, so the engines can trust what
// they receive), starts the engine for the opcode and waits for its done or
// err. NOP layers complete without an engine. FC is forwarded as CONV2D with
// k = 1, pad = 0, in_h = in_w = 1, stride 0.
//
// done_o and err_o are single-cycle pulses; busy_o is already low in the
// cycle they are asserted. cur_layer_o points at the current layer, and at
// the failing layer after an error. total_cycles_o counts cycles while busy,
// layer_cycles_o[i] the busy cycles spent on layer i, array_busy_cycles_o the
// busy cycles in which array_valid_i was high. All three clear on start.
//
// Check order, first hit wins: ERR_BAD_OP, ERR_BAD_PARAM, ERR_BAD_OUT_DIM,
// ERR_BAD_ADDR. The checks are a three-stage pipeline on the fetched
// (registered) descriptor.
module npu_sequencer
  import npu_pkg::*;
(
  input  logic                        clk_i,
  input  logic                        rst_ni,

  input  logic                        start_i,
  input  logic [4:0]                  layer_cnt_i,
  output logic                        desc_rd_en_o,
  output logic [3:0]                  desc_rd_idx_o,
  output logic [3:0]                  desc_rd_word_o,
  input  logic                        desc_rd_gnt_i,
  input  logic                        desc_rd_valid_i,
  input  logic [31:0]                 desc_rd_data_i,

  output logic                        busy_o,
  output logic                        done_o,
  output logic                        err_o,
  output err_e                        err_code_o,
  output logic [3:0]                  cur_layer_o,
  output logic [NUM_DESC-1:0][31:0]   layer_cycles_o,
  output logic [31:0]                 total_cycles_o,
  output logic [31:0]                 array_busy_cycles_o,
  input  logic                        array_valid_i,

  output desc_t                       eng_desc_o,
  output logic                        conv_start_o,
  input  logic                        conv_done_i,
  input  logic                        conv_err_i,
  input  err_e                        conv_err_code_i,
  output logic                        pool_start_o,
  input  logic                        pool_done_i,
  input  logic                        pool_err_i,
  input  err_e                        pool_err_code_i,
  output logic                        argmax_start_o,
  input  logic                        argmax_done_i,
  input  logic                        argmax_err_i,
  input  err_e                        argmax_err_code_i
);

  // ---------------------------------------------------------------------------
  // Descriptor validation
  // ---------------------------------------------------------------------------
  function automatic logic [16:0] pad_to_arr(input logic [15:0] c);
    // round up to a multiple of ARR_N (power of two)
    return (17'(c) + 17'(ARR_N - 1)) & ~17'(ARR_N - 1);
  endfunction

  // Descriptor checks: one datapath for every op, pipelined over three
  // registered stages so that no path holds more than one multiplier or one
  // wide add (the checks run once per layer, CHECK_LAT cycles).
  //   A  operands that differ between ops (ARGMAX and FC are 1x1 images,
  //      MAXPOOL halves the size and keeps C_in), the size products, and the
  //      parameter, size and alignment flags
  //   B  byte counts (size x channels, K_total x padded C_out)
  //   C  end addresses against the buffer sizes, and the error code in the
  //      documented order: BAD_OP, BAD_PARAM, BAD_OUT_DIM, BAD_ADDR
  // All widths are exact, so no sum or product can wrap.
  localparam int unsigned CHECK_LAT = 3;

  typedef struct packed {
    logic        nop, bad_op, is_mac, is_pool;
    logic        param_err;            // all parameter checks but the K_total limit
    logic        dim_err, align_err;
    logic [15:0] c_in, c_out_eff;
    logic [16:0] n_pad;
    logic [21:0] k_total;
    logic [31:0] in_hw;
    logic [33:0] out_hw;
    logic [31:0] in_base, out_base, w_base, b_base;
  } chk_a_t;

  typedef struct packed {
    logic        nop, bad_op, is_mac, is_pool;
    logic        param_err, dim_err, align_err;
    logic [47:0] in_bytes;
    logic [49:0] out_bytes;
    logic [38:0] w_bytes;
    logic [18:0] b_bytes;
    logic [31:0] in_base, out_base, w_base, b_base;
  } chk_b_t;

  function automatic chk_a_t check_stage_a(input desc_t d);
    chk_a_t      a;
    logic        is_conv, is_fc, is_argmax;
    logic [3:0]  k, pad;
    logic [5:0]  kk;
    logic [15:0] in_h, in_w;
    logic [16:0] oh, ow;
    // Zero points are int8 by construction and have no further constraint.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [15:0] unused_zp;
    /* verilator lint_on UNUSEDSIGNAL */
    unused_zp = {d.z_in, d.z_out};

    is_conv   = (d.op == OP_CONV2D);
    is_fc     = (d.op == OP_FC);
    is_argmax = (d.op == OP_ARGMAX);
    a.nop     = (d.op == OP_NOP);
    a.is_mac  = is_conv || is_fc;
    a.is_pool = (d.op == OP_MAXPOOL2X2);
    a.bad_op  = !a.nop && !a.is_mac && !a.is_pool && !is_argmax;

    k           = is_conv ? d.k   : 4'd1;
    pad         = is_conv ? d.pad : 4'd0;
    kk          = 6'(k) * 6'(k);
    in_h        = (is_fc || is_argmax) ? 16'd1 : d.in_h;
    in_w        = (is_fc || is_argmax) ? 16'd1 : d.in_w;
    a.c_in      = d.c_in;
    a.c_out_eff = a.is_pool ? d.c_in : d.c_out;
    if (a.is_pool) begin
      oh = {2'b00, in_h[15:1]};
      ow = {2'b00, in_w[15:1]};
    end else begin
      oh = 17'(in_h) + 17'(pad) * 17'd2 - 17'(k) + 17'd1;   // only used when dim_err is clear
      ow = 17'(in_w) + 17'(pad) * 17'd2 - 17'(k) + 17'd1;
    end
    a.n_pad   = pad_to_arr(d.c_out);
    a.k_total = 22'(d.c_in) * 22'(kk);
    a.in_hw   = 32'(in_h) * 32'(in_w);
    a.out_hw  = 34'(oh) * 34'(ow);
    a.in_base = d.in_base; a.out_base = d.out_base; a.w_base = d.w_base; a.b_base = d.b_base;

    if (is_argmax) begin
      a.param_err = (d.c_in == 16'd0) || (d.c_in > 16'(MAX_ARGMAX_C));
      a.dim_err   = 1'b0;
      a.align_err = 1'b0;
    end else if (a.is_pool) begin
      a.param_err = (d.c_in == 16'd0);
      a.dim_err   = (d.in_h < 16'd2) || (d.in_w < 16'd2);
      a.align_err = 1'b0;
    end else begin
      a.param_err = (k == 4'd0) || (k > 4'(MAX_KERNEL)) || (pad > 4'(MAX_PAD))
                 || (is_conv && d.stride_m1 != 2'd0)
                 || (d.c_in == 16'd0) || (d.c_out == 16'd0)
                 || (d.rq_shift == 6'd0) || (d.rq_shift == 6'd63)             // documented range 1..62
                 || (d.rq_mult[31] != 1'b0) || (d.rq_mult[30] != 1'b1)       // [2^30, 2^31)
                 || (d.act_min > d.act_max);                                   // signed compare
      a.dim_err   = (17'(in_h) + 17'(pad) * 17'd2 < 17'(k)) || (17'(in_w) + 17'(pad) * 17'd2 < 17'(k));
      a.align_err = (d.w_base[2:0] != 3'b000) || (d.b_base[2:0] != 3'b000);   // both 8-byte aligned
    end
    return a;
  endfunction

  function automatic chk_b_t check_stage_b(input chk_a_t a);
    chk_b_t b;
    b.nop = a.nop; b.bad_op = a.bad_op; b.is_mac = a.is_mac; b.is_pool = a.is_pool;
    b.param_err = a.param_err || (a.is_mac && a.k_total > 22'(MAX_K_TOTAL));
    b.dim_err   = a.dim_err;
    b.align_err = a.align_err;
    b.in_bytes  = 48'(a.in_hw) * 48'(a.c_in);
    b.out_bytes = 50'(a.out_hw) * 50'(a.c_out_eff);
    b.w_bytes   = 39'(a.k_total) * 39'(a.n_pad);
    b.b_bytes   = {a.n_pad, 2'b00};
    b.in_base = a.in_base; b.out_base = a.out_base; b.w_base = a.w_base; b.b_base = a.b_base;
    return b;
  endfunction

  function automatic err_e check_stage_c(input chk_b_t b);
    logic in_over, out_over, w_over, b_over, addr_err;
    in_over  = (49'(b.in_base)  + 49'(b.in_bytes))  > 49'(ABUF_BYTES);
    out_over = (51'(b.out_base) + 51'(b.out_bytes)) > 51'(ABUF_BYTES);
    w_over   = (40'(b.w_base)   + 40'(b.w_bytes))   > 40'(WBUF_BYTES);
    b_over   = (33'(b.b_base)   + 33'(b.b_bytes))   > 33'(WBUF_BYTES);
    if (b.is_mac)       addr_err = b.align_err || in_over || out_over || w_over || b_over;
    else if (b.is_pool) addr_err = in_over || out_over;
    else                addr_err = in_over;                                     // ARGMAX
    if (b.nop)       return ERR_NONE;
    if (b.bad_op)    return ERR_BAD_OP;
    if (b.param_err) return ERR_BAD_PARAM;
    if (b.dim_err)   return ERR_BAD_OUT_DIM;
    if (addr_err)    return ERR_BAD_ADDR;
    return ERR_NONE;
  endfunction

  // Descriptor as handed to the engine: FC becomes a 1x1 CONV2D.
  function automatic desc_t engine_view(input desc_t d);
    desc_t e;
    e = d;
    if (d.op == OP_FC) begin
      e.k         = 4'd1;
      e.pad       = 4'd0;
      e.stride_m1 = 2'd0;
      e.in_h      = 16'd1;
      e.in_w      = 16'd1;
    end
    return e;
  endfunction

  // ---------------------------------------------------------------------------
  // Sequencer FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {S_IDLE, S_FETCH, S_CHK_WAIT, S_CHECK, S_RUN, S_NEXT} state_e;

  state_e      state_q;
  logic        busy_q;
  logic [3:0]  cur_q;
  logic [4:0]  cnt_q;
  op_e         active_op_q;

  desc_t cur_desc_c;
  logic  eng_done_c, eng_err_c;
  err_e  eng_err_code_c;
  logic  start_ok_c;

  // Descriptor fetch: words are requested in order 0 .. DESC_USED_WORDS-1 and
  // shifted in from the top as they arrive, so after the last one word w sits
  // at index w. Fetching through the register file's 32-bit read port replaces
  // a combinational select of a whole descriptor out of the table, the largest
  // mux in the design, and registers the input of the checks below.
  localparam int unsigned WCNT_W = $clog2(DESC_USED_WORDS + 1);
  desc_words_t       cur_words_q;
  logic [WCNT_W-1:0] fetch_req_q, fetch_rcv_q;   // words requested / received
  logic              fetch_last_c;

  assign desc_rd_en_o   = (state_q == S_FETCH) && (fetch_req_q < WCNT_W'(DESC_USED_WORDS));
  assign desc_rd_idx_o  = cur_q;
  assign desc_rd_word_o = 4'(fetch_req_q);
  assign fetch_last_c   = desc_rd_valid_i && (fetch_rcv_q == WCNT_W'(DESC_USED_WORDS - 1));
  assign cur_desc_c     = desc_from_words(cur_words_q);

  // Check pipeline: free running on the fetched descriptor, which is stable
  // from the end of the fetch until the next layer's fetch.
  chk_a_t chk_a_q;
  chk_b_t chk_b_q;
  err_e   chk_err_q;
  logic [1:0] chk_wait_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      chk_a_q   <= '0;
      chk_b_q   <= '0;
      chk_err_q <= ERR_NONE;
    end else begin
      chk_a_q   <= check_stage_a(cur_desc_c);
      chk_b_q   <= check_stage_b(chk_a_q);
      chk_err_q <= check_stage_c(chk_b_q);
    end
  end

  assign start_ok_c = start_i && (state_q == S_IDLE) && (layer_cnt_i != 5'd0) && (layer_cnt_i <= 5'(NUM_DESC));

  always_comb begin
    case (active_op_q)
      OP_MAXPOOL2X2: begin
        eng_done_c = pool_done_i;   eng_err_c = pool_err_i;   eng_err_code_c = pool_err_code_i;
      end
      OP_ARGMAX: begin
        eng_done_c = argmax_done_i; eng_err_c = argmax_err_i; eng_err_code_c = argmax_err_code_i;
      end
      default: begin
        eng_done_c = conv_done_i;   eng_err_c = conv_err_i;   eng_err_code_c = conv_err_code_i;
      end
    endcase
  end

  assign busy_o      = busy_q;
  assign cur_layer_o = cur_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q        <= S_IDLE;
      busy_q         <= 1'b0;
      cur_q          <= '0;
      cnt_q          <= '0;
      active_op_q    <= OP_NOP;
      done_o         <= 1'b0;
      err_o          <= 1'b0;
      err_code_o     <= ERR_NONE;
      eng_desc_o     <= '0;
      cur_words_q    <= '0;
      fetch_req_q    <= '0;
      fetch_rcv_q    <= '0;
      chk_wait_q     <= '0;
      conv_start_o   <= 1'b0;
      pool_start_o   <= 1'b0;
      argmax_start_o <= 1'b0;
    end else begin
      done_o         <= 1'b0;
      err_o          <= 1'b0;
      conv_start_o   <= 1'b0;
      pool_start_o   <= 1'b0;
      argmax_start_o <= 1'b0;

      case (state_q)
        S_IDLE: begin
          if (start_i) begin
            if (!start_ok_c) begin
              err_o      <= 1'b1;
              err_code_o <= ERR_BAD_LAYER_CNT;
            end else begin
              busy_q     <= 1'b1;
              cur_q      <= '0;
              cnt_q      <= layer_cnt_i;
              err_code_o <= ERR_NONE;
              fetch_req_q <= '0;
              fetch_rcv_q <= '0;
              state_q    <= S_FETCH;
            end
          end
        end

        S_FETCH: begin
          if (desc_rd_en_o && desc_rd_gnt_i) fetch_req_q <= fetch_req_q + WCNT_W'(1);
          if (desc_rd_valid_i) begin
            cur_words_q <= {desc_rd_data_i, cur_words_q[DESC_USED_WORDS-1:1]};
            fetch_rcv_q <= fetch_rcv_q + WCNT_W'(1);
          end
          if (fetch_last_c) begin
            chk_wait_q <= 2'(CHECK_LAT - 1);
            state_q    <= S_CHK_WAIT;
          end
        end

        S_CHK_WAIT: begin   // the check pipeline fills
          if (chk_wait_q == 2'd0) state_q <= S_CHECK;
          else                    chk_wait_q <= chk_wait_q - 2'd1;
        end

        S_CHECK: begin
          if (chk_err_q != ERR_NONE) begin
            err_o      <= 1'b1;
            err_code_o <= chk_err_q;
            busy_q     <= 1'b0;
            state_q    <= S_IDLE;
          end else if (cur_desc_c.op == OP_NOP) begin
            state_q <= S_NEXT;
          end else begin
            eng_desc_o  <= engine_view(cur_desc_c);
            active_op_q <= cur_desc_c.op;
            case (cur_desc_c.op)
              OP_MAXPOOL2X2: pool_start_o   <= 1'b1;
              OP_ARGMAX:     argmax_start_o <= 1'b1;
              default:       conv_start_o   <= 1'b1;   // CONV2D and FC
            endcase
            state_q <= S_RUN;
          end
        end

        S_RUN: begin
          if (eng_err_c) begin
            err_o      <= 1'b1;
            err_code_o <= eng_err_code_c;
            busy_q     <= 1'b0;
            state_q    <= S_IDLE;
          end else if (eng_done_c) begin
            state_q <= S_NEXT;
          end
        end

        S_NEXT: begin
          if (5'(cur_q) + 5'd1 == cnt_q) begin
            busy_q  <= 1'b0;
            done_o  <= 1'b1;
            state_q <= S_IDLE;
          end else begin
            cur_q       <= cur_q + 4'd1;
            fetch_req_q <= '0;
            fetch_rcv_q <= '0;
            state_q     <= S_FETCH;
          end
        end

        default: state_q <= S_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------------------
  // Performance counters
  // ---------------------------------------------------------------------------
  // One incrementer on the selected layer counter, written back with a decoded
  // enable (same reason as the descriptor select above).
  logic [31:0] layer_cycles_inc_c;
  always_comb begin
    layer_cycles_inc_c = '0;
    for (int i = 0; i < NUM_DESC; i++) begin
      layer_cycles_inc_c |= layer_cycles_o[i] & {32{cur_q == 4'(i)}};
    end
    layer_cycles_inc_c = layer_cycles_inc_c + 32'd1;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      layer_cycles_o      <= '0;
      total_cycles_o      <= '0;
      array_busy_cycles_o <= '0;
    end else if (start_ok_c) begin
      layer_cycles_o      <= '0;
      total_cycles_o      <= '0;
      array_busy_cycles_o <= '0;
    end else if (busy_q) begin
      total_cycles_o       <= total_cycles_o + 32'd1;
      for (int i = 0; i < NUM_DESC; i++) begin
        if (cur_q == 4'(i)) layer_cycles_o[i] <= layer_cycles_inc_c;
      end
      if (array_valid_i) array_busy_cycles_o <= array_busy_cycles_o + 32'd1;
    end
  end

endmodule

`default_nettype wire
