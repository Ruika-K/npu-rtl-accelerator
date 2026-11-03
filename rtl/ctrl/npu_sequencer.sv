`default_nettype none

// Layer sequencer (docs/architecture.md sections 7.3, 7.4; error codes in
// docs/programming_model.md section 5).
//
// On start_i it walks descriptors 0 .. layer_cnt_i-1. For each layer it first
// validates the descriptor (all static checks live here, so the engines can
// trust what they receive), then starts the engine for the opcode and waits
// for its done or err. NOP layers complete without an engine. FC is forwarded
// as CONV2D with k = 1, pad = 0, in_h = in_w = 1, stride 0.
//
// done_o and err_o are single-cycle pulses; busy_o is already low in the
// cycle they are asserted. cur_layer_o points at the current layer, and at
// the failing layer after an error. total_cycles_o counts cycles while busy,
// layer_cycles_o[i] the busy cycles spent on layer i, array_busy_cycles_o the
// busy cycles in which array_valid_i was high. All three clear on start.
//
// Check order, first hit wins: ERR_BAD_OP, ERR_BAD_PARAM, ERR_BAD_OUT_DIM,
// ERR_BAD_ADDR. The checks are combinational on the current descriptor.
module npu_sequencer
  import npu_pkg::*;
(
  input  logic                        clk_i,
  input  logic                        rst_ni,

  input  logic                        start_i,
  input  logic [4:0]                  layer_cnt_i,
  input  desc_t [NUM_DESC-1:0]        desc_i,

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

  // One datapath for every op: the operands that differ between ops are
  // selected first, so each product is built once (ARGMAX and FC are 1x1
  // images, MAXPOOL halves the size and keeps C_in). The error is then picked
  // in the documented order: BAD_OP, BAD_PARAM, BAD_OUT_DIM, BAD_ADDR.
  // All widths are exact, so no sum or product can wrap.
  function automatic err_e check_desc(input desc_t d);
    logic        is_conv, is_fc, is_pool, is_argmax, is_mac;
    logic [3:0]  k, pad;
    logic [15:0] in_h, in_w, c_out_eff;
    logic [16:0] oh, ow, n_pad;
    logic [21:0] k_total;                 // c_in (16) * k*k (6)
    logic [31:0] in_hw;
    logic [33:0] out_hw;
    logic [48:0] in_end;
    logic [50:0] out_end;
    logic [48:0] w_end, b_end;
    logic        param_err, dim_err, addr_err;
    // Zero points are int8 by construction and have no further constraint.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [15:0] unused_zp;
    /* verilator lint_on UNUSEDSIGNAL */
    unused_zp = {d.z_in, d.z_out};

    is_conv   = (d.op == OP_CONV2D);
    is_fc     = (d.op == OP_FC);
    is_pool   = (d.op == OP_MAXPOOL2X2);
    is_argmax = (d.op == OP_ARGMAX);
    is_mac    = is_conv || is_fc;

    // operands
    k         = is_conv ? d.k   : 4'd1;
    pad       = is_conv ? d.pad : 4'd0;
    in_h      = (is_fc || is_argmax) ? 16'd1 : d.in_h;
    in_w      = (is_fc || is_argmax) ? 16'd1 : d.in_w;
    c_out_eff = is_pool ? d.c_in : d.c_out;
    if (is_pool) begin
      oh = {2'b00, in_h[15:1]};
      ow = {2'b00, in_w[15:1]};
    end else begin
      oh = 17'(in_h) + 17'(pad) * 17'd2 - 17'(k) + 17'd1;   // only used when dim_err is clear
      ow = 17'(in_w) + 17'(pad) * 17'd2 - 17'(k) + 17'd1;
    end
    n_pad   = pad_to_arr(d.c_out);
    k_total = 22'(d.c_in) * 22'(k) * 22'(k);

    // sizes and end addresses (one product each)
    in_hw   = 32'(in_h) * 32'(in_w);
    out_hw  = 34'(oh) * 34'(ow);
    in_end  = 49'(d.in_base) + 49'(in_hw) * 49'(d.c_in);
    out_end = 51'(d.out_base) + 51'(out_hw) * 51'(c_out_eff);
    w_end   = 49'(d.w_base) + 49'(k_total) * 49'(n_pad);
    b_end   = 49'(d.b_base) + 49'(n_pad) * 49'd4;

    // checks per class, in the documented order inside each class
    if (is_argmax) begin
      param_err = (d.c_in == 16'd0) || (d.c_in > 16'(MAX_ARGMAX_C));
      dim_err   = 1'b0;
      addr_err  = (in_end > 49'(ABUF_BYTES));
    end else if (is_pool) begin
      param_err = (d.c_in == 16'd0);
      dim_err   = (d.in_h < 16'd2) || (d.in_w < 16'd2);
      addr_err  = (in_end > 49'(ABUF_BYTES)) || (out_end > 51'(ABUF_BYTES));
    end else begin
      param_err = (k == 4'd0) || (k > 4'(MAX_KERNEL)) || (pad > 4'(MAX_PAD))
               || (is_conv && d.stride_m1 != 2'd0)
               || (d.c_in == 16'd0) || (d.c_out == 16'd0)
               || (k_total > 22'(MAX_K_TOTAL))
               || (d.rq_shift == 6'd0) || (d.rq_shift == 6'd63)              // documented range 1..62
               || (d.rq_mult[31] != 1'b0) || (d.rq_mult[30] != 1'b1)        // [2^30, 2^31)
               || (d.act_min > d.act_max);                                    // signed compare
      dim_err   = (17'(in_h) + 17'(pad) * 17'd2 < 17'(k)) || (17'(in_w) + 17'(pad) * 17'd2 < 17'(k));
      addr_err  = (d.w_base[2:0] != 3'b000) || (d.b_base[2:0] != 3'b000)     // both 8-byte aligned
               || (in_end > 49'(ABUF_BYTES)) || (out_end > 51'(ABUF_BYTES))
               || (w_end > 49'(WBUF_BYTES)) || (b_end > 49'(WBUF_BYTES));
    end

    if (d.op == OP_NOP)                 return ERR_NONE;
    if (!is_mac && !is_pool && !is_argmax) return ERR_BAD_OP;
    if (param_err)                      return ERR_BAD_PARAM;
    if (dim_err)                        return ERR_BAD_OUT_DIM;
    if (addr_err)                       return ERR_BAD_ADDR;
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
  typedef enum logic [1:0] {S_IDLE, S_CHECK, S_RUN, S_NEXT} state_e;

  state_e      state_q;
  logic        busy_q;
  logic [3:0]  cur_q;
  logic [4:0]  cnt_q;
  op_e         active_op_q;

  desc_t cur_desc_c;
  err_e  check_c;
  logic  eng_done_c, eng_err_c;
  err_e  eng_err_code_c;
  logic  start_ok_c;

  // One-hot AND-OR select of the current descriptor. A plain desc_i[cur_q]
  // means the same thing, but some flows (sv2v + Yosys) flatten the array and
  // build a barrel shifter for a variable index whose stride is not a power of
  // two, at roughly ten times the area of this mux.
  always_comb begin
    logic [$bits(desc_t)-1:0] sel;
    sel = '0;
    for (int i = 0; i < NUM_DESC; i++) begin
      sel |= desc_i[i] & {$bits(desc_t){cur_q == 4'(i)}};
    end
    cur_desc_c = desc_t'(sel);
  end
  assign check_c = check_desc(cur_desc_c);
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
              state_q    <= S_CHECK;
            end
          end
        end

        S_CHECK: begin
          if (check_c != ERR_NONE) begin
            err_o      <= 1'b1;
            err_code_o <= check_c;
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
            cur_q   <= cur_q + 4'd1;
            state_q <= S_CHECK;
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
