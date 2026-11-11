`default_nettype none

// Assertions for npu_sequencer, bound into every instance: engine starts are
// exclusive single pulses, a run ends with exactly one of done or err, the
// performance counters are consistent, and the scenarios (every opcode, every
// error code, the layer-count limits) are reached.
module npu_sequencer_sva
  import npu_pkg::*;
(
  input logic                      clk_i,
  input logic                      rst_ni,
  input logic                      start_i,
  input logic [4:0]                layer_cnt_i,
  input logic                      desc_rd_en_o,
  input logic                      desc_rd_gnt_i,
  input logic                      busy_o,
  input logic                      done_o,
  input logic                      err_o,
  input err_e                      err_code_o,
  input logic [3:0]                cur_layer_o,
  input logic [NUM_DESC-1:0][31:0] layer_cycles_o,
  input logic [31:0]               total_cycles_o,
  input logic [31:0]               array_busy_cycles_o,
  input desc_t                     eng_desc_o,
  input logic                      conv_start_o,
  input logic                      conv_done_i,
  input logic                      pool_start_o,
  input logic                      pool_done_i,
  input logic                      argmax_start_o,
  input logic                      argmax_done_i,
  input logic [4:0]                cnt_q
);

  logic any_start_c, any_done_c;
  assign any_start_c = conv_start_o || pool_start_o || argmax_start_o;
  assign any_done_c  = conv_done_i || pool_done_i || argmax_done_i;

  logic        busy_prev_q, running_q;   // running: an engine was started and is not done
  logic [31:0] total_prev_q;
  logic [35:0] layer_sum_c;
  always_comb begin
    layer_sum_c = '0;
    for (int i = 0; i < NUM_DESC; i++) layer_sum_c += 36'(layer_cycles_o[i]);
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy_prev_q <= 1'b0; running_q <= 1'b0; total_prev_q <= '0;
    end else begin
      busy_prev_q  <= busy_o;
      total_prev_q <= total_cycles_o;
      if (any_start_c)                 running_q <= 1'b1;
      else if (any_done_c || !busy_o)  running_q <= 1'b0;
    end
  end

  // Engine starts
  a_start_onehot: assert property (@(posedge clk_i) disable iff (!rst_ni)
      $onehot0({conv_start_o, pool_start_o, argmax_start_o}))
    else $error("more than one engine started");
  a_start_pulse: assert property (@(posedge clk_i) disable iff (!rst_ni) any_start_c |=> !any_start_c)
    else $error("engine start longer than one cycle");
  a_start_when_free: assert property (@(posedge clk_i) disable iff (!rst_ni) any_start_c |-> busy_o && !running_q)
    else $error("engine started while idle or while another engine runs");
  a_start_op: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (!conv_start_o || eng_desc_o.op == OP_CONV2D || eng_desc_o.op == OP_FC)
      && (!pool_start_o || eng_desc_o.op == OP_MAXPOOL2X2) && (!argmax_start_o || eng_desc_o.op == OP_ARGMAX))
    else $error("engine started with another engine's opcode");
  // FC reaches conv_engine as a 1x1 convolution over a 1x1 image
  a_fc_view: assert property (@(posedge clk_i) disable iff (!rst_ni)
      conv_start_o && eng_desc_o.op == OP_FC |-> eng_desc_o.k == 4'd1 && eng_desc_o.pad == 4'd0
                                                 && eng_desc_o.in_h == 16'd1 && eng_desc_o.in_w == 16'd1)
    else $error("FC descriptor not rewritten to the 1x1 convolution view");
  m_done_expected: assume property (@(posedge clk_i) disable iff (!rst_ni) any_done_c |-> running_q)
    else $error("engine done without a running engine");

  // End of a run
  a_done_err_excl: assert property (@(posedge clk_i) disable iff (!rst_ni) !(done_o && err_o))
    else $error("done and err together");
  a_end_pulse: assert property (@(posedge clk_i) disable iff (!rst_ni) done_o || err_o |=> !done_o && !err_o)
    else $error("done or err longer than one cycle");
  a_busy_end: assert property (@(posedge clk_i) disable iff (!rst_ni) busy_prev_q && !busy_o |-> done_o || err_o)
    else $error("busy fell without done or err");
  a_end_only_at_fall: assert property (@(posedge clk_i) disable iff (!rst_ni)
      done_o |-> busy_prev_q && !busy_o)
    else $error("done without a run ending");
  a_err_code: assert property (@(posedge clk_i) disable iff (!rst_ni) err_o |-> err_code_o != ERR_NONE)
    else $error("err without an error code");

  // Progress and counters
  a_layer_in_range: assert property (@(posedge clk_i) disable iff (!rst_ni) busy_o |-> 5'(cur_layer_o) < cnt_q)
    else $error("current layer beyond the layer count");
  a_fetch_only_busy: assert property (@(posedge clk_i) disable iff (!rst_ni) desc_rd_en_o |-> busy_o)
    else $error("descriptor fetch while idle");
  a_total_counts: assert property (@(posedge clk_i) disable iff (!rst_ni)
      busy_prev_q && !(start_i && !busy_o) |-> total_cycles_o == total_prev_q + 32'd1)
    else $error("total_cycles did not count a busy cycle");
  a_layer_sum: assert property (@(posedge clk_i) disable iff (!rst_ni) layer_sum_c == 36'(total_cycles_o))
    else $error("per-layer cycle counts do not add up to total_cycles");
  a_array_le_total: assert property (@(posedge clk_i) disable iff (!rst_ni) array_busy_cycles_o <= total_cycles_o)
    else $error("array_busy_cycles above total_cycles");

  // Scenarios
  c_op_conv:      cover property (@(posedge clk_i) disable iff (!rst_ni) conv_start_o && eng_desc_o.op == OP_CONV2D);
  c_op_fc:        cover property (@(posedge clk_i) disable iff (!rst_ni) conv_start_o && eng_desc_o.op == OP_FC);
  c_op_pool:      cover property (@(posedge clk_i) disable iff (!rst_ni) pool_start_o);
  c_op_argmax:    cover property (@(posedge clk_i) disable iff (!rst_ni) argmax_start_o);
  c_err_op:       cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && err_code_o == ERR_BAD_OP);
  c_err_param:    cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && err_code_o == ERR_BAD_PARAM);
  c_err_addr:     cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && err_code_o == ERR_BAD_ADDR);
  c_err_cnt:      cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && err_code_o == ERR_BAD_LAYER_CNT);
  c_err_out_dim:  cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && err_code_o == ERR_BAD_OUT_DIM);
  c_err_late:     cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && cur_layer_o != 4'd0);
  c_one_layer:    cover property (@(posedge clk_i) disable iff (!rst_ni) done_o && cnt_q == 5'd1);
  c_max_layers:   cover property (@(posedge clk_i) disable iff (!rst_ni) done_o && cnt_q == 5'(NUM_DESC));
  c_start_busy:   cover property (@(posedge clk_i) disable iff (!rst_ni) start_i && busy_o);
  c_fetch_stall:  cover property (@(posedge clk_i) disable iff (!rst_ni) desc_rd_en_o && !desc_rd_gnt_i);

endmodule

bind npu_sequencer npu_sequencer_sva u_sva (.*);

`default_nettype wire
