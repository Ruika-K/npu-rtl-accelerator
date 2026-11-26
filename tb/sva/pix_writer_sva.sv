`default_nettype none

// Assertions for pix_writer, bound into every instance: it writes only when
// the memory port is granted, never takes a stream byte while a pack waits to
// be written, and reports each frame with one pulse.
module pix_writer_sva
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input logic         clk_i,
  input logic         rst_ni,
  input logic         en_i,
  input logic         fifo_valid_i,
  input logic         fifo_ready_o,
  input logic         wr_en_o,
  input logic [N-1:0] wr_be_o,
  input logic         wr_ready_i,
  input logic         frame_done_o,
  input logic         sof_err_o
);

  logic flush_wait_c;   // a pack is complete and the port is not granted
  assign flush_wait_c = en_i && !fifo_ready_o && !wr_en_o;

  a_wr_when_granted: assert property (@(posedge clk_i) disable iff (!rst_ni) wr_en_o |-> wr_ready_i && wr_be_o != '0)
    else $error("write without the grant, or with no byte enabled");
  a_wr_be_prefix: assert property (@(posedge clk_i) disable iff (!rst_ni) wr_en_o |-> (wr_be_o & (wr_be_o + N'(1))) == '0)
    else $error("byte enables are not a run of low lanes");
  a_no_take_in_flush: assert property (@(posedge clk_i) disable iff (!rst_ni) wr_en_o |-> !fifo_ready_o)
    else $error("stream byte accepted while a pack is being written");
  a_disabled_quiet: assert property (@(posedge clk_i) disable iff (!rst_ni) !en_i |-> !fifo_ready_o)
    else $error("stream accepted while disabled");
  a_done_pulse: assert property (@(posedge clk_i) disable iff (!rst_ni) frame_done_o |=> !frame_done_o)
    else $error("frame_done longer than one cycle");

  c_flush_wait:   cover property (@(posedge clk_i) disable iff (!rst_ni) flush_wait_c && !wr_ready_i);
  c_partial_pack: cover property (@(posedge clk_i) disable iff (!rst_ni) wr_en_o && wr_be_o != '1);
  c_full_pack:    cover property (@(posedge clk_i) disable iff (!rst_ni) wr_en_o && wr_be_o == '1);
  c_sof_err:      cover property (@(posedge clk_i) disable iff (!rst_ni) sof_err_o);
  c_source_stall: cover property (@(posedge clk_i) disable iff (!rst_ni) en_i && fifo_ready_o && !fifo_valid_i);

endmodule

bind pix_writer pix_writer_sva #(.N(N)) u_sva (.*);

`default_nettype wire
