`default_nettype none

// Assertions for npu_top, bound into every instance: who owns the shared
// memory ports. Exactly one of the engines, the DMA, the host and the pixel
// stream may use them at a time.
module npu_top_sva (
  input logic clk_i,
  input logic rst_ni,
  input logic busy,
  input logic conv_busy,
  input logic pool_busy,
  input logic argmax_busy,
  input logic dma_busy,
  input logic h_abuf_wr_en,
  input logic h_abuf_rd_en,
  input logic h_wbuf_wr_en,
  input logic h_wbuf_rd_en,
  input logic wbuf_wr_en,
  input logic pix_wr_en_i,
  input logic pix_wr_ready_o,
  input logic irq_o
);

  logic eng_busy_c;
  assign eng_busy_c = conv_busy || pool_busy || argmax_busy;

  a_one_engine: assert property (@(posedge clk_i) disable iff (!rst_ni)
      $onehot0({conv_busy, pool_busy, argmax_busy}))
    else $error("two engines busy together");
  a_engine_in_run: assert property (@(posedge clk_i) disable iff (!rst_ni) eng_busy_c |-> busy)
    else $error("an engine is busy outside a run");
  a_dma_excl: assert property (@(posedge clk_i) disable iff (!rst_ni) !(dma_busy && (busy || eng_busy_c)))
    else $error("DMA and a run active together");
  a_host_mem_idle: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (h_abuf_wr_en || h_abuf_rd_en || h_wbuf_wr_en || h_wbuf_rd_en) |-> !busy && !dma_busy)
    else $error("host memory access granted while a run or the DMA is active");
  a_weights_stable: assert property (@(posedge clk_i) disable iff (!rst_ni) wbuf_wr_en |-> !busy)
    else $error("weight buffer written during a run");
  a_pix_ready: assert property (@(posedge clk_i) disable iff (!rst_ni)
      pix_wr_ready_o |-> !busy && !dma_busy && !h_abuf_wr_en)
    else $error("pixel stream granted while the activation buffer write port is in use");

  // The pixel source writes only when granted (a write without the grant
  // would be dropped); pix_writer_sva covers the source waiting for it.
  m_pix_when_ready: assume property (@(posedge clk_i) disable iff (!rst_ni) pix_wr_en_i |-> pix_wr_ready_o)
    else $error("pixel write without pix_wr_ready_o");

  c_pix_write:     cover property (@(posedge clk_i) disable iff (!rst_ni) pix_wr_en_i && pix_wr_ready_o);
  c_dma_load:      cover property (@(posedge clk_i) disable iff (!rst_ni) dma_busy && wbuf_wr_en);
  c_irq:           cover property (@(posedge clk_i) disable iff (!rst_ni) irq_o);

endmodule

bind npu_top npu_top_sva u_sva (.*);

`default_nettype wire
