`default_nettype none

// Assertions for npu_top, bound into every instance: who owns the shared
// memory ports (exactly one of the engines, the DMA, the host and the pixel
// stream may use them at a time), and coherent STATUS / DMA_STATUS reads.
module npu_top_sva (
  input logic        clk_i,
  input logic        rst_ni,
  /* verilator lint_off UNUSEDSIGNAL */   // only the address window, STATUS bits and byte 0 are observed
  input logic [31:0] s_axil_awaddr_i,
  input logic        s_axil_awvalid_i,
  input logic        s_axil_awready_o,
  input logic [31:0] s_axil_wdata_i,
  input logic [3:0]  s_axil_wstrb_i,
  input logic [31:0] s_axil_araddr_i,
  input logic        s_axil_arvalid_i,
  input logic        s_axil_arready_o,
  input logic [31:0] s_axil_rdata_o,
  /* verilator lint_on UNUSEDSIGNAL */
  input logic [1:0]  s_axil_rresp_o,
  input logic        s_axil_rvalid_o,
  input logic        s_axil_rready_i,
  input logic start,
  input logic dma_start,
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

  // ---------------------------------------------------------------------------
  // STATUS coherence: after an accepted START (DMA start), and until software
  // clears DONE / ERR with a write to STATUS (DMA_STATUS), a read of it never
  // shows BUSY, DONE and ERR all clear. The engines drop busy in the cycle
  // they report done, so this needs the status read to include that pulse.
  // ---------------------------------------------------------------------------
  localparam logic [19:0] A_STATUS = 20'h008, A_DMA_STATUS = 20'h0D0;
  logic        run_armed_q, dma_armed_q;
  logic [19:0] rd_addr_q;
  logic        run_armed_rd_q, dma_armed_rd_q;
  logic        aw_hs_c, ar_hs_c, r_hs_c, wr_clear_c, wr_dma_clear_c;
  assign aw_hs_c        = s_axil_awvalid_i && s_axil_awready_o;     // AW and W are accepted together
  assign ar_hs_c        = s_axil_arvalid_i && s_axil_arready_o;
  assign r_hs_c         = s_axil_rvalid_o && s_axil_rready_i;
  assign wr_clear_c     = aw_hs_c && s_axil_awaddr_i[19:0] == A_STATUS && s_axil_wstrb_i[0] && s_axil_wdata_i[2:1] != 2'b00;
  assign wr_dma_clear_c = aw_hs_c && s_axil_awaddr_i[19:0] == A_DMA_STATUS && s_axil_wstrb_i[0]
                          && s_axil_wdata_i[2:1] != 2'b00;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      run_armed_q <= 1'b0; dma_armed_q <= 1'b0; rd_addr_q <= '0; run_armed_rd_q <= 1'b0; dma_armed_rd_q <= 1'b0;
    end else begin
      if (start)           run_armed_q <= 1'b1;
      else if (wr_clear_c) run_armed_q <= 1'b0;
      if (dma_start)           dma_armed_q <= 1'b1;
      else if (wr_dma_clear_c) dma_armed_q <= 1'b0;
      if (ar_hs_c) begin
        rd_addr_q      <= s_axil_araddr_i[19:0];
        run_armed_rd_q <= run_armed_q;
        dma_armed_rd_q <= dma_armed_q;
      end
    end
  end

  a_status_coherent: assert property (@(posedge clk_i) disable iff (!rst_ni)
      r_hs_c && rd_addr_q == A_STATUS && run_armed_rd_q && s_axil_rresp_o == 2'b00 |-> s_axil_rdata_o[2:0] != 3'b000)
    else $error("STATUS read after START shows neither BUSY nor DONE nor ERR");
  a_dma_status_coherent: assert property (@(posedge clk_i) disable iff (!rst_ni)
      r_hs_c && rd_addr_q == A_DMA_STATUS && dma_armed_rd_q && s_axil_rresp_o == 2'b00 |-> s_axil_rdata_o[2:0] != 3'b000)
    else $error("DMA_STATUS read after a DMA start shows neither BUSY nor DONE nor ERR");

  c_status_busy:     cover property (@(posedge clk_i) disable iff (!rst_ni) r_hs_c && rd_addr_q == A_STATUS && s_axil_rdata_o[0]);
  c_dma_status_busy: cover property (@(posedge clk_i) disable iff (!rst_ni) r_hs_c && rd_addr_q == A_DMA_STATUS && s_axil_rdata_o[0]);

  c_pix_write:     cover property (@(posedge clk_i) disable iff (!rst_ni) pix_wr_en_i && pix_wr_ready_o);
  c_dma_load:      cover property (@(posedge clk_i) disable iff (!rst_ni) dma_busy && wbuf_wr_en);
  c_irq:           cover property (@(posedge clk_i) disable iff (!rst_ni) irq_o);

endmodule

bind npu_top npu_top_sva u_sva (.*);

`default_nettype wire
