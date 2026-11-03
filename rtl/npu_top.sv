`default_nettype none

// NPU v1 top level (docs/architecture.md section 2).
// Host side: AXI4-Lite slave and a level interrupt. Inside: register file and
// descriptor table, layer sequencer, the three engines, the two on-chip
// memories, and the muxes that give the memory ports to the active engine
// or, while idle, to the host.
module npu_top
  import npu_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [31:0] s_axil_awaddr_i,
  input  logic [2:0]  s_axil_awprot_i,
  input  logic        s_axil_awvalid_i,
  output logic        s_axil_awready_o,
  input  logic [31:0] s_axil_wdata_i,
  input  logic [3:0]  s_axil_wstrb_i,
  input  logic        s_axil_wvalid_i,
  output logic        s_axil_wready_o,
  output logic [1:0]  s_axil_bresp_o,
  output logic        s_axil_bvalid_o,
  input  logic        s_axil_bready_i,
  input  logic [31:0] s_axil_araddr_i,
  input  logic [2:0]  s_axil_arprot_i,
  input  logic        s_axil_arvalid_i,
  output logic        s_axil_arready_o,
  output logic [31:0] s_axil_rdata_o,
  output logic [1:0]  s_axil_rresp_o,
  output logic        s_axil_rvalid_o,
  input  logic        s_axil_rready_i,

  // Stream write port into the activation buffer (chip-level pixel writer).
  // Granted only while no layer program runs and the host is not writing.
  input  logic                    pix_wr_en_i,
  input  logic [ABUF_AW-1:0]      pix_wr_addr_i,
  input  logic [ARR_N-1:0][7:0]   pix_wr_data_i,
  input  logic [ARR_N-1:0]        pix_wr_be_i,
  output logic                    pix_wr_ready_o,

  // AXI4 master of the DMA (64-bit data, INCR bursts, single ID)
  output logic [31:0] m_axi_awaddr_o,
  output logic [7:0]  m_axi_awlen_o,
  output logic [2:0]  m_axi_awsize_o,
  output logic [1:0]  m_axi_awburst_o,
  output logic        m_axi_awvalid_o,
  input  logic        m_axi_awready_i,
  output logic [63:0] m_axi_wdata_o,
  output logic [7:0]  m_axi_wstrb_o,
  output logic        m_axi_wlast_o,
  output logic        m_axi_wvalid_o,
  input  logic        m_axi_wready_i,
  input  logic [1:0]  m_axi_bresp_i,
  input  logic        m_axi_bvalid_i,
  output logic        m_axi_bready_o,
  output logic [31:0] m_axi_araddr_o,
  output logic [7:0]  m_axi_arlen_o,
  output logic [2:0]  m_axi_arsize_o,
  output logic [1:0]  m_axi_arburst_o,
  output logic        m_axi_arvalid_o,
  input  logic        m_axi_arready_i,
  input  logic [63:0] m_axi_rdata_i,
  input  logic [1:0]  m_axi_rresp_i,
  input  logic        m_axi_rlast_i,
  input  logic        m_axi_rvalid_i,
  output logic        m_axi_rready_o,

  output logic        irq_o
);

  localparam int unsigned N = ARR_N;

  // ---------------------------------------------------------------------------
  // Host path
  // ---------------------------------------------------------------------------
  logic        wr_req, wr_ack, wr_err, rd_req, rd_ack, rd_err;
  logic [19:0] wr_addr, rd_addr;
  logic [31:0] wr_data, rd_data;
  logic [3:0]  wr_strb;

  axil_slave u_axil (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .s_axil_awaddr_i (s_axil_awaddr_i), .s_axil_awprot_i (s_axil_awprot_i), .s_axil_awvalid_i (s_axil_awvalid_i), .s_axil_awready_o (s_axil_awready_o),
      .s_axil_wdata_i (s_axil_wdata_i), .s_axil_wstrb_i (s_axil_wstrb_i), .s_axil_wvalid_i (s_axil_wvalid_i), .s_axil_wready_o (s_axil_wready_o),
      .s_axil_bresp_o (s_axil_bresp_o), .s_axil_bvalid_o (s_axil_bvalid_o), .s_axil_bready_i (s_axil_bready_i),
      .s_axil_araddr_i (s_axil_araddr_i), .s_axil_arprot_i (s_axil_arprot_i), .s_axil_arvalid_i (s_axil_arvalid_i), .s_axil_arready_o (s_axil_arready_o),
      .s_axil_rdata_o (s_axil_rdata_o), .s_axil_rresp_o (s_axil_rresp_o), .s_axil_rvalid_o (s_axil_rvalid_o), .s_axil_rready_i (s_axil_rready_i),
      .wr_req_o (wr_req), .wr_addr_o (wr_addr), .wr_data_o (wr_data), .wr_strb_o (wr_strb), .wr_ack_i (wr_ack), .wr_err_i (wr_err),
      .rd_req_o (rd_req), .rd_addr_o (rd_addr), .rd_ack_i (rd_ack), .rd_data_i (rd_data), .rd_err_i (rd_err)
  );

  logic        hm_req, hm_we, hm_sel, hm_ack;
  logic [19:0] hm_addr;
  logic [31:0] hm_wdata, hm_rdata;
  logic [3:0]  hm_be;

  logic        start, busy, done, err;
  logic [4:0]  layer_cnt;
  desc_t [NUM_DESC-1:0] desc;
  err_e        err_code;
  logic [3:0]  cur_layer;
  logic [NUM_DESC-1:0][31:0] layer_cycles;
  logic [31:0] total_cycles, array_busy_cycles;
  logic        argmax_valid;
  logic [3:0]  argmax_idx;
  logic [NUM_LOGITS-1:0][7:0] argmax_logits;
  logic        dma_start, dma_dir, dma_dst_sel, dma_busy, dma_done, dma_err;
  logic [31:0] dma_src;
  logic [19:0] dma_dst, dma_len;
  err_e        dma_err_code;

  npu_regs u_regs (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .wr_req_i (wr_req), .wr_addr_i (wr_addr), .wr_data_i (wr_data), .wr_strb_i (wr_strb), .wr_ack_o (wr_ack), .wr_err_o (wr_err),
      .rd_req_i (rd_req), .rd_addr_i (rd_addr), .rd_ack_o (rd_ack), .rd_data_o (rd_data), .rd_err_o (rd_err),
      .hm_req_o (hm_req), .hm_we_o (hm_we), .hm_sel_o (hm_sel), .hm_addr_o (hm_addr), .hm_wdata_o (hm_wdata), .hm_be_o (hm_be),
      .hm_ack_i (hm_ack), .hm_rdata_i (hm_rdata),
      .start_o (start), .layer_cnt_o (layer_cnt), .desc_o (desc),
      .busy_i (busy), .done_i (done), .err_i (err), .err_code_i (err_code), .cur_layer_i (cur_layer),
      .layer_cycles_i (layer_cycles), .total_cycles_i (total_cycles), .array_busy_cycles_i (array_busy_cycles),
      .dma_start_o (dma_start), .dma_dir_o (dma_dir), .dma_dst_sel_o (dma_dst_sel), .dma_src_o (dma_src),
      .dma_dst_o (dma_dst), .dma_len_o (dma_len), .dma_busy_i (dma_busy), .dma_done_i (dma_done),
      .dma_err_i (dma_err), .dma_err_code_i (dma_err_code),
      .argmax_valid_i (argmax_valid), .argmax_idx_i (argmax_idx), .argmax_logits_i (argmax_logits),
      .irq_o (irq_o)
  );

  // ---------------------------------------------------------------------------
  // Sequencer and engines
  // ---------------------------------------------------------------------------
  desc_t eng_desc;
  logic  conv_start, conv_busy, conv_done, array_valid;
  logic  pool_start, pool_busy, pool_done;
  logic  argmax_start, argmax_busy, argmax_done;

  npu_sequencer u_seq (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (start), .layer_cnt_i (layer_cnt), .desc_i (desc),
      .busy_o (busy), .done_o (done), .err_o (err), .err_code_o (err_code), .cur_layer_o (cur_layer),
      .layer_cycles_o (layer_cycles), .total_cycles_o (total_cycles), .array_busy_cycles_o (array_busy_cycles),
      .array_valid_i (array_valid),
      .eng_desc_o (eng_desc),
      .conv_start_o (conv_start), .conv_done_i (conv_done), .conv_err_i (1'b0), .conv_err_code_i (ERR_NONE),
      .pool_start_o (pool_start), .pool_done_i (pool_done), .pool_err_i (1'b0), .pool_err_code_i (ERR_NONE),
      .argmax_start_o (argmax_start), .argmax_done_i (argmax_done), .argmax_err_i (1'b0), .argmax_err_code_i (ERR_NONE)
  );

  // Memory ports of each client (c conv, p pool, a argmax, h host, d DMA)
  logic               c_abuf_rd_en, p_abuf_rd_en, a_abuf_rd_en, h_abuf_rd_en, d_abuf_rd_en;
  logic [ABUF_AW-1:0] c_abuf_rd_addr, p_abuf_rd_addr, a_abuf_rd_addr, h_abuf_rd_addr, d_abuf_rd_addr;
  logic               c_abuf_wr_en, p_abuf_wr_en, h_abuf_wr_en, d_abuf_wr_en;
  logic [ABUF_AW-1:0] c_abuf_wr_addr, p_abuf_wr_addr, h_abuf_wr_addr, d_abuf_wr_addr;
  logic [N-1:0][7:0]  c_abuf_wr_data, p_abuf_wr_data, h_abuf_wr_data, d_abuf_wr_data;
  logic [N-1:0]       c_abuf_wr_be, p_abuf_wr_be, h_abuf_wr_be, d_abuf_wr_be;
  logic               c_wbuf_rd_en, h_wbuf_rd_en, d_wbuf_rd_en;
  logic [WBUF_WAW-1:0] c_wbuf_rd_addr, h_wbuf_rd_addr, d_wbuf_rd_addr;
  logic               h_wbuf_wr_en, d_wbuf_wr_en;
  logic [WBUF_WAW-1:0] h_wbuf_wr_addr, d_wbuf_wr_addr;
  logic [63:0]        h_wbuf_wr_data, d_wbuf_wr_data;
  logic [7:0]         h_wbuf_wr_be, d_wbuf_wr_be;

  // Shared memory ports
  logic               abuf_rd_en, abuf_rd_valid, abuf_wr_en;
  logic [ABUF_AW-1:0] abuf_rd_addr, abuf_wr_addr;
  logic [N-1:0][7:0]  abuf_rd_data, abuf_wr_data;
  logic [N-1:0]       abuf_wr_be;
  logic               wbuf_rd_en, wbuf_rd_valid, wbuf_wr_en;
  logic [WBUF_WAW-1:0] wbuf_rd_addr, wbuf_wr_addr;
  logic [63:0]        wbuf_rd_data, wbuf_wr_data;
  logic [7:0]         wbuf_wr_be;

  npu_dma u_dma (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (dma_start), .dir_i (dma_dir), .dst_sel_i (dma_dst_sel), .src_i (dma_src), .dst_off_i (dma_dst), .len_i (dma_len),
      .busy_o (dma_busy), .done_o (dma_done), .err_o (dma_err), .err_code_o (dma_err_code),
      .abuf_wr_en_o (d_abuf_wr_en), .abuf_wr_addr_o (d_abuf_wr_addr), .abuf_wr_data_o (d_abuf_wr_data), .abuf_wr_be_o (d_abuf_wr_be),
      .abuf_rd_en_o (d_abuf_rd_en), .abuf_rd_addr_o (d_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .wbuf_wr_en_o (d_wbuf_wr_en), .wbuf_wr_addr_o (d_wbuf_wr_addr), .wbuf_wr_data_o (d_wbuf_wr_data), .wbuf_wr_be_o (d_wbuf_wr_be),
      .wbuf_rd_en_o (d_wbuf_rd_en), .wbuf_rd_addr_o (d_wbuf_rd_addr), .wbuf_rd_data_i (wbuf_rd_data), .wbuf_rd_valid_i (wbuf_rd_valid),
      .m_axi_awaddr_o (m_axi_awaddr_o), .m_axi_awlen_o (m_axi_awlen_o), .m_axi_awsize_o (m_axi_awsize_o), .m_axi_awburst_o (m_axi_awburst_o),
      .m_axi_awvalid_o (m_axi_awvalid_o), .m_axi_awready_i (m_axi_awready_i),
      .m_axi_wdata_o (m_axi_wdata_o), .m_axi_wstrb_o (m_axi_wstrb_o), .m_axi_wlast_o (m_axi_wlast_o), .m_axi_wvalid_o (m_axi_wvalid_o), .m_axi_wready_i (m_axi_wready_i),
      .m_axi_bresp_i (m_axi_bresp_i), .m_axi_bvalid_i (m_axi_bvalid_i), .m_axi_bready_o (m_axi_bready_o),
      .m_axi_araddr_o (m_axi_araddr_o), .m_axi_arlen_o (m_axi_arlen_o), .m_axi_arsize_o (m_axi_arsize_o), .m_axi_arburst_o (m_axi_arburst_o),
      .m_axi_arvalid_o (m_axi_arvalid_o), .m_axi_arready_i (m_axi_arready_i),
      .m_axi_rdata_i (m_axi_rdata_i), .m_axi_rresp_i (m_axi_rresp_i), .m_axi_rlast_i (m_axi_rlast_i), .m_axi_rvalid_i (m_axi_rvalid_i), .m_axi_rready_o (m_axi_rready_o)
  );

  conv_engine #(.N(N)) u_conv (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (conv_start), .desc_i (eng_desc), .busy_o (conv_busy), .done_o (conv_done), .array_valid_o (array_valid),
      .abuf_rd_en_o (c_abuf_rd_en), .abuf_rd_addr_o (c_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .abuf_wr_en_o (c_abuf_wr_en), .abuf_wr_addr_o (c_abuf_wr_addr), .abuf_wr_data_o (c_abuf_wr_data), .abuf_wr_be_o (c_abuf_wr_be),
      .wbuf_rd_en_o (c_wbuf_rd_en), .wbuf_rd_addr_o (c_wbuf_rd_addr), .wbuf_rd_data_i (wbuf_rd_data), .wbuf_rd_valid_i (wbuf_rd_valid)
  );

  pool_engine #(.N(N)) u_pool (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (pool_start), .desc_i (eng_desc), .busy_o (pool_busy), .done_o (pool_done),
      .abuf_rd_en_o (p_abuf_rd_en), .abuf_rd_addr_o (p_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .abuf_wr_en_o (p_abuf_wr_en), .abuf_wr_addr_o (p_abuf_wr_addr), .abuf_wr_data_o (p_abuf_wr_data), .abuf_wr_be_o (p_abuf_wr_be)
  );

  argmax_engine #(.N(N)) u_argmax (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (argmax_start), .desc_i (eng_desc), .busy_o (argmax_busy), .done_o (argmax_done),
      .abuf_rd_en_o (a_abuf_rd_en), .abuf_rd_addr_o (a_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .result_valid_o (argmax_valid), .result_idx_o (argmax_idx), .logits_o (argmax_logits)
  );

  host_mem_adapter u_hma (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .hm_req_i (hm_req), .hm_we_i (hm_we), .hm_sel_i (hm_sel), .hm_addr_i (hm_addr), .hm_wdata_i (hm_wdata), .hm_be_i (hm_be),
      .hm_ack_o (hm_ack), .hm_rdata_o (hm_rdata),
      .abuf_rd_en_o (h_abuf_rd_en), .abuf_rd_addr_o (h_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .abuf_wr_en_o (h_abuf_wr_en), .abuf_wr_addr_o (h_abuf_wr_addr), .abuf_wr_data_o (h_abuf_wr_data), .abuf_wr_be_o (h_abuf_wr_be),
      .wbuf_rd_en_o (h_wbuf_rd_en), .wbuf_rd_addr_o (h_wbuf_rd_addr), .wbuf_rd_data_i (wbuf_rd_data), .wbuf_rd_valid_i (wbuf_rd_valid),
      .wbuf_wr_en_o (h_wbuf_wr_en), .wbuf_wr_addr_o (h_wbuf_wr_addr), .wbuf_wr_data_o (h_wbuf_wr_data), .wbuf_wr_be_o (h_wbuf_wr_be)
  );

  // ---------------------------------------------------------------------------
  // Port muxes: the active engine owns the ports; the DMA owns them while it
  // runs (only started while the NPU is idle); the host gets them otherwise
  // (npu_regs already rejects host memory accesses while busy or DMA busy);
  // the pixel stream writes when nobody else writes.
  // ---------------------------------------------------------------------------
  always_comb begin
    wbuf_wr_en = h_wbuf_wr_en; wbuf_wr_addr = h_wbuf_wr_addr; wbuf_wr_data = h_wbuf_wr_data; wbuf_wr_be = h_wbuf_wr_be;
    if (conv_busy) begin
      abuf_rd_en = c_abuf_rd_en; abuf_rd_addr = c_abuf_rd_addr;
      abuf_wr_en = c_abuf_wr_en; abuf_wr_addr = c_abuf_wr_addr; abuf_wr_data = c_abuf_wr_data; abuf_wr_be = c_abuf_wr_be;
      wbuf_rd_en = c_wbuf_rd_en; wbuf_rd_addr = c_wbuf_rd_addr;
    end else if (pool_busy) begin
      abuf_rd_en = p_abuf_rd_en; abuf_rd_addr = p_abuf_rd_addr;
      abuf_wr_en = p_abuf_wr_en; abuf_wr_addr = p_abuf_wr_addr; abuf_wr_data = p_abuf_wr_data; abuf_wr_be = p_abuf_wr_be;
      wbuf_rd_en = 1'b0;         wbuf_rd_addr = '0;
    end else if (argmax_busy) begin
      abuf_rd_en = a_abuf_rd_en; abuf_rd_addr = a_abuf_rd_addr;
      abuf_wr_en = 1'b0;         abuf_wr_addr = '0;            abuf_wr_data = '0;            abuf_wr_be = '0;
      wbuf_rd_en = 1'b0;         wbuf_rd_addr = '0;
    end else if (dma_busy) begin
      abuf_rd_en = d_abuf_rd_en; abuf_rd_addr = d_abuf_rd_addr;
      abuf_wr_en = d_abuf_wr_en; abuf_wr_addr = d_abuf_wr_addr; abuf_wr_data = d_abuf_wr_data; abuf_wr_be = d_abuf_wr_be;
      wbuf_rd_en = d_wbuf_rd_en; wbuf_rd_addr = d_wbuf_rd_addr;
      wbuf_wr_en = d_wbuf_wr_en; wbuf_wr_addr = d_wbuf_wr_addr; wbuf_wr_data = d_wbuf_wr_data; wbuf_wr_be = d_wbuf_wr_be;
    end else begin
      abuf_rd_en = h_abuf_rd_en; abuf_rd_addr = h_abuf_rd_addr;
      wbuf_rd_en = h_wbuf_rd_en; wbuf_rd_addr = h_wbuf_rd_addr;
      if (h_abuf_wr_en || busy) begin
        abuf_wr_en = h_abuf_wr_en; abuf_wr_addr = h_abuf_wr_addr; abuf_wr_data = h_abuf_wr_data; abuf_wr_be = h_abuf_wr_be;
      end else begin
        abuf_wr_en = pix_wr_en_i; abuf_wr_addr = pix_wr_addr_i; abuf_wr_data = pix_wr_data_i; abuf_wr_be = pix_wr_be_i;
      end
    end
  end

  assign pix_wr_ready_o = !busy && !dma_busy && !conv_busy && !pool_busy && !argmax_busy && !h_abuf_wr_en;

  act_buffer #(.N(N)) u_abuf (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .rd_en_i (abuf_rd_en), .rd_addr_i (abuf_rd_addr), .rd_data_o (abuf_rd_data), .rd_valid_o (abuf_rd_valid),
      .wr_en_i (abuf_wr_en), .wr_addr_i (abuf_wr_addr), .wr_data_i (abuf_wr_data), .wr_be_i (abuf_wr_be)
  );

  weight_buffer u_wbuf (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .rd_en_i (wbuf_rd_en), .rd_addr_i (wbuf_rd_addr), .rd_data_o (wbuf_rd_data), .rd_valid_o (wbuf_rd_valid),
      .wr_en_i (wbuf_wr_en), .wr_addr_i (wbuf_wr_addr), .wr_data_i (wbuf_wr_data), .wr_be_i (wbuf_wr_be)
  );

endmodule

`default_nettype wire
