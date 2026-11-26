`default_nettype none

// Assertions for npu_dma, bound into every instance: AXI4 master protocol on
// the m_axi port, the burst rules the DMA promises (64-bit INCR bursts of at
// most MAX_BURST beats that never cross a 4 KB boundary), and the transfer
// length. m_* constrain the AXI4 slave, a_* are guarantees, c_* scenarios.
module npu_dma_sva #(
  parameter int unsigned MAX_BURST = 16
) (
  input logic        clk_i,
  input logic        rst_ni,
  input logic        start_i,
  input logic [19:0] len_i,
  input logic        busy_o,
  input logic        done_o,
  input logic        err_o,
  input logic        abuf_wr_en_o,
  input logic        wbuf_wr_en_o,
  input logic        abuf_rd_en_o,
  input logic        wbuf_rd_en_o,
  input logic [31:0] m_axi_awaddr_o,
  input logic [7:0]  m_axi_awlen_o,
  input logic [2:0]  m_axi_awsize_o,
  input logic [1:0]  m_axi_awburst_o,
  input logic        m_axi_awvalid_o,
  input logic        m_axi_awready_i,
  input logic [63:0] m_axi_wdata_o,
  input logic [7:0]  m_axi_wstrb_o,
  input logic        m_axi_wlast_o,
  input logic        m_axi_wvalid_o,
  input logic        m_axi_wready_i,
  /* verilator lint_off UNUSEDSIGNAL */   // xRESP[1] alone tells an error
  input logic [1:0]  m_axi_bresp_i,
  /* verilator lint_on UNUSEDSIGNAL */
  input logic        m_axi_bvalid_i,
  input logic        m_axi_bready_o,
  input logic [31:0] m_axi_araddr_o,
  input logic [7:0]  m_axi_arlen_o,
  input logic [2:0]  m_axi_arsize_o,
  input logic [1:0]  m_axi_arburst_o,
  input logic        m_axi_arvalid_o,
  input logic        m_axi_arready_i,
  /* verilator lint_off UNUSEDSIGNAL */   // xRESP[1] alone tells an error
  input logic [1:0]  m_axi_rresp_i,
  /* verilator lint_on UNUSEDSIGNAL */
  input logic        m_axi_rlast_i,
  input logic        m_axi_rvalid_i,
  input logic        m_axi_rready_o,
  input logic [19:0] remaining_q
);

  logic aw_hs_c, w_hs_c, b_hs_c, ar_hs_c, r_hs_c;
  assign aw_hs_c = m_axi_awvalid_o && m_axi_awready_i;
  assign w_hs_c  = m_axi_wvalid_o && m_axi_wready_i;
  assign b_hs_c  = m_axi_bvalid_i && m_axi_bready_o;
  assign ar_hs_c = m_axi_arvalid_o && m_axi_arready_i;
  assign r_hs_c  = m_axi_rvalid_i && m_axi_rready_o;

  // Reference model of the burst in flight and of the whole transfer
  logic        rd_out_q, wr_out_q;      // a read / write burst is outstanding
  logic        w_done_q;                // all W beats of the write burst sent
  logic [7:0]  len_q, beat_q;           // AxLEN and beats seen of the burst
  logic [19:0] xfer_len_q;              // len_i of the transfer
  logic [17:0] xfer_beats_q;            // beats moved since start
  logic        failed_q;                // an error response was seen

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_out_q <= 1'b0; wr_out_q <= 1'b0; w_done_q <= 1'b0; len_q <= '0; beat_q <= '0;
      xfer_len_q <= '0; xfer_beats_q <= '0; failed_q <= 1'b0;
    end else begin
      if (start_i && !busy_o) begin
        xfer_len_q <= len_i; xfer_beats_q <= '0; failed_q <= 1'b0;
      end
      if (ar_hs_c) begin rd_out_q <= 1'b1; len_q <= m_axi_arlen_o; beat_q <= '0; end
      if (aw_hs_c) begin wr_out_q <= 1'b1; w_done_q <= 1'b0; len_q <= m_axi_awlen_o; beat_q <= '0; end
      if (r_hs_c) begin
        beat_q <= beat_q + 8'd1; xfer_beats_q <= xfer_beats_q + 18'd1;
        if (m_axi_rresp_i[1]) failed_q <= 1'b1;
        if (m_axi_rlast_i) rd_out_q <= 1'b0;
      end
      if (w_hs_c) begin
        beat_q <= beat_q + 8'd1; xfer_beats_q <= xfer_beats_q + 18'd1;
        if (m_axi_wlast_o) w_done_q <= 1'b1;
      end
      if (b_hs_c) begin
        wr_out_q <= 1'b0;
        if (m_axi_bresp_i[1]) failed_q <= 1'b1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Environment: AXI4 slave
  // ---------------------------------------------------------------------------
  m_r_expected: assume property (@(posedge clk_i) disable iff (!rst_ni) m_axi_rvalid_i |-> rd_out_q)
    else $error("R beat without an outstanding read burst");
  m_r_last: assume property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_rvalid_i |-> m_axi_rlast_i == (beat_q == len_q))
    else $error("RLAST not on the last beat of the burst");
  m_b_expected: assume property (@(posedge clk_i) disable iff (!rst_ni) m_axi_bvalid_i |-> wr_out_q && w_done_q)
    else $error("B response before the last W beat");

  // ---------------------------------------------------------------------------
  // Guarantees: channel handshakes
  // ---------------------------------------------------------------------------
  a_aw_hold: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_awvalid_o && !m_axi_awready_i |=> m_axi_awvalid_o && $stable({m_axi_awaddr_o, m_axi_awlen_o}))
    else $error("AW dropped or changed before AWREADY");
  a_ar_hold: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_arvalid_o && !m_axi_arready_i |=> m_axi_arvalid_o && $stable({m_axi_araddr_o, m_axi_arlen_o}))
    else $error("AR dropped or changed before ARREADY");
  a_w_hold: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_wvalid_o && !m_axi_wready_i |=> m_axi_wvalid_o && $stable({m_axi_wdata_o, m_axi_wlast_o}))
    else $error("W dropped or changed before WREADY");

  // Guarantees: burst shape
  a_aw_legal: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_awvalid_o |-> m_axi_awsize_o == 3'b011 && m_axi_awburst_o == 2'b01 && m_axi_awaddr_o[2:0] == 3'b000
                          && 32'(m_axi_awlen_o) < MAX_BURST
                          && 32'(m_axi_awaddr_o[11:0]) + ((32'(m_axi_awlen_o) + 32'd1) << 3) <= 32'h1000)
    else $error("illegal write burst: size, type, alignment, length or 4 KB crossing");
  a_ar_legal: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_arvalid_o |-> m_axi_arsize_o == 3'b011 && m_axi_arburst_o == 2'b01 && m_axi_araddr_o[2:0] == 3'b000
                          && 32'(m_axi_arlen_o) < MAX_BURST
                          && 32'(m_axi_araddr_o[11:0]) + ((32'(m_axi_arlen_o) + 32'd1) << 3) <= 32'h1000)
    else $error("illegal read burst: size, type, alignment, length or 4 KB crossing");
  a_w_shape: assert property (@(posedge clk_i) disable iff (!rst_ni)
      m_axi_wvalid_o |-> wr_out_q && !w_done_q && m_axi_wstrb_o == 8'hFF && m_axi_wlast_o == (beat_q == len_q))
    else $error("W beat outside a write burst, partial strobe, or WLAST misplaced");

  // Guarantees: one burst at a time, ready only for an expected beat
  a_one_burst: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (rd_out_q || wr_out_q) |-> !m_axi_arvalid_o && !m_axi_awvalid_o)
    else $error("address issued while a burst is outstanding");
  a_rready: assert property (@(posedge clk_i) disable iff (!rst_ni) m_axi_rready_o |-> rd_out_q)
    else $error("RREADY without an outstanding read burst");
  a_bready: assert property (@(posedge clk_i) disable iff (!rst_ni) m_axi_bready_o |-> wr_out_q && w_done_q)
    else $error("BREADY before the last W beat");

  // Guarantees: transfer level
  a_idle_quiet: assert property (@(posedge clk_i) disable iff (!rst_ni)
      !busy_o |-> !m_axi_awvalid_o && !m_axi_arvalid_o && !m_axi_wvalid_o && !m_axi_rready_o && !m_axi_bready_o
                  && !abuf_wr_en_o && !wbuf_wr_en_o && !abuf_rd_en_o && !wbuf_rd_en_o)
    else $error("bus or memory activity while idle");
  a_done_err: assert property (@(posedge clk_i) disable iff (!rst_ni) !(done_o && err_o))
    else $error("done and err together");
  a_done_len: assert property (@(posedge clk_i) disable iff (!rst_ni)
      done_o |-> !failed_q && {xfer_beats_q, 3'b000} == 21'(xfer_len_q))
    else $error("done after an error response or with a byte count other than len");
  a_err_cause: assert property (@(posedge clk_i) disable iff (!rst_ni)
      err_o |-> failed_q || xfer_beats_q == '0)
    else $error("err without an error response or a rejected start");
  a_mem_wr_on_r: assert property (@(posedge clk_i) disable iff (!rst_ni)
      (abuf_wr_en_o || wbuf_wr_en_o) |-> r_hs_c && !(abuf_wr_en_o && wbuf_wr_en_o))
    else $error("buffer write without an R beat, or to both buffers");

  // ---------------------------------------------------------------------------
  // Scenarios
  // ---------------------------------------------------------------------------
  c_rd_burst_max:    cover property (@(posedge clk_i) disable iff (!rst_ni) ar_hs_c && 32'(m_axi_arlen_o) == MAX_BURST - 1);
  c_wr_burst_max:    cover property (@(posedge clk_i) disable iff (!rst_ni) aw_hs_c && 32'(m_axi_awlen_o) == MAX_BURST - 1);
  c_rd_burst_single: cover property (@(posedge clk_i) disable iff (!rst_ni) ar_hs_c && m_axi_arlen_o == 8'd0);
  c_wr_burst_single: cover property (@(posedge clk_i) disable iff (!rst_ni) aw_hs_c && m_axi_awlen_o == 8'd0);
  // a burst cut short by the 4 KB boundary with more data to move after it
  c_rd_split_4k:     cover property (@(posedge clk_i) disable iff (!rst_ni)
      ar_hs_c && 32'(m_axi_arlen_o) < MAX_BURST - 1
      && 32'(remaining_q) > ((32'(m_axi_arlen_o) + 32'd1) << 3)
      && 32'(m_axi_araddr_o[11:0]) + ((32'(m_axi_arlen_o) + 32'd1) << 3) == 32'h1000);
  c_wr_split_4k:     cover property (@(posedge clk_i) disable iff (!rst_ni)
      aw_hs_c && 32'(m_axi_awlen_o) < MAX_BURST - 1
      && 32'(remaining_q) > ((32'(m_axi_awlen_o) + 32'd1) << 3)
      && 32'(m_axi_awaddr_o[11:0]) + ((32'(m_axi_awlen_o) + 32'd1) << 3) == 32'h1000);
  c_ar_stall:        cover property (@(posedge clk_i) disable iff (!rst_ni) m_axi_arvalid_o && !m_axi_arready_i);
  c_aw_stall:        cover property (@(posedge clk_i) disable iff (!rst_ni) m_axi_awvalid_o && !m_axi_awready_i);
  c_w_stall:         cover property (@(posedge clk_i) disable iff (!rst_ni) m_axi_wvalid_o && !m_axi_wready_i);
  c_r_stall:         cover property (@(posedge clk_i) disable iff (!rst_ni) m_axi_rready_o && !m_axi_rvalid_i);
  c_rd_error:        cover property (@(posedge clk_i) disable iff (!rst_ni) r_hs_c && m_axi_rresp_i[1]);
  c_wr_error:        cover property (@(posedge clk_i) disable iff (!rst_ni) b_hs_c && m_axi_bresp_i[1]);
  c_bad_param:       cover property (@(posedge clk_i) disable iff (!rst_ni) err_o && xfer_beats_q == '0 && !failed_q);

endmodule

bind npu_dma npu_dma_sva #(.MAX_BURST(MAX_BURST)) u_sva (.*);

`default_nettype wire
