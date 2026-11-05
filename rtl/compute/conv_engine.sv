`default_nettype none

// CONV2D / FC engine (docs/architecture.md section 5).
//
// For each n_tile, each output row and each group of N output columns
// (an m_tile that never crosses an output row) it streams K_total beats
// through the systolic array: beat k = (ci, kh, kw) reads N consecutive input
// bytes starting at in_base + ci*H*W + (oh+kh-pad)*W + (ow0+kw-pad), lanes
// outside the image or the output row are replaced by z_in, and the matching
// weight word W[k][n_tile*N .. +N-1] from the weight buffer. Tiles with
// K_total < N are padded with idle beats so the tile period is at least N.
// The drained columns go through N requant units and are written back as N
// consecutive bytes of one output channel.
//
// Biases are prefetched per n_tile into one of two register banks (bank =
// n_tile[0]) during idle cycles, so the previous n_tile can still drain with
// its own biases. b_base must therefore be 8-byte aligned.
//
// Descriptor fields are trusted: the sequencer has validated them. FC arrives
// as a 1x1 CONV2D. Done is pulsed once the last column has been written.
module conv_engine
  import npu_pkg::*;
#(
  parameter int unsigned N = ARR_N
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,

  input  logic                      start_i,
  input  desc_t                     desc_i,
  output logic                      busy_o,
  output logic                      done_o,
  output logic                      array_valid_o,

  output logic                      abuf_rd_en_o,
  output logic [ABUF_AW-1:0]        abuf_rd_addr_o,
  input  logic [N-1:0][7:0]         abuf_rd_data_i,
  input  logic                      abuf_rd_valid_i,
  output logic                      abuf_wr_en_o,
  output logic [ABUF_AW-1:0]        abuf_wr_addr_o,
  output logic [N-1:0][7:0]         abuf_wr_data_o,
  output logic [N-1:0]              abuf_wr_be_o,

  output logic                      wbuf_rd_en_o,
  output logic [WBUF_WAW-1:0]       wbuf_rd_addr_o,
  input  logic [63:0]               wbuf_rd_data_i,
  input  logic                      wbuf_rd_valid_i
);

  localparam int unsigned LANE_W  = $clog2(N);
  localparam int unsigned IDX_W   = $clog2(N);
  localparam int unsigned BIAS_WORDS = N / 2;          // 64-bit words holding N int32 biases
  localparam int unsigned FLUSH_CYCLES = 2 * N + 8;    // skew + drain + requant + write + slack

  // ---------------------------------------------------------------------------
  // Layer parameters, registered at start
  // ---------------------------------------------------------------------------
  logic [15:0]             h_q, w_q, cin_q, cout_q;
  logic [3:0]              k_q, pad_q;
  logic [ABUF_AW-1:0]      in_base_q, out_base_q;
  logic [WBUF_WAW-1:0]     w_word_q, b_word_q;
  logic signed [ACC_W-1:0] rq_mult_q;
  logic [RQ_SHIFT_W-1:0]   rq_shift_q;
  logic signed [7:0]       z_in_q, z_out_q, act_min_q, act_max_q;

  // Derived, registered in SETUP (kk, k_total, ohw and period: in the setup
  // product pipeline below)
  logic [16:0]             oh_q, ow_q;        // output size
  logic [31:0]             hw_q, ohw_q;       // H*W, OH*OW
  logic [7:0]              kk_q;              // K * K
  logic [21:0]             k_total_q;         // C_in * K * K
  logic [12:0]             n_tiles_q;         // ceil(C_out / N)
  logic [13:0]             ow_tiles_q;        // ceil(OW / N)
  logic [21:0]             period_q;          // max(K_total, N)

  // ---------------------------------------------------------------------------
  // Generator state
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {S_IDLE, S_SETUP, S_PREFETCH, S_RUN, S_FLUSH} state_e;
  state_e state_q;

  logic [12:0] nt_q;        // n_tile
  logic [16:0] oh_i_q;      // output row
  logic [13:0] owt_q;       // output column tile
  logic [15:0] ci_q;
  logic [3:0]  kh_q, kw_q;
  logic [21:0] kidx_q;      // beat index inside the tile, 0 .. period-1
  logic [31:0] ci_off_q;    // ci * H * W
  logic [31:0] kh_off_q;    // kh * W
  logic signed [32:0] row_base_q;  // (oh - pad) * W, may be negative
  // Running addresses, so that no multiplier sits in the beat or tile path
  logic [19:0]         pad_w_q;     // pad * W, registered in SETUP
  logic [WBUF_WAW-1:0] w_nt_q;      // weight word of beat 0 of the n_tile: w_word + nt
  logic [WBUF_WAW-1:0] w_ptr_q;     // weight word of this beat: w_word + kidx * n_tiles + nt
  logic [ABUF_AW-1:0]  col_nt_q;    // out_base + nt * N * OHW
  logic [ABUF_AW-1:0]  col_row_q;   // out_base + nt * N * OHW + oh * OW
  logic [2:0]  pf_cnt_q;    // bias prefetch word counter
  logic [5:0]  flush_q;

  // Bias banks: [bank][column]
  logic [1:0][N-1:0][31:0] bias_q;

  // ---------------------------------------------------------------------------
  // Beat (generator output, registered)
  // ---------------------------------------------------------------------------
  logic              g_active_q;      // an activation / weight read is issued this cycle
  logic              g_first_q, g_last_q, g_tile_start_q;
  logic [ABUF_AW-1:0] g_addr_q;
  logic [WBUF_WAW-1:0] g_waddr_q;
  logic [N-1:0]      g_mask_q;
  // Tile description pushed to the queue at tile start
  logic [ABUF_AW-1:0] g_col_base_q;   // address of column 0 of the tile's first channel
  logic [N-1:0]      g_ow_mask_q;     // lanes inside the output row
  logic              g_bank_q;        // bias bank of the n_tile (nt[0])
  logic [N-1:0]      g_cmask_q;       // columns of the n_tile whose output channel exists

  // ---------------------------------------------------------------------------
  // Setup arithmetic (combinational on the registered parameters)
  // ---------------------------------------------------------------------------
  logic [16:0] oh_c, ow_c;
  assign oh_c      = 17'(h_q) + 17'(pad_q) * 17'd2 - 17'(k_q) + 17'd1;
  assign ow_c      = 17'(w_q) + 17'(pad_q) * 17'd2 - 17'(k_q) + 17'd1;

  // Per-beat address and mask (combinational on the counters)
  logic signed [32:0] a_c;            // lane 0 input address, may be negative or past the image
  logic signed [17:0] ih_c, iw0_c;
  logic               row_ok_c;
  logic [N-1:0]       mask_c, ow_mask_c;
  logic [16:0]        ow0_c;
  assign ow0_c  = 17'(owt_q) << LANE_W;
  assign ih_c   = 18'(oh_i_q) + 18'(kh_q) - 18'(pad_q);
  assign iw0_c  = 18'(ow0_c) + 18'(kw_q) - 18'(pad_q);
  assign row_ok_c = (ih_c >= 18'sd0) && (ih_c < 18'(h_q));
  assign a_c    = 33'(in_base_q) + 33'(ci_off_q) + row_base_q + 33'(kh_off_q) + 33'(iw0_c);
  always_comb begin
    for (int i = 0; i < N; i++) begin
      logic signed [17:0] iw;
      iw = iw0_c + 18'(i);
      ow_mask_c[i] = (ow0_c + 17'(i)) < ow_q;
      mask_c[i]    = row_ok_c && (iw >= 18'sd0) && (iw < 18'(w_q)) && ow_mask_c[i];
    end
  end

  // Output channels of the current n_tile that exist (nt*N + j < C_out). Kept
  // in the tile queue so that the drain side selects one bit instead of
  // adding and comparing channel numbers.
  logic [N-1:0] cmask_c;
  always_comb begin
    for (int j = 0; j < N; j++) cmask_c[j] = ((32'(nt_q) << LANE_W) + 32'(j)) < 32'(cout_q);
  end

  // Column base of the tile: out_base + (nt*N)*OHW + oh*OW + ow0, from the
  // running row base (all address sums wrap at the buffer width on purpose)
  logic [ABUF_AW-1:0] col_base_c;
  assign col_base_c = col_row_q + ABUF_AW'(ow0_c);

  logic last_kw_c, last_kh_c, last_beat_c, last_tile_c, last_owt_c, last_oh_c, active_c;
  assign active_c    = kidx_q < k_total_q;
  assign last_kw_c   = (kw_q == k_q - 4'd1);
  assign last_kh_c   = (kh_q == k_q - 4'd1);
  assign last_beat_c = (kidx_q == period_q - 22'd1);
  assign last_owt_c  = (owt_q == ow_tiles_q - 14'd1);
  assign last_oh_c   = (oh_i_q == oh_q - 17'd1);
  assign last_tile_c = last_owt_c && last_oh_c && (nt_q == n_tiles_q - 13'd1);

  assign busy_o = (state_q != S_IDLE);

  // Bias prefetch: one read per cycle while the beat read port is free; the
  // data returns a cycle later tagged with the word index.
  logic       pf_issue_c, pf_rd_q;
  logic [1:0] pf_rd_idx_q;
  logic [IDX_W-1:0] pf_col_c, pf_col1_c;
  assign pf_issue_c = (state_q == S_PREFETCH) && !g_active_q && (pf_cnt_q < 3'(BIAS_WORDS));
  assign pf_col_c   = IDX_W'({pf_rd_idx_q, 1'b0});
  assign pf_col1_c  = IDX_W'({pf_rd_idx_q, 1'b1});

  // The sequencer consumed op and stride; the address sums are truncated to
  // the buffer width on purpose (the sequencer bounded them).
  /* verilator lint_off UNUSEDSIGNAL */
  logic [5:0]  unused_desc;
  logic [17:0] unused_a_hi;
  assign unused_desc   = {desc_i.stride_m1, desc_i.op};
  assign unused_a_hi   = a_c[32:ABUF_AW];
  /* verilator lint_on UNUSEDSIGNAL */

  // Setup products, one multiply per stage: K*K, then C_in*K*K and OH*OW,
  // then max(K_total, N). They are recomputed every cycle from registers that
  // only change in IDLE and SETUP, and the first beat of a layer is issued at
  // least BIAS_WORDS + 2 cycles after SETUP (the bias prefetch sits between
  // them), so all three stages have settled before RUN reads them. Doing the
  // products in SETUP itself put a multiplier, a second multiplier and a
  // compare-and-select in one cycle, the longest path of the design on ECP5.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      kk_q <= '0; k_total_q <= '0; ohw_q <= '0; period_q <= '0;
    end else begin
      kk_q      <= 8'(k_q) * 8'(k_q);
      k_total_q <= 22'(cin_q) * 22'(kk_q);
      ohw_q     <= 32'(oh_q) * 32'(ow_q);
      period_q  <= (k_total_q < 22'(N)) ? 22'(N) : k_total_q;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_IDLE;
      h_q <= '0; w_q <= '0; cin_q <= '0; cout_q <= '0; k_q <= '0; pad_q <= '0;
      in_base_q <= '0; out_base_q <= '0; w_word_q <= '0; b_word_q <= '0;
      rq_mult_q <= '0; rq_shift_q <= '0; z_in_q <= '0; z_out_q <= '0; act_min_q <= '0; act_max_q <= '0;
      oh_q <= '0; ow_q <= '0; hw_q <= '0; n_tiles_q <= '0; ow_tiles_q <= '0;
      nt_q <= '0; oh_i_q <= '0; owt_q <= '0; ci_q <= '0; kh_q <= '0; kw_q <= '0; kidx_q <= '0;
      ci_off_q <= '0; kh_off_q <= '0; row_base_q <= '0; pf_cnt_q <= '0; flush_q <= '0;
      pad_w_q <= '0; w_nt_q <= '0; w_ptr_q <= '0; col_nt_q <= '0; col_row_q <= '0;
      bias_q <= '0; pf_rd_q <= 1'b0; pf_rd_idx_q <= '0;
      g_active_q <= 1'b0; g_first_q <= 1'b0; g_last_q <= 1'b0; g_tile_start_q <= 1'b0;
      g_addr_q <= '0; g_waddr_q <= '0; g_mask_q <= '0; g_col_base_q <= '0; g_ow_mask_q <= '0;
      g_bank_q <= 1'b0; g_cmask_q <= '0;
      done_o <= 1'b0;
    end else begin
      done_o         <= 1'b0;
      g_active_q     <= 1'b0;
      g_first_q      <= 1'b0;
      g_last_q       <= 1'b0;
      g_tile_start_q <= 1'b0;

      // bias capture, one cycle after each prefetch read
      pf_rd_q     <= pf_issue_c;
      pf_rd_idx_q <= pf_cnt_q[1:0];
      if (pf_rd_q && wbuf_rd_valid_i) begin
        bias_q[nt_q[0]][pf_col_c]  <= wbuf_rd_data_i[31:0];
        bias_q[nt_q[0]][pf_col1_c] <= wbuf_rd_data_i[63:32];
      end

      case (state_q)
        S_IDLE: begin
          if (start_i) begin
            h_q        <= desc_i.in_h;
            w_q        <= desc_i.in_w;
            cin_q      <= desc_i.c_in;
            cout_q     <= desc_i.c_out;
            k_q        <= desc_i.k;
            pad_q      <= desc_i.pad;
            in_base_q  <= desc_i.in_base[ABUF_AW-1:0];
            out_base_q <= desc_i.out_base[ABUF_AW-1:0];
            w_word_q   <= desc_i.w_base[WBUF_WAW+2:3];
            b_word_q   <= desc_i.b_base[WBUF_WAW+2:3];
            rq_mult_q  <= desc_i.rq_mult;
            rq_shift_q <= desc_i.rq_shift;
            z_in_q     <= desc_i.z_in;
            z_out_q    <= desc_i.z_out;
            act_min_q  <= desc_i.act_min;
            act_max_q  <= desc_i.act_max;
            state_q    <= S_SETUP;
          end
        end

        S_SETUP: begin
          oh_q       <= oh_c;
          ow_q       <= ow_c;
          hw_q       <= 32'(h_q) * 32'(w_q);
          n_tiles_q  <= 13'((32'(cout_q) + 32'(N - 1)) >> LANE_W);
          ow_tiles_q <= 14'((32'(ow_c) + 32'(N - 1)) >> LANE_W);
          pad_w_q    <= 20'(pad_q) * 20'(w_q);
          w_nt_q     <= w_word_q;
          col_nt_q   <= out_base_q;
          col_row_q  <= out_base_q;
          nt_q <= '0; oh_i_q <= '0; owt_q <= '0;
          pf_cnt_q <= '0;
          state_q <= S_PREFETCH;
        end

        S_PREFETCH: begin
          // BIAS_WORDS read cycles (waiting for the previous beat's read to
          // leave the port), then one cycle for the last capture
          if (pf_issue_c) pf_cnt_q <= pf_cnt_q + 3'd1;
          if (pf_cnt_q == 3'(BIAS_WORDS)) begin
            // start the first tile of this n_tile
            ci_q <= '0; kh_q <= '0; kw_q <= '0; kidx_q <= '0;
            ci_off_q <= '0; kh_off_q <= '0;
            row_base_q <= -33'(pad_w_q);     // oh is 0 at the start of every n_tile
            w_ptr_q    <= w_nt_q;
            state_q <= S_RUN;
          end
        end

        S_RUN: begin
          // emit this beat
          g_active_q     <= active_c;
          g_first_q      <= (kidx_q == 22'd0);
          g_last_q       <= (kidx_q == k_total_q - 22'd1);
          g_tile_start_q <= (kidx_q == 22'd0);
          g_addr_q       <= a_c[ABUF_AW-1:0];
          g_waddr_q      <= w_ptr_q;
          g_mask_q       <= mask_c;
          g_col_base_q   <= col_base_c;
          g_ow_mask_q    <= ow_mask_c;
          g_bank_q       <= nt_q[0];
          g_cmask_q      <= cmask_c;

          // advance the (ci, kh, kw) walk on active beats
          if (active_c) begin
            if (last_kw_c) begin
              kw_q <= '0;
              if (last_kh_c) begin
                kh_q     <= '0;
                kh_off_q <= '0;
                ci_q     <= ci_q + 16'd1;
                ci_off_q <= ci_off_q + hw_q;
              end else begin
                kh_q     <= kh_q + 4'd1;
                kh_off_q <= kh_off_q + 32'(w_q);
              end
            end else begin
              kw_q <= kw_q + 4'd1;
            end
          end

          // tile / row / n_tile advance
          if (last_beat_c) begin
            kidx_q  <= '0;
            w_ptr_q <= w_nt_q;               // next tile of the same n_tile
            ci_q <= '0; kh_q <= '0; kw_q <= '0; ci_off_q <= '0; kh_off_q <= '0;
            if (last_tile_c) begin
              flush_q <= 6'(FLUSH_CYCLES);
              state_q <= S_FLUSH;
            end else if (last_owt_c) begin
              owt_q <= '0;
              if (last_oh_c) begin
                oh_i_q     <= '0;
                row_base_q <= -33'(pad_w_q);
                nt_q       <= nt_q + 13'd1;
                w_nt_q     <= w_nt_q + WBUF_WAW'(1);
                col_nt_q   <= col_nt_q + ABUF_AW'(ohw_q << LANE_W);
                col_row_q  <= col_nt_q + ABUF_AW'(ohw_q << LANE_W);
                pf_cnt_q   <= '0;
                state_q    <= S_PREFETCH;
              end else begin
                oh_i_q     <= oh_i_q + 17'd1;
                row_base_q <= row_base_q + 33'(w_q);
                col_row_q  <= col_row_q + ABUF_AW'(ow_q);
              end
            end else begin
              owt_q <= owt_q + 14'd1;
            end
          end else begin
            kidx_q  <= kidx_q + 22'd1;
            w_ptr_q <= w_ptr_q + WBUF_WAW'(n_tiles_q);
          end
        end

        S_FLUSH: begin
          if (flush_q == 6'd0) begin
            done_o  <= 1'b1;
            state_q <= S_IDLE;
          end else begin
            flush_q <= flush_q - 6'd1;
          end
        end

        default: state_q <= S_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------------------
  // Memory reads driven by the beat
  // ---------------------------------------------------------------------------
  assign abuf_rd_en_o   = g_active_q;
  assign abuf_rd_addr_o = g_addr_q;
  assign wbuf_rd_en_o   = g_active_q || pf_issue_c;
  assign wbuf_rd_addr_o = g_active_q ? g_waddr_q
                                     : (b_word_q + WBUF_WAW'(32'(nt_q) * 32'(BIAS_WORDS) + 32'(pf_cnt_q)));

  // ---------------------------------------------------------------------------
  // Alignment of flags / masks with the activation data (ABUF_RD_LAT cycles)
  // and of the weight word (WBUF_RD_LAT cycles earlier)
  // ---------------------------------------------------------------------------
  logic [ABUF_RD_LAT-1:0]          d_active_q, d_first_q, d_last_q;
  logic [ABUF_RD_LAT-1:0][N-1:0]   d_mask_q;
  logic [63:0]                     w_hold_q;   // weight word delayed to the activation data

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      d_active_q <= '0; d_first_q <= '0; d_last_q <= '0; d_mask_q <= '0; w_hold_q <= '0;
    end else begin
      d_active_q <= {d_active_q[ABUF_RD_LAT-2:0], g_active_q};
      d_first_q  <= {d_first_q[ABUF_RD_LAT-2:0],  g_first_q};
      d_last_q   <= {d_last_q[ABUF_RD_LAT-2:0],   g_last_q};
      d_mask_q   <= {d_mask_q[ABUF_RD_LAT-2:0],   g_mask_q};
      if (wbuf_rd_valid_i) w_hold_q <= wbuf_rd_data_i;   // held for ABUF_RD_LAT - WBUF_RD_LAT = 1 cycle
    end
  end

  // Array inputs, registered
  logic               arr_valid_q, arr_first_q, arr_last_q;
  logic [N-1:0][7:0]  arr_act_q, arr_wgt_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      arr_valid_q <= 1'b0; arr_first_q <= 1'b0; arr_last_q <= 1'b0; arr_act_q <= '0; arr_wgt_q <= '0;
    end else begin
      arr_valid_q <= d_active_q[ABUF_RD_LAT-1];
      arr_first_q <= d_first_q[ABUF_RD_LAT-1];
      arr_last_q  <= d_last_q[ABUF_RD_LAT-1];
      for (int i = 0; i < N; i++) begin
        arr_act_q[i] <= d_mask_q[ABUF_RD_LAT-1][i] ? abuf_rd_data_i[i] : z_in_q;
        arr_wgt_q[i] <= w_hold_q[8*i +: 8];
      end
    end
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_rd_valid;
  assign unused_rd_valid = abuf_rd_valid_i;   // the delay line replaces it
  /* verilator lint_on UNUSEDSIGNAL */

  assign array_valid_o = arr_valid_q;

  // ---------------------------------------------------------------------------
  // Systolic array
  // ---------------------------------------------------------------------------
  logic              drain_valid;
  logic [IDX_W-1:0]  drain_idx;
  logic [N-1:0][31:0] drain_data;

  systolic_array #(.N(N)) u_array (
      .clk_i           (clk_i),
      .rst_ni          (rst_ni),
      .in_valid_i      (arr_valid_q),
      .in_first_i      (arr_first_q),
      .in_last_i       (arr_last_q),
      .act_i           (arr_act_q),
      .wgt_i           (arr_wgt_q),
      .drain_valid_o   (drain_valid),
      .drain_col_idx_o (drain_idx),
      .drain_data_o    (drain_data)
  );

  // ---------------------------------------------------------------------------
  // Tile queue: pushed at tile start, popped when its drain begins
  // ---------------------------------------------------------------------------
  localparam int unsigned TQ_DEPTH = 4;
  typedef struct packed {
    logic [ABUF_AW-1:0] col_base;
    logic [N-1:0]       ow_mask;
    logic [N-1:0]       cmask;
    logic               bank;
  } tile_t;
  tile_t [TQ_DEPTH-1:0] tq_q;
  logic [1:0] tq_wr_q, tq_rd_q;

  // Drain-side registers for the tile being drained
  logic [ABUF_AW-1:0] dr_col_addr_q;    // address of the column being requantized
  logic [N-1:0]       dr_ow_mask_q;
  logic [N-1:0]       dr_cmask_q;
  logic               dr_bank_q;
  logic               drain_start_c;
  assign drain_start_c = drain_valid && (drain_idx == '0);

  tile_t tq_head_c;
  assign tq_head_c = tq_q[tq_rd_q];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tq_q <= '0; tq_wr_q <= '0; tq_rd_q <= '0;
      dr_col_addr_q <= '0; dr_ow_mask_q <= '0; dr_cmask_q <= '0; dr_bank_q <= 1'b0;
    end else begin
      if (g_tile_start_q) begin
        tq_q[tq_wr_q] <= '{col_base: g_col_base_q, ow_mask: g_ow_mask_q, cmask: g_cmask_q, bank: g_bank_q};
        tq_wr_q <= tq_wr_q + 2'd1;
      end
      if (drain_start_c) begin
        dr_col_addr_q <= tq_head_c.col_base + ohw_q[ABUF_AW-1:0];   // address of column 1
        dr_ow_mask_q  <= tq_head_c.ow_mask;
        dr_cmask_q    <= tq_head_c.cmask;
        dr_bank_q     <= tq_head_c.bank;
        tq_rd_q       <= tq_rd_q + 2'd1;
      end else if (drain_valid) begin
        dr_col_addr_q <= dr_col_addr_q + ohw_q[ABUF_AW-1:0];
      end
    end
  end

  // The column being drained this cycle: the queue head on the first column,
  // the running registers afterwards.
  logic [ABUF_AW-1:0] col_addr_c;
  logic [N-1:0]       col_ow_mask_c;
  logic [N-1:0]       col_cmask_c;
  logic               col_bank_c;
  logic               col_wr_c;          // channel exists
  assign col_addr_c    = drain_start_c ? tq_head_c.col_base : dr_col_addr_q;
  assign col_ow_mask_c = drain_start_c ? tq_head_c.ow_mask  : dr_ow_mask_q;
  assign col_cmask_c   = drain_start_c ? tq_head_c.cmask    : dr_cmask_q;
  assign col_bank_c    = drain_start_c ? tq_head_c.bank     : dr_bank_q;
  assign col_wr_c      = drain_valid && col_cmask_c[drain_idx];

  // ---------------------------------------------------------------------------
  // Requantization, N lanes, and the write-back delay line
  // ---------------------------------------------------------------------------
  logic [N-1:0]      rq_valid;
  logic [N-1:0][7:0] rq_q;

  for (genvar i = 0; i < N; i++) begin : g_rq
    requant_unit u_rq (
        .clk_i     (clk_i),
        .rst_ni    (rst_ni),
        .valid_i   (drain_valid),
        .acc_i     (drain_data[i]),
        .bias_i    (bias_q[col_bank_c][drain_idx]),
        .mult_i    (rq_mult_q),
        .shift_i   (rq_shift_q),
        .z_out_i   (z_out_q),
        .act_min_i (act_min_q),
        .act_max_i (act_max_q),
        .valid_o   (rq_valid[i]),
        .q_o       (rq_q[i])
    );
  end

  logic [RQ_PIPE_LAT-1:0]              wr_pending_q;
  logic [RQ_PIPE_LAT-1:0][ABUF_AW-1:0] wr_addr_pipe_q;
  logic [RQ_PIPE_LAT-1:0][N-1:0]       wr_be_pipe_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_pending_q <= '0; wr_addr_pipe_q <= '0; wr_be_pipe_q <= '0;
    end else begin
      wr_pending_q   <= {wr_pending_q[RQ_PIPE_LAT-2:0], col_wr_c};
      wr_addr_pipe_q <= {wr_addr_pipe_q[RQ_PIPE_LAT-2:0], col_addr_c};
      wr_be_pipe_q   <= {wr_be_pipe_q[RQ_PIPE_LAT-2:0], col_ow_mask_c};
    end
  end

  assign abuf_wr_en_o   = wr_pending_q[RQ_PIPE_LAT-1] && (&rq_valid);
  assign abuf_wr_addr_o = wr_addr_pipe_q[RQ_PIPE_LAT-1];
  assign abuf_wr_be_o   = wr_be_pipe_q[RQ_PIPE_LAT-1];
  assign abuf_wr_data_o = rq_q;

endmodule

`default_nettype wire
