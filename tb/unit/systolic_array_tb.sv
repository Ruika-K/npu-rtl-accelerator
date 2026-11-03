`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for systolic_array.
//   - Drives GEMMs Out[M][N] = In[M][K] x W[K][N] tile by tile exactly as the
//     conv_engine will: m_tiles of N rows, n_tiles of N columns, K beats per
//     tile, random idle cycles anywhere, and N-K idle cycles after a tile when
//     K < N so that the tile period is at least N.
//   - Unused lanes of a partial tile carry random garbage; their results are
//     ignored, as the engine's writer mask will do.
//   - Checks every drained value against an SV integer matrix multiply, the
//     drain column order, that drain_valid never drops mid-drain, and the exact
//     drain start cycle (N-1 edges after the last beat is consumed at the input).
//   - Phase 3 reads GEMM cases from model/gen_gemm_vectors.py (+gemm_vectors=)
//     and also cross-checks the SV reference against the Python results.
// Plusargs: +seed=<int> (default 1), +n=<random cases> (default 200).
module systolic_array_tb
  import npu_pkg::*;
;

  localparam int unsigned N     = ARR_N;
  localparam int unsigned IDX_W = $clog2(N);
  localparam int MAX_M = 64;
  localparam int MAX_K = 512;
  localparam int MAX_N = 32;
  localparam time CLK_PERIOD = 10ns;

  logic clk_i;
  logic rst_ni;
  logic in_valid_i, in_first_i, in_last_i;
  logic [N-1:0][ACT_W-1:0] act_i;
  logic [N-1:0][WGT_W-1:0] wgt_i;
  logic drain_valid_o;
  logic [IDX_W-1:0] drain_col_idx_o;
  logic [N-1:0][ACC_W-1:0] drain_data_o;

  systolic_array #(.N(N)) dut (
      .clk_i           (clk_i),
      .rst_ni          (rst_ni),
      .in_valid_i      (in_valid_i),
      .in_first_i      (in_first_i),
      .in_last_i       (in_last_i),
      .act_i           (act_i),
      .wgt_i           (wgt_i),
      .drain_valid_o   (drain_valid_o),
      .drain_col_idx_o (drain_col_idx_o),
      .drain_data_o    (drain_data_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  // Cycle index of the most recent rising edge.
  int unsigned cycle = 0;
  always_ff @(posedge clk_i) cycle <= cycle + 1;

  // ---------------------------------------------------------------------------
  // Current case
  // ---------------------------------------------------------------------------
  logic signed [ACT_W-1:0] a_mat   [MAX_M][MAX_K];
  logic signed [WGT_W-1:0] w_mat   [MAX_K][MAX_N];
  logic signed [ACC_W-1:0] exp_mat [MAX_M][MAX_N];
  int cur_m, cur_k, cur_n;

  int n_cases = 0, n_tiles = 0, n_checked = 0, n_errors = 0, n_vec_cases = 0;

  // Tiles in flight: (m0, n0) pushed at tile start, last-beat cycle at the last beat.
  int q_m0[$], q_n0[$];
  int unsigned q_lc[$];

  // ---------------------------------------------------------------------------
  // Monitor on the falling edge
  // ---------------------------------------------------------------------------
  logic mon_en = 1'b0;
  int exp_idx = -1;          // expected drain_col_idx_o, -1 when not draining
  int cur_m0, cur_n0;
  int unsigned cur_lc;

  // Blocking assignments are intended here: this is a checker, not a flop.
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      if (drain_valid_o) begin
        if (exp_idx < 0) begin
          if (q_m0.size() == 0) begin
            n_errors++;
            $display("ERROR @%0t: drain started with no tile in flight", $time);
          end else begin
            cur_m0 = q_m0.pop_front();
            cur_n0 = q_n0.pop_front();
            cur_lc = q_lc.pop_front();
            if (drain_col_idx_o !== '0) begin
              n_errors++;
              $display("ERROR @%0t: drain started at column %0d", $time, drain_col_idx_o);
            end
            if (cycle != cur_lc + N - 1) begin
              n_errors++;
              $display("ERROR @%0t: drain started at cycle %0d, expected %0d (last beat at %0d)",
                       $time, cycle, cur_lc + N - 1, cur_lc);
            end
            exp_idx = 0;
          end
        end
        if (exp_idx >= 0) begin
          if (int'(drain_col_idx_o) != exp_idx) begin
            n_errors++;
            if (n_errors < 20)
              $display("ERROR @%0t: drain column %0d, expected %0d", $time, drain_col_idx_o, exp_idx);
          end
          for (int i = 0; i < N; i++) begin
            if (cur_m0 + i < cur_m && cur_n0 + exp_idx < cur_n) begin
              n_checked++;
              if (drain_data_o[i] !== exp_mat[cur_m0 + i][cur_n0 + exp_idx]) begin
                n_errors++;
                if (n_errors < 20)
                  $display("ERROR @%0t: Out[%0d][%0d] = %0d, expected %0d (M=%0d K=%0d N=%0d)",
                           $time, cur_m0 + i, cur_n0 + exp_idx, $signed(drain_data_o[i]),
                           exp_mat[cur_m0 + i][cur_n0 + exp_idx], cur_m, cur_k, cur_n);
              end
            end
          end
          exp_idx++;
          if (exp_idx == N) exp_idx = -1;
        end
      end else if (exp_idx >= 0) begin
        n_errors++;
        $display("ERROR @%0t: drain_valid_o dropped after column %0d", $time, exp_idx - 1);
        exp_idx = -1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Drivers (falling edge, blocking assignments)
  // ---------------------------------------------------------------------------
  task automatic idle_beat();
    @(negedge clk_i);
    in_valid_i = 1'b0;
    in_first_i = 1'($urandom_range(1));   // garbage flags must be ignored
    in_last_i  = 1'($urandom_range(1));
    for (int i = 0; i < N; i++) begin
      act_i[i] = 8'($urandom());
      wgt_i[i] = 8'($urandom());
    end
  endtask

  task automatic compute_expected();
    for (int m = 0; m < cur_m; m++) begin
      for (int n = 0; n < cur_n; n++) begin
        int acc;
        acc = 0;
        for (int k = 0; k < cur_k; k++) acc += int'(a_mat[m][k]) * int'(w_mat[k][n]);
        exp_mat[m][n] = acc;
      end
    end
  endtask

  // Fill the used region of a_mat / w_mat. mode: 0 full random, 1 all -128,
  // 2 random extremes, 3 small values.
  function automatic logic signed [7:0] rand_val(input int mode);
    case (mode)
      1:       return -8'sd128;
      2:       return ($urandom_range(1) == 0) ? -8'sd128 : 8'sd127;
      3:       return 8'($urandom_range(6)) - 8'sd3;
      default: return 8'($urandom());
    endcase
  endfunction

  task automatic set_case(input int m, input int k, input int n, input int mode_a, input int mode_w);
    cur_m = m; cur_k = k; cur_n = n;
    for (int mm = 0; mm < m; mm++)
      for (int kk = 0; kk < k; kk++) a_mat[mm][kk] = rand_val(mode_a);
    for (int kk = 0; kk < k; kk++)
      for (int nn = 0; nn < n; nn++) w_mat[kk][nn] = rand_val(mode_w);
  endtask

  // Drive every tile of the current case, exactly as conv_engine will.
  task automatic run_case();
    int tiles_m, tiles_n;
    tiles_m = (cur_m + N - 1) / N;
    tiles_n = (cur_n + N - 1) / N;
    compute_expected();
    for (int nt = 0; nt < tiles_n; nt++) begin
      for (int mt = 0; mt < tiles_m; mt++) begin
        q_m0.push_back(mt * N);
        q_n0.push_back(nt * N);
        for (int k = 0; k < cur_k; k++) begin
          while ($urandom_range(4) == 0) idle_beat();
          @(negedge clk_i);
          in_valid_i = 1'b1;
          in_first_i = (k == 0);
          in_last_i  = (k == cur_k - 1);
          for (int i = 0; i < N; i++) begin
            act_i[i] = (mt * N + i < cur_m) ? a_mat[mt * N + i][k] : 8'($urandom());
            wgt_i[i] = (nt * N + i < cur_n) ? w_mat[k][nt * N + i] : 8'($urandom());
          end
          if (k == cur_k - 1) q_lc.push_back(cycle + 1);
        end
        for (int p = cur_k; p < N; p++) idle_beat();   // tile period >= N
        n_tiles++;
      end
    end
    n_cases++;
  endtask

  task automatic wait_drained();
    repeat (2 * N + 4) idle_beat();
    @(negedge clk_i);
    if (q_m0.size() != 0 || exp_idx >= 0) begin
      n_errors++;
      $display("ERROR @%0t: %0d tiles never drained (M=%0d K=%0d N=%0d)", $time, q_m0.size(), cur_m, cur_k, cur_n);
      q_m0.delete(); q_n0.delete(); q_lc.delete();
      exp_idx = -1;
    end
  endtask

  task automatic gen_random_case();
    int m, k, n, mode_a, mode_w;
    m = $urandom_range(MAX_M, 1);
    case ($urandom_range(9))
      0, 1, 2, 3, 4: k = $urandom_range(16, 1);
      5, 6, 7:       k = $urandom_range(128, 17);
      default:       k = $urandom_range(MAX_K, 129);
    endcase
    n = $urandom_range(MAX_N, 1);
    mode_a = ($urandom_range(5) < 4) ? 0 : $urandom_range(3, 1);
    mode_w = ($urandom_range(5) < 4) ? 0 : $urandom_range(3, 1);
    set_case(m, k, n, mode_a, mode_w);
  endtask

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  int seed = 1;
  int n_random = 200;
  string vec_path = "";

  initial begin
    int fd, code, v, m, k, n, py_mismatch;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 200;
    void'($value$plusargs("gemm_vectors=%s", vec_path));
    void'($urandom(seed));
    $display("systolic_array_tb: N=%0d seed=%0d n_random=%0d vectors='%s'", N, seed, n_random, vec_path);

    in_valid_i = 1'b0; in_first_i = 1'b0; in_last_i = 1'b0; act_i = '0; wgt_i = '0;
    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    // Phase 1: directed shapes.
    set_case(1, 1, 1, 0, 0);          run_case(); wait_drained();   // K = 1, single element
    set_case(8, 1, 8, 0, 0);          run_case(); wait_drained();   // K = 1, full tile, period exactly N
    set_case(8, 8, 8, 0, 0);          run_case(); wait_drained();   // K = N
    set_case(8, 7, 8, 0, 0);          run_case(); wait_drained();   // K = N - 1
    set_case(8, 9, 8, 0, 0);          run_case(); wait_drained();   // K = N + 1
    set_case(13, 25, 6, 0, 0);        run_case(); wait_drained();   // partial m and n tiles, conv1-like K
    set_case(28, 150, 16, 0, 0);      run_case(); wait_drained();   // conv2-like
    set_case(1, 400, 24, 0, 0);       run_case(); wait_drained();   // fc-like: one row, many n tiles
    set_case(MAX_M, MAX_K, MAX_N, 1, 1); run_case(); wait_drained(); // all -128, largest sums
    set_case(16, 300, 16, 2, 2);      run_case(); wait_drained();   // random extremes
    set_case(5, 3, 3, 3, 3);          run_case(); wait_drained();

    // Phase 2: random shapes and data.
    for (int c = 0; c < n_random; c++) begin
      gen_random_case();
      run_case();
      wait_drained();
    end

    // Phase 3: Python-generated GEMMs.
    if (vec_path != "") begin
      fd = $fopen(vec_path, "r");
      if (fd == 0) begin
        n_errors++;
        $display("ERROR: cannot open vector file '%s'", vec_path);
      end else begin
        while (!$feof(fd)) begin
          code = $fscanf(fd, "%d %d %d\n", m, k, n);
          if (code != 3) break;
          if (m > MAX_M || k > MAX_K || n > MAX_N) begin
            n_errors++;
            $display("ERROR: vector case %0dx%0dx%0d exceeds testbench limits", m, k, n);
            break;
          end
          cur_m = m; cur_k = k; cur_n = n;
          for (int mm = 0; mm < m; mm++)
            for (int kk = 0; kk < k; kk++) begin code = $fscanf(fd, "%d", v); a_mat[mm][kk] = 8'(v); end
          for (int kk = 0; kk < k; kk++)
            for (int nn = 0; nn < n; nn++) begin code = $fscanf(fd, "%d", v); w_mat[kk][nn] = 8'(v); end
          compute_expected();
          py_mismatch = 0;
          for (int mm = 0; mm < m; mm++)
            for (int nn = 0; nn < n; nn++) begin
              code = $fscanf(fd, "%d", v);
              if (exp_mat[mm][nn] != v) py_mismatch++;
            end
          if (py_mismatch != 0) begin
            n_errors++;
            $display("ERROR: SV reference disagrees with Python on %0d elements (M=%0d K=%0d N=%0d)",
                     py_mismatch, m, k, n);
          end
          n_vec_cases++;
          run_case();
          wait_drained();
        end
        $fclose(fd);
      end
    end

    if (n_errors == 0) begin
      $display("PASS: systolic_array_tb cases=%0d (python=%0d) tiles=%0d checked=%0d",
               n_cases, n_vec_cases, n_tiles, n_checked);
      $finish;
    end else begin
      $display("FAIL: systolic_array_tb errors=%0d cases=%0d checked=%0d", n_errors, n_cases, n_checked);
      $fatal(1, "systolic_array_tb failed");
    end
  end

  initial begin
    #2s;
    $fatal(1, "TIMEOUT: systolic_array_tb did not finish");
  end

endmodule

`default_nettype wire
