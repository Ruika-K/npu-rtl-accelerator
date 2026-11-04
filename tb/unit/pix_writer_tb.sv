`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for pix_writer: random frames (random base, length
// not a multiple of N, random gaps and random port stalls) written into a
// byte model through a modelled N-byte write port; sof restart and a byte
// before any sof.
// Plusargs: +seed=<int> (default 1), +n=<frames> (default 200).
module pix_writer_tb
  import npu_pkg::*;
;

  localparam int unsigned N  = ARR_N;
  localparam int unsigned AW = ABUF_AW;
  localparam time CLK_PERIOD = 10ns;

  logic clk_i = 1'b0, rst_ni = 1'b0;
  logic en_i = 1'b1;
  logic [AW-1:0] base_i = '0, len_i = '0;
  logic fifo_valid_i = 1'b0, fifo_ready_o;
  logic [8:0] fifo_data_i = '0;
  logic wr_en_o; logic [AW-1:0] wr_addr_o; logic [N-1:0][7:0] wr_data_o; logic [N-1:0] wr_be_o;
  logic wr_ready_i = 1'b1;
  logic frame_done_o, sof_err_o;

  pix_writer dut (
      .clk_i (clk_i), .rst_ni (rst_ni), .en_i (en_i), .base_i (base_i), .len_i (len_i),
      .fifo_valid_i (fifo_valid_i), .fifo_ready_o (fifo_ready_o), .fifo_data_i (fifo_data_i),
      .wr_en_o (wr_en_o), .wr_addr_o (wr_addr_o), .wr_data_o (wr_data_o), .wr_be_o (wr_be_o), .wr_ready_i (wr_ready_i),
      .frame_done_o (frame_done_o), .sof_err_o (sof_err_o)
  );

  always #(CLK_PERIOD / 2) clk_i = ~clk_i;

  int n_frames = 0, n_errors = 0, n_checked = 0, done_pulses = 0, sof_errs = 0;
  logic [7:0] mem [ABUF_BYTES];

  // Memory model and port stall on the falling edge: a write happened at the
  // previous rising edge when wr_en_o was high there (wr_en_o = flush && ready).
  logic wr_fire_pending = 1'b0; logic [AW-1:0] a_pend; logic [N-1:0][7:0] d_pend; logic [N-1:0] be_pend;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (rst_ni) begin
      if (wr_fire_pending) for (int i = 0; i < N; i++) if (be_pend[i]) mem[AW'(a_pend + AW'(i))] = d_pend[i];
      if (frame_done_o) done_pulses++;
      if (sof_err_o) sof_errs++;
      wr_ready_i = ($urandom_range(2) != 0);
      wr_fire_pending = wr_en_o;   // sampled after ready settled? no: wr_en_o depends on ready; recompute below
      #1;
      wr_fire_pending = wr_en_o; a_pend = wr_addr_o; d_pend = wr_data_o; be_pend = wr_be_o;
    end
  end
  /* verilator lint_on BLKSEQ */

  // Source: present one {sof, byte}; hold until accepted.
  task automatic push(input bit sof, input logic [7:0] b);
    bit acc;
    @(negedge clk_i);
    fifo_valid_i = 1'b1; fifo_data_i = {sof, b};
    forever begin
      #2; acc = fifo_valid_i && fifo_ready_o;
      @(posedge clk_i); @(negedge clk_i);
      if (acc) break;
    end
    fifo_valid_i = 1'b0;
    while ($urandom_range(3) == 0) @(negedge clk_i);
  endtask

  task automatic send_frame(input int base, input int len, output logic [7:0] data [4096]);
    base_i = AW'(base); len_i = AW'(len);
    for (int i = 0; i < len; i++) begin
      data[i] = 8'($urandom());
      push(i == 0, data[i]);
    end
  endtask

  // Frames are counted, not edge-detected: the done pulse of a short frame can
  // land before this task is entered.
  int expected_done = 0;
  task automatic wait_done();
    int guard = 0;
    expected_done++;
    while (done_pulses < expected_done && guard < 10000) begin @(negedge clk_i); guard++; end
    if (guard >= 10000) begin n_errors++; $display("ERROR: no frame_done (have %0d, expected %0d)", done_pulses, expected_done); end
    @(negedge clk_i);
  endtask

  task automatic check_frame(input int base, input int len, ref logic [7:0] data [4096]);
    int mism = 0;
    for (int i = 0; i < len; i++) begin n_checked++; if (mem[AW'(base + i)] !== data[i]) mism++; end
    // guard bytes around the frame must be untouched (0xEE)
    for (int i = 1; i <= N; i++) begin
      n_checked++;
      if (mem[AW'(base - i)] !== 8'hEE) mism++;
      if (mem[AW'(base + len + i - 1)] !== 8'hEE) mism++;
    end
    if (mism != 0) begin n_errors++; $display("ERROR: frame base %0d len %0d: %0d bytes wrong", base, len, mism); end
    n_frames++;
  endtask

  int seed = 1, n_random = 200;
  logic [7:0] fdata [4096];

  int lens [6] = '{1, N - 1, N, N + 1, 2 * N, 784};

  initial begin
    int base, len, e0;
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 200;
    void'($urandom(seed));
    for (int i = 0; i < ABUF_BYTES; i++) mem[i] = 8'hEE;
    repeat (3) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);

    // directed: lengths 1, N-1, N, N+1, 2N, 784 at odd bases
    for (int k = 0; k < 6; k++) begin
      len = lens[k];
      base = 100 + 37 * k;
      send_frame(base, len, fdata); wait_done(); check_frame(base, len, fdata);
      for (int i = 0; i < len; i++) mem[AW'(base + i)] = 8'hEE;
    end
    // byte before any sof is dropped and flagged; the next proper frame is fine
    begin
      e0 = sof_errs;
      push(1'b0, 8'h55);
      @(negedge clk_i); @(negedge clk_i);
      if (sof_errs != e0 + 1) begin n_errors++; $display("ERROR: byte before sof not flagged"); end
      send_frame(500, 20, fdata); wait_done(); check_frame(500, 20, fdata);
      for (int i = 0; i < 20; i++) mem[AW'(500 + i)] = 8'hEE;
    end
    // sof in the middle of a frame restarts it
    begin
      e0 = sof_errs;
      base_i = AW'(700); len_i = AW'(12);
      push(1'b1, 8'h01); push(1'b0, 8'h02); push(1'b0, 8'h03);        // 3 bytes of a frame that is abandoned
      for (int i = 0; i < 12; i++) begin fdata[i] = 8'(16'hA0 + i); push(i == 0, fdata[i]); end
      wait_done();
      if (sof_errs != e0 + 1) begin n_errors++; $display("ERROR: mid-frame sof not flagged (%0d)", sof_errs - e0); end
      check_frame(700, 12, fdata);
      for (int i = 0; i < 12; i++) mem[AW'(700 + i)] = 8'hEE;
    end
    // random frames
    for (int f = 0; f < n_random; f++) begin
      len = $urandom_range(64, 1);
      base = $urandom_range(ABUF_BYTES - 128, 16);
      send_frame(base, len, fdata); wait_done(); check_frame(base, len, fdata);
      for (int i = 0; i < len; i++) mem[AW'(base + i)] = 8'hEE;
    end

    if (n_errors == 0) begin
      $display("PASS: pix_writer_tb frames=%0d checked=%0d done_pulses=%0d", n_frames, n_checked, done_pulses);
      $finish;
    end else begin
      $display("FAIL: pix_writer_tb errors=%0d frames=%0d", n_errors, n_frames);
      $fatal(1, "pix_writer_tb failed");
    end
  end

  initial begin
    #100ms;
    $fatal(1, "TIMEOUT: pix_writer_tb did not finish");
  end

endmodule

`default_nettype wire
