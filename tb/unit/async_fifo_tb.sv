`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for async_fifo: two unrelated clocks (the ratio is
// swept), random stalls on both sides, a queue model for order and content,
// and checks that the FIFO never accepts more than DEPTH entries in flight
// and never shows data when the model is empty.
// Plusargs: +seed=<int> (default 1), +n=<words per phase> (default 4000).
module async_fifo_tb;

  localparam int unsigned WIDTH = 16;
  localparam int unsigned DEPTH = 16;
  localparam int unsigned AW    = $clog2(DEPTH);

  logic wr_clk = 1'b0, rd_clk = 1'b0;
  logic wr_rst_n = 1'b0, rd_rst_n = 1'b0;
  logic wr_valid = 1'b0, wr_ready, rd_valid, rd_ready = 1'b0;
  logic [WIDTH-1:0] wr_data = '0, rd_data;

  async_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) dut (
      .wr_clk_i (wr_clk), .wr_rst_ni (wr_rst_n), .wr_valid_i (wr_valid), .wr_ready_o (wr_ready), .wr_data_i (wr_data),
      .rd_clk_i (rd_clk), .rd_rst_ni (rd_rst_n), .rd_valid_o (rd_valid), .rd_ready_i (rd_ready), .rd_data_o (rd_data)
  );

  // Clock periods in ps, changed per phase.
  int wr_half = 5000, rd_half = 3300;
  always #(wr_half * 1ps) wr_clk = ~wr_clk;
  always #(rd_half * 1ps) rd_clk = ~rd_clk;

  int n_pushed = 0, n_popped = 0, n_errors = 0, n_checked = 0;
  logic [WIDTH-1:0] model[$];
  int wr_stall = 2, rd_stall = 2;   // stall with probability 1/(N+1); 0 = never stall
  logic wr_enable = 1'b1;

  // Write driver: decide on the falling edge; a transfer happens at the next
  // rising edge when valid and ready are both high there. ready is registered
  // in the DUT, so sampling it here is what the rising edge will see.
  logic [WIDTH-1:0] next_data = 16'h0001;
  logic wr_fire_pending = 1'b0;
  /* verilator lint_off BLKSEQ */
  always @(negedge wr_clk) begin
    if (wr_rst_n) begin
      if (wr_fire_pending) begin
        model.push_back(wr_data);
        n_pushed++;
        next_data = next_data * 16'd3 + 16'd7;
      end
      // the DUT must never accept a write with DEPTH entries stored
      if (wr_fire_pending && (dut.wr_bin_q - dut.rd_bin_q) > (AW+1)'(DEPTH)) begin
        n_errors++; $display("ERROR @%0t: overflow, occupancy %0d", $time, dut.wr_bin_q - dut.rd_bin_q);
      end
      wr_valid = wr_enable && ((wr_stall == 0) || ($urandom_range(wr_stall) != 0));
      wr_data  = next_data;
      wr_fire_pending = wr_valid && wr_ready;
    end
  end

  // Read side: decide rd_ready on the falling edge and remember the word that
  // the next rising edge will pop; check it on the following falling edge.
  logic rd_fire_pending = 1'b0;
  logic [WIDTH-1:0] rd_data_pending;
  always @(negedge rd_clk) begin
    if (rd_rst_n) begin
      if (rd_fire_pending) begin
        n_checked++;
        if (model.size() == 0) begin
          n_errors++; $display("ERROR @%0t: pop while the model is empty (underflow)", $time);
        end else begin
          logic [WIDTH-1:0] exp;
          exp = model.pop_front();
          if (rd_data_pending !== exp) begin n_errors++; if (n_errors < 10) $display("ERROR @%0t: popped %h, expected %h", $time, rd_data_pending, exp); end
        end
        n_popped++;
      end
      rd_ready = (rd_stall == 0) || ($urandom_range(rd_stall) != 0);
      rd_fire_pending = rd_valid && rd_ready;
      rd_data_pending = rd_data;
    end
  end
  /* verilator lint_on BLKSEQ */

  task automatic phase(input int wh, input int rh, input int ws, input int rs, input int words, input string what);
    int target;
    wr_half = wh; rd_half = rh; wr_stall = ws; rd_stall = rs;
    target = n_popped + words;
    while (n_popped < target) @(negedge rd_clk);
    $display("phase %s: wr %0d ps, rd %0d ps, pushed %0d popped %0d", what, 2 * wh, 2 * rh, n_pushed, n_popped);
  endtask

  int seed = 1, n_words = 4000;

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_words)) n_words = 4000;
    void'($urandom(seed));
    $display("async_fifo_tb: seed=%0d n=%0d", seed, n_words);

    #100ns;
    wr_rst_n = 1'b1; rd_rst_n = 1'b1;
    @(negedge wr_clk); @(negedge rd_clk);

    phase(5000, 3300, 2, 2, n_words, "rd faster");
    phase(3300, 5000, 2, 2, n_words, "wr faster, FIFO fills");
    phase(5000, 5010, 8, 1, n_words, "nearly equal, writer rarely stalls");
    phase(1700, 9100, 1, 3, n_words, "wr much faster");
    phase(9100, 1700, 3, 1, n_words, "rd much faster");
    phase(5000, 5000, 0, 0, n_words, "same period, no stalls");

    // drain
    wr_enable = 1'b0;
    repeat (2) @(negedge wr_clk);
    repeat (60) @(negedge rd_clk);
    if (model.size() != 0) begin n_errors++; $display("ERROR: %0d entries never came out", model.size()); end

    if (n_errors == 0 && n_popped > 0) begin
      $display("PASS: async_fifo_tb popped=%0d checked=%0d", n_popped, n_checked);
      $finish;
    end else begin
      $display("FAIL: async_fifo_tb errors=%0d", n_errors);
      $fatal(1, "async_fifo_tb failed");
    end
  end

  initial begin
    #200ms;
    $fatal(1, "TIMEOUT: async_fifo_tb did not finish");
  end

endmodule

`default_nettype wire
