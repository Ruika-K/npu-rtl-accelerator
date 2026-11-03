`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for act_buffer.
//   - Random unaligned 8-byte writes with random byte enables and random
//     unaligned 8-byte reads, checked against a byte-array model.
//   - Reads and writes issued in the same cycle to different rows, a read
//     issued the cycle after a write to the same bytes, back-to-back reads
//     every cycle, and the fixed 2-cycle read latency.
// Plusargs: +seed=<int> (default 1), +n=<random ops> (default 20000).
module act_buffer_tb
  import npu_pkg::*;
;

  localparam int unsigned N  = ARR_N;
  localparam int unsigned AW = ABUF_AW;
  localparam time CLK_PERIOD = 10ns;

  logic clk_i, rst_ni;
  logic rd_en_i; logic [AW-1:0] rd_addr_i; logic [N-1:0][7:0] rd_data_o; logic rd_valid_o;
  logic wr_en_i; logic [AW-1:0] wr_addr_i; logic [N-1:0][7:0] wr_data_i; logic [N-1:0] wr_be_i;

  act_buffer dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .rd_en_i (rd_en_i), .rd_addr_i (rd_addr_i), .rd_data_o (rd_data_o), .rd_valid_o (rd_valid_o),
      .wr_en_i (wr_en_i), .wr_addr_i (wr_addr_i), .wr_data_i (wr_data_i), .wr_be_i (wr_be_i)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_checked = 0, n_errors = 0, n_writes = 0, n_reads = 0;

  // Byte model. Bytes never written hold 0 after the directed clear below.
  logic [7:0] model [ABUF_BYTES];

  // Expected read data queue (pushed when a read is issued, popped at rd_valid).
  logic [N*8-1:0] exp_q[$];
  logic [AW-1:0]  addr_q[$];

  // Latency check: rd_en delayed by 2 must equal rd_valid.
  logic [1:0] rd_pipe_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) rd_pipe_q <= '0;
    else         rd_pipe_q <= {rd_pipe_q[0], rd_en_i};
  end

  logic mon_en = 1'b0;
  logic [N*8-1:0] exp_val;
  logic [AW-1:0]  exp_addr;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      if (rd_valid_o !== rd_pipe_q[1]) begin
        n_errors++; $display("ERROR @%0t: rd_valid_o=%b expected %b (latency)", $time, rd_valid_o, rd_pipe_q[1]);
      end
      if (rd_valid_o) begin
        if (exp_q.size() == 0) begin
          n_errors++; $display("ERROR @%0t: rd_valid without a pending read", $time);
        end else begin
          exp_val  = exp_q.pop_front();
          exp_addr = addr_q.pop_front();
          n_checked++;
          if (rd_data_o !== exp_val) begin
            n_errors++;
            if (n_errors < 20)
              $display("ERROR @%0t: read @%h = %h, expected %h", $time, exp_addr, rd_data_o, exp_val);
          end
        end
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // Issue a read: expected value is the model content at issue time; the DUT
  // must return what was in the array at the SRAM edge, so a write issued in
  // the same cycle to the same bytes is excluded by the stimulus.
  task automatic issue_read(input logic [AW-1:0] a);
    logic [N*8-1:0] e;
    for (int i = 0; i < N; i++) e[8*i +: 8] = model[AW'(a + AW'(i))];
    exp_q.push_back(e);
    addr_q.push_back(a);
    rd_en_i   = 1'b1;
    rd_addr_i = a;
    n_reads++;
  endtask

  task automatic issue_write(input logic [AW-1:0] a, input logic [N*8-1:0] d, input logic [N-1:0] be);
    wr_en_i   = 1'b1;
    wr_addr_i = a;
    for (int i = 0; i < N; i++) begin
      wr_data_i[i] = d[8*i +: 8];
      if (be[i]) model[AW'(a + AW'(i))] = d[8*i +: 8];
    end
    wr_be_i = be;
    n_writes++;
  endtask

  // Rows touched by an N-byte access at address a: a>>3 and possibly +1.
  function automatic bit rows_overlap(input logic [AW-1:0] a, input logic [AW-1:0] b);
    logic [AW-4:0] ra0, ra1, rb0, rb1;
    ra0 = a[AW-1:3]; ra1 = AW-3'(a[AW-1:3] + 1);
    rb0 = b[AW-1:3]; rb1 = AW-3'(b[AW-1:3] + 1);
    return (ra0 == rb0) || (ra0 == rb1) || (ra1 == rb0) || (ra1 == rb1);
  endfunction

  int seed = 1;
  int n_random = 20000;

  initial begin
    logic [AW-1:0] a, b;
    logic [N*8-1:0] d;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 20000;
    void'($urandom(seed));
    $display("act_buffer_tb: seed=%0d n_random=%0d", seed, n_random);

    rd_en_i = 1'b0; rd_addr_i = '0; wr_en_i = 1'b0; wr_addr_i = '0; wr_data_i = '0; wr_be_i = '0;
    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    // Clear the whole buffer with aligned full writes so the model and the
    // array agree everywhere.
    for (int r = 0; r < ABUF_BYTES / N; r++) begin
      @(negedge clk_i);
      rd_en_i = 1'b0;
      issue_write(AW'(r * N), '0, '1);
    end
    for (int i = 0; i < ABUF_BYTES; i++) model[i] = 8'h0;

    // Directed: every rotation, write then read the cycle after, read the same cycle as an unrelated write.
    for (int rot = 0; rot < N; rot++) begin
      a = AW'(16'h0100 + rot);
      @(negedge clk_i); rd_en_i = 1'b0; wr_en_i = 1'b0;
      d = {8'h88, 8'h77, 8'h66, 8'h55, 8'h44, 8'h33, 8'h22, 8'h11} ^ {8{8'(rot)}};
      issue_write(a, d, '1);
      @(negedge clk_i); wr_en_i = 1'b0;
      issue_read(a);                                   // read the cycle after the write
      @(negedge clk_i); rd_en_i = 1'b0;
      issue_read(AW'(a - 3));                          // overlapping, different rotation
      issue_write(AW'(16'h2000 + rot * 8 + 5), ~d, N'($urandom()));   // unrelated row, same cycle
      @(negedge clk_i); rd_en_i = 1'b0; wr_en_i = 1'b0;
    end
    // Edge of the buffer: wrap is allowed but must be consistent with the model.
    @(negedge clk_i); issue_write(AW'(ABUF_BYTES - 3), 64'hA1A2_A3A4_A5A6_A7A8, '1);
    @(negedge clk_i); wr_en_i = 1'b0; issue_read(AW'(ABUF_BYTES - 3));
    @(negedge clk_i); rd_en_i = 1'b0; issue_read(AW'(0));
    @(negedge clk_i); rd_en_i = 1'b0;

    // Random: each cycle maybe a read, maybe a write, never touching the same rows.
    for (int i = 0; i < n_random; i++) begin
      @(negedge clk_i);
      rd_en_i = 1'b0; wr_en_i = 1'b0;
      a = ($urandom_range(3) == 0) ? AW'($urandom()) : AW'($urandom_range(1023));
      if ($urandom_range(2) != 0) issue_read(a);
      if ($urandom_range(2) != 0) begin
        b = ($urandom_range(3) == 0) ? AW'($urandom()) : AW'($urandom_range(1023));
        if (!rd_en_i || !rows_overlap(a, b)) issue_write(b, {2{$urandom()}}, N'($urandom()));
      end
    end
    @(negedge clk_i); rd_en_i = 1'b0; wr_en_i = 1'b0;
    repeat (4) @(negedge clk_i);

    if (exp_q.size() != 0) begin n_errors++; $display("ERROR: %0d reads never returned", exp_q.size()); end

    if (n_errors == 0) begin
      $display("PASS: act_buffer_tb checked=%0d writes=%0d reads=%0d", n_checked, n_writes, n_reads);
      $finish;
    end else begin
      $display("FAIL: act_buffer_tb errors=%0d checked=%0d", n_errors, n_checked);
      $fatal(1, "act_buffer_tb failed");
    end
  end

  initial begin
    #100ms;
    $fatal(1, "TIMEOUT: act_buffer_tb did not finish");
  end

endmodule

`default_nettype wire
