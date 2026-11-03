`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for pe.
//   - random tiles of length K in 1..48 with first/last flags, random bubbles
//     carrying garbage data and garbage flags (must be ignored);
//   - shadow_o compared with an int32 reference at every last_o;
//   - forwarded a/w/flags checked as an exact one-cycle delay;
//   - directed tiles at the int8 extremes, including K = 1.
// Plusargs: +seed=<int> (default 1), +n=<tiles> (default 2000).
module pe_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i;
  logic rst_ni;
  logic valid_i, first_i, last_i;
  logic signed [ACT_W-1:0] a_i;
  logic signed [WGT_W-1:0] w_i;
  logic valid_o, first_o, last_o;
  logic signed [ACT_W-1:0] a_o;
  logic signed [WGT_W-1:0] w_o;
  logic signed [ACC_W-1:0] shadow_o;

  pe dut (
      .clk_i    (clk_i),
      .rst_ni   (rst_ni),
      .valid_i  (valid_i),
      .first_i  (first_i),
      .last_i   (last_i),
      .a_i      (a_i),
      .w_i      (w_i),
      .valid_o  (valid_o),
      .first_o  (first_o),
      .last_o   (last_o),
      .a_o      (a_o),
      .w_o      (w_o),
      .shadow_o (shadow_o)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_tiles   = 0;
  int n_checked = 0;
  int n_errors  = 0;
  logic signed [ACC_W-1:0] exp_q[$];

  // One-cycle delayed copies of the inputs for the forwarding check.
  logic valid_d, first_d, last_d;
  logic signed [ACT_W-1:0] a_d;
  logic signed [WGT_W-1:0] w_d;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_d <= 1'b0; first_d <= 1'b0; last_d <= 1'b0; a_d <= '0; w_d <= '0;
    end else begin
      valid_d <= valid_i; first_d <= first_i; last_d <= last_i; a_d <= a_i; w_d <= w_i;
    end
  end

  logic mon_en = 1'b0;
  logic signed [ACC_W-1:0] exp_val;
  // Blocking assignments are intended here: this is a checker, not a flop.
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      if (valid_o !== valid_d || first_o !== first_d || last_o !== last_d ||
          a_o !== a_d || w_o !== w_d) begin
        n_errors++;
        if (n_errors < 20)
          $display("ERROR @%0t: forwarding mismatch v/f/l/a/w = %b%b%b %0d %0d, expected %b%b%b %0d %0d",
                   $time, valid_o, first_o, last_o, a_o, w_o, valid_d, first_d, last_d, a_d, w_d);
      end
      if (valid_o && last_o) begin
        if (exp_q.size() == 0) begin
          n_errors++;
          $display("ERROR @%0t: last_o with no expected shadow queued", $time);
        end else begin
          exp_val = exp_q.pop_front();
          n_checked++;
          if (shadow_o !== exp_val) begin
            n_errors++;
            if (n_errors < 20)
              $display("ERROR @%0t: shadow_o=%0d expected %0d", $time, shadow_o, exp_val);
          end
        end
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // Inputs are driven on the falling edge with blocking assignments, so the
  // DUT samples them on the next rising edge in every simulator alike.
  task automatic bubble();
    @(negedge clk_i);
    valid_i = 1'b0;
    first_i = 1'($urandom_range(1));   // garbage flags must be ignored
    last_i  = 1'($urandom_range(1));
    a_i     = 8'($urandom());
    w_i     = 8'($urandom());
  endtask

  // Drive one tile; mode selects the operand distribution.
  task automatic tile(input int k_len, input int mode);
    logic signed [ACT_W-1:0] a;
    logic signed [WGT_W-1:0] w;
    logic signed [ACC_W-1:0] acc_ref;
    acc_ref = '0;
    for (int k = 0; k < k_len; k++) begin
      while ($urandom_range(4) == 0) bubble();
      case (mode)
        0: begin a = 8'($urandom()); w = 8'($urandom()); end
        1: begin a = -8'sd128; w = -8'sd128; end
        2: begin a = -8'sd128; w = 8'sd127; end
        default: begin a = 8'sd127; w = 8'sd127; end
      endcase
      @(negedge clk_i);
      valid_i = 1'b1;
      first_i = (k == 0);
      last_i  = (k == k_len - 1);
      a_i     = a;
      w_i     = w;
      acc_ref = (k == 0) ? ACC_W'(a * w) : acc_ref + ACC_W'(a * w);
    end
    exp_q.push_back(acc_ref);
    n_tiles++;
  endtask

  int seed = 1;
  int n_random = 2000;

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 2000;
    void'($urandom(seed));
    $display("pe_tb: seed=%0d n_random=%0d", seed, n_random);

    valid_i = 1'b0; first_i = 1'b0; last_i = 1'b0; a_i = '0; w_i = '0;
    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    // Directed: K = 1 (first and last in the same cycle) and the extremes.
    tile(1, 0);
    tile(1, 1);
    tile(48, 1);   // 48 * 16384 = 786432
    tile(48, 2);   // 48 * -16256
    tile(48, 3);
    tile(2, 0);

    for (int i = 0; i < n_random; i++) begin
      tile($urandom_range(48, 1), ($urandom_range(7) == 0) ? $urandom_range(3, 1) : 0);
    end

    bubble();
    repeat (3) @(posedge clk_i);
    @(negedge clk_i);

    if (exp_q.size() != 0) begin
      n_errors++;
      $display("ERROR: %0d expected shadows never appeared", exp_q.size());
    end

    if (n_errors == 0) begin
      $display("PASS: pe_tb tiles=%0d checked=%0d", n_tiles, n_checked);
      $finish;
    end else begin
      $display("FAIL: pe_tb errors=%0d tiles=%0d", n_errors, n_tiles);
      $fatal(1, "pe_tb failed");
    end
  end

  initial begin
    #100ms;
    $fatal(1, "TIMEOUT: pe_tb did not finish");
  end

endmodule

`default_nettype wire
