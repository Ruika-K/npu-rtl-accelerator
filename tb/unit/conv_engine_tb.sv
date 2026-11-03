`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for conv_engine with the real act_buffer and
// weight_buffer. Cases come from model/gen_layer_vectors.py
// (+layer_vectors=<path>): the testbench loads the input tensor, the weights
// and the biases through the memory write ports, fills the neighbourhood of
// the output region with a pattern, starts the engine with the descriptor,
// waits for done, reads the output region back and compares it byte for byte
// with the Python model. Writes outside the output region and writes while
// not busy are errors.
module conv_engine_tb
  import npu_pkg::*;
;

  localparam int unsigned N  = ARR_N;
  localparam int unsigned AW = ABUF_AW;
  localparam int unsigned WW = WBUF_WAW;
  localparam time CLK_PERIOD = 10ns;
  localparam int MAX_BYTES = 65536;

  logic clk_i, rst_ni;

  // engine <-> memories
  logic start_i = 1'b0;
  desc_t desc_i = '0;
  logic busy_o, done_o, array_valid_o;
  logic e_abuf_rd_en; logic [AW-1:0] e_abuf_rd_addr;
  logic e_abuf_wr_en; logic [AW-1:0] e_abuf_wr_addr; logic [N-1:0][7:0] e_abuf_wr_data; logic [N-1:0] e_abuf_wr_be;
  logic e_wbuf_rd_en; logic [WW-1:0] e_wbuf_rd_addr;
  // TB <-> memories
  logic t_abuf_rd_en = 1'b0; logic [AW-1:0] t_abuf_rd_addr = '0;
  logic t_abuf_wr_en = 1'b0; logic [AW-1:0] t_abuf_wr_addr = '0; logic [N-1:0][7:0] t_abuf_wr_data = '0; logic [N-1:0] t_abuf_wr_be = '0;
  logic t_wbuf_wr_en = 1'b0; logic [WW-1:0] t_wbuf_wr_addr = '0; logic [63:0] t_wbuf_wr_data = '0; logic [7:0] t_wbuf_wr_be = '0;
  // memory ports
  logic abuf_rd_en; logic [AW-1:0] abuf_rd_addr; logic [N-1:0][7:0] abuf_rd_data; logic abuf_rd_valid;
  logic abuf_wr_en; logic [AW-1:0] abuf_wr_addr; logic [N-1:0][7:0] abuf_wr_data; logic [N-1:0] abuf_wr_be;
  logic wbuf_rd_en; logic [WW-1:0] wbuf_rd_addr; logic [63:0] wbuf_rd_data; logic wbuf_rd_valid;

  assign abuf_rd_en   = busy_o ? e_abuf_rd_en   : t_abuf_rd_en;
  assign abuf_rd_addr = busy_o ? e_abuf_rd_addr : t_abuf_rd_addr;
  assign abuf_wr_en   = busy_o ? e_abuf_wr_en   : t_abuf_wr_en;
  assign abuf_wr_addr = busy_o ? e_abuf_wr_addr : t_abuf_wr_addr;
  assign abuf_wr_data = busy_o ? e_abuf_wr_data : t_abuf_wr_data;
  assign abuf_wr_be   = busy_o ? e_abuf_wr_be   : t_abuf_wr_be;
  assign wbuf_rd_en   = e_wbuf_rd_en;
  assign wbuf_rd_addr = e_wbuf_rd_addr;

  conv_engine u_eng (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .start_i (start_i), .desc_i (desc_i), .busy_o (busy_o), .done_o (done_o), .array_valid_o (array_valid_o),
      .abuf_rd_en_o (e_abuf_rd_en), .abuf_rd_addr_o (e_abuf_rd_addr), .abuf_rd_data_i (abuf_rd_data), .abuf_rd_valid_i (abuf_rd_valid),
      .abuf_wr_en_o (e_abuf_wr_en), .abuf_wr_addr_o (e_abuf_wr_addr), .abuf_wr_data_o (e_abuf_wr_data), .abuf_wr_be_o (e_abuf_wr_be),
      .wbuf_rd_en_o (e_wbuf_rd_en), .wbuf_rd_addr_o (e_wbuf_rd_addr), .wbuf_rd_data_i (wbuf_rd_data), .wbuf_rd_valid_i (wbuf_rd_valid)
  );

  act_buffer u_abuf (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .rd_en_i (abuf_rd_en), .rd_addr_i (abuf_rd_addr), .rd_data_o (abuf_rd_data), .rd_valid_o (abuf_rd_valid),
      .wr_en_i (abuf_wr_en), .wr_addr_i (abuf_wr_addr), .wr_data_i (abuf_wr_data), .wr_be_i (abuf_wr_be)
  );

  weight_buffer u_wbuf (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .rd_en_i (wbuf_rd_en), .rd_addr_i (wbuf_rd_addr), .rd_data_o (wbuf_rd_data), .rd_valid_o (wbuf_rd_valid),
      .wr_en_i (t_wbuf_wr_en), .wr_addr_i (t_wbuf_wr_addr), .wr_data_i (t_wbuf_wr_data), .wr_be_i (t_wbuf_wr_be)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_cases = 0, n_checked = 0, n_errors = 0;
  int unsigned cyc_total = 0, cyc_array = 0;

  // ---------------------------------------------------------------------------
  // Current case
  // ---------------------------------------------------------------------------
  int c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, c_in_base, c_out_base, c_w_base, c_b_base;
  int c_mult, c_shift, c_zin, c_zout, c_amin, c_amax;
  int n_in, n_w, n_b, n_out;
  logic [7:0] in_bytes  [MAX_BYTES];
  logic [7:0] w_bytes   [MAX_BYTES];
  logic [7:0] b_bytes   [MAX_BYTES];
  logic [7:0] out_bytes [MAX_BYTES];
  localparam int GUARD = 16;

  // Engine write monitor: every enabled byte must land inside the output region,
  // and writes must only happen while busy.
  logic mon_en = 1'b0;
  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      if (busy_o) begin
        cyc_total++;
        if (array_valid_o) cyc_array++;
      end
      if (e_abuf_wr_en && !busy_o) begin n_errors++; $display("ERROR @%0t: engine write while idle", $time); end
      if (e_abuf_wr_en && busy_o) begin
        for (int i = 0; i < N; i++) begin
          if (e_abuf_wr_be[i]) begin
            int a;
            a = int'(e_abuf_wr_addr) + i;
            if (a < c_out_base || a >= c_out_base + n_out) begin
              n_errors++;
              if (n_errors < 20) $display("ERROR @%0t: write to %0d outside output region [%0d, %0d)", $time, a, c_out_base, c_out_base + n_out);
            end
          end
        end
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Memory access tasks (falling edge, engine idle)
  // ---------------------------------------------------------------------------
  task automatic abuf_write8(input int addr, input logic [N-1:0][7:0] d, input logic [N-1:0] be);
    @(negedge clk_i);
    t_abuf_wr_en = 1'b1; t_abuf_wr_addr = AW'(addr); t_abuf_wr_data = d; t_abuf_wr_be = be;
    @(negedge clk_i);
    t_abuf_wr_en = 1'b0;
  endtask

  task automatic abuf_load(input int base, input int n, ref logic [7:0] src [MAX_BYTES]);
    int i;
    logic [N-1:0][7:0] d; logic [N-1:0] be;
    i = 0;
    while (i < n) begin
      d = '0; be = '0;
      for (int l = 0; l < N; l++) if (i + l < n) begin d[l] = src[i + l]; be[l] = 1'b1; end
      abuf_write8(base + i, d, be);
      i += N;
    end
  endtask

  task automatic abuf_fill(input int base, input int n, input logic [7:0] v);
    int i;
    logic [N-1:0][7:0] d; logic [N-1:0] be;
    i = 0;
    while (i < n) begin
      d = '0; be = '0;
      for (int l = 0; l < N; l++) if (i + l < n) begin d[l] = v; be[l] = 1'b1; end
      abuf_write8(base + i, d, be);
      i += N;
    end
  endtask

  task automatic wbuf_load(input int base, input int n, ref logic [7:0] src [MAX_BYTES]);
    int i;
    logic [63:0] d; logic [7:0] be;
    if (base % 8 != 0) begin n_errors++; $display("ERROR: wbuf load base %0d not 8-byte aligned", base); end
    i = 0;
    while (i < n) begin
      d = '0; be = '0;
      for (int l = 0; l < 8; l++) if (i + l < n) begin d[8*l +: 8] = src[i + l]; be[l] = 1'b1; end
      @(negedge clk_i);
      t_wbuf_wr_en = 1'b1; t_wbuf_wr_addr = WW'((base + i) >> 3); t_wbuf_wr_data = d; t_wbuf_wr_be = be;
      i += 8;
    end
    @(negedge clk_i);
    t_wbuf_wr_en = 1'b0;
  endtask

  // Read n bytes starting at base into dst (8 per read, 2-cycle latency, pipelined).
  task automatic abuf_read(input int base, input int n, ref logic [7:0] dst [MAX_BYTES]);
    int issued, got;
    issued = 0; got = 0;
    while (got < n) begin
      @(negedge clk_i);
      if (issued < n) begin t_abuf_rd_en = 1'b1; t_abuf_rd_addr = AW'(base + issued); issued += N; end
      else t_abuf_rd_en = 1'b0;
      if (abuf_rd_valid) begin
        for (int l = 0; l < N; l++) if (got + l < n) dst[got + l] = abuf_rd_data[l];
        got += N;
      end
    end
    t_abuf_rd_en = 1'b0;
    @(negedge clk_i);
  endtask

  // ---------------------------------------------------------------------------
  // Run one case
  // ---------------------------------------------------------------------------
  logic [7:0] got_bytes [MAX_BYTES];

  task automatic run_case();
    desc_t d;
    int guard, mism, cyc0, cyc_arr0;
    // descriptor
    d = '0;
    d.op = op_e'(c_op); d.k = 4'(c_k); d.pad = 4'(c_pad); d.stride_m1 = 2'd0;
    d.in_h = 16'(c_in_h); d.in_w = 16'(c_in_w); d.c_in = 16'(c_cin); d.c_out = 16'(c_cout);
    d.in_base = 32'(c_in_base); d.out_base = 32'(c_out_base); d.w_base = 32'(c_w_base); d.b_base = 32'(c_b_base);
    d.rq_mult = 32'(c_mult); d.rq_shift = 6'(c_shift);
    d.z_in = 8'(c_zin); d.z_out = 8'(c_zout); d.act_min = 8'(c_amin); d.act_max = 8'(c_amax);
    if (c_op == int'(OP_FC)) begin d.k = 4'd1; d.pad = 4'd0; d.in_h = 16'd1; d.in_w = 16'd1; end   // the sequencer's FC view

    // memories
    abuf_load(c_in_base, n_in, in_bytes);
    abuf_fill(c_out_base - GUARD, n_out + 2 * GUARD, 8'hA5);
    wbuf_load(c_w_base, n_w, w_bytes);
    wbuf_load(c_b_base, n_b, b_bytes);

    // run
    cyc0 = cyc_total; cyc_arr0 = cyc_array;
    @(negedge clk_i);
    desc_i = d; start_i = 1'b1;
    @(negedge clk_i);
    start_i = 1'b0;
    guard = 0;
    while (!done_o && guard < 2000000) begin @(negedge clk_i); guard++; end
    if (guard >= 2000000) begin n_errors++; $display("ERROR: case %0d did not finish", n_cases); end
    if (busy_o) begin n_errors++; $display("ERROR: busy still high with done"); end
    @(negedge clk_i);

    // compare the output region and its guard bands
    abuf_read(c_out_base - GUARD, n_out + 2 * GUARD, got_bytes);
    mism = 0;
    for (int i = 0; i < n_out + 2 * GUARD; i++) begin
      logic [7:0] exp;
      exp = (i < GUARD || i >= GUARD + n_out) ? 8'hA5 : out_bytes[i - GUARD];
      n_checked++;
      if (got_bytes[i] !== exp) begin
        mism++;
        if (mism <= 4) $display("ERROR: case %0d byte %0d (addr %0d) = %0d, expected %0d", n_cases, i - GUARD, c_out_base - GUARD + i, got_bytes[i], exp);
      end
    end
    if (mism != 0) begin
      n_errors++;
      $display("ERROR: case %0d (op=%0d k=%0d pad=%0d in=%0dx%0d cin=%0d cout=%0d): %0d bytes differ",
               n_cases, c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, mism);
    end
    $display("case %0d: op=%0d k=%0d pad=%0d in=%0dx%0d cin=%0d cout=%0d out=%0d bytes, %0d cycles, array busy %0d, %s",
             n_cases, c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, n_out, cyc_total - cyc0, cyc_array - cyc_arr0,
             (mism == 0) ? "ok" : "MISMATCH");
    n_cases++;
  endtask

  // ---------------------------------------------------------------------------
  // Vector file parsing
  // ---------------------------------------------------------------------------
  task automatic read_bytes(input int fd, output int n, ref logic [7:0] dst [MAX_BYTES]);
    int code, v;
    code = $fscanf(fd, "%d", n);
    if (n > MAX_BYTES) begin n_errors++; $display("ERROR: section of %0d bytes exceeds the testbench limit", n); n = MAX_BYTES; end
    for (int i = 0; i < n; i++) begin code = $fscanf(fd, "%d", v); dst[i] = 8'(v); end
  endtask

  string vec_path = "";

  initial begin
    int fd, code;
    string tag;

    void'($value$plusargs("layer_vectors=%s", vec_path));
    $display("conv_engine_tb: vectors='%s'", vec_path);

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    if (vec_path == "") begin
      n_errors++; $display("ERROR: +layer_vectors not given");
    end else begin
      fd = $fopen(vec_path, "r");
      if (fd == 0) begin n_errors++; $display("ERROR: cannot open '%s'", vec_path); end
      else begin
        n_in = 0; n_w = 0; n_b = 0; n_out = 0;
        while (!$feof(fd)) begin
          code = $fscanf(fd, "%s", tag);
          if (code != 1) break;
          case (tag)
            "CASE": begin
              code = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                             c_op, c_k, c_pad, c_in_h, c_in_w, c_cin, c_cout, c_in_base, c_out_base, c_w_base, c_b_base,
                             c_mult, c_shift, c_zin, c_zout, c_amin, c_amax);
              n_in = 0; n_w = 0; n_b = 0; n_out = 0;
            end
            "IN":  read_bytes(fd, n_in, in_bytes);
            "W":   read_bytes(fd, n_w, w_bytes);
            "B":   read_bytes(fd, n_b, b_bytes);
            "OUT": begin
              read_bytes(fd, n_out, out_bytes);
              if (c_op == int'(OP_CONV2D) || c_op == int'(OP_FC)) run_case();
            end
            default: begin n_errors++; $display("ERROR: unknown tag '%s'", tag); end
          endcase
        end
        $fclose(fd);
      end
    end

    if (n_errors == 0 && n_cases > 0) begin
      $display("PASS: conv_engine_tb cases=%0d checked=%0d", n_cases, n_checked);
      $finish;
    end else begin
      $display("FAIL: conv_engine_tb errors=%0d cases=%0d", n_errors, n_cases);
      $fatal(1, "conv_engine_tb failed");
    end
  end

  initial begin
    #2s;
    $fatal(1, "TIMEOUT: conv_engine_tb did not finish");
  end

endmodule

`default_nettype wire
