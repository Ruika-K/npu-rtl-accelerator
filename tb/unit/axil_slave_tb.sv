`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for axil_slave.
//   - axil_master_bfm drives the AXI4-Lite side with every AW/W ordering and
//     random response back-pressure.
//   - A responder model answers the internal bus after a random delay, keeps a
//     memory, and flags err for addresses with bit 19 set.
//   - A protocol monitor checks the AXI4-Lite rules the slave must obey and
//     that the internal bus carries one transaction at a time.
// Plusargs: +seed=<int> (default 1), +n=<random ops> (default 2000).
module axil_slave_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;

  logic clk_i, rst_ni;

  logic [31:0] awaddr; logic awvalid, awready;
  logic [31:0] wdata;  logic [3:0] wstrb; logic wvalid, wready;
  logic [1:0]  bresp;  logic bvalid, bready;
  logic [31:0] araddr; logic arvalid, arready;
  logic [31:0] rdata;  logic [1:0] rresp; logic rvalid, rready;

  logic        wr_req, wr_ack, wr_err;
  logic [19:0] wr_addr;
  logic [31:0] wr_data;
  logic [3:0]  wr_strb;
  logic        rd_req, rd_ack, rd_err;
  logic [19:0] rd_addr;
  logic [31:0] rd_data;

  axil_slave dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .s_axil_awaddr_i (awaddr), .s_axil_awprot_i (3'b000), .s_axil_awvalid_i (awvalid), .s_axil_awready_o (awready),
      .s_axil_wdata_i (wdata), .s_axil_wstrb_i (wstrb), .s_axil_wvalid_i (wvalid), .s_axil_wready_o (wready),
      .s_axil_bresp_o (bresp), .s_axil_bvalid_o (bvalid), .s_axil_bready_i (bready),
      .s_axil_araddr_i (araddr), .s_axil_arprot_i (3'b000), .s_axil_arvalid_i (arvalid), .s_axil_arready_o (arready),
      .s_axil_rdata_o (rdata), .s_axil_rresp_o (rresp), .s_axil_rvalid_o (rvalid), .s_axil_rready_i (rready),
      .wr_req_o (wr_req), .wr_addr_o (wr_addr), .wr_data_o (wr_data), .wr_strb_o (wr_strb),
      .wr_ack_i (wr_ack), .wr_err_i (wr_err),
      .rd_req_o (rd_req), .rd_addr_o (rd_addr), .rd_ack_i (rd_ack), .rd_data_i (rd_data), .rd_err_i (rd_err)
  );

  axil_master_bfm u_bfm (
      .clk_i (clk_i),
      .awaddr_o (awaddr), .awvalid_o (awvalid), .awready_i (awready),
      .wdata_o (wdata), .wstrb_o (wstrb), .wvalid_o (wvalid), .wready_i (wready),
      .bresp_i (bresp), .bvalid_i (bvalid), .bready_o (bready),
      .araddr_o (araddr), .arvalid_o (arvalid), .arready_i (arready),
      .rdata_i (rdata), .rresp_i (rresp), .rvalid_i (rvalid), .rready_o (rready)
  );

  initial begin
    clk_i = 1'b0;
    forever #(CLK_PERIOD / 2) clk_i = ~clk_i;
  end

  int n_checked = 0, n_errors = 0, n_writes = 0, n_reads = 0;

  // ---------------------------------------------------------------------------
  // Responder model for the internal bus
  // ---------------------------------------------------------------------------
  logic [31:0] mem [logic [19:0]];
  function automatic logic is_err_addr(input logic [19:0] a);
    return |(a & 20'h80000);
  endfunction
  function automatic logic [31:0] default_data(input logic [19:0] a);
    return 32'h5A5A_0000 ^ {12'h0, a};
  endfunction

  logic        wr_pend_q, rd_pend_q;
  int          wr_cnt_q, rd_cnt_q;
  logic [19:0] wr_addr_q, rd_addr_q;
  logic [31:0] wr_data_q;
  logic [3:0]  wr_strb_q;
  int          req_log[$];   // 0 = write, 1 = read, in order of arrival

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ack <= 1'b0; wr_err <= 1'b0; rd_ack <= 1'b0; rd_err <= 1'b0; rd_data <= '0;
      wr_pend_q <= 1'b0; rd_pend_q <= 1'b0; wr_cnt_q <= 0; rd_cnt_q <= 0;
      wr_addr_q <= '0; rd_addr_q <= '0; wr_data_q <= '0; wr_strb_q <= '0;
    end else begin
      wr_ack <= 1'b0;
      rd_ack <= 1'b0;
      if (wr_req) begin
        wr_pend_q <= 1'b1; wr_cnt_q <= $urandom_range(3);
        wr_addr_q <= wr_addr; wr_data_q <= wr_data; wr_strb_q <= wr_strb;
        req_log.push_back(0);
      end else if (wr_pend_q) begin
        if (wr_cnt_q == 0) begin
          if (!is_err_addr(wr_addr_q)) begin
            logic [31:0] old;
            old = (mem.exists(wr_addr_q) != 0) ? mem[wr_addr_q] : default_data(wr_addr_q);
            for (int b = 0; b < 4; b++) if (wr_strb_q[b]) old[8*b +: 8] = wr_data_q[8*b +: 8];
            mem[wr_addr_q] = old;
          end
          wr_ack <= 1'b1; wr_err <= is_err_addr(wr_addr_q); wr_pend_q <= 1'b0;
        end else wr_cnt_q <= wr_cnt_q - 1;
      end
      if (rd_req) begin
        rd_pend_q <= 1'b1; rd_cnt_q <= $urandom_range(3); rd_addr_q <= rd_addr;
        req_log.push_back(1);
      end else if (rd_pend_q) begin
        if (rd_cnt_q == 0) begin
          rd_data <= (mem.exists(rd_addr_q) != 0) ? mem[rd_addr_q] : default_data(rd_addr_q);
          rd_ack <= 1'b1; rd_err <= is_err_addr(rd_addr_q); rd_pend_q <= 1'b0;
        end else rd_cnt_q <= rd_cnt_q - 1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Protocol monitor
  // ---------------------------------------------------------------------------
  logic mon_en = 1'b0;
  logic p_bvalid, p_bready, p_rvalid, p_rready, p_awvalid, p_wvalid;
  logic [1:0] p_bresp, p_rresp;
  logic [31:0] p_rdata;
  logic [31:0] p_awaddr, p_wdata;
  int outstanding = 0;   // internal-bus requests without ack

  /* verilator lint_off BLKSEQ */
  always @(negedge clk_i) begin
    if (mon_en) begin
      // slave outputs must hold while not accepted
      if (p_bvalid && !p_bready) begin
        if (!bvalid || bresp !== p_bresp) begin n_errors++; $display("ERROR @%0t: B channel changed before bready", $time); end
      end
      if (p_rvalid && !p_rready) begin
        if (!rvalid || rresp !== p_rresp || rdata !== p_rdata) begin n_errors++; $display("ERROR @%0t: R channel changed before rready", $time); end
      end
      // master (BFM) outputs must hold too; this guards the BFM itself
      if (p_awvalid && !awready && awvalid && awaddr !== p_awaddr) begin n_errors++; $display("ERROR @%0t: BFM changed awaddr while waiting", $time); end
      if (p_wvalid && !wready && wvalid && wdata !== p_wdata) begin n_errors++; $display("ERROR @%0t: BFM changed wdata while waiting", $time); end
      // ready rules
      if (awready !== wready) begin n_errors++; $display("ERROR @%0t: awready != wready", $time); end
      if (awready && !(awvalid && wvalid)) begin n_errors++; $display("ERROR @%0t: awready without both AW and W valid", $time); end
      if (arready && !arvalid) begin n_errors++; $display("ERROR @%0t: arready without arvalid", $time); end
      if (awready && arready) begin n_errors++; $display("ERROR @%0t: read and write accepted in the same cycle", $time); end
      // internal bus: single transaction, pulses
      if (wr_req && rd_req) begin n_errors++; $display("ERROR @%0t: wr_req and rd_req together", $time); end
      if ((wr_req || rd_req) && outstanding != 0) begin n_errors++; $display("ERROR @%0t: request while previous not acked", $time); end
      if (wr_req || rd_req) outstanding++;
      if (wr_ack || rd_ack) outstanding--;
    end
    p_bvalid = bvalid; p_bready = bready; p_bresp = bresp;
    p_rvalid = rvalid; p_rready = rready; p_rresp = rresp; p_rdata = rdata;
    p_awvalid = awvalid; p_wvalid = wvalid;
    p_awaddr = awaddr; p_wdata = wdata;
  end
  /* verilator lint_on BLKSEQ */

  // ---------------------------------------------------------------------------
  // Scoreboard helpers
  // ---------------------------------------------------------------------------
  logic [31:0] exp_mem [logic [19:0]];

  task automatic do_write(input logic [19:0] a, input logic [31:0] d, input logic [3:0] strb);
    logic [1:0] resp;
    logic [31:0] old;
    u_bfm.write({12'h0, a}, d, strb, resp);
    n_writes++;
    n_checked++;
    if (is_err_addr(a)) begin
      if (resp !== 2'b10) begin n_errors++; $display("ERROR: write to %h expected SLVERR, got %b", a, resp); end
    end else begin
      if (resp !== 2'b00) begin n_errors++; $display("ERROR: write to %h expected OKAY, got %b", a, resp); end
      old = (exp_mem.exists(a) != 0) ? exp_mem[a] : default_data(a);
      for (int b = 0; b < 4; b++) if (strb[b]) old[8*b +: 8] = d[8*b +: 8];
      exp_mem[a] = old;
      if ((mem.exists(a) == 0) || mem[a] !== old) begin
        n_errors++;
        $display("ERROR: after write %h <= %h strb %b, responder holds %h, expected %h", a, d, strb,
                 mem.exists(a) ? mem[a] : 32'hx, old);
      end
    end
  endtask

  task automatic do_read(input logic [19:0] a);
    logic [1:0] resp;
    logic [31:0] data, exp;
    u_bfm.read({12'h0, a}, data, resp);
    n_reads++;
    n_checked++;
    exp = (exp_mem.exists(a) != 0) ? exp_mem[a] : default_data(a);
    if (is_err_addr(a)) begin
      if (resp !== 2'b10) begin n_errors++; $display("ERROR: read %h expected SLVERR, got %b", a, resp); end
    end else begin
      if (resp !== 2'b00) begin n_errors++; $display("ERROR: read %h expected OKAY, got %b", a, resp); end
      if (data !== exp) begin n_errors++; $display("ERROR: read %h = %h, expected %h", a, data, exp); end
    end
  endtask

  function automatic logic [19:0] rand_addr();
    logic [19:0] a;
    a = 20'($urandom());
    a[1:0] = 2'b00;
    if ($urandom_range(3) != 0) a[19] = 1'b0;          // mostly legal
    if ($urandom_range(1) == 0) a[18:6] = '0;          // hit the same few words often
    return a;
  endfunction

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  int seed = 1;
  int n_random = 2000;

  initial begin
    logic [31:0] data;
    logic [1:0]  resp;

    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", n_random)) n_random = 2000;
    void'($urandom(seed));
    $display("axil_slave_tb: seed=%0d n_random=%0d", seed, n_random);

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);
    mon_en = 1'b1;

    if (bvalid || rvalid || awready || wready || arready) begin
      n_errors++; $display("ERROR: outputs active after reset");
    end

    // Phase 1: every ordering, no back-pressure.
    for (int ord = 0; ord < 3; ord++) begin
      u_bfm.aw_w_order = ord;
      u_bfm.max_resp_delay = 0;
      do_write(20'h00010 + 20'(ord) * 4, 32'h1111_0000 + 32'(ord), 4'hF);
      do_read(20'h00010 + 20'(ord) * 4);
      do_write(20'h00010, 32'hAABB_CCDD, 4'b0101);   // byte strobes
      do_read(20'h00010);
      do_write(20'h80000, 32'hDEAD_BEEF, 4'hF);       // error address
      do_read(20'h80000);
      do_read(20'h00014);
    end

    // Phase 2: read and write requested in the same cycle; the write must win
    // and the read must follow with the data the write did not touch.
    u_bfm.aw_w_order = 0;
    req_log.delete();
    u_bfm.write_with_ar(32'h0000_0020, 32'h0BAD_F00D, 4'hF, 32'h0000_0010, resp);
    n_checked++;
    if (resp !== 2'b00) begin n_errors++; $display("ERROR: simultaneous write resp %b", resp); end
    exp_mem[20'h00020] = 32'h0BAD_F00D;
    u_bfm.read_complete(data, resp);
    n_checked++;
    if (resp !== 2'b00 || data !== exp_mem[20'h00010]) begin
      n_errors++; $display("ERROR: pending read returned %h/%b, expected %h/00", data, resp, exp_mem[20'h00010]);
    end
    if (req_log.size() != 2 || req_log[0] != 0 || req_log[1] != 1) begin
      n_errors++; $display("ERROR: expected write then read on the internal bus, got %0d entries", req_log.size());
    end
    if ((mem.exists(20'h00020) == 0) || mem[20'h00020] !== 32'h0BAD_F00D) begin
      n_errors++; $display("ERROR: simultaneous write did not land");
    end

    // Phase 3: random traffic with random ordering and back-pressure.
    u_bfm.aw_w_order = 3;
    for (int i = 0; i < n_random; i++) begin
      u_bfm.max_resp_delay = $urandom_range(3);
      if ($urandom_range(1) == 0) do_write(rand_addr(), $urandom(), 4'($urandom_range(15, 1)));
      else                        do_read(rand_addr());
    end

    // Phase 4: a slow master. The response is ready three cycles after the
    // request at the earliest, so only a longer delay makes B and R wait for
    // BREADY / RREADY (the cover properties of axil_slave_sva showed that the
    // delays of phase 3 never did).
    for (int i = 0; i < 40; i++) begin
      u_bfm.max_resp_delay = $urandom_range(12, 6);
      if (i % 2 == 0) do_write(rand_addr(), $urandom(), 4'($urandom_range(15, 1)));
      else            do_read(rand_addr());
    end
    u_bfm.max_resp_delay = 0;

    repeat (4) @(negedge clk_i);
    if (outstanding != 0) begin n_errors++; $display("ERROR: %0d internal-bus requests left without ack", outstanding); end

    if (n_errors == 0) begin
      $display("PASS: axil_slave_tb checked=%0d writes=%0d reads=%0d", n_checked, n_writes, n_reads);
      $finish;
    end else begin
      $display("FAIL: axil_slave_tb errors=%0d checked=%0d", n_errors, n_checked);
      $fatal(1, "axil_slave_tb failed");
    end
  end

  initial begin
    #200ms;
    $fatal(1, "TIMEOUT: axil_slave_tb did not finish");
  end

endmodule

`default_nettype wire
