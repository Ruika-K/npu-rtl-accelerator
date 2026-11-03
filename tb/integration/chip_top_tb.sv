`timescale 1ns/1ps
`default_nettype none

// Integration test for chip_top with two unrelated clocks.
//   +program=<model/data/<net>/program.txt>  +images=<n> (default 4)
//   - APB master loads the program through the APB -> AXI4-Lite bridge.
//   - Frames arrive on the pixel stream (pixel clock), cross the async FIFO
//     and land in the activation buffer; FRAME_DONE then START; the class
//     must equal the integer model's prediction for every image.
//   - Also: chip ID, a frame streamed while the NPU is busy (back-pressure
//     through the FIFO, then the frame completes), a byte before sof
//     (SOF_ERR), an overrun from a source that ignores ready, the core clock
//     gate (core window returns PSLVERR while gated, works again after).
module chip_top_tb
  import npu_pkg::*;
;

  localparam time SYS_PERIOD = 10ns;    // 100 MHz
  localparam time PIX_PERIOD = 27ns;    // about 37 MHz, unrelated
  localparam int  MAX_BYTES  = 65536;

  logic sys_clk = 1'b0, pix_clk = 1'b0, rst_n = 1'b0;
  always #(SYS_PERIOD / 2) sys_clk = ~sys_clk;
  always #(PIX_PERIOD / 2) pix_clk = ~pix_clk;

  logic psel, penable, pwrite, pready, pslverr;
  logic [31:0] paddr, pwdata, prdata;
  logic [3:0] pstrb;
  logic pix_valid = 1'b0, pix_sof = 1'b0, pix_ready;
  logic [7:0] pix_data = '0;
  logic irq;

  chip_top dut (
      .sys_clk_i (sys_clk), .pix_clk_i (pix_clk), .rst_ni (rst_n),
      .s_apb_psel_i (psel), .s_apb_penable_i (penable), .s_apb_pwrite_i (pwrite), .s_apb_paddr_i (paddr),
      .s_apb_pwdata_i (pwdata), .s_apb_pstrb_i (pstrb), .s_apb_pready_o (pready), .s_apb_prdata_o (prdata), .s_apb_pslverr_o (pslverr),
      .pix_valid_i (pix_valid), .pix_sof_i (pix_sof), .pix_data_i (pix_data), .pix_ready_o (pix_ready),
      .m_axi_awaddr_o (), .m_axi_awlen_o (), .m_axi_awsize_o (), .m_axi_awburst_o (), .m_axi_awvalid_o (), .m_axi_awready_i (1'b0),
      .m_axi_wdata_o (), .m_axi_wstrb_o (), .m_axi_wlast_o (), .m_axi_wvalid_o (), .m_axi_wready_i (1'b0),
      .m_axi_bresp_i (2'b00), .m_axi_bvalid_i (1'b0), .m_axi_bready_o (),
      .m_axi_araddr_o (), .m_axi_arlen_o (), .m_axi_arsize_o (), .m_axi_arburst_o (), .m_axi_arvalid_o (), .m_axi_arready_i (1'b0),
      .m_axi_rdata_i ('0), .m_axi_rresp_i (2'b00), .m_axi_rlast_i (1'b0), .m_axi_rvalid_i (1'b0), .m_axi_rready_o (),
      .irq_o (irq)
  );

  apb_master_bfm u_apb (
      .clk_i (sys_clk), .psel_o (psel), .penable_o (penable), .pwrite_o (pwrite), .paddr_o (paddr), .pwdata_o (pwdata),
      .pstrb_o (pstrb), .pready_i (pready), .prdata_i (prdata), .pslverr_i (pslverr)
  );

  int n_images = 0, n_checked = 0, n_errors = 0, n_agree = 0, n_correct = 0;

  // NPU core window (AXI4-Lite map) and chip registers
  localparam logic [31:0] A_CTRL = 32'h004, A_STATUS = 32'h008, A_LCNT = 32'h00C, A_RESULT = 32'h010, A_DESC = 32'h100;
  localparam logic [31:0] C_ID = 32'h100000, C_CLK = 32'h100004, C_PIXCTRL = 32'h100008, C_PIXBASE = 32'h10000C;
  localparam logic [31:0] C_PIXLEN = 32'h100010, C_PIXSTAT = 32'h100014, C_PIXFRAMES = 32'h100018;

  task automatic wr(input logic [31:0] a, input logic [31:0] d, input logic [3:0] strb = 4'hF, input bit exp_err = 1'b0);
    logic e;
    u_apb.write(a, d, strb, e);
    if (e !== exp_err) begin n_errors++; $display("ERROR: write %h slverr %b, expected %b", a, e, exp_err); end
  endtask

  task automatic rd(input logic [31:0] a, output logic [31:0] d, input bit exp_err = 1'b0);
    logic e;
    u_apb.read(a, d, e);
    if (e !== exp_err) begin n_errors++; $display("ERROR: read %h slverr %b, expected %b", a, e, exp_err); end
  endtask

  task automatic check(input bit cond, input string what);
    n_checked++;
    if (!cond) begin n_errors++; $display("ERROR @%0t: %s", $time, what); end
  endtask

  // Pixel source (pixel clock): respects ready unless rude is set.
  bit rude = 1'b0;
  task automatic send_byte(input bit sof, input logic [7:0] b);
    bit acc;
    @(negedge pix_clk);
    pix_valid = 1'b1; pix_sof = sof; pix_data = b;
    forever begin
      #1; acc = pix_valid && (pix_ready || rude);
      @(posedge pix_clk); @(negedge pix_clk);
      if (acc) break;
    end
    pix_valid = 1'b0; pix_sof = 1'b0;
    while ($urandom_range(2) == 0) @(negedge pix_clk);
  endtask

  task automatic send_frame(input int n, ref logic [7:0] data [MAX_BYTES], input bit with_sof = 1'b1);
    for (int i = 0; i < n; i++) send_byte(with_sof && (i == 0), data[i]);
  endtask

  task automatic wait_frame_done();
    logic [31:0] v; int guard = 0;
    do begin rd(C_PIXSTAT, v); guard++; end while (!v[0] && guard < 20000);
    check(v[0] == 1'b1, "FRAME_DONE");
    wr(C_PIXSTAT, 32'h1);
  endtask

  task automatic run_and_check(input int expected, input int label, input int idx);
    logic [31:0] status, v; int guard = 0;
    wr(A_CTRL, 32'h1);
    do begin rd(A_STATUS, status); guard++; end while (status[2:1] == 2'b00 && guard < 100000);
    check(status[1] == 1'b1 && status[2] == 1'b0, $sformatf("image %0d: DONE without ERR (status %h)", idx, status));
    rd(A_RESULT, v);
    n_images++;
    if (int'(v[3:0]) == expected) n_agree++;
    else begin n_errors++; $display("ERROR: image %0d: NPU says %0d, integer model says %0d (label %0d)", idx, v[3:0], expected, label); end
    if (int'(v[3:0]) == label) n_correct++;
    wr(A_STATUS, 32'h2);
  endtask

  function automatic int hex_to_bytes(input string s, ref logic [7:0] dst [MAX_BYTES]);
    int n = s.len() / 2;
    if (n > MAX_BYTES) n = MAX_BYTES;
    for (int i = 0; i < n; i++) dst[i] = 8'(s.substr(2 * i, 2 * i + 1).atohex());
    return n;
  endfunction

  logic [7:0] img [MAX_BYTES];
  logic [7:0] img2 [MAX_BYTES];

  string prog_path = "";
  int max_images = 4;

  initial begin
    int fd, code, idx, label, expected, in_base, nbytes, nwords, n, n2, exp2, lab2;
    logic [31:0] v, w;
    string tag, hex;
    bit have_second;

    void'($value$plusargs("program=%s", prog_path));
    void'($value$plusargs("images=%d", max_images));
    $display("chip_top_tb: program='%s' images=%0d", prog_path, max_images);

    #120ns;
    rst_n = 1'b1;
    repeat (4) @(negedge sys_clk);

    rd(C_ID, v);           check(v == 32'h4E50_5543, "chip ID");
    rd(32'h0, v);          check(v == NPU_ID, "NPU ID through the bridge");
    rd(C_CLK, v);          check(v == 32'h1, "core clock enabled after reset");

    // ---- clock gate: core window blocked while gated ----
    wr(C_CLK, 32'h0);
    rd(32'h0, v, 1'b1);    // PSLVERR, no hang
    wr(A_LCNT, 32'h1, 4'hF, 1'b1);
    rd(C_ID, v);           check(v == 32'h4E50_5543, "chip registers alive while the core is gated");
    wr(C_CLK, 32'h1);
    repeat (2) @(negedge sys_clk);
    rd(32'h0, v);          check(v == NPU_ID, "core back after the clock gate");

    // ---- program ----
    fd = 0;
    if (prog_path != "") fd = $fopen(prog_path, "r");
    if (fd == 0) begin n_errors++; $display("ERROR: cannot open program"); end
    else begin
      in_base = 0; nbytes = 0; have_second = 1'b0;
      while (!$feof(fd)) begin
        code = $fscanf(fd, "%s", tag);
        if (code != 1) break;
        case (tag)
          "LAYERS": begin code = $fscanf(fd, "%d", n); wr(A_LCNT, 32'(n)); end
          "DESC": begin
            code = $fscanf(fd, "%d", idx);
            for (int k = 0; k < DESC_USED_WORDS; k++) begin code = $fscanf(fd, "%d", w); wr(A_DESC + 32'(idx * 64 + k * 4), w); end
          end
          "WORDS": begin
            code = $fscanf(fd, "%d", nwords);
            for (int i = 0; i < nwords; i++) begin code = $fscanf(fd, "%h", w); wr(32'(WBUF_BASE) + 32'(4 * i), w); end
          end
          "IMAGE": begin
            code = $fscanf(fd, "%d %d %d %d %d %s", idx, label, expected, in_base, nbytes, hex);
            if (n_images >= max_images) continue;
            if (n_images == 0) begin
              wr(C_PIXBASE, 32'(in_base));
              wr(C_PIXLEN, 32'(nbytes));
              wr(C_PIXCTRL, 32'h3);    // EN + IRQ_EN
            end
            n = hex_to_bytes(hex, img);
            send_frame(n, img);
            wait_frame_done();
            run_and_check(expected, label, idx);
            // keep the second image for the back-pressure test
            if (!have_second && n_images == 1) begin have_second = 1'b1; n2 = n; exp2 = expected; lab2 = label;
              for (int i = 0; i < n; i++) img2[i] = img[i]; end
          end
          "TRACE": begin code = $fscanf(fd, "%d %d %d %d %s", idx, idx, idx, idx, hex); end
          default: begin n_errors++; $display("ERROR: unknown tag '%s'", tag); end
        endcase
      end
      $fclose(fd);
    end

    // ---- frame streamed while the NPU is busy: FIFO back-pressure, then completes ----
    if (have_second) begin
      logic [31:0] status; int guard;
      send_frame(n2, img2); wait_frame_done();           // load image 2 ...
      wr(A_CTRL, 32'h1);                                 // ... start it ...
      send_frame(n2, img2);                              // ... and stream it again while busy
      guard = 0;
      do begin rd(A_STATUS, status); guard++; end while (status[1] == 1'b0 && guard < 100000);
      check(status[1] == 1'b1, "run completed with a frame waiting in the FIFO");
      rd(A_RESULT, v); check(int'(v[3:0]) == exp2, "result of the run that was interrupted by a frame");
      wr(A_STATUS, 32'h2);
      wait_frame_done();                                 // the waiting frame drains after DONE
      run_and_check(exp2, lab2, 1);
      rd(C_PIXFRAMES, v); check(v == 32'(n_images + 1), $sformatf("PIX_FRAMES = %0d", v));
    end

    // ---- byte before sof ----
    send_byte(1'b0, 8'h5A);
    repeat (6) @(negedge sys_clk);
    rd(C_PIXSTAT, v); check(v[1] == 1'b1, "SOF_ERR set"); wr(C_PIXSTAT, 32'h2);
    rd(C_PIXSTAT, v); check(v[1] == 1'b0, "SOF_ERR cleared");

    // ---- overrun: a rude source with the writer disabled ----
    wr(C_PIXCTRL, 32'h0);
    rude = 1'b1;
    for (int i = 0; i < 40; i++) send_byte(i == 0, 8'(i));
    rude = 1'b0;
    repeat (6) @(negedge sys_clk);
    rd(C_PIXSTAT, v); check(v[2] == 1'b1, "OVERRUN set"); wr(C_PIXSTAT, 32'h4);
    rd(C_PIXSTAT, v); check(v[2] == 1'b0, "OVERRUN cleared");
    check(irq == 1'b0, "no interrupt pending at the end");

    $display("images %0d: %0d agree with the integer model, %0d correct vs labels", n_images, n_agree, n_correct);
    if (n_errors == 0 && n_images > 0) begin
      $display("PASS: chip_top_tb images=%0d agree=%0d checked=%0d", n_images, n_agree, n_checked);
      $finish;
    end else begin
      $display("FAIL: chip_top_tb errors=%0d images=%0d", n_errors, n_images);
      $fatal(1, "chip_top_tb failed");
    end
  end

  initial begin
    #20s;
    $fatal(1, "TIMEOUT: chip_top_tb did not finish");
  end

endmodule

`default_nettype wire
