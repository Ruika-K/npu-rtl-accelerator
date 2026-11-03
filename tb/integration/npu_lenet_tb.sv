`timescale 1ns/1ps
`default_nettype none

// Whole-network test for npu_top: LeNet-5 from model/quantize.py.
//   +program=<model/data/<net>/program.txt>  +images=<n>  (default: all)
//   - Loads the weight-buffer image and the descriptors once over AXI4-Lite.
//   - For every image: writes the input, starts, polls STATUS, reads RESULT
//     and compares the class with the integer model's prediction (bit-exact
//     end to end means every prediction must agree, not just the accuracy).
//   - For the images that carry TRACE lines, every layer's output region is
//     read back and compared byte for byte.
//   - Reports accuracy against the labels and a per-layer cycle table from
//     the performance counters of the first image.
module npu_lenet_tb
  import npu_pkg::*;
;

  localparam time CLK_PERIOD = 10ns;
  localparam int  MAX_BYTES  = 65536;
  localparam int  MAX_WORDS  = 32768;

  logic clk_i, rst_ni;
  logic [31:0] awaddr; logic awvalid, awready;
  logic [31:0] wdata;  logic [3:0] wstrb; logic wvalid, wready;
  logic [1:0]  bresp;  logic bvalid, bready;
  logic [31:0] araddr; logic arvalid, arready;
  logic [31:0] rdata;  logic [1:0] rresp; logic rvalid, rready;
  logic irq_o;

  npu_top dut (
      .clk_i (clk_i), .rst_ni (rst_ni),
      .s_axil_awaddr_i (awaddr), .s_axil_awprot_i (3'b000), .s_axil_awvalid_i (awvalid), .s_axil_awready_o (awready),
      .s_axil_wdata_i (wdata), .s_axil_wstrb_i (wstrb), .s_axil_wvalid_i (wvalid), .s_axil_wready_o (wready),
      .s_axil_bresp_o (bresp), .s_axil_bvalid_o (bvalid), .s_axil_bready_i (bready),
      .s_axil_araddr_i (araddr), .s_axil_arprot_i (3'b000), .s_axil_arvalid_i (arvalid), .s_axil_arready_o (arready),
      .s_axil_rdata_o (rdata), .s_axil_rresp_o (rresp), .s_axil_rvalid_o (rvalid), .s_axil_rready_i (rready),
      .pix_wr_en_i (1'b0), .pix_wr_addr_i ('0), .pix_wr_data_i ('0), .pix_wr_be_i ('0), .pix_wr_ready_o (),
      .irq_o (irq_o)
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

  int n_images = 0, n_checked = 0, n_errors = 0, n_correct = 0, n_agree = 0, n_traces = 0;

  localparam logic [31:0] A_CTRL = 32'h004, A_STATUS = 32'h008, A_LCNT = 32'h00C, A_RESULT = 32'h010;
  localparam logic [31:0] A_PTOTAL = 32'h060, A_PARRAY = 32'h064, A_PLAYER = 32'h080, A_DESC = 32'h100;

  task automatic wr32(input logic [31:0] a, input logic [31:0] d, input logic [3:0] strb = 4'hF);
    logic [1:0] resp;
    u_bfm.write(a, d, strb, resp);
    if (resp !== 2'b00) begin n_errors++; $display("ERROR: write %h resp %b", a, resp); end
  endtask

  task automatic rd32(input logic [31:0] a, output logic [31:0] d);
    logic [1:0] resp;
    u_bfm.read(a, d, resp);
    if (resp !== 2'b00) begin n_errors++; $display("ERROR: read %h resp %b", a, resp); end
  endtask

  task automatic check(input bit cond, input string what);
    n_checked++;
    if (!cond) begin n_errors++; $display("ERROR @%0t: %s", $time, what); end
  endtask

  task automatic write_bytes(input logic [31:0] base, input int offset, input int n, ref logic [7:0] src [MAX_BYTES]);
    int a, i;
    logic [31:0] d; logic [3:0] strb;
    a = (offset / 4) * 4;
    while (a < offset + n) begin
      d = '0; strb = '0;
      for (int b = 0; b < 4; b++) begin
        i = a + b - offset;
        if (i >= 0 && i < n) begin d[8*b +: 8] = src[i]; strb[b] = 1'b1; end
      end
      wr32(base + 32'(a), d, strb);
      a += 4;
    end
  endtask

  task automatic read_bytes(input logic [31:0] base, input int offset, input int n, ref logic [7:0] dst [MAX_BYTES]);
    int a, i;
    logic [31:0] d;
    a = (offset / 4) * 4;
    while (a < offset + n) begin
      rd32(base + 32'(a), d);
      for (int b = 0; b < 4; b++) begin
        i = a + b - offset;
        if (i >= 0 && i < n) dst[i] = d[8*b +: 8];
      end
      a += 4;
    end
  endtask

  // Hex string (two characters per byte) -> bytes
  function automatic int hex_to_bytes(input string s, ref logic [7:0] dst [MAX_BYTES]);
    int n;
    n = s.len() / 2;
    if (n > MAX_BYTES) n = MAX_BYTES;
    for (int i = 0; i < n; i++) dst[i] = 8'(s.substr(2 * i, 2 * i + 1).atohex());
    return n;
  endfunction

  logic [7:0] buf_a [MAX_BYTES];
  logic [7:0] buf_b [MAX_BYTES];

  // ---------------------------------------------------------------------------
  // Test sequence
  // ---------------------------------------------------------------------------
  string prog_path = "";
  int max_images = 1 << 30;
  int n_layers = 0;

  initial begin
    int fd, code, idx, label, expected, in_base, nbytes, nwords, img, layer, base, n, mism, guard;
    logic [31:0] v, status, w, ptotal, parray;
    logic [31:0] player [NUM_DESC];
    string tag, hex;
    bit first_image_done;

    void'($value$plusargs("program=%s", prog_path));
    void'($value$plusargs("images=%d", max_images));
    $display("npu_lenet_tb: program='%s' images=%0d", prog_path, max_images);
    u_bfm.aw_w_order = 0;

    rst_ni = 1'b0;
    repeat (4) @(negedge clk_i);
    rst_ni = 1'b1;
    repeat (2) @(negedge clk_i);

    first_image_done = 1'b0;
    fd = 0;
    if (prog_path != "") fd = $fopen(prog_path, "r");   // not a ternary: Verilator 5.020 mis-generates it
    if (fd == 0) begin
      n_errors++; $display("ERROR: cannot open program '%s'", prog_path);
    end else begin
      while (!$feof(fd)) begin
        code = $fscanf(fd, "%s", tag);
        if (code != 1) break;
        case (tag)
          "LAYERS": begin
            code = $fscanf(fd, "%d", n_layers);
            wr32(A_LCNT, 32'(n_layers));
          end
          "DESC": begin
            code = $fscanf(fd, "%d", idx);
            for (int k = 0; k < DESC_USED_WORDS; k++) begin
              code = $fscanf(fd, "%d", w);
              wr32(A_DESC + 32'(idx * 64 + k * 4), w);
            end
          end
          "WORDS": begin
            code = $fscanf(fd, "%d", nwords);
            for (int i = 0; i < nwords; i++) begin
              code = $fscanf(fd, "%h", w);
              wr32(32'(WBUF_BASE) + 32'(4 * i), w);
            end
            $display("weights: %0d words loaded", nwords);
          end
          "IMAGE": begin
            code = $fscanf(fd, "%d %d %d %d %d %s", idx, label, expected, in_base, nbytes, hex);
            if (n_images >= max_images) continue;
            n = hex_to_bytes(hex, buf_a);
            check(n == nbytes, "image byte count");
            write_bytes(32'(ABUF_BASE), in_base, n, buf_a);
            wr32(A_CTRL, 32'h1);
            guard = 0;
            do begin rd32(A_STATUS, status); guard++; end while (status[2:1] == 2'b00 && guard < 100000);
            check(status[1] == 1'b1 && status[2] == 1'b0, $sformatf("image %0d: DONE without ERR (status %h)", idx, status));
            rd32(A_RESULT, v);
            check(v[8] == 1'b1, $sformatf("image %0d: RESULT.VALID", idx));
            n_images++;
            if (int'(v[3:0]) == expected) n_agree++;
            else begin n_errors++; $display("ERROR: image %0d: NPU says %0d, integer model says %0d (label %0d)", idx, v[3:0], expected, label); end
            if (int'(v[3:0]) == label) n_correct++;
            if (!first_image_done) begin
              first_image_done = 1'b1;
              rd32(A_PTOTAL, ptotal); rd32(A_PARRAY, parray);
              for (int l = 0; l < n_layers; l++) rd32(A_PLAYER + 32'(4 * l), player[l]);
              $display("perf image %0d: total %0d cycles, array busy %0d cycles", idx, ptotal, parray);
              for (int l = 0; l < n_layers; l++) $display("perf layer %0d: %0d cycles", l, player[l]);
            end
            wr32(A_STATUS, 32'h2);
          end
          "TRACE": begin
            code = $fscanf(fd, "%d %d %d %d %s", img, layer, base, nbytes, hex);
            if (img >= max_images) continue;
            n = hex_to_bytes(hex, buf_a);
            read_bytes(32'(ABUF_BASE), base, n, buf_b);
            mism = 0;
            for (int i = 0; i < n; i++) begin
              n_checked++;
              if (buf_b[i] !== buf_a[i]) begin mism++; if (mism < 4) $display("ERROR: image %0d layer %0d byte %0d = %0d, expected %0d", img, layer, i, buf_b[i], buf_a[i]); end
            end
            if (mism != 0) n_errors++;
            n_traces++;
            $display("trace image %0d layer %0d: %0d bytes %s", img, layer, n, (mism == 0) ? "match" : "MISMATCH");
          end
          default: begin n_errors++; $display("ERROR: unknown tag '%s'", tag); end
        endcase
      end
      $fclose(fd);
    end

    $display("images %0d: %0d agree with the integer model, %0d correct vs labels (%.2f%%), %0d layer traces",
             n_images, n_agree, n_correct, n_images ? 100.0 * n_correct / n_images : 0.0, n_traces);
    if (n_errors == 0 && n_images > 0) begin
      $display("PASS: npu_lenet_tb images=%0d agree=%0d correct=%0d traces=%0d checked=%0d", n_images, n_agree, n_correct, n_traces, n_checked);
      $finish;
    end else begin
      $display("FAIL: npu_lenet_tb errors=%0d images=%0d", n_errors, n_images);
      $fatal(1, "npu_lenet_tb failed");
    end
  end

  initial begin
    #20s;
    $fatal(1, "TIMEOUT: npu_lenet_tb did not finish");
  end

endmodule

`default_nettype wire
