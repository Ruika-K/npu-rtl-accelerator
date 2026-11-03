`timescale 1ns/1ps
`default_nettype none

// AXI4 slave memory model for the testbenches: 64-bit data, INCR bursts,
// single outstanding transaction per direction, random ready / valid stalls,
// and an optional error window that answers SLVERR. The memory is a byte
// array addressed by the low ADDR_BITS of the AXI address; the testbench
// reads and writes it directly (mem[]).
module axi4_slave_mem #(
  parameter int unsigned ADDR_BITS = 20
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [31:0] awaddr_i,
  input  logic [7:0]  awlen_i,
  input  logic [2:0]  awsize_i,
  input  logic [1:0]  awburst_i,
  input  logic        awvalid_i,
  output logic        awready_o,
  input  logic [63:0] wdata_i,
  input  logic [7:0]  wstrb_i,
  input  logic        wlast_i,
  input  logic        wvalid_i,
  output logic        wready_o,
  output logic [1:0]  bresp_o,
  output logic        bvalid_o,
  input  logic        bready_i,
  input  logic [31:0] araddr_i,
  input  logic [7:0]  arlen_i,
  input  logic [2:0]  arsize_i,
  input  logic [1:0]  arburst_i,
  input  logic        arvalid_i,
  output logic        arready_o,
  output logic [63:0] rdata_o,
  output logic [1:0]  rresp_o,
  output logic        rlast_o,
  output logic        rvalid_o,
  input  logic        rready_i
);

  localparam int unsigned BYTES = 1 << ADDR_BITS;
  logic [7:0] mem [BYTES];

  int max_stall = 3;                 // random stall cycles on each handshake
  logic [31:0] err_lo = 32'hFFFF_FFFF, err_hi = 32'hFFFF_FFFF;   // [lo, hi): SLVERR window
  int n_aw = 0, n_ar = 0, n_wbeats = 0, n_rbeats = 0, n_proto_errors = 0;

  function automatic bit in_err(input logic [31:0] a);
    return (a >= err_lo) && (a < err_hi);
  endfunction

  // ---------------------------------------------------------------------------
  // Write channel
  // ---------------------------------------------------------------------------
  logic [31:0] wr_addr_q; logic [7:0] wr_left_q; logic wr_active_q; logic wr_err_q;
  int wr_stall_q;

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      awready_o <= 1'b0; wready_o <= 1'b0; bvalid_o <= 1'b0; bresp_o <= 2'b00;
      wr_addr_q <= '0; wr_left_q <= '0; wr_active_q <= 1'b0; wr_err_q <= 1'b0; wr_stall_q <= 0;
    end else begin
      // AW
      if (awvalid_i && awready_o) begin
        awready_o   <= 1'b0;
        wr_addr_q   <= awaddr_i;
        wr_left_q   <= awlen_i + 8'd1;
        wr_active_q <= 1'b1;
        wr_err_q    <= in_err(awaddr_i);
        n_aw++;
        if (awsize_i != 3'b011 || awburst_i != 2'b01) begin n_proto_errors++; $display("ERROR @%0t: AW size/burst %0d/%0d", $time, awsize_i, awburst_i); end
        if (awaddr_i[2:0] != 3'b000) begin n_proto_errors++; $display("ERROR @%0t: AW unaligned %h", $time, awaddr_i); end
        if ((awaddr_i[11:0] + 12'(awlen_i) * 12'd8) < awaddr_i[11:0]) begin n_proto_errors++; $display("ERROR @%0t: AW burst crosses 4 KB", $time); end
      end else if (!wr_active_q && !bvalid_o && awvalid_i) begin
        if (wr_stall_q == 0) begin awready_o <= 1'b1; wr_stall_q <= $urandom_range(max_stall); end
        else wr_stall_q <= wr_stall_q - 1;
      end else awready_o <= 1'b0;
      // W
      if (wvalid_i && wready_o) begin
        wready_o <= 1'b0;
        if (!wr_err_q) for (int b = 0; b < 8; b++) if (wstrb_i[b]) mem[(wr_addr_q + 32'(b)) & (BYTES - 1)] = wdata_i[8*b +: 8];
        wr_addr_q <= wr_addr_q + 32'd8;
        wr_left_q <= wr_left_q - 8'd1;
        n_wbeats++;
        if (wlast_i != (wr_left_q == 8'd1)) begin n_proto_errors++; $display("ERROR @%0t: WLAST mismatch", $time); end
        if (wr_left_q == 8'd1) begin wr_active_q <= 1'b0; bvalid_o <= 1'b1; bresp_o <= wr_err_q ? 2'b10 : 2'b00; end
      end else if (wr_active_q && wvalid_i) begin
        if (wr_stall_q == 0) begin wready_o <= 1'b1; wr_stall_q <= $urandom_range(max_stall); end
        else wr_stall_q <= wr_stall_q - 1;
      end else wready_o <= 1'b0;
      // B
      if (bvalid_o && bready_i) bvalid_o <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------
  // Read channel
  // ---------------------------------------------------------------------------
  logic [31:0] rd_addr_q; logic [7:0] rd_left_q; logic rd_active_q; logic rd_err_q;
  int rd_stall_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      arready_o <= 1'b0; rvalid_o <= 1'b0; rdata_o <= '0; rresp_o <= 2'b00; rlast_o <= 1'b0;
      rd_addr_q <= '0; rd_left_q <= '0; rd_active_q <= 1'b0; rd_err_q <= 1'b0; rd_stall_q <= 0;
    end else begin
      if (arvalid_i && arready_o) begin
        arready_o   <= 1'b0;
        rd_addr_q   <= araddr_i;
        rd_left_q   <= arlen_i + 8'd1;
        rd_active_q <= 1'b1;
        rd_err_q    <= in_err(araddr_i);
        n_ar++;
        if (arsize_i != 3'b011 || arburst_i != 2'b01) begin n_proto_errors++; $display("ERROR @%0t: AR size/burst", $time); end
        if (araddr_i[2:0] != 3'b000) begin n_proto_errors++; $display("ERROR @%0t: AR unaligned %h", $time, araddr_i); end
        if ((araddr_i[11:0] + 12'(arlen_i) * 12'd8) < araddr_i[11:0]) begin n_proto_errors++; $display("ERROR @%0t: AR burst crosses 4 KB", $time); end
      end else if (!rd_active_q && arvalid_i) begin
        if (rd_stall_q == 0) begin arready_o <= 1'b1; rd_stall_q <= $urandom_range(max_stall); end
        else rd_stall_q <= rd_stall_q - 1;
      end else arready_o <= 1'b0;

      if (rvalid_o && rready_i) begin
        rvalid_o <= 1'b0;
        n_rbeats++;
        if (rlast_o) rd_active_q <= 1'b0;
      end else if (rd_active_q && !rvalid_o) begin
        if (rd_stall_q == 0) begin
          for (int b = 0; b < 8; b++) rdata_o[8*b +: 8] <= mem[(rd_addr_q + 32'(b)) & (BYTES - 1)];
          rresp_o   <= rd_err_q ? 2'b10 : 2'b00;
          rlast_o   <= (rd_left_q == 8'd1);
          rvalid_o  <= 1'b1;
          rd_addr_q <= rd_addr_q + 32'd8;
          rd_left_q <= rd_left_q - 8'd1;
          rd_stall_q <= $urandom_range(max_stall);
        end else rd_stall_q <= rd_stall_q - 1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */

endmodule

`default_nettype wire
