`default_nettype none

// Shared parameters, encodings and types for the NPU v1.
// Source of truth: docs/architecture.md (sections 3, 4, 9) and
// docs/programming_model.md (sections 1 to 3). Change the documents first.
//
// Parameters that a given module does not use are expected; the package is
// imported by every RTL module, so UNUSEDPARAM is disabled for the whole file.
/* verilator lint_off UNUSEDPARAM */
package npu_pkg;

  // ---------------------------------------------------------------------------
  // Synthesis-time parameters (architecture.md section 9)
  // ---------------------------------------------------------------------------
  localparam int unsigned ARR_N       = 8;       // systolic array edge, also act_buffer bank count
  localparam int unsigned ABUF_BYTES  = 32768;   // activation buffer
  localparam int unsigned WBUF_BYTES  = 131072;  // weight buffer
  localparam int unsigned NUM_DESC    = 16;      // descriptor table entries
  localparam int unsigned ABUF_RD_LAT = 2;       // act_buffer read latency, cycles
  localparam int unsigned WBUF_RD_LAT = 1;       // weight_buffer read latency, cycles

  localparam int unsigned ABUF_AW  = $clog2(ABUF_BYTES);      // byte address width
  localparam int unsigned WBUF_WAW = $clog2(WBUF_BYTES / 8);  // 64-bit word address width

  // ---------------------------------------------------------------------------
  // Numeric formats (architecture.md section 3)
  // ---------------------------------------------------------------------------
  localparam int unsigned ACT_W      = 8;   // int8 activation
  localparam int unsigned WGT_W      = 8;   // int8 weight
  localparam int unsigned PROD_W     = 16;  // int8 x int8 product
  localparam int unsigned ACC_W      = 32;  // int32 accumulator, bias, requant multiplier
  localparam int unsigned RQ_SHIFT_W = 6;   // requant shift, documented range 1..62
  localparam int unsigned RQ_PIPE_LAT = 3;  // requant_unit fixed pipeline latency

  // ---------------------------------------------------------------------------
  // Host address map, 20-bit internal address (programming_model.md section 1)
  // ---------------------------------------------------------------------------
  localparam logic [19:0] REG_BASE  = 20'h00000;
  localparam logic [19:0] ABUF_BASE = 20'h10000;
  localparam logic [19:0] WBUF_BASE = 20'h20000;

  localparam logic [31:0] NPU_ID = 32'h4E50_5501;  // "NPU" + version 0x01

  // Register offsets inside the 4 KB register window (programming_model.md section 2)
  localparam logic [11:0] REG_ID              = 12'h000;
  localparam logic [11:0] REG_CTRL            = 12'h004;
  localparam logic [11:0] REG_STATUS          = 12'h008;
  localparam logic [11:0] REG_LAYER_CNT       = 12'h00C;
  localparam logic [11:0] REG_RESULT          = 12'h010;
  localparam logic [11:0] REG_LOGITS_BASE     = 12'h020;  // 16 words
  localparam logic [11:0] REG_PERF_TOTAL      = 12'h060;
  localparam logic [11:0] REG_PERF_ARRAY_BUSY = 12'h064;
  localparam logic [11:0] REG_PERF_LAYER_BASE = 12'h080;  // NUM_DESC words
  localparam logic [11:0] REG_DESC_BASE       = 12'h100;  // NUM_DESC x 16 words
  localparam int unsigned DESC_WORDS          = 16;

  // ---------------------------------------------------------------------------
  // Encodings
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    OP_NOP        = 4'd0,
    OP_CONV2D     = 4'd1,
    OP_MAXPOOL2X2 = 4'd2,
    OP_FC         = 4'd3,
    OP_ARGMAX     = 4'd4
  } op_e;

  typedef enum logic [7:0] {
    ERR_NONE          = 8'h00,
    ERR_BAD_OP        = 8'h01,
    ERR_BAD_PARAM     = 8'h02,
    ERR_BAD_ADDR      = 8'h03,
    ERR_BAD_LAYER_CNT = 8'h04,
    ERR_BAD_OUT_DIM   = 8'h05
  } err_e;

  // ---------------------------------------------------------------------------
  // Layer descriptor (programming_model.md section 3)
  // Fields are listed from the highest word down so that the packed order reads
  // W9 .. W0 top to bottom. The register file maps AXI words onto these fields.
  // ---------------------------------------------------------------------------
  typedef struct packed {
    // W9
    logic signed [ACT_W-1:0]      act_max;
    logic signed [ACT_W-1:0]      act_min;
    logic signed [ACT_W-1:0]      z_out;
    logic signed [ACT_W-1:0]      z_in;
    // W8
    logic [RQ_SHIFT_W-1:0]        rq_shift;
    // W7
    logic signed [ACC_W-1:0]      rq_mult;
    // W6 .. W3
    logic [31:0]                  b_base;
    logic [31:0]                  w_base;
    logic [31:0]                  out_base;
    logic [31:0]                  in_base;
    // W2
    logic [15:0]                  c_out;
    logic [15:0]                  c_in;
    // W1
    logic [15:0]                  in_w;
    logic [15:0]                  in_h;
    // W0
    logic [1:0]                   stride_m1;
    logic [3:0]                   pad;
    logic [3:0]                   k;
    op_e                          op;
  } desc_t;

endpackage
/* verilator lint_on UNUSEDPARAM */

`default_nettype wire
