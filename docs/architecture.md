# NPU v1 Architecture

Version: v1 (2026-10-29). This document exists before the RTL: any interface change is made here first, then in the code.
Companion documents: `docs/programming_model.md` (registers, descriptors, host flow),
`docs/verification.md` (verification plan), `docs/decisions.md` (why each key decision was made and what was rejected).

## 0. One-sentence summary

A layer-descriptor-driven INT8 CNN inference accelerator: an 8x8 output-stationary systolic array,
on-chip activation / weight SRAM, an AXI4-Lite host interface, and integer arithmetic that matches a Python
golden model bit for bit. The same RTL runs the MNIST and CIFAR-10 versions of LeNet-5 by changing only the
descriptors and the weights.

## 1. Scope

### 1.1 Goals

| Item | v1 decision |
| --- | --- |
| Target networks | LeNet-5 class small CNNs. One descriptor set and weight set each for MNIST (28x28x1) and CIFAR-10 (32x32x3) |
| Operators | CONV2D (K = 1..7, stride 1, pad 0..7, fused bias / requant / clamp), MAXPOOL 2x2 stride 2, FC (as a 1x1 CONV2D special case), ARGMAX |
| Numerics | int8 activations (asymmetric, with zero point), int8 weights (symmetric, per-tensor), int32 accumulation, integer requantization |
| Compute core | 8x8 output-stationary systolic array; parameter `ARR_N` can be 4 or 16 |
| On-chip memory | 32 KB activation buffer, 128 KB weight buffer |
| Host interface | AXI4-Lite slave, 32-bit data, covering the registers and both on-chip memories |
| Verification | numpy integer golden model, bit-exact per layer; Verilator reproducible locally and in GitHub Actions; VCS as an additional check |

### 1.2 Non-goals (explicitly out of v1)

- Training, floating point, dynamic shapes, a compiler, multi-core, a DRAM controller.
- Stride-2 convolution (downsampling is done by maxpool), per-channel quantization, depthwise convolution, batch > 1.
- PLL, SPI, multiple clock domains, pads: left for the M6 `chip_top` wrapper.
- External-memory DMA (AXI4 master): M6. In v1 the host writes weights and images into on-chip memory over AXI4-Lite.

Section 10 gives an extension path for each of these; the v1 interfaces reserve room for them.

## 2. Top-level structure

```text
                      AXI4-Lite slave (s_axil_*)                 irq_o
                               |                                   ^
  +----------------------------v-----------------------------------+-----------+
  | npu_top                                                                    |
  |                                                                            |
  |  axil_slave ---> npu_regs ------------------> npu_sequencer                |
  |  (protocol)      ctrl/status/result            fetch descriptor per layer, |
  |                  descriptor table x16          start the engine, wait done |
  |                  perf counters                        |                    |
  |                      |                 +--------------+--------------+     |
  |         host access to on-chip memory  |              |              |     |
  |                      |            conv_engine    pool_engine   argmax_engine|
  |                      |            +----------+                             |
  |                      |            |conv_addr_gen                           |
  |                      |            |systolic_array 8x8                      |
  |                      |            |requant_unit x8                         |
  |                      |            |act_writer |                            |
  |                      v            +----------+                             |
  |     +----------- act_buffer (32 KB, 8 byte banks, 1R1W, unaligned 8-byte)  |
  |     +----------- weight_buffer (128 KB, 64-bit words, byte enables)        |
  +----------------------------------------------------------------------------+
```

Module-to-file mapping (one primary module per file, file name equals module name):

| Module | File | Responsibility |
| --- | --- | --- |
| `npu_top` | `rtl/npu_top.sv` | Top-level wiring; externally only clk / rst / AXI4-Lite / irq |
| `npu_pkg` | `rtl/npu_pkg.sv` | Parameters, opcodes, descriptor struct, address-map constants |
| `axil_slave` | `rtl/host/axil_slave.sv` | AXI4-Lite protocol to an internal single-cycle req/ack bus |
| `npu_regs` | `rtl/host/npu_regs.sv` | Registers, descriptor table, perf counters, host routing to on-chip memory |
| `npu_sequencer` | `rtl/ctrl/npu_sequencer.sv` | Layer-loop FSM, error checks, per-layer timing |
| `conv_engine` | `rtl/compute/conv_engine.sv` | Tile loops, address stream, drain and write-back for CONV2D / FC |
| `conv_addr_gen` | `rtl/compute/conv_addr_gen.sv` | Expands (n_tile, oh, ow0, k) into read addresses, lane masks and control flags |
| `systolic_array` | `rtl/compute/systolic_array.sv` | ARR_N x ARR_N PE array, edge skew, shadow accumulators and column drain |
| `pe` | `rtl/compute/pe.sv` | int8 x int8 multiply-accumulate, int32 accumulator, shadow register |
| `requant_unit` | `rtl/compute/requant_unit.sv` | Bias, fixed-point multiply, rounding shift, zero point, clamp |
| `pool_engine` | `rtl/compute/pool_engine.sv` | 2x2 stride-2 maxpool |
| `argmax_engine` | `rtl/compute/argmax_engine.sv` | Reads the logits, outputs the index of the maximum |
| `act_buffer` | `rtl/mem/act_buffer.sv` | Activation SRAM built from 8 byte banks; 8-byte reads and writes at any byte address |
| `weight_buffer` | `rtl/mem/weight_buffer.sv` | 64-bit wide weight SRAM with byte enables |
| `sram_1r1w` | `rtl/mem/sram_1r1w.sv` | Behavioural 1-read 1-write SRAM; the only boundary to replace with a technology macro |

## 3. Numeric formats

### 3.1 Quantization scheme

Affine quantization with the same structure as TFLite int8 inference:

```text
real = scale * (q - zero_point)
```

| Tensor | Type | Zero point | Scale granularity |
| --- | --- | --- | --- |
| Input image | int8 | -128 (pixel 0..255 minus 128) | per-tensor, scale = 1/255 |
| Layer activations | int8 | any, from calibration | per-tensor |
| Weights | int8 | 0 (symmetric) | per-tensor |
| Bias | int32 | 0 | scale = s_in * s_w |
| Accumulator | int32 | - | - |

Accumulator safety: the longest dot product is FC1 with length 400; each term is at most 128 * 127, so the sum is
about 6.5e6, far below 2^31. v1 bounds K_total at 4096 (see the error codes in `programming_model.md`).

### 3.2 Zero-point folding

The real convolution is `sum_k (q_in[k] - z_in) * q_w[k]`. So that the PE only computes int8 x int8, the
zero-point correction is folded into the bias by the export script:

```text
bias_folded[n] = bias_q[n] - z_in * sum_k q_w[k, n]
acc[n]         = sum_k q_in[k] * q_w[k, n]          (the only thing the hardware does)
```

This holds only if padded input positions carry the value `z_in`, not 0. `conv_addr_gen` fills out-of-range
lanes with `z_in`.

### 3.3 Requantization

Once per output element:

```text
sum     = acc + bias_folded[n]                        // 33-bit signed
prod    = sum * rq_mult                               // rq_mult is a positive int32 in [2^30, 2^31)
rounded = (prod + (1 << (rq_shift - 1))) >>> rq_shift // arithmetic shift, round half up
q_out   = clamp(rounded + z_out, act_min, act_max)    // int8 result
```

- `rq_mult / 2^rq_shift` approximates the real multiplier `s_in * s_w / s_out`; the export script normalizes it.
  `rq_shift` is 1..62. `rq_shift = 0` is outside the legal range, but both the RTL and the model define it as
  "no rounding term, no shift" so that an illegal descriptor cannot produce X.
- `act_min <= act_max` is required. If violated, the order is: above `act_max` gives `act_max`, otherwise below
  `act_min` gives `act_min`, otherwise the value itself.
- ReLU is fused as `act_min = z_out` (the quantized value of real 0). Without ReLU, `act_min = -128, act_max = 127`.
- The Python golden model implements exactly these four steps with Python big integers; that is the basis of bit-exactness.

### 3.4 Maxpool and argmax

- Quantization is monotonic, so maxpool compares int8 values directly and the output keeps the input scale and
  zero point; no requantization.
- argmax does a signed comparison of the 10 int8 logits from FC3; ties go to the lowest index.

## 4. Tensor layout and on-chip memory

### 4.1 Layout

- All activations use CHW layout (channel-major, then row, then column), one byte per element, at
  `base + c*H*W + h*W + w`.
- The FC input vector is the natural flattening of the previous CHW tensor, which is the same order PyTorch
  produces with `view(-1)` on NCHW; no reordering is needed.
- Weights are stored in the GEMM view: `W[K_total][N_pad8]`, where `K_total = C_in * K * K`, k is expanded in
  `(ci, kh, kw)` order with kw innermost, and `N_pad8` is C_out rounded up to a multiple of 8 with the extra
  columns zero. One row `W[k][n0 .. n0+7]` is therefore exactly one aligned 64-bit word.
- The bias is an int32 array `bias[N_pad8]` in the weight buffer, pointed to by the descriptor's `b_base`.

### 4.2 act_buffer

- 32 KB, built from 8 banks of 4 KB x 8-bit 1R1W SRAM; the bank number is the low 3 bits of the byte address.
- Read port: 8 consecutive bytes starting at any byte address A, issued in one cycle, returned after a fixed
  2-cycle latency. Internally bank b reads index `(A + ((b - A[2:0]) mod 8)) >> 3` and the output is rotated by
  `A[2:0]` back into natural order.
- Write port: the same unaligned 8-byte write with an 8-bit byte enable.
- Two clients: the engines (conv / pool / argmax share one read and one write port) and the host. Host accesses
  while the NPU is busy are rejected with SLVERR.
- Why 8 byte banks: 8 consecutive addresses always fall into 8 distinct banks, so stride-1 convolution never has a
  bank conflict. This follows directly from the output-stationary dataflow, see `decisions.md`.

### 4.3 weight_buffer

- 128 KB, 64 bits wide, 16K words, one read port plus one write port with byte enables (a 32-bit host write uses half a word).
- The read port feeds the conv_engine weight stream, one word per cycle (8 int8 values, the 8 columns of one n_tile).
- The bias is read from here too. To avoid contending with the weight stream for the read port, the 8 biases of an
  n_tile are prefetched into one of two register banks (bank = n_tile[0]) in the idle cycles before its first tile,
  taking 4 word reads; `b_base` is therefore 8-byte aligned. The other bank still serves the previous n_tile's drain.

### 4.4 Capacity budget (LeNet-5 MNIST, int8)

| Tensor | Size | Location |
| --- | --- | --- |
| Input 28x28x1 | 784 B | act_buffer |
| conv1 output 6x28x28 | 4704 B | act_buffer |
| pool1 output 6x14x14 | 1176 B | act_buffer |
| conv2 output 16x10x10 | 1600 B | act_buffer |
| pool2 / fc1 / fc2 / fc3 outputs | 400 / 120 / 84 / 10 B | act_buffer |
| All weights (N padded to 8) | about 62.5 KB | weight_buffer |
| All biases | about 1 KB | weight_buffer |

Activations total under 9 KB, so every tensor can stay resident without overlap and no ping-pong management is
needed; weights plus alignment gaps are about 67 KB.

## 5. Compute core: 8x8 output-stationary systolic array

### 5.1 GEMM view

A CONV2D layer is `Out[M][N] = In[M][K_total] x W[K_total][N]`, with `M = H_out * W_out` (output pixels),
`N = C_out`, `K_total = C_in * K * K`. `In[m][k]` is never materialized in memory; `conv_addr_gen` generates its
addresses on demand (im2col by address generation). FC is the special case `H = W = 1, K = 1`, so `K_total = C_in`.

### 5.2 PE and array

- PE(i, j) owns one output element `Out[m0 + i][n0 + j]`; its int32 accumulator stays inside the PE (output-stationary).
- Each cycle, activations flow left to right along rows (`a` is delayed one cycle per PE) and weights flow top to
  bottom along columns (`w` likewise). The edges skew `a[i]` by i cycles and `w[j]` by j cycles, so
  `In[m0+i][k]` and `W[k][n0+j]` meet at PE(i, j).
- Control flags travel with `a`: `valid` (data present this cycle), `first` (k = 0, load the accumulator after
  clearing), `last` (k = K_total - 1, copy the accumulator into the shadow register).
- A tile is the K_total cycles of k = 0 .. K_total - 1 for a fixed (n_tile, oh, ow0). Tiles run back to back with no idle cycles.

### 5.3 Drain

- `last` reaches PE(i, j) at `tile_start + K_total - 1 + i + j`. The shadow register is written only in that cycle
  and then holds for K_total cycles.
- Column drain: in the cycle after the clock edge at which PE(ARR_N-1, 0) consumes `last` (ARR_N - 1 edges after
  the edge that consumed that beat at the array input), `drain_valid_o` rises and one whole column
  `j = 0, 1, ..., ARR_N-1` (8 int32 values, one per m) is presented per cycle through an 8:1 mux directly from the
  shadows; there is no shift chain. Column j is presented exactly when its row ARR_N-1 shadow becomes valid, so the
  mux output is combinational from the shadow registers and the consumer registers it.
- No-conflict condition: tile period `P >= ARR_N`. Hence `P = max(K_total, ARR_N)`; when K_total is below 8,
  `conv_addr_gen` inserts idle cycles.
- The 8 values of one drained column belong to the same output channel `n = n0 + j`, which in CHW layout are 8
  consecutive bytes: exactly one unaligned `act_buffer` write. This is why the drain is by column, not by row.

### 5.4 conv_engine loop nest

```text
for n_tile in 0 .. ceil(C_out / 8) - 1:
    prefetch bias[n_tile*8 .. +7] into registers (4 cycles)
    for oh in 0 .. H_out - 1:
        for ow0 in 0, 8, 16, ... < W_out:            # an m_tile never crosses an output row
            for ci, kh, kw (i.e. k = 0 .. K_total - 1):
                ih    = oh - pad + kh
                A     = in_base + ci*H*W + ih*W + (ow0 - pad + kw)
                lane i input = (ih out of range, or ow0+i-pad+kw out of range, or ow0+i >= W_out) ? z_in : act[A + i]
                weight word  = w_base_word + k * (N_pad8 / 8) + n_tile
            drain 8 columns -> requant x8 -> write out_base + n*H_out*W_out + oh*W_out + ow0, lane mask ow0+i < W_out
```

- Read addresses and control flags go through a pipeline matched to the SRAM read latency before entering the array.
- Write-back overlaps the next tile's compute, because the drain takes only 8 cycles and `P >= 8`.
- Drained results for padding columns `n >= C_out` are discarded.

### 5.5 Throughput estimate (LeNet-5 MNIST, cycles)

| Layer | n_tile | oh | ow_tile | K_total | Cycles | Useful MACs |
| --- | --- | --- | --- | --- | --- | --- |
| conv1 5x5, 1->6, pad 2 | 1 | 28 | 4 | 25 | 2800 | 117.6K |
| pool1 | - | - | - | - | about 700 | - |
| conv2 5x5, 6->16 | 2 | 10 | 2 | 150 | 6000 | 240K |
| pool2 | - | - | - | - | about 350 | - |
| fc1 400->120 | 15 | 1 | 1 | 400 | 6000 | 48K |
| fc2 120->84 | 11 | 1 | 1 | 120 | 1320 | 10.1K |
| fc3 84->10 | 2 | 1 | 1 | 84 | 168 | 0.84K |
| Total | | | | | about 17.5K | 416.5K |

Peak is 64 MAC/cycle, the ideal is 6.5K cycles, so the expected utilization is about 37%. All three losses are
structural and worth understanding: C_out = 6 uses only 6 of 8 columns; with W_out = 28 the last m_tile uses only
4 lanes; FC has a single output pixel, uses 1 of 8 rows, and is bound by weight bandwidth (8 bytes of weights per
cycle support at most 8 MACs). The perf counters record per-layer cycles and array-active cycles so that the
README can report measured numbers.

## 6. pool_engine and argmax_engine

- `pool_engine`: for each channel and each output row, reads input rows 2r and 2r+1, 16 bytes each (two 8-byte
  reads), produces 8 outputs and does one 8-byte write. Output size rounds down, as PyTorch does by default.
- `argmax_engine`: reads `c_in` int8 values (v1 limit c_in <= 16) from `in_base`, compares signed, writes the index
  to the RESULT register and the first 16 logits to the LOGITS registers for debugging.

## 7. Module interfaces

Naming: inputs end in `_i`, outputs in `_o`, the active-low reset is `rst_ni`. Multi-bit vectors use packed arrays.

### 7.1 npu_top

| Port | Dir | Description |
| --- | --- | --- |
| `clk_i`, `rst_ni` | in | Single clock domain, asynchronous active-low reset (synchronized outside the top) |
| `s_axil_aw*_i/o`, `s_axil_w*`, `s_axil_b*`, `s_axil_ar*`, `s_axil_r*` | - | AXI4-Lite slave, 32-bit ADDR (low 20 bits used internally), 32-bit DATA |
| `irq_o` | out | Level interrupt, `STATUS.DONE & CTRL.IRQ_EN` |

### 7.2 Internal bus from axil_slave to npu_regs

| Signal | Description |
| --- | --- |
| `wr_req_o`, `wr_addr_o[19:0]`, `wr_data_o[31:0]`, `wr_strb_o[3:0]` | Single-cycle write request |
| `wr_ack_i`, `wr_err_i` | Response from the next cycle on; err maps to SLVERR |
| `rd_req_o`, `rd_addr_o[19:0]` | Single-cycle read request |
| `rd_ack_i`, `rd_data_i[31:0]`, `rd_err_i` | Read response |

The AXI4-Lite side handles one transaction at a time with no outstanding requests. A write is accepted only when
AW and W are both valid; when a write and a read are requested in the same cycle the write is taken first. The
responder answers a request at least one cycle later; the err flags map to SLVERR. `awprot` / `arprot` are
ignored and only the low 20 address bits are decoded.

`npu_regs` forwards accesses to the two memory windows on a host memory port while the NPU is idle:

| Signal | Description |
| --- | --- |
| `hm_req_o`, `hm_we_o`, `hm_sel_o` | Single-cycle request; `hm_sel_o` = 0 act_buffer, 1 weight_buffer |
| `hm_addr_o[19:0]`, `hm_wdata_o[31:0]`, `hm_be_o[3:0]` | Byte offset inside the selected buffer (4-byte aligned), write data, byte enables |
| `hm_ack_i`, `hm_rdata_i[31:0]` | Completion, at least one cycle after the request |

The adapter from this 32-bit port to the 64-bit buffer ports is part of M4.

### 7.3 npu_regs to npu_sequencer

| Signal | Description |
| --- | --- |
| `start_i` (pulse), `layer_cnt_i[4:0]` | Produced by CTRL.START; `npu_regs` already suppresses START while busy |
| `desc_i[NUM_DESC]` (array of `desc_t`) | Read-only view of the descriptor table, decoded from the stored words |
| `busy_o`, `done_o` (pulse), `err_o` (pulse), `err_code_o`, `cur_layer_o[3:0]` | Status; busy is already low in the cycle done or err pulses |
| `layer_cycles_o[NUM_DESC][31:0]`, `total_cycles_o[31:0]`, `array_busy_cycles_o[31:0]` | Perf counters, cleared on an accepted start; `array_busy_cycles_o` counts busy cycles with `array_valid_i` high |
| `array_valid_i` | The systolic array's `in_valid`, from conv_engine |

`npu_regs` also receives the argmax result (`argmax_valid_i`, `argmax_idx_i[3:0]`, `argmax_logits_i[16][7:0]`)
and drives `irq_o = DONE & IRQ_EN`.

### 7.4 npu_sequencer to the engines

One `eng_desc_o` (the current layer's `desc_t`, held for the whole layer) shared by the engines, and per engine a
`start` pulse plus `done` (pulse), `err` (pulse) and `err_code` back. CONV2D and FC both start `conv_engine`; FC is
forwarded as a 1x1 CONV2D with `in_h = in_w = 1`, `pad = 0`, stride 0. All static descriptor checks (opcode,
parameter ranges, output size, address ranges and alignment) are done once in the sequencer before the start pulse,
in the order ERR_BAD_OP, ERR_BAD_PARAM, ERR_BAD_OUT_DIM, ERR_BAD_ADDR; an engine only reports errors it detects
while running. NOP layers complete without an engine.

### 7.5 Engine to act_buffer

| Signal | Description |
| --- | --- |
| `rd_en_i`, `rd_addr_i[14:0]` | Byte address, any alignment |
| `rd_data_o[63:0]`, `rd_valid_o` | Fixed 2-cycle latency, lane i at `[8*i +: 8]` |
| `wr_en_i`, `wr_addr_i[14:0]`, `wr_data_i[63:0]`, `wr_be_i[7:0]` | Single-cycle write |

### 7.6 Engine to weight_buffer

| Signal | Description |
| --- | --- |
| `rd_en_i`, `rd_word_addr_i[13:0]` | 64-bit word address |
| `rd_data_o[63:0]`, `rd_valid_o` | Fixed 1-cycle latency |

### 7.7 systolic_array

Parameter `N` (default `ARR_N`).

| Port | Description |
| --- | --- |
| `in_valid_i`, `in_first_i`, `in_last_i` | k-position flags for this cycle's input vectors; `first` / `last` only count when `valid` is 1 |
| `act_i[N][7:0]`, `wgt_i[N][7:0]` | Unskewed inputs; lane i of `act_i` is row i, lane j of `wgt_i` is column j; the skew happens inside the module |
| `drain_valid_o` | High for N consecutive cycles per tile, starting N - 1 edges after the edge that consumed the tile's `last` beat |
| `drain_col_idx_o[$clog2(N)-1:0]` | 0 .. N-1 during a drain |
| `drain_data_o[N][31:0]` | Column `drain_col_idx_o` of the shadows, lane i = row i; combinational from the shadow registers |

Contract: at least N cycles between consecutive `last` beats at the input (tile period `P >= N`), counting idle
cycles. Idle cycles may appear anywhere, including inside a tile. A `last` that arrives while a drain is in progress
restarts the drain and corrupts the previous tile; that is a producer bug, not a condition the array recovers from.

### 7.8 requant_unit

Inputs `acc_i[31:0]`, `bias_i[31:0]`, `mult_i[31:0]`, `shift_i[5:0]`, `z_out_i[7:0]`, `act_min_i[7:0]`,
`act_max_i[7:0]`, `valid_i`; outputs `q_o[7:0]`, `valid_o`. Fixed 3-stage pipeline (`RQ_PIPE_LAT`) split along
the formula in section 3.3: S1 does the 33-bit bias add, S2 the 33 x 32 signed multiply, S3 the rounding shift,
zero-point add and clamp. `valid_o` is exactly `valid_i` delayed by 3 cycles; data registers only toggle when valid.

### 7.9 pe

| Port | Dir | Description |
| --- | --- | --- |
| `valid_i`, `first_i`, `last_i` | in | Control flags that travel with the activation along the row. `first` / `last` are only meaningful when `valid` is 1 |
| `a_i[7:0]` | in | int8 activation from the left neighbour |
| `w_i[7:0]` | in | int8 weight from the neighbour above |
| `valid_o`, `first_o`, `last_o`, `a_o[7:0]` | out | Forwarded to the right neighbour with a fixed 1-cycle delay, not gated by valid |
| `w_o[7:0]` | out | Forwarded to the neighbour below with a fixed 1-cycle delay |
| `shadow_o[31:0]` | out | Accumulation result captured in the `last` cycle, held until the next `last` |

Behaviour: when `valid_i` is 1, `acc <= first_i ? a*w : acc + a*w`, and if `last_i` is also 1 then
`shadow <= the same new value`. With `K_total = 1`, `first` and `last` coincide and the shadow receives the single
product. The forwarding registers update unconditionally every cycle so that the array skew is a fixed delay line.

## 8. Clock, reset, interrupt, errors

- v1 is a single clock domain. `rst_ni` is synchronized outside `npu_top`; inside, every `always_ff` uses an
  asynchronous reset with synchronous release.
- The interrupt is a level signal; it deasserts after software writes 1 to clear `STATUS.DONE`.
- Errors are fatal: the sequencer stops, `STATUS.ERR` is set, `ERR_CODE` gives the reason and `CUR_LAYER` the
  failing layer; software writes 1 to clear and restarts with START.
- Host access to on-chip memory while busy: SLVERR, no effect, and `STATUS.HOST_CONFLICT` is set (non-fatal, debug only).

## 9. Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `ARR_N` | 8 | Array edge, also the act_buffer bank count; must be a power of two |
| `ABUF_BYTES` | 32768 | Activation buffer capacity |
| `WBUF_BYTES` | 131072 | Weight buffer capacity |
| `NUM_DESC` | 16 | Descriptor table entries |
| `ABUF_RD_LAT` | 2 | act_buffer read latency |
| `WBUF_RD_LAT` | 1 | weight_buffer read latency |

## 10. Known limitations and extension paths

| Limitation | Reason | Extension |
| --- | --- | --- |
| Stride must be 1 | 8 lanes with stride 2 land in 4 banks and conflict | Read over two cycles, or double the bank count |
| Per-tensor quantization | Fewer descriptor fields and no requant lookup | Store mult / shift per channel in the weight buffer and prefetch them with the bias |
| FC utilization 1/8 | A single output pixel occupies one row; weight-bandwidth bound | Batch several images, or split K across the 8 rows and add a reduction tree |
| No DMA | v1 focuses on the compute core | M6 adds an AXI4 master that moves weights and images from external memory into the two buffers |
| Single clock domain | Focus on the compute core | M6 `chip_top` adds a pixel-stream async FIFO, an APB or SPI configuration bridge and clock gating |
| Weights re-read per m_tile | Inherent cost of output-stationary | On-chip SRAM bandwidth is sufficient here; with external memory, add a weight cache or move to weight-stationary |
