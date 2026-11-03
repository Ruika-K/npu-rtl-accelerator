# NPU v1 Programming Model

Everything the host sees over AXI4-Lite: address map, registers, descriptor format, run sequence, error codes.
Companion to `docs/architecture.md`. Once a register address is referenced by both the RTL and the Python scripts
it can only be appended to, never moved.

## 1. Address map

AXI4-Lite decodes only the low 20 address bits: a 1 MB window.

| Range | Size | Contents | Access |
| --- | --- | --- | --- |
| `0x00000 - 0x00FFF` | 4 KB | Registers and descriptor table | Any time |
| `0x10000 - 0x17FFF` | 32 KB | act_buffer (byte addressed; 32-bit accesses must be 4-byte aligned) | Only while the NPU is idle |
| `0x20000 - 0x3FFFF` | 128 KB | weight_buffer (byte addressed; 32-bit accesses must be 4-byte aligned) | Only while the NPU is idle |
| Other | - | Unmapped | Returns SLVERR |

`in_base / out_base` in a descriptor are byte offsets inside act_buffer; `w_base / b_base` are byte offsets inside
weight_buffer. The host adds the window base `0x10000` or `0x20000` when writing.

## 2. Registers

All registers are 32 bits. Types: RO read-only, RW read-write, W1P write-1-to-pulse (reads as 0), W1C write-1-to-clear.

| Offset | Name | Bits | Type | Reset | Description |
| --- | --- | --- | --- | --- | --- |
| `0x000` | ID | [31:0] | RO | `0x4E505501` | ASCII "NPU" + version 0x01 |
| `0x004` | CTRL | [0] START | W1P | 0 | Start one inference; ignored while busy. An accepted START clears DONE, ERR, ERR_CODE, RESULT.VALID and the perf counters |
| | | [1] IRQ_EN | RW | 0 | Enable `irq_o` |
| `0x008` | STATUS | [0] BUSY | RO | 0 | Running |
| | | [1] DONE | W1C | 0 | One inference finished |
| | | [2] ERR | W1C | 0 | Stopped on error; see ERR_CODE and CUR_LAYER |
| | | [3] HOST_CONFLICT | W1C | 0 | A host access to on-chip memory was rejected while busy |
| | | [11:8] CUR_LAYER | RO | 0 | Index of the current (or failing) descriptor |
| | | [23:16] ERR_CODE | RO | 0 | See section 5 |
| `0x00C` | LAYER_CNT | [4:0] | RW | 0 | Number of descriptors to execute, 1 to 16 |
| `0x010` | RESULT | [3:0] ARGMAX | RO | 0 | Result of the most recent ARGMAX |
| | | [8] VALID | RO | 0 | Cleared on START, set when ARGMAX completes |
| `0x020 - 0x05C` | LOGITS[0..15] | [31:0] | RO | 0 | int8 values read by ARGMAX, sign-extended to 32 bits |
| `0x060` | PERF_TOTAL | [31:0] | RO | 0 | Cycles from START to DONE |
| `0x064` | PERF_ARRAY_BUSY | [31:0] | RO | 0 | Cycles with the systolic array's `in_valid` high |
| `0x080 - 0x0BC` | PERF_LAYER[0..15] | [31:0] | RO | 0 | Cycles per descriptor |
| `0x100 - 0x4FF` | DESC[0..15] | 16 words | RW | 0 | Descriptor table, 64 bytes per entry, see section 3 |

The perf registers clear on START. Undefined offsets inside the 4 KB register window read as 0 and ignore writes
(no error, for forward compatibility); writes to read-only registers are ignored. Reserved bits of descriptor words
read as 0. Misaligned addresses (bits [1:0] not zero) return SLVERR in every window.

## 3. Descriptor format

Each descriptor is 16 32-bit words at offset `0x100 + 64 * idx`.

| Word | Bits | Field | Description |
| --- | --- | --- | --- |
| W0 | [3:0] | OP | 0 NOP, 1 CONV2D, 2 MAXPOOL2X2, 3 FC, 4 ARGMAX |
| | [7:4] | K | Kernel size 1..7. FC fixes it to 1 |
| | [11:8] | PAD | Same padding on all four sides, 0..7, filled with Z_IN |
| | [13:12] | STRIDE_M1 | Stride minus 1; must be 0 in v1 |
| | [31:14] | Reserved | Write 0 |
| W1 | [15:0] | IN_H | Input height. 1 for FC |
| | [31:16] | IN_W | Input width. 1 for FC |
| W2 | [15:0] | C_IN | Input channels. Input vector length for FC, number of logits for ARGMAX |
| | [31:16] | C_OUT | Output channels. Output vector length for FC |
| W3 | [31:0] | IN_BASE | Byte offset inside act_buffer |
| W4 | [31:0] | OUT_BASE | Byte offset inside act_buffer |
| W5 | [31:0] | W_BASE | Byte offset inside weight_buffer, 8-byte aligned |
| W6 | [31:0] | B_BASE | Byte offset inside weight_buffer, 4-byte aligned |
| W7 | [31:0] | RQ_MULT | Positive int32 in `[2^30, 2^31)` |
| W8 | [5:0] | RQ_SHIFT | 1 to 62 |
| W9 | [7:0] | Z_IN | Input zero point, int8 |
| | [15:8] | Z_OUT | Output zero point, int8 |
| | [23:16] | ACT_MIN | Output lower bound, int8. Equals Z_OUT for ReLU |
| | [31:24] | ACT_MAX | Output upper bound, int8 |
| W10 - W15 | | Reserved | Write 0 |

Fields used per OP:

| OP | Uses | Ignores |
| --- | --- | --- |
| CONV2D | All | STRIDE_M1 must be 0 |
| FC | Same as CONV2D; hardware forces K = 1, PAD = 0, IN_H = IN_W = 1 | |
| MAXPOOL2X2 | IN_H, IN_W, C_IN, IN_BASE, OUT_BASE | Everything else |
| ARGMAX | C_IN (<= 16), IN_BASE | Everything else |

Output sizes are computed by hardware: `OUT = IN + 2*PAD - K + 1` (CONV2D), `OUT = IN / 2` rounded down (MAXPOOL).

## 4. Run sequence

1. After reset, read ID and check for `0x4E505501`.
2. Write the weight blob produced by the export script to weight_buffer starting at `0x20000`. The blob already
   contains all layers' weights, biases and alignment gaps, so a single sequential write is enough.
3. Write DESC[0 .. L-1] and LAYER_CNT = L.
4. Write the input image (int8, already minus 128, CHW) to act_buffer at DESC[0].IN_BASE.
5. Optionally set CTRL.IRQ_EN = 1.
6. Write CTRL.START = 1.
7. Wait for `irq_o` or poll STATUS.DONE. Read RESULT and LOGITS; read PERF_* for performance data.
8. Write 1 to clear STATUS.DONE. For the next image repeat steps 4, 6, 7 and 8; weights and descriptors stay resident.

## 5. Error codes

| ERR_CODE | Meaning |
| --- | --- |
| `0x00` | No error |
| `0x01` | Illegal OP |
| `0x02` | Unsupported parameter: K = 0 or > 7, PAD > 7, STRIDE_M1 != 0 (CONV2D), C_IN or C_OUT = 0, K_total > 4096, RQ_SHIFT = 0 or 63, RQ_MULT outside `[2^30, 2^31)`, ACT_MIN > ACT_MAX, ARGMAX C_IN = 0 or > 16 |
| `0x03` | Address out of range or misaligned: input, output, weight or bias region exceeds its buffer, W_BASE not 8-byte aligned, B_BASE not 4-byte aligned |
| `0x04` | LAYER_CNT = 0 or > 16 |
| `0x05` | Output size <= 0 (K > IN + 2*PAD) |

On error the sequencer stops immediately, BUSY clears, ERR is set and CUR_LAYER points at the failing descriptor.
The checks run in the order `0x01`, `0x02`, `0x05`, `0x03`; the first failing check sets the code. `0x04` is
checked at START before any descriptor. All checks happen before a layer's engine starts, so a layer that fails has
not touched memory. ERR_CODE keeps its value after ERR is cleared and resets on the next accepted START.

## 6. Example: LeNet-5 on MNIST

act_buffer layout (byte offsets):

| Tensor | Offset | Size |
| --- | --- | --- |
| Input 1x28x28 | `0x0000` | 784 |
| conv1 output 6x28x28 | `0x0400` | 4704 |
| pool1 output 6x14x14 | `0x1800` | 1176 |
| conv2 output 16x10x10 | `0x1D00` | 1600 |
| pool2 output 400 | `0x2400` | 400 |
| fc1 output 120 | `0x2600` | 120 |
| fc2 output 84 | `0x2700` | 84 |
| fc3 output 10 | `0x2780` | 10 |

weight_buffer layout (byte offsets, produced by the export script; shown as an example):

| Layer | Weights `[K_total][N_pad8]` | Offset | Bias | Offset |
| --- | --- | --- | --- | --- |
| conv1 | 25 x 8 = 200 B | `0x00000` | 8 x 4 B | `0x00100` |
| conv2 | 150 x 16 = 2400 B | `0x00200` | 16 x 4 B | `0x00C00` |
| fc1 | 400 x 120 = 48000 B | `0x01000` | 120 x 4 B | `0x0CC00` |
| fc2 | 120 x 88 = 10560 B | `0x0D000` | 88 x 4 B | `0x0FA00` |
| fc3 | 84 x 16 = 1344 B | `0x10000` | 16 x 4 B | `0x10600` |

Descriptor table (LAYER_CNT = 8):

| idx | OP | K | PAD | IN_H x IN_W | C_IN -> C_OUT | IN_BASE -> OUT_BASE | W_BASE / B_BASE | ReLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CONV2D | 5 | 2 | 28 x 28 | 1 -> 6 | `0x0000 -> 0x0400` | `0x00000 / 0x00100` | yes |
| 1 | MAXPOOL | - | - | 28 x 28 | 6 | `0x0400 -> 0x1800` | - | - |
| 2 | CONV2D | 5 | 0 | 14 x 14 | 6 -> 16 | `0x1800 -> 0x1D00` | `0x00200 / 0x00C00` | yes |
| 3 | MAXPOOL | - | - | 10 x 10 | 16 | `0x1D00 -> 0x2400` | - | - |
| 4 | FC | 1 | 0 | 1 x 1 | 400 -> 120 | `0x2400 -> 0x2600` | `0x01000 / 0x0CC00` | yes |
| 5 | FC | 1 | 0 | 1 x 1 | 120 -> 84 | `0x2600 -> 0x2700` | `0x0D000 / 0x0FA00` | yes |
| 6 | FC | 1 | 0 | 1 x 1 | 84 -> 10 | `0x2700 -> 0x2780` | `0x10000 / 0x10600` | no |
| 7 | ARGMAX | - | - | - | 10 | `0x2780` | - | - |

The CIFAR-10 version differs in only two places: DESC[0] becomes IN 32 x 32, C_IN 3, PAD 0, and the conv1 weights
become 75 x 8 bytes; every other descriptor has exactly the same shape and differs only in RQ_MULT / RQ_SHIFT / Z_*
and in the weight contents. This is what "one RTL, two networks" means concretely.
