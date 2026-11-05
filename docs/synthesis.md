# Synthesis Results

Open-source flow, run by `.github/workflows/synth.yml` on every push (see `synth/README.md`):
`sv2v` translates the SystemVerilog to Verilog-2005, Yosys 0.33 synthesizes it, nextpnr-ecp5 places and routes
the ECP5 netlist. These are estimates for sizing and for checking that every construct is synthesizable, not a
tape-out: the SRAMs are the behavioural `sram_1r1w` inferred as block RAM, there are no I/O constraints, and the
default parameters (`ARR_N = 8`, 32 KB activation buffer, 128 KB weight buffer) are used.

## Summary

Core `npu_top` at commit `93eac9a`. ECP5 is routed with nextpnr's router2 for three placement seeds; the Fmax
column gives the worst and best of the three (see Timing).

| Target | Logic | Flip-flops | Block RAM | Multipliers | Fmax |
| --- | --- | --- | --- | --- | --- |
| Xilinx 7-series, `synth_xilinx` (no place and route) | 17866 LUT + 3196 MUXF7/8 | 14469 | 40 RAMB36 + 1 RAMB18 | 111 DSP48E1 | - |
| Lattice ECP5-85k (LFE5U-85F, speed 8), `synth_ecp5 -abc9` + nextpnr-ecp5 | 24516 LUT4 cells (29%) | 13913 (17%) | 81 DP16KD (39%) | 114 MULT18X18D (73%) | 55.9 to 62.0 MHz |

The ECP5 run places the core inside `synth/npu_top_ooc.sv`, which feeds the 273 input bits from a shift register
and XOR-reduces the 211 output bits so that the core fits the package pins. The wrapper adds 485 flip-flops and
a small XOR tree to the ECP5 numbers; it does not change the core's logic.

## Xilinx 7-series per module

Cells per instance, including submodules (`synth/stat_table.py`). MUXF counts the MUXF7 / MUXF8 cells that
combine two or four LUTs into one wider function.

| Module | Instances | LUT | MUXF | CARRY4 | FF | DSP48E1 | RAMB18 | RAMB36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_top` | 1 | 17866 | 3196 | 1422 | 14469 | 111 | 1 | 40 |
| `conv_engine` | 1 | 12978 | 2495 | 1038 | 10378 | 101 | 0 | 0 |
| &nbsp;&nbsp;`systolic_array` | 1 | 4934 | 0 | 513 | 5785 | 64 | 0 | 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;`pe` | 64 | 65 | 0 | 8 | 83 | 1 | 0 | 0 |
| &nbsp;&nbsp;`requant_unit` | 8 | 544 | 186 | 36 | 371 | 4 | 0 | 0 |
| `npu_regs` | 1 | 1361 | 112 | 20 | 592 | 0 | 1 | 0 |
| `npu_sequencer` | 1 | 931 | 8 | 114 | 1998 | 8 | 0 | 0 |
| `pool_engine` | 1 | 727 | 12 | 123 | 597 | 2 | 0 | 0 |
| `argmax_engine` | 1 | 662 | 470 | 33 | 318 | 0 | 0 | 0 |
| `act_buffer` | 1 | 391 | 0 | 58 | 273 | 0 | 0 | 8 |
| `npu_dma` | 1 | 254 | 11 | 36 | 162 | 0 | 0 | 0 |
| `host_mem_adapter` | 1 | 50 | 0 | 0 | 37 | 0 | 0 | 0 |
| `axil_slave` | 1 | 17 | 0 | 0 | 113 | 0 | 0 | 0 |
| `weight_buffer` | 1 | 8 | 0 | 0 | 1 | 0 | 0 | 32 |

## How the area came down

The first per-module report showed that two thirds of the logic were muxes nobody had designed. Each row is a
commit on the M6 branch; xc7 from the per-module report, ECP5 from nextpnr's device utilisation.

| Commit | Change | xc7 LUT | xc7 MUXF | DSP48E1 | ECP5 LUT4 cells | ECP5 result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `2c321e7` | First flow, before the DMA | - | - | - | 145254 (173%) | Does not fit |
| `377663d` | First per-module report, DMA included | 51217 | 7820 | 123 | - | - |
| `c5cb72a` | One-hot muxes for variable-index selects of wide arrays | 25181 | 6091 | 122 | 59154 (70%) | Logic fits; 484 core I/O bits do not |
| `eabc27a` | Running addresses in `conv_engine`; out-of-context wrapper | 25290 | 6200 | 118 | 60404 (72%) | Placement legalization fails |
| `641e322` | Narrow requant tail; shared descriptor-check datapath | 24914 | 7410 | 112 | 61278 (73%) | Placement legalization fails |
| `996164e` | Sequencer fetches the descriptor word by word | 21112 | 4287 | 112 | 66143 (79%) | Placement fails |
| `d08daa1` | ECP5 logic mapped with `abc9` (same RTL) | 21112 | 4287 | 112 | 33752 (40%) | Routed, 28.8 MHz |
| `75472a4` | Descriptor table in a 1R1W RAM with per-word valid bits (ADR-017) | 18364 | 3277 | 111 | 24970 (29%) | Routed, 56.2 MHz |
| `cb95def` | Registered pool write port, channel mask in the tile queue, requant multiply in two halves | 17886 | 3216 | 111 | 24247 (28%) | Routed, 50.8 to 60.5 MHz over three seeds |
| `93eac9a` | Registered `act_buffer` requests, descriptor check stage O split from the products | 17866 | 3196 | 111 | 24516 (29%) | Routed, 55.9 to 62.0 MHz over three seeds |

What each step taught:

- **Variable-index selects.** `desc_i[cur_q]` (16 structs of about 300 bits) and `desc_words_q[idx][word]`
  (16 x 10 words) are plain muxes in the source. sv2v flattens the arrays and computes the bit offset as
  `index * width`; when the width is not a power of two, Yosys turns that into a barrel shifter across the whole
  array. The sequencer went from 19.9K to 3.3K LUTs and the register file from 14.9K to 5.5K by writing the
  selects as one-hot AND-OR muxes. Later the register file's read mux indexes a 16-words-per-entry view with a
  power-of-two stride, which maps better still.
- **Mux mapping on ECP5.** Synthesizing the descriptor selects alone showed that this Yosys version maps wide
  muxes to ECP5 at several times the ideal cost (a 16:1 select of 320 bits used about 25K cells). The fix that
  holds for any flow is to not build the mux: the sequencer now fetches ten 32-bit words through the register
  file's existing read port, 11 cycles per layer (ADR-016). That alone did not make ECP5 fit, because the default
  ABC mapping of the remaining muxes grew; mapping with `abc9` halved the ECP5 logic for the same RTL, with
  identical flip-flop, DSP and block-RAM counts. (Yosys 0.33's ABC9 reports one aborted ABC call as a warning;
  nextpnr finds no unmapped cell and the register counts match the default mapping, so no logic was lost.)
- **Arithmetic.** The beat and tile address products in `conv_engine` became running registers (four
  multipliers fewer in per-cycle paths), the descriptor check builds each size product once for all opcodes, and
  the requantization tail saturates to 14 bits before the zero point and the clamp, which is exact.

## Timing

Post-route Fmax of the single clock on the ECP5-85k, speed grade 8, against a 100 MHz target. The routed builds
and what each one's critical path was:

| Commit | Change | Router | ECP5 LUT4 cells | Fmax | Critical path |
| --- | --- | --- | ---: | ---: | --- |
| `d08daa1` | First routed build | router1 | 33752 (40%) | 28.8 MHz | Fetched descriptor through the size products and the end-address compares to the sequencer state (32 ns) |
| `95e6fe1` | Descriptor check as a three-stage pipeline | router1 | 33875 (40%) | 42.4 MHz | requant stage 3: 65-bit shift, saturation, rounding and zero-point adds, clamp (19.7 ns, 34 cells) |
| `cd30631` | requant tail split into two stages; router2 | router2 | 35590 (42%) | 34.8 MHz | Host register read: address decode, the shared 160-word descriptor mux, `rd_data_o` (3.7 ns logic, 20.0 ns routing) |
| `75472a4` | Descriptor table in a RAM | router2 | 24970 (29%) | 56.2 MHz | Into an `act_buffer` bank's write data through two carry chains, the port mux and the byte rotation (17.0 ns, 11.7 ns of it routing); `pool_engine` writes its 2x2 maximum (two signed compares) unregistered on that port |

The flow uses router2 since `cd30631`: router1 took 16 to 20 minutes on the earlier builds but more than 90 minutes
on `7520872`, while router2 routes the same design in about 11 minutes. router2 makes longer wires on this design,
so its Fmax is not directly comparable with the router1 rows; the critical path is what matters for the next step.

### Placement seeds

The rows above are one placement each (seed 1). Since `51b842d` the workflow routes three seeds in parallel, and
the spread is large: the same RTL ranges over 16 MHz, and each seed exposes a different near-critical path. So a
fix is judged by the worst of the three, not by one run.

| RTL | Seed 1 | Seed 2 | Seed 3 | Worst |
| --- | --- | --- | --- | ---: |
| `main` after ADR-017 (`5cc8e3d`) | 56.2 MHz: `pool_engine` 2x2 maximum into the `act_buffer` write port | 45.9 MHz: drain side, queued n_tile to channel number to the C_out compare | 40.4 MHz: requant 33 x 32 multiply in one cycle | 40.4 MHz |
| Registered pool write port only (`51b842d`) | 49.2 MHz: requant shift-and-saturate stage | 39.2 MHz: drain side as before | 40.9 MHz: drain side as before | 39.2 MHz |
| Round two (`cb95def`) | 60.5 MHz: descriptor check stage A | 50.8 MHz: `act_buffer` read address through the port mux and the bank row increment | 52.6 MHz: descriptor check stage A | 50.8 MHz |
| Round three (`93eac9a`) | 62.0 MHz | 60.9 MHz | 55.9 MHz | 55.9 MHz |

The second row is the lesson: registering the pool output was right, but alone it did not move the worst case,
because two other paths were nearly as long. Round two fixed those as well, at no cycle cost:

- `conv_engine` pushes a mask of the columns whose output channel exists into its tile queue, computed by the
  beat generator; the drain side selects one bit instead of adding and comparing 16-bit channel numbers.
- `requant_unit` forms the two 33 x 16-bit halves of the multiply in one stage and adds them in the next
  (latency 5); `conv_engine`'s flush window already covered the extra cycle.

Reading the timing:

- Each fix moved the critical path to a different block, which is the sign of a design without one dominant
  structural problem left. The three paths so far were all "too much in one cycle" (two levels of products and
  compares, a shift plus three adds) and were fixed by pipelining logic that runs once per layer or once per
  output, at a cost of 3 cycles per layer and 1 cycle of requant latency.
- The register-read path was mostly routing: 5120 descriptor flip-flops spread over the device and a 32-bit mux
  gathering them. Moving the table into a block RAM behind the existing read port (ADR-017) removed both, and
  with the same router Fmax went from 34.8 to 56.2 MHz while the logic fell from 42% to 29% of the device.
- Round three removed both remaining paths. `act_buffer` registers the per-bank addresses, write data and byte
  enables next to its SRAMs and makes the lane rotation after them combinational, so the read latency stays two
  cycles and no engine changes. The descriptor check's first stage was split into operand selection and the
  size products (one more cycle per layer).
- In all three round-three seeds the critical path is now the same one: requant stage 3, from the shift-amount
  register into the eight 65-bit shifters (15.8 to 17.7 ns, 12.6 to 14.2 ns of it routing). The eight units
  hold identical copies of that register, Yosys merges them into one, and the single register then drives all
  eight shifters spread around the DSP columns. Fixing it needs either register duplication that synthesis keeps
  (a tool-specific attribute) or a shift split over two stages; both are recorded as next steps.
- These are FPGA numbers from an open flow. The same RTL in a standard-cell process would close far higher;
  the value here is that every path has been through a real placer and router and the worst ones are known.

## Reading the numbers

- The systolic array is 64 `pe` instances: one 8x8 multiplier (one DSP), a 32-bit accumulator and a 32-bit shadow
  register each, 83 flip-flops per PE.
- The eight `requant_unit` instances carry a 33x32-bit multiplier each (four DSP48E1) and a 65-bit barrel shifter,
  the widest arithmetic in the design. A tape-out would size both to the real precision need (see `decisions.md`
  on the TFLite-style multiplier).
- `npu_regs` keeps the 16 x 16-word descriptor table in one RAMB18 (ADR-017; until `75472a4` it was 5120
  flip-flops and a 32-bit read mux). Its 592 flip-flops are the control and DMA registers, the logits, the
  per-word valid bits (one per RAM word, kept at a power-of-two count so the index needs no multiplier) and the
  read path.
- Block RAM: the activation buffer is eight 4 KB byte banks (8 RAMB36 on xc7) and the weight buffer one 128 KB
  64-bit memory (32 RAMB36).
