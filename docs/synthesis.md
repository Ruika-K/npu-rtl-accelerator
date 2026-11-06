# Synthesis Results

Open-source flow, run by `.github/workflows/synth.yml` on every push (see `synth/README.md`):
`sv2v` translates the SystemVerilog to Verilog-2005, Yosys 0.33 synthesizes it, nextpnr-ecp5 places and routes
the ECP5 netlist. These are estimates for sizing and for checking that every construct is synthesizable, not a
tape-out: the SRAMs are the behavioural `sram_1r1w` inferred as block RAM, there are no I/O constraints, and the
default parameters (`ARR_N = 8`, 32 KB activation buffer, 128 KB weight buffer) are used.

## Summary

Core `npu_top` at commit `b9210b7`. ECP5 is routed with nextpnr's router2 for three placement seeds; the Fmax
column gives the worst and best of the three (see Timing). The ECP5 script keeps each requant unit's own copy
of the layer constants (ADR-018), which accounts for about 900 of its flip-flops; the xc7 run does not.

| Target | Logic | Flip-flops | Block RAM | Multipliers | Fmax |
| --- | --- | --- | --- | --- | --- |
| Xilinx 7-series, `synth_xilinx` (no place and route) | 16050 LUT + 1936 MUXF7/8 | 14971 | 40 RAMB36 + 1 RAMB18 | 110 DSP48E1 | - |
| Lattice ECP5-85k (LFE5U-85F, speed 8), `synth_ecp5 -abc9` + nextpnr-ecp5 | 21653 LUT4 cells (25%) | 15437 (18%) | 81 DP16KD (39%) | 113 MULT18X18D (72%) | 58.8 to 60.6 MHz |

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
| `99104df` | Requant layer constants kept per unit (ADR-018), `conv_engine` setup products pipelined, running bias prefetch pointer | 17987 | 3329 | 110 | 27310 (32%) | Routed, 57.1 to 60.5 MHz over three seeds |
| `1e543d1` | Requant shift over two stages, current layer's cycle count in a local register | 16112 | 2040 | 110 | 21910 (26%) | Routed, 58.5 to 67.2 MHz over three seeds |
| `b9210b7` | Register-file address windows decoded with equality tests | 16050 | 1936 | 110 | 21653 (25%) | Routed, 58.8 to 60.6 MHz over three seeds |

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
| Round four, shift amount kept per unit (`b6884bf`) | 60.0 MHz: `conv_engine` setup, C_in * K * K and the tile-period select | 53.6 MHz: the same path | Not finished: router2 still running after 70 minutes, superseded | - |
| Round four, setup products pipelined (`f46e958`) | 59.8 MHz: pool read address into the `act_buffer` bank rows (12.3 ns routing) | 56.0 MHz: descriptor RAM port select in `npu_regs` | 54.9 MHz: the requant multiplier, again one merged register for eight units | 54.9 MHz |
| Round four, all requant constants kept (`d64fd0c`) | 62.7 MHz: descriptor RAM port select | 50.2 MHz: bias prefetch address adder into the weight SRAM address pins (13.5 ns routing) | 64.3 MHz: requant shift and saturate, inside one unit | 50.2 MHz |
| Round four (`99104df`) | 57.1 MHz: requant shift and saturate, inside one unit | 60.5 MHz: requant low-half multiply, inside one unit | 57.6 MHz: requant shift and saturate, inside one unit | 57.1 MHz |
| Round five, shift over two stages (`dc3b7cc`) | 56.0 MHz: the per-layer cycle counter loop in `npu_sequencer` (labelled with AXI and register-file signals, see below) | 65.8 MHz: the same path | 60.9 MHz: the same path | 56.0 MHz |
| Round five (`1e543d1`) | 67.2 MHz: requant low-half multiply, inside one unit | 58.5 MHz: host read address decode into the descriptor RAM | 58.8 MHz: requant high-half multiply, inside one unit | 58.5 MHz |
| Round six, address windows by equality (`b9210b7`) | 59.4 MHz: requant low-half multiply | 58.8 MHz: host register read mux into its output register (all LUTs and routing) | 60.6 MHz: requant high-half multiply | 58.8 MHz |
| Round six, four DSP-sized partial products (`a1352bc`, reverted) | Not finished: router2 still running when the job limit stopped it after two hours | 59.0 MHz: the weight buffer's read enable into the clock enables of its block RAMs (14.3 ns routing) | 58.2 MHz: requant stage 1, bias select and add | - |

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
- After round three every seed failed on the same path: requant stage 3, from the shift-amount register into
  the eight 65-bit shifters (15.8 to 17.7 ns, 12.6 to 14.2 ns of it routing). The eight units hold identical
  copies of that register, Yosys merges them into one, and the single register drove all eight shifters spread
  around the DSP columns.
- Round four took four commits, each judged by a full three-seed sweep:
  - The ECP5 script marks the requant shift-amount registers `keep` after `proc` (ADR-018). That path left
    both seeds that finished, and the next one was `conv_engine`'s SETUP cycle: C_in * K * K (two chained multiplies), then
    max(K_total, N), about 17 ns.
  - Those products now come from a three-stage pipeline that recomputes them every cycle (K * K; C_in * K * K
    and OH * OW; the tile period). Their inputs change only in IDLE and SETUP, and the bias prefetch puts at
    least six cycles between SETUP and the first beat, so no cycle count changed.
  - The next seed failed on the requant multiplier, merged into one register in the same way. Rather than fix
    the members of that class one at a time, the `keep` rule now covers all 17 layer-constant registers of
    `requant_unit` (multiplier, shift amount, zero point and clamp bounds at every stage), for about 900
    flip-flops.
  - One seed then exposed the bias prefetch address, an adder `b_word + nt * 4 + pf_cnt` in front of the weight
    buffer's port mux and the address pins of its block RAMs. Every n_tile reads exactly four consecutive
    words, so a pointer loaded in SETUP and incremented per read gives the same addresses without the adder.
  - The worst seed is now 57.1 MHz (was 55.9) and the spread is the narrowest so far (57.1 to 60.5 MHz). All
    three critical paths are inside one requant unit: the shift-and-saturate stage and the low-half multiply,
    still 11.3 to 12.6 ns of routing each. The next step there is a shift over two stages.
- The ECP5 cell count rose from 24655 to 27310 with the last commit, while the xc7 count for the same change
  moved by 57 LUTs. That points at the ECP5 mapping rather than the RTL (Yosys 0.33's ABC9 aborts one call in
  every run, see above); the exact cause has not been isolated.
- Round five split the requant shift and then removed a path the reports had mislabelled:
  - Stage 3 becomes two stages. Stage 2b forms shift - 1 next to the product add; stage 3a shifts by the
    multiple of 8 and keeps the 21 low bits, a flag that the bits above are all equal, and the sign; stage 3b
    shifts by the remaining 0..7 and saturates. The saturated value is bit-identical to the one-step shift (an
    off-line check over every shift amount and 491k products around every boundary). Requant latency 5 -> 6;
    `conv_engine`'s flush window is now derived from the latencies (`2N + ABUF_RD_LAT + RQ_PIPE_LAT`, the
    measured minimum), which keeps it at 24 cycles, so no cycle count changed. The narrow shifter also took
    190 LUTs out of each requant unit on xc7.
  - With the shift gone, all three seeds reported a path labelled with `m_axi_bvalid_i`, `m_axi_arvalid` and the
    register file's read data, signals the RTL does not connect combinationally. The labels are names nextpnr
    gives cells after neighbouring nets. The workflow now prints the raw path with placement coordinates, and
    `nextpnr_summary.py` lists the source lines the path's cells were built from. Those showed the real path:
    the per-layer cycle counter in `npu_sequencer`, where one incrementer served all sixteen counters by
    gathering `layer_cycles_o[cur_q]` through a 16:1 select and writing the sum back, a loop through 512
    flip-flops that crossed the die twice (x 56 -> 19 -> 65 -> 23, 12.5 ns of routing). The current layer's
    count now lives in one local register that restarts when the layer advances; the counters hold the same
    values in every cycle, and the sequencer lost 318 LUTs on xc7.
  - The worst seed is 58.5 MHz (was 57.1), the best 67.2 MHz, and the ECP5 logic is down to 26% of the device.
    Two seeds now fail in the requant multiply (`s1_sum_q` into the DSP halves, about 5 ns of routing before
    the DSP), the third on the host read address decode into the descriptor RAM.
- Round six took one fix, one negative result and a flow fix:
  - Yosys 0.33 maps some magnitude compares against constants to carry chains on ECP5 and folds others into a
    LUT (in isolated tests `a < 20'h01000` became 10 CCU2C cells and the 12-bit logits-window test 6, while the
    12-bit descriptor-window test folded into one LUT). The
    register file now decodes every address window with equality tests: a window is a run of aligned granules
    (the lowest set bit of base | size), one compare each, which is exact for any base and size. An exhaustive
    comparison with the old decode over all 2^20 addresses agrees, also for non-power-of-two buffers. The
    descriptor-RAM path left the reports, and the worst seed moved from 58.5 to 58.8 MHz with the narrowest
    spread so far (58.8 to 60.6 MHz).
  - Splitting the requant multiply into four products that each fit one 18 x 18 multiplier (registered at
    the DSPs, summed in stage 2b, same latency) removed the multiply paths, but the paths behind them were as
    long: the weight buffer's read enable fanning out to its block RAMs, the stage-1 bias select and add. And
    one seed did not finish routing: router2 kept going until the two-hour job limit, the second such stall on
    this design. By the worst-seed rule that is no improvement, at 264 more flip-flops, so it was reverted.
    The workflow now stops a place and route after 45 minutes with an error naming the seed.
- Where this leaves timing: after six rounds every remaining critical path is 13 to 16 ns of which 11 to 14 ns
  is routing, and the paths sit in different blocks from seed to seed (the requant multiply, the register read
  mux, the buffers' address and enable fan-out). The design spreads over most of the device's multipliers and
  block RAM, so the long wires are structural. Further gains would come from physical measures, such as
  duplicating the buffer enables and addresses per group of block RAMs or constraining placement, rather than
  from more pipelining.
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
