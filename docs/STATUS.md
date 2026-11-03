# Project Status

Shared handoff between work sessions. Update it after every meaningful change; project facts live here, not in chat history.

## Current milestone

M6 in progress: the open-source synthesis flow runs in CI (numbers in `docs/synthesis.md`), and `chip_top` wraps the core with reset synchronizers, a clock gate, an APB bridge and a pixel stream crossing clock domains, verified with two clocks. CIFAR-10 still waits for its dataset. Next: AXI4 master DMA.

## Completed

- Repository skeleton: RTL / testbench / filelist / docs layout, Verilator-based CI, VCS make targets, MIT license.
- v1 architecture defined (2026-10-29): `docs/architecture.md`, `docs/programming_model.md`, `docs/verification.md`,
  `docs/decisions.md`. Key points: descriptor-driven, INT8 integer inference, 8x8 output-stationary systolic array,
  AXI4-Lite host interface, bit-exact Python golden model, Verilator CI.
- M1 (2026-11-03):
  - RTL: `rtl/npu_pkg.sv` (parameters, opcodes, error codes, `desc_t`, address-map constants),
    `rtl/compute/requant_unit.sv` (3-stage pipeline), `rtl/compute/pe.sv`. All three tops lint clean under Verilator `-Wall`.
  - Model: `model/npu_ref.py` (`requant`, `quantize_multiplier`, `pe_mac` / `pe_tile`),
    `model/gen_requant_vectors.py`, `model/test_npu_ref.py` (21 hand-computed cases).
  - Testbenches: `tb/common/npu_tb_pkg.sv` (SV reference functions), `tb/unit/requant_unit_tb.sv`, `tb/unit/pe_tb.sv`.
  - Flow: Makefile targets `lint` / `model-test` / `vectors` / `unit` / `smoke` / `ci` and `vcs-unit-<module>`;
    one filelist per unit testbench; GitHub Actions runs `make ci`.

- M2 (2026-11-03):
  - RTL: `rtl/compute/systolic_array.sv`: N x N grid of `pe`, per-row and per-column input skew, column drain
    from the shadow registers through a mux, `P >= N` tile-period contract. Lint clean under `-Wall`.
  - Model: `matmul_int` in `model/npu_ref.py`; `model/gen_gemm_vectors.py` (9 directed + random GEMM cases).
  - Testbench: `tb/unit/systolic_array_tb.sv` drives GEMMs tile by tile exactly as `conv_engine` will and checks
    data, column order, drain continuity and the exact drain start cycle.

- M3 (2026-11-03):
  - RTL: `rtl/host/axil_slave.sv` (AXI4-Lite to the internal req/ack bus, write wins same-cycle arbitration),
    `rtl/host/npu_regs.sv` (registers, 16 x 10-word descriptor table with byte strobes and reserved masks,
    host routing to the memory windows with busy rejection, argmax result, irq), `rtl/ctrl/npu_sequencer.sv`
    (layer loop, all static descriptor checks in a fixed order, FC as 1x1 CONV2D, three perf counters).
    `npu_pkg` gained the descriptor word mapping and parameter limits. All lint clean under `-Wall`.
  - Model: `desc_pack` / `desc_unpack` / `desc_check` in `model/npu_ref.py`, mirroring the sequencer's checks;
    `model/gen_desc_vectors.py`.
  - Testbenches: `tb/common/axil_master_bfm.sv` (reusable AXI4-Lite master), `tb/unit/axil_slave_tb.sv`,
    `tb/unit/npu_regs_tb.sv`, `tb/unit/npu_sequencer_tb.sv`.
  - Spec refinements recorded in `programming_model.md`: START clears DONE / ERR / ERR_CODE / RESULT.VALID;
    error 0x02 also covers PAD > 7, RQ_SHIFT, RQ_MULT range and ACT_MIN > ACT_MAX; check order documented.

- M4 (2026-11-03):
  - RTL: `rtl/mem/sram_1r1w.sv`, `act_buffer.sv` (8 byte banks, unaligned 8-byte access, 2-cycle reads),
    `weight_buffer.sv`; `rtl/host/host_mem_adapter.sv`; `rtl/compute/conv_engine.sv` (tile loops with
    incremental addressing, latency-matched pipeline into the systolic array, tile queue, double-buffered bias
    prefetch, 8 requant lanes, write-back), `pool_engine.sv`, `argmax_engine.sv`; `rtl/npu_top.sv` wiring the
    host path, sequencer, engines, memories and the busy-driven port muxes. All lint clean under `-Wall`.
  - Spec refinement: `b_base` is 8-byte aligned (sequencer, model, docs).
  - Model: `conv2d_layer`, `fc_layer`, `maxpool2x2`, `argmax`, memory-image helpers; `model/gen_layer_vectors.py`.
  - Testbenches: `tb/unit/act_buffer_tb.sv`, `conv_engine_tb.sv`, `pool_engine_tb.sv`, `argmax_engine_tb.sv`
    (real memories, guard bands); `tb/integration/npu_top_tb.sv` replaces the reset smoke test and drives
    everything through AXI4-Lite (`make integ`).

- M5 (2026-11-03):
  - `model/train_lenet.py`: float32 LeNet-5 trained with numpy only (im2col, Adam); MNIST 4 epochs reach 98.7% on
    the 10k test set in about 3 minutes on 4 cores.
  - `model/quantize.py`: calibration on 500 training images, per-tensor symmetric weights, asymmetric activations,
    folded zero points, mult / shift, memory layout and descriptors; exports `model/data/mnist/program.txt`
    (descriptors, 16208 weight words, 100 images with the integer model's prediction, full traces of two images),
    `net.json`, float16 checkpoint. Integer program accuracy 97.9% on 2000 images, equal to float32.
  - `model/npu_ref.py`: `run_program`, the memory-level executor used for every expected value.
  - `tb/integration/npu_lenet_tb.sv` (`make integ-mnist`): weights and descriptors once, then per image load,
    run, compare the class with the model; layer traces byte for byte; perf counters printed.
  - README performance table from the counters: 18635 cycles per image, 35% of peak.

- M6 (2026-11-03), first two items:
  - Synthesis: `synth/` scripts and `.github/workflows/synth.yml` (sv2v + Yosys `synth_xilinx` for resources,
    Yosys `synth_ecp5` + nextpnr-ecp5 for a post-route Fmax), results in `docs/synthesis.md`.
  - `rtl/chip/`: `rst_sync`, `clk_gate`, `cdc_sync`, `async_fifo` (Gray pointers), `apb2axil`, `pix_writer`,
    `chip_regs`, `chip_top`; `npu_top` gained a stream write port granted while idle.
  - Testbenches: `tb/unit/async_fifo_tb.sv` (six clock ratios, 24k words), `tb/unit/pix_writer_tb.sv` (208 frames),
    `tb/integration/chip_top_tb.sv` (`make integ-chip`: APB loads the program, frames on the pixel clock, a frame
    during a run, rude source, clock gate; 5 MNIST images agree with the model).

## Milestone plan

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | `npu_pkg`, `requant_unit`, `pe`, Python requant model, unit testbenches | Done |
| M2 | `systolic_array` GEMM checked against the Python integer model | Done |
| M3 | `axil_slave`, `npu_regs`, `npu_sequencer`, error codes | Done |
| M4 | Memories, `conv_engine`, `pool_engine`, `argmax_engine`, `npu_top`; single layers bit-exact | Done |
| M5 | LeNet-5 end to end on MNIST and CIFAR-10, training and quantization scripts, CI green, README performance table | MNIST done; CIFAR-10 pending dataset |
| M6 | Synthesis flow and numbers, `chip_top` (CDC / APB bridge / clock gating), AXI4 master DMA | Synthesis and chip_top done, DMA not started |

## Verification evidence

| Date | Tool | Command | Result |
| --- | --- | --- | --- |
| 2026-11-03 | Verilator 5.020, Python 3.13 | `make lint` | `npu_top`, `requant_unit`, `pe`: zero warnings under `-Wall` |
| 2026-11-03 | Python 3.13 | `make model-test` | 21 hand-computed requant cases, PE extreme-value tiles, `quantize_multiplier` normalization all pass |
| 2026-11-03 | Verilator 5.020 | `make unit-requant_unit` (seed 1) | PASS, 120034 checked: 22 hand-computed, 20000 random, 100012 Python vectors; latency fixed at 3 cycles |
| 2026-11-03 | Verilator 5.020 | `make unit-pe` (seed 1) | PASS, 2006 tiles including K = 1 and int8 extremes; forwarding delay checked every cycle |
| 2026-11-03 | Verilator 5.020 | `make unit SEED=7 REQUANT_N=20000` (testbench seed and Python vectors regenerated) | Both testbenches PASS |
| 2026-11-03 | Verilator 5.020 | Negative test: one Python vector's expected value corrupted | Testbench reports FAIL, proving the checker works |
| 2026-11-03 | Verilator 5.020 | `make clean && make ci` | All pass; this is the full GitHub Actions flow |
| 2026-11-03 | Verilator 5.020 | `make smoke` | SMOKE PASS |
| 2026-11-03 | Verilator 5.020 | `make unit-systolic_array` (seed 1) | PASS, 250 GEMM cases (11 directed, 200 random, 39 Python), 2343 tiles, 110113 elements checked; drain start cycle exact in every tile |
| 2026-11-03 | Verilator 5.020 | `make unit-systolic_array SEED=7` | PASS with regenerated Python vectors |
| 2026-11-03 | Verilator 5.020 | Negative test: one Python GEMM expected value corrupted | Testbench reports FAIL (SV-vs-Python cross-check) |
| 2026-11-03 | Verilator 5.020 | Negative test: testbench copy with one expected matrix element off by one | Monitor reports the mismatching element in every case (RTL-vs-expected check) |
| 2026-11-03 | Verilator 5.020 | `make clean && make ci` after M2 | All pass |
| 2026-11-03 | Verilator 5.020 | `make unit-axil_slave` (seed 1) | PASS, 2023 checks: 980 writes, 1041 reads, every AW/W ordering, random back-pressure, same-cycle arbitration |
| 2026-11-03 | Verilator 5.020 | `make unit-npu_regs` (seed 1) | PASS, 1153 checks: reset values, W1P / W1C / RO, descriptor words, memory windows, SLVERR cases, HOST_CONFLICT, 500 random accesses |
| 2026-11-03 | Verilator 5.020 | `make unit-npu_sequencer` (seed 1) | PASS, 2928 checks: LeNet program, NOP, start while busy, every error code, engine error, cycle-exact counters, 2000 Python descriptors with matching codes |
| 2026-11-03 | Verilator 5.020 | The three M3 testbenches with `SEED=7` | PASS |
| 2026-11-03 | Verilator 5.020 | Negative tests: one Python descriptor code flipped; register TB copy with a wrong expected ID | Both report FAIL |
| 2026-11-03 | Verilator 5.020 | `make clean && make ci` after M3 | All pass, 47 s |
| 2026-11-03 | Verilator 5.020 | `make unit-act_buffer` | PASS, 13386 unaligned reads against a byte model, 2-cycle latency |
| 2026-11-03 | Verilator 5.020 | `make unit-conv_engine` | PASS, 40 CONV2D / FC cases (K 1..7, pad 0..3, partial tiles, FC up to 400x120) bit-exact, 17648 bytes incl. guard bands; cycle counts match the section 5.5 model (FC 400->120: 6115 cycles) |
| 2026-11-03 | Verilator 5.020 | `make unit-pool_engine`, `make unit-argmax_engine` | PASS, 14 pool cases incl. odd sizes, 14 argmax cases incl. ties |
| 2026-11-03 | Verilator 5.020 | `make integ` | PASS, 68 single-layer cases through AXI4-Lite, 21870 checks; error descriptor, host access while busy, perf counters, irq |
| 2026-11-03 | Verilator 5.020 | The M4 testbenches with `SEED=7` | PASS |
| 2026-11-03 | Verilator 5.020 | Negative test: one expected output byte corrupted in the layer vectors | `make integ` reports FAIL |
| 2026-11-03 | Verilator 5.020 | `make clean && make ci` after M4 | All 12 testbenches pass, 78 s |
| 2026-11-03 | numpy 2.5 | `python3 model/train_lenet.py --net mnist` | float32 test accuracy 98.68% after 4 epochs |
| 2026-11-03 | Python 3.13 | `python3 model/quantize.py --net mnist --report 2000` | int8 program 97.9% on 2000 images, float32 97.9% on the same images |
| 2026-11-03 | Verilator 5.020 | `make integ-mnist` (100 images) | PASS: 100 of 100 predictions equal the integer model, 99 match the label, 14 layer traces byte-exact, 18635 cycles per image |
| 2026-11-03 | Verilator 5.020 | `make unit-async_fifo`, `make unit-pix_writer` (seeds 1 and 7) | PASS: 24005 words over six clock ratios; 208 frames incl. sof errors |
| 2026-11-03 | Verilator 5.020 | `make integ-chip` | PASS: 5 images through APB and the pixel stream agree with the model; back-pressure, clock gate, SOF_ERR, OVERRUN checked |
| 2026-11-03 | Verilator 5.020 | `make clean && make ci` after chip_top | 16 testbenches pass, 106 s |

## Next steps

1. CIFAR-10: run `python3 model/train_lenet.py --net cifar10` and `python3 model/quantize.py --net cifar10` on a
   machine that can download the Toronto archive, commit `model/data/cifar10/`, add `cifar10` to `NETS` in the
   Makefile. No RTL change is expected: conv1 has C_in 3 and pad 0, everything else is identical.
2. M6 options: AXI4 master DMA, `chip_top` with CDC and a configuration bridge, FPGA synthesis numbers.
3. Run the unit and integration testbenches with VCS and record the results in the table below.

## Known limitations

- The open flow is Verilator; a commercial simulator (VCS) is available to the author on a separate machine and is used
  as an additional check only. Results are recorded here; no files are transferred from that machine.
- The design is derived from the requirements documented in `docs/`; no third-party proprietary specifications are used.

## Decision log

Full reasoning in `docs/decisions.md`; this is an index.

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-07-13 | Use the `.sv` extension for SystemVerilog | Unambiguous language, simulator friendly |
| 2026-07-13 | Keep RTL and testbenches in separate trees | Clear synthesis boundary, separate organization of verification code |
| 2026-07-13 | The local repository is the single source of truth | Nothing is taken from the machine that hosts the commercial simulator |
| 2026-10-29 | Descriptor-driven programmable compute core (ADR-001) | One RTL for different networks |
| 2026-10-29 | INT8 integer inference, TFLite-style requantization (ADR-002/003) | Small area, bit-exact |
| 2026-10-29 | 8x8 output-stationary systolic array (ADR-004/005/006) | Simple memory access, no partial-sum SRAM |
| 2026-10-29 | AXI4-Lite host interface, DMA deferred to M6 (ADR-007) | SoC integration standard; verify the compute core first |
| 2026-10-29 | Verilator as the primary simulator (ADR-010) | Reproducible by anyone, runs in CI |
| 2026-10-29 | Support both MNIST and CIFAR-10 (ADR-011) | Demonstrates programmability, covers the RGB path |
| 2026-10-29 | One branch and one pull request per milestone (ADR-013) | The PR history records the engineering process |

## VCS run log

| Date | Commit | VCS version | Test | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| - | - | - | - | Not yet run | - |
