# Project Status

Shared handoff between work sessions. Update it after every meaningful change; project facts live here, not in chat history.

## Current milestone

M1 complete: `requant_unit`, `pe`, the requant part of the Python golden model, Verilator CI. Next: M2, the systolic array.

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

## Milestone plan

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | `npu_pkg`, `requant_unit`, `pe`, Python requant model, unit testbenches | Done |
| M2 | `systolic_array` GEMM checked against numpy | Not started |
| M3 | `axil_slave`, `npu_regs`, `npu_sequencer`, error codes | Not started |
| M4 | `act_buffer`, `weight_buffer`, `conv_addr_gen`, `conv_engine`, `pool_engine`, `argmax_engine`; single layers bit-exact | Not started |
| M5 | LeNet-5 end to end on MNIST and CIFAR-10, training and quantization scripts, CI green, README performance table | Not started |
| M6 | Optional: AXI4 master DMA, `chip_top` (CDC / configuration bridge / clock gating), FPGA synthesis data | Not started |

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

## Next steps

1. M2: `rtl/compute/systolic_array.sv` with edge skew, shadow column drain and the `P = max(K_total, ARR_N)` idle-cycle rule;
   a GEMM reference in `model/npu_ref.py`; a unit testbench covering K from 1 to 512 and M both a multiple of 8 and not.
2. Run `make vcs-unit-requant_unit` and `make vcs-unit-pe` with VCS and record the results in the table below.

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
