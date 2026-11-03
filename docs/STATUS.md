# Project Status

Shared handoff between work sessions. Update it after every meaningful change; project facts live here, not in chat history.

## Current milestone

v1 architecture defined. Next: M1, the requantization unit and the processing element.

## Completed

- Repository skeleton: RTL / testbench / filelist / docs layout, Verilator-based CI, VCS make targets, MIT license.
- v1 architecture defined (2026-10-29): `docs/architecture.md`, `docs/programming_model.md`, `docs/verification.md`,
  `docs/decisions.md`. Key points: descriptor-driven, INT8 integer inference, 8x8 output-stationary systolic array,
  AXI4-Lite host interface, bit-exact Python golden model, Verilator CI.

## Milestone plan

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | `npu_pkg`, `requant_unit`, `pe`, Python requant model, unit testbenches | Not started |
| M2 | `systolic_array` GEMM checked against numpy | Not started |
| M3 | `axil_slave`, `npu_regs`, `npu_sequencer`, error codes | Not started |
| M4 | `act_buffer`, `weight_buffer`, `conv_addr_gen`, `conv_engine`, `pool_engine`, `argmax_engine`; single layers bit-exact | Not started |
| M5 | LeNet-5 end to end on MNIST and CIFAR-10, training and quantization scripts, CI green, README performance table | Not started |
| M6 | Optional: AXI4 master DMA, `chip_top` (CDC / configuration bridge / clock gating), FPGA synthesis data | Not started |

## Next steps

1. Review the four v1 documents and freeze the interfaces.
2. M1: `rtl/npu_pkg.sv`, `rtl/compute/requant_unit.sv`, `rtl/compute/pe.sv`, the requant function of
   `model/npu_ref.py`, unit testbenches, Verilator unit-test targets in the Makefile and CI.
3. Run the same filelists with VCS and record the results in the table below.

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
