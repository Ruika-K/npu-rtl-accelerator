# NPU RTL Accelerator

[![CI](https://github.com/Ruika-K/npu-rtl-accelerator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruika-K/npu-rtl-accelerator/actions/workflows/ci.yml)

A from-scratch SystemVerilog implementation of a small INT8 CNN inference accelerator. The goal is not just RTL
that simulates, but a documented architecture, a bit-exact golden model, and a verification flow that anyone can
reproduce with open-source tools.

## Highlights

- **Programmable, descriptor-driven.** One RTL runs LeNet-5 on both MNIST and CIFAR-10; only the layer
  descriptors and the weights change.
- **Integer-only datapath.** int8 x int8 multiply, int32 accumulate, TFLite-style integer requantization.
  No floating point anywhere, so every layer matches the Python golden model bit for bit.
- **8x8 output-stationary systolic array.** Partial sums stay in the PEs; activations are read as unaligned
  8-byte vectors from a byte-banked SRAM with zero bank conflicts.
- **On-chip memory and a standard host port.** 32 KB activation buffer, 128 KB weight buffer, AXI4-Lite slave
  for configuration and loading.
- **Reproducible verification.** Pure-integer Python reference, self-checking SystemVerilog testbenches,
  Verilator lint and simulation in GitHub Actions.

## Repository layout

```text
.
├── rtl/                     # Synthesizable SystemVerilog
│   ├── npu_pkg.sv           # Parameters, encodings, descriptor type
│   ├── npu_top.sv           # Top level: host path, sequencer, engines, memories, port muxes
│   ├── compute/             # requant_unit, pe, systolic_array, conv_engine, pool_engine, argmax_engine
│   ├── host/                # axil_slave, npu_regs, host_mem_adapter
│   ├── ctrl/                # npu_sequencer
│   └── mem/                 # sram_1r1w, act_buffer, weight_buffer
├── tb/
│   ├── common/              # SV reference functions, AXI4-Lite master BFM
│   ├── unit/                # One self-checking testbench per module
│   └── integration/         # npu_top through AXI4-Lite
├── model/                   # Python integer golden model, vector generators, self-test
├── sim/filelists/           # Filelists shared by Verilator and VCS
├── docs/                    # Architecture, programming model, verification plan, decisions, status
├── .github/workflows/       # CI
├── CONTRIBUTING.md          # Engineering conventions
└── Makefile                 # lint / unit / smoke / ci targets
```

## Quick start

Requirements: Verilator 5.x and Python 3 (no other dependencies for simulation).

```sh
make ci                 # lint + Python self-test + all unit testbenches + top-level integration test
```

```sh
make unit-requant_unit  # one unit testbench
```

With a commercial simulator (VCS) the same filelists are used:

```sh
make vcs-unit-pe
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Scope, numeric formats, tensor layout, dataflow, module partition, interfaces, throughput estimate |
| [docs/programming_model.md](docs/programming_model.md) | Address map, registers, descriptor format, run sequence, error codes, LeNet-5 example |
| [docs/verification.md](docs/verification.md) | Golden model, testbench levels, assertions, tools, definition of done per milestone |
| [docs/decisions.md](docs/decisions.md) | Why each key decision was made and what was rejected |
| [docs/STATUS.md](docs/STATUS.md) | Current milestone, verification evidence, next steps |

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | `npu_pkg`, `requant_unit`, `pe`, Python requant model, unit testbenches, Verilator CI | Done |
| M2 | `systolic_array` GEMM checked against the Python integer model | Done |
| M3 | `axil_slave`, `npu_regs`, `npu_sequencer`, error codes | Done |
| M4 | Memories, `conv_engine`, `pool_engine`, `argmax_engine`, `npu_top`; single layers bit-exact through AXI4-Lite | Done |
| M5 | LeNet-5 end to end on MNIST and CIFAR-10, training and quantization scripts, performance table | Next |
| M6 | Optional: AXI4 master DMA, `chip_top` (CDC, configuration bridge, clock gating), FPGA synthesis numbers | |

## Originality

All RTL, models and documentation in this repository are the author's original work, derived from the
requirements documented in `docs/`. No third-party proprietary code or specifications are included.

## License

MIT, see [LICENSE](LICENSE).
