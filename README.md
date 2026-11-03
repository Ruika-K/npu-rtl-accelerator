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
  Verilator lint and simulation in GitHub Actions, and a whole LeNet-5 that matches the model bit for bit.

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
make ci                 # lint + Python self-test + all unit testbenches + layer cases + LeNet-5 on MNIST
```

```sh
make unit-requant_unit  # one unit testbench
```

```sh
make integ-mnist IMAGES=10   # the network test on fewer images
```

Regenerating the network data needs only numpy: `python3 model/train_lenet.py --net mnist` trains float32
LeNet-5 (MNIST downloads automatically), then `python3 model/quantize.py --net mnist` quantizes, builds the
descriptors and writes `model/data/mnist/program.txt`. CIFAR-10 uses the same two commands with `--net cifar10`
once `cifar-10-python.tar.gz` is in `model/out/datasets/`.

With a commercial simulator (VCS) the same filelists are used:

```sh
make vcs-unit-pe
```

## Measured performance (LeNet-5, MNIST, Verilator)

One image through the 8-layer program, read from the performance counters (`make integ-mnist`):

| Layer | Shape | Cycles | Useful MACs | Note |
| --- | --- | --- | --- | --- |
| conv1 5x5, 1->6, pad 2 | 28x28x1 -> 28x28x6 | 2835 | 117.6K | 6 of 8 columns used, last column tile 4 of 8 lanes |
| pool1 | -> 14x14x6 | 1350 | - | sequential FSM |
| conv2 5x5, 6->16 | -> 10x10x16 | 6041 | 240K | 10 of 16 lanes in the last column tile |
| pool2 | -> 5x5x16 | 646 | - | |
| fc1 400->120 | | 6119 | 48K | weight-bandwidth bound: 1 of 8 rows |
| fc2 120->84 | | 1415 | 10.1K | |
| fc3 84->10 | | 209 | 0.8K | |
| argmax | | 20 | - | |
| **Total** | | **18635** | **416.5K** | array busy 16288 cycles; 35% of the 64 MAC/cycle peak |

All 100 test images in `model/data/mnist/program.txt` give the same class as the integer model (99 of 100 match the
label); the first two images are also checked layer by layer, byte for byte. The float32 model scores 97.9% and
the int8 program 97.9% on the same 2000 test images (`model/data/mnist/net.json`).

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
| M5 | LeNet-5 end to end: numpy training, quantization and export, MNIST bit-exact on 100 images, performance table | MNIST done, CIFAR-10 pending its dataset |
| M6 | Optional: AXI4 master DMA, `chip_top` (CDC, configuration bridge, clock gating), FPGA synthesis numbers | Next |

## Originality

All RTL, models and documentation in this repository are the author's original work, derived from the
requirements documented in `docs/`. No third-party proprietary code or specifications are included.

## License

MIT, see [LICENSE](LICENSE).
