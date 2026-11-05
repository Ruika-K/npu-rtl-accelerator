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
- **On-chip memory and standard SoC ports.** 32 KB activation buffer, 128 KB weight buffer, an AXI4-Lite slave
  for configuration and an AXI4 master DMA (64-bit bursts, 4 KB rule) for loading and storing. A chip-level
  wrapper adds an APB bridge, a pixel stream crossing clock domains through a Gray-pointer FIFO, reset
  synchronizers and a core clock gate.
- **Synthesized in CI.** sv2v + Yosys report the area per module on every push and nextpnr-ecp5 places and routes
  the core; the first report drove a round of RTL changes that cut the logic by more than half
  ([docs/synthesis.md](docs/synthesis.md)).
- **Reproducible verification.** Pure-integer Python reference, self-checking SystemVerilog testbenches,
  Verilator lint and simulation in GitHub Actions, and a whole LeNet-5 that matches the model bit for bit.

## Repository layout

```text
.
├── rtl/                     # Synthesizable SystemVerilog
│   ├── npu_pkg.sv           # Parameters, encodings, descriptor type
│   ├── npu_top.sv           # Top level: host path, sequencer, engines, memories, port muxes
│   ├── compute/             # requant_unit, pe, systolic_array, conv_engine, pool_engine, argmax_engine
│   ├── host/                # axil_slave, npu_regs, host_mem_adapter, npu_dma (AXI4 master)
│   ├── ctrl/                # npu_sequencer
│   ├── mem/                 # sram_1r1w, act_buffer, weight_buffer
│   └── chip/                # chip_top: reset sync, clock gate, APB bridge, async FIFO, pixel writer
├── tb/
│   ├── common/              # SV reference functions, AXI4-Lite and APB master BFMs, AXI4 slave memory model
│   ├── unit/                # One self-checking testbench per module
│   └── integration/         # npu_top through AXI4-Lite, LeNet-5 networks, chip_top with two clocks
├── model/                   # Python integer golden model, vector generators, self-test
├── sim/filelists/           # Filelists shared by Verilator and VCS
├── docs/                    # Architecture, programming model, verification plan, decisions, synthesis, status
├── synth/                   # Yosys scripts (sv2v + synth_xilinx / synth_ecp5 + nextpnr)
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
descriptors and writes `model/data/mnist/program.txt`. CIFAR-10 uses the same two commands with `--net cifar10`;
the trainer reads either the original `cifar-10-python.tar.gz` or the fast.ai mirror `cifar10.tgz` (PNG files,
decoded with Pillow) from `model/out/datasets/`, and downloads one of them if neither is there.

With a commercial simulator (VCS) the same filelists are used:

```sh
make vcs-unit-pe
```

## Measured performance (LeNet-5, MNIST, Verilator)

One image through the 8-layer program, read from the performance counters (`make integ-mnist`):

| Layer | Shape | Cycles | Useful MACs | Note |
| --- | --- | --- | --- | --- |
| conv1 5x5, 1->6, pad 2 | 28x28x1 -> 28x28x6 | 2850 | 117.6K | 6 of 8 columns used, last column tile 4 of 8 lanes |
| pool1 | -> 14x14x6 | 1365 | - | sequential FSM |
| conv2 5x5, 6->16 | -> 10x10x16 | 6056 | 240K | 10 of 16 lanes in the last column tile |
| pool2 | -> 5x5x16 | 661 | - | |
| fc1 400->120 | | 6134 | 48K | weight-bandwidth bound: 1 of 8 rows |
| fc2 120->84 | | 1430 | 10.1K | |
| fc3 84->10 | | 224 | 0.8K | |
| argmax | | 35 | - | |
| **Total** | | **18755** | **416.5K** | array busy 16288 cycles; 35% of the 64 MAC/cycle peak |

All 100 test images in `model/data/mnist/program.txt` give the same class as the integer model (99 of 100 match the
label); the first two images are also checked layer by layer, byte for byte. The float32 model scores 97.9% and
the int8 program 97.9% on the same 2000 test images (`model/data/mnist/net.json`).

### CIFAR-10 on the same RTL

Only conv1 changes (32x32x3 input, no padding, 75 weight rows instead of 25); every other layer has the same shape
and the same cycle count (`make integ-cifar10`):

| Layer | Shape | Cycles | Useful MACs | Note |
| --- | --- | --- | --- | --- |
| conv1 5x5, 3->6, pad 0 | 32x32x3 -> 28x28x6 | 8450 | 352.8K | K_total 75 per tile instead of 25 |
| pool1 to argmax | as above | 15905 | 299K | identical to MNIST |
| **Total** | | **24355** | **651.7K** | array busy 21888 cycles; 42% of the 64 MAC/cycle peak |

All 100 test images in `model/data/cifar10/program.txt` (10 per class) give the same class as the integer model, 59
match the label, and the first two are checked layer by layer. LeNet-5 is a small network for CIFAR-10: float32
reaches 63.0% on the 10k test set, and on the first 2000 test images the int8 program scores 61.8% against 61.75%
for float32 (`model/data/cifar10/net.json`), so quantization costs nothing; the accuracy is the network's, not the
hardware's.

## Synthesis (open-source flow)

Measured by `.github/workflows/synth.yml` on every push; details and the history in
[docs/synthesis.md](docs/synthesis.md).

| Target | Logic | Flip-flops | Block RAM | Multipliers | Fmax |
| --- | --- | --- | --- | --- | --- |
| Xilinx 7-series, Yosys `synth_xilinx` | 18.0K LUT + 3.3K MUXF | 14.5K | 40 RAMB36 + 1 RAMB18 | 110 DSP48E1 | - |
| Lattice ECP5-85k, Yosys + nextpnr, placed and routed | 27.3K LUT4 (32%) | 15.0K | 81 DP16KD | 113 MULT18X18D | 57.1 to 60.5 MHz (3 seeds) |

The first per-module report put the core at 51K LUTs, two thirds of it in the sequencer and the register file:
variable-index selects of wide arrays, which sv2v and Yosys turn into barrel shifters. Rewriting those selects,
fetching the current descriptor word by word instead of selecting it whole, pipelining the two longest paths
and moving the descriptor table into a RAM, then removing the paths each routed build exposed, brought the core
to the numbers above for 15 extra cycles per layer. Fmax is quoted over three placement seeds because one seed can
hide a path that another exposes. At the worst of the three, one MNIST image takes about 0.33 ms.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Scope, numeric formats, tensor layout, dataflow, module partition, interfaces, throughput estimate |
| [docs/programming_model.md](docs/programming_model.md) | Address map, registers, descriptor format, run sequence, error codes, LeNet-5 example |
| [docs/verification.md](docs/verification.md) | Golden model, testbench levels, assertions, tools, definition of done per milestone |
| [docs/decisions.md](docs/decisions.md) | Why each key decision was made and what was rejected |
| [docs/synthesis.md](docs/synthesis.md) | FPGA resource and timing estimates from the open-source flow |
| [docs/STATUS.md](docs/STATUS.md) | Current milestone, verification evidence, next steps |

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | `npu_pkg`, `requant_unit`, `pe`, Python requant model, unit testbenches, Verilator CI | Done |
| M2 | `systolic_array` GEMM checked against the Python integer model | Done |
| M3 | `axil_slave`, `npu_regs`, `npu_sequencer`, error codes | Done |
| M4 | Memories, `conv_engine`, `pool_engine`, `argmax_engine`, `npu_top`; single layers bit-exact through AXI4-Lite | Done |
| M5 | LeNet-5 end to end: numpy training, quantization and export, MNIST and CIFAR-10 bit-exact on 100 images each, performance tables | Done |
| M6 | Open-source synthesis flow and numbers, `chip_top` (CDC, APB bridge, clock gating), AXI4 master DMA | Done |

## Originality

All RTL, models and documentation in this repository are the author's original work, derived from the
requirements documented in `docs/`. No third-party proprietary code or specifications are included.

## License

MIT, see [LICENSE](LICENSE).
