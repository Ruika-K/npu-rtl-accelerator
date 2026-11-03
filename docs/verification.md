# NPU v1 Verification Plan

One goal: every layer output of the RTL matches the Python integer golden model bit for bit, and anyone who clones
the repository can reproduce that with a single command.

## 1. Reference model and data

Directory `model/`:

| File | Purpose | Depends on |
| --- | --- | --- |
| `train_lenet.py` | Trains fp32 LeNet-5 in PyTorch, one checkpoint each for MNIST and CIFAR-10 | torch, torchvision. Optional; simulation runs without it |
| `quantize.py` | Post-training quantization: symmetric per-tensor weights, activation scale and zero point from calibration min/max; computes RQ_MULT / RQ_SHIFT; folds the zero point into the bias; emits the weight blob, descriptor table and quantization JSON | numpy, reads the checkpoint |
| `npu_ref.py` | numpy / pure-integer golden model. Implements every formula of `architecture.md` section 3 with integers only; can dump each layer | numpy |
| `gen_vectors.py` | Picks N test images, emits input hex, expected per-layer output hex and the expected argmax | numpy |
| `data/<net>/weights.hex`, `desc.json`, `vectors/` | Generated results committed to the repository so that simulation does not need PyTorch | - |

Rules:

- `npu_ref.py` is the only source of truth. PyTorch only produces weights; every expected value after that comes from the integer model.
- Committed data files are generated with a fixed random seed; the scripts can regenerate them and the result can be diffed.
- Committed data stays under 1 MB. Unit-level random vectors (such as the 100k requant vectors from
  `gen_requant_vectors.py`) are generated into `build/` at build time and are not committed.
- `model/test_npu_ref.py` holds hand-computed cases and runs with `make model-test`; the same hand-computed values
  are repeated in the matching SystemVerilog testbench, giving a four-way cross-check:
  hand calculation, Python, SV reference function, RTL.

## 2. Verification levels

### 2.1 Unit level (one self-checking testbench per module, `tb/unit/<module>_tb.sv`)

| Module | Stimulus | Check |
| --- | --- | --- |
| `requant_unit` | Random acc / bias / mult / shift / zero point, plus directed cases for saturation, rounding exactly at 0.5 and shift extremes | Compared one by one with the SV integer reference function; that function is in turn cross-checked against Python |
| `pe` | Random int8 sequences with first / last flags | Accumulated value and shadow timing |
| `systolic_array` | Random `In[M][K]`, `W[K][8]`, K from 1 to 512, M both a multiple of 8 and not | Drained columns equal the numpy matrix product; idle cycles correct for K < 8 |
| `act_buffer` | Random unaligned 8-byte reads and writes, byte enables, same-cycle read and write to different addresses | Compared with a byte-array model in SV |
| `conv_addr_gen` | Random H / W / C_in / C_out / K / pad | Address stream, lane masks and first / last match a Python-generated sequence cycle by cycle |
| `axil_slave` + `npu_regs` | AXI4-Lite reads and writes, W1P / W1C semantics, memory access while busy | Response codes, register values, HOST_CONFLICT |
| `npu_sequencer` | Stub engines (immediate or delayed done), illegal descriptors | Layer order, perf counts, error code and CUR_LAYER |
| `pool_engine`, `argmax_engine` | Random tensors | Match numpy |

### 2.2 Integration level (`tb/integration/`)

1. Single CONV2D layer: random shape and weights, loaded over AXI4-Lite, run, output tensor read back and compared with the golden model.
2. Single FC, MAXPOOL and ARGMAX likewise.
3. Full 8-layer LeNet-5 on MNIST: read back and compare layer by layer; then run 100 images and require the
   accuracy to equal that of `npu_ref.py` exactly.
4. LeNet-5 on CIFAR-10: same, with the other data files.
5. Error injection: at least one case per error code.

### 2.3 Assertions (SVA in `tb/common/`, bound to the RTL)

- AXI4-Lite: once valid is high it must not drop or change data before ready; bresp / rresp are only OKAY or SLVERR.
- Memory: engine writes and host writes never happen in the same cycle; read addresses stay in range.
- Array: the number of valid cycles between `first` and `last` equals K_total; shadows are not overwritten during a drain (tile period >= ARR_N).
- Sequencer: no engine start while busy is 0; done and err are mutually exclusive.

## 3. Tools and how to run

| Tool | Use | Where |
| --- | --- | --- |
| Verilator 5 (`--timing`) | Lint, unit and integration simulation | Local, GitHub Actions |
| VCS | Additional run of the same filelists, recorded in `docs/STATUS.md` | Separate machine available to the author |
| Python 3 + numpy | Golden model, vector generation | Local, GitHub Actions |
| PyTorch | Training only, not in CI | Local, optional |

Constraints:

- Testbenches use the SystemVerilog subset Verilator supports: procedural testbenches, queues, `$readmemh`,
  `$fopen` / `$fscanf`; no classes and no UVM.
- Shared reference functions live in `tb/common/npu_tb_pkg.sv` and correspond one to one with `model/npu_ref.py`.
- Stimulus is driven on the falling clock edge with blocking assignments, and checks happen on the falling edge too.
  The DUT then samples identical values under Verilator and VCS, avoiding the semantic difference of non-blocking
  assignments inside initial blocks between simulators.
- Every testbench ends with a `PASS:` or `FAIL:` line and exits via `$finish` / `$fatal`; the Makefile judges by the `PASS` line in the log.
- Each unit testbench has its own filelist `sim/filelists/unit_<module>.f`, shared by Verilator and VCS;
  `make unit-<module>` runs Verilator, `make vcs-unit-<module>` runs VCS.
- `make lint`, `make model-test`, `make unit` and `make smoke` make up `make ci`, which GitHub Actions runs in full; `make integ` is added in M4.
- RTL lint is `-Wall` with warnings fatal; testbench builds tolerate warnings, but known checker-style warnings are
  annotated with a local `lint_off` and a reason.
- Coverage: one functional-point review with Verilator `--coverage` in M5, without chasing a number.

## 4. Definition of done per milestone

| Milestone | Done when |
| --- | --- |
| M1 | `requant_unit` and `pe` unit testbenches pass; the requant function of `npu_ref.py` agrees with the SV reference function on 100k random samples |
| M2 | `systolic_array` is bit-exact on 200 random GEMMs, including K < 8 and K > 256 |
| M3 | `axil_slave`, `npu_regs` and `npu_sequencer` unit testbenches pass; every error code is covered |
| M4 | Single-layer CONV2D / FC / MAXPOOL / ARGMAX integration testbenches are bit-exact |
| M5 | LeNet-5 on MNIST and CIFAR-10 is bit-exact end to end; accuracy equals the golden model; CI green; README has the performance table |
| M6 | Optional items each define their own: DMA integration testbench, chip_top CDC testbench, FPGA synthesis report |
