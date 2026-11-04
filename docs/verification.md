# NPU v1 Verification Plan

One goal: every layer output of the RTL matches the Python integer golden model bit for bit, and anyone who clones
the repository can reproduce that with a single command.

## 1. Reference model and data

Directory `model/`:

| File | Purpose | Depends on |
| --- | --- | --- |
| `train_lenet.py` | Trains fp32 LeNet-5 with numpy only (im2col convolutions, Adam), one run each for MNIST and CIFAR-10; writes `model/out/<net>/fp32.npz`, `calib.npz`, `test.npz` | numpy; MNIST downloads automatically, CIFAR-10 needs the Toronto archive. Optional: simulation runs without it |
| `quantize.py` | Post-training quantization: symmetric per-tensor weights, activation range from a calibration set (post-ReLU ranges start at 0); computes RQ_MULT / RQ_SHIFT; folds the zero point into the bias; builds the memory layout and descriptors; writes `model/data/<net>/program.txt`, `net.json` and a float16 copy of the checkpoint | numpy, reads the trainer's output or the committed float16 checkpoint |
| `npu_ref.py` | Pure-integer golden model: requant, GEMM, conv2d / fc / maxpool / argmax layers, memory-image helpers, descriptor checks, and `run_program`, which executes a descriptor list against the two memories exactly like the hardware | none |
| `gen_requant_vectors.py`, `gen_gemm_vectors.py`, `gen_desc_vectors.py`, `gen_layer_vectors.py` | Unit and single-layer vectors generated into `build/` at build time | none |
| `data/<net>/program.txt` | Committed: descriptors, the weight-buffer image, 100 test images with the integer model's prediction and label, and every layer's output bytes for the first two images | - |

Rules:

- `npu_ref.py` is the only source of truth. PyTorch only produces weights; every expected value after that comes from the integer model.
- Committed data files are generated with a fixed random seed; the scripts can regenerate them and the result can be diffed.
- Committed data stays small: one `program.txt` per network (weights, 100 images, two full traces) plus the
  float16 checkpoint. Unit-level random vectors are generated into `build/` at build time and are not committed.
- `model/test_npu_ref.py` holds hand-computed cases and runs with `make model-test`; the same hand-computed values
  are repeated in the matching SystemVerilog testbench, giving a four-way cross-check:
  hand calculation, Python, SV reference function, RTL.

## 2. Verification levels

### 2.1 Unit level (one self-checking testbench per module, `tb/unit/<module>_tb.sv`)

| Module | Stimulus | Check |
| --- | --- | --- |
| `requant_unit` | Random acc / bias / mult / shift / zero point, plus directed cases for saturation, rounding exactly at 0.5 and shift extremes, and boundary vectors on both sides of the internal 14-bit saturation and of the int8 window for every shift | Compared one by one with the SV integer reference function; that function is in turn cross-checked against Python |
| `pe` | Random int8 sequences with first / last flags | Accumulated value and shadow timing |
| `systolic_array` | GEMMs driven tile by tile as `conv_engine` will: M up to 64, K from 1 to 512, N up to 32, partial tiles with random garbage in unused lanes, random idle cycles anywhere, N-K idle cycles after tiles with K < N; data modes include all -128 and random extremes | Every drained element equals the SV integer matrix product (itself cross-checked against Python GEMM vectors); column order 0..N-1; `drain_valid_o` never drops mid-drain; the drain starts exactly N-1 edges after the last beat |
| `act_buffer` | Random unaligned 8-byte reads and writes with byte enables, same-cycle read and write to different rows, read the cycle after a write, buffer-edge wrap, back-to-back reads | Compared with a byte-array model in SV; fixed 2-cycle latency |
| `conv_engine`, `pool_engine`, `argmax_engine` | Single-layer cases from `model/gen_layer_vectors.py` (directed kernel sizes, paddings, partial tiles, FC shapes, odd pool sizes, argmax ties, plus random), with the real memories; images loaded through the memory write ports, a guard band around the output region | Output region bit-exact with the Python model, guard bands untouched, no writes outside the region or while idle; argmax index and logits |
| `axil_slave` | `tb/common/axil_master_bfm.sv` drives every AW/W ordering, random response back-pressure and a same-cycle read/write; a responder model answers after random delays and flags errors by address | Data and strobes reach the responder, read data and response codes, AXI4-Lite rules (valid held until ready, readies only with valids, no simultaneous read and write accept), one internal-bus transaction at a time, write wins arbitration |
| `npu_regs` (behind `axil_slave`) | Testbench plays the sequencer, the argmax engine and both memories; directed register sequences plus random mixed traffic against a reference | Reset values, RW / RO / W1P / W1C, START side effects, irq, descriptor words with byte strobes and reserved masks, the sequencer's descriptor read port (the whole table walked twice while the host keeps reading descriptor words: grant only without a same-cycle host read, data one cycle after the grant), memory-window routing, SLVERR for unmapped / misaligned / busy accesses, HOST_CONFLICT |
| `npu_sequencer` | Stub engines with random latency or injected errors, a model of the register file's descriptor read port with random grant stalls, a LeNet-5 program, NOP layers, start while busy, one directed case per check, descriptors from `model/gen_desc_vectors.py` | Engine order and forwarded descriptors (FC view), done / busy / cur_layer timing, every error code with the failing layer, the three counters against a cycle-exact model, error codes equal to the Python model on 2000 random descriptors, descriptor requests only for the current layer and in word order |
| `pool_engine`, `argmax_engine` | Random tensors | Match numpy |
| `async_fifo` | Two unrelated clocks swept over six ratios, random stalls on both sides, 24k words | Order and content against a queue model, no underflow, occupancy from the DUT pointers never above DEPTH |
| `pix_writer` | Random frames at random bases with lengths that are not multiples of 8, random port stalls, a byte before sof, a sof inside a frame | Bytes land exactly in the frame region (guard bytes untouched), frame-done count, error pulses |
| `npu_dma` (inside `make integ-layers`) | The biggest layer case replayed with DMA loads of image, weights and biases from `axi4_slave_mem`, the layer run, the output stored back by DMA; a host access and a START during a DMA; misaligned length, out-of-range destination, SLVERR on read and on write | Stored output equals the model; HOST_CONFLICT and ignored START; error codes 2 and 3; the memory model sees no protocol violation |
| `chip_top` (`make integ-chip`) | Two clocks (100 MHz system, about 37 MHz pixel); APB loads the program through the bridge; frames arrive on the pixel stream; a frame streamed during a run; a rude source; the clock gate | Every prediction equals the integer model; FRAME_DONE and PIX_FRAMES; back-pressure through the FIFO; SOF_ERR and OVERRUN; PSLVERR on the core window while gated |

### 2.2 Integration level (`tb/integration/npu_top_tb.sv`, `make integ`)

1. Single CONV2D, FC, MAXPOOL and ARGMAX layers from the same layer vectors: images loaded over AXI4-Lite with
   32-bit writes and strobes, descriptor and LAYER_CNT written, START, STATUS polled, output region and guard bands
   read back and compared with the golden model; ARGMAX checked through RESULT and LOGITS. (M4)
2. Around the cases: ID, an error descriptor in layer 1 of a 2-layer program (ERR, ERR_CODE, CUR_LAYER) followed by
   normal runs, a host memory read rejected while busy with HOST_CONFLICT, perf counters, DONE W1C and irq. (M4)
3. Full 8-layer LeNet-5 (`tb/integration/npu_lenet_tb.sv`, `make integ-<net>`): weights and descriptors loaded
   once, then per image the input is written, the run started and RESULT compared with the integer model's
   prediction; every prediction must agree (bit-exact end to end, not merely the same accuracy). For the first two
   images every layer's output region is read back and compared byte for byte. The per-layer cycle counts of the
   first image are printed for the README table. (M5, MNIST; CIFAR-10 with its own data files)
4. Error injection: at least one case per error code at the unit level (M3) and an error descriptor with recovery
   at the top level (M4).

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
- `make lint`, `make model-test`, `make unit` and `make integ` make up `make ci`, which GitHub Actions runs in full.
- RTL lint is `-Wall` with warnings fatal; testbench builds tolerate warnings, but known checker-style warnings are
  annotated with a local `lint_off` and a reason.
- Coverage: one functional-point review with Verilator `--coverage` in M5, without chasing a number.

## 4. Definition of done per milestone

| Milestone | Done when |
| --- | --- |
| M1 | `requant_unit` and `pe` unit testbenches pass; the requant function of `npu_ref.py` agrees with the SV reference function on 100k random samples |
| M2 | `systolic_array` is bit-exact on 200 random GEMMs, including K < 8 and K > 256 |
| M3 | `axil_slave`, `npu_regs` and `npu_sequencer` unit testbenches pass; every error code is covered; descriptor checks agree with the Python model |
| M4 | Single-layer CONV2D / FC / MAXPOOL / ARGMAX cases are bit-exact at the engine level and through AXI4-Lite at the top level |
| M5 | LeNet-5 is bit-exact end to end on every tested image (MNIST; CIFAR-10 once its training data is available); CI green; README has the performance table |
| M6 | Optional items each define their own: DMA integration testbench, chip_top CDC testbench, FPGA synthesis report |
