# Design Decision Record

Each entry answers four questions: what the choice was, what was chosen, why, and what was given up at what cost.
This file records the "why"; the "what" lives in `architecture.md` and `programming_model.md`.
New decisions are appended at the end. A superseded decision is marked "superseded by ADR-xxx", never deleted.

## ADR-001 Programmable, descriptor-driven, instead of one hard-wired datapath per layer

- Choice: all convolution and fully-connected layers share one `conv_engine`; each layer's shape, addresses and
  quantization parameters come from a descriptor table written by the host.
- Why: the point of an accelerator is that one piece of hardware runs different networks. A hard-wired datapath and
  RAM per layer has to be rewritten for every new network; it is a circuit drawing of one network. A descriptor
  table is the smallest common denominator of real NPUs (a simplified form of an instruction or command queue) and
  is the line between "built an accelerator" and "built one fixed circuit".
- Rejected: hard-wired layers are easier to tune individually but cannot be reused; a real instruction set and
  compiler are beyond what a single-person project can keep under control.
- Cost: a general address generator and parameter checking, which is the most complex control logic in v1.

## ADR-002 INT8 integer inference, not INT8 storage with FP16 compute

- Choice: int8 x int8 multiply, int32 accumulate, integer requantization back to int8. No floating point anywhere.
- Why: this is what TFLite, TensorRT INT8 and most edge NPUs do. An int8 multiplier is far smaller than an FP16
  multiply-add, accumulation has no rounding error, and results can be matched against software bit for bit. An
  FP16 path needs a dequantization unit and FP16 MACs; it is expensive and makes bit-exactness nearly impossible.
- Rejected: FP16 has more precision and is less sensitive to quantization quality. For a LeNet-class network the
  int8 accuracy loss is negligible.
- Cost: a custom quantization export script, and quantization arithmetic specified down to the last bit.

## ADR-003 Fold the zero point into the bias; the PE does only int8 x int8

- Choice: the export script computes `bias - z_in * sum(w)`; the PE never sees a zero point; padding is filled with `z_in`.
- Why: the PE is replicated 64 times. Dropping a 9-bit subtractor and a zero-point broadcast per PE keeps the
  multiplier at 8x8. This is the standard gemmlowp / TFLite trick; correctness depends on a single convention,
  "padding value equals z_in".
- Rejected: subtracting `q - z_in` at the PE input is conceptually simpler but costs one subtractor and one extra bit per PE.
- Cost: the address generator must substitute `z_in`, not 0, for out-of-range lanes; the testbench checks this convention explicitly.

## ADR-004 Output-stationary dataflow, not weight-stationary

This is the most important architectural decision in v1.

- Choice: each PE owns one output element; activations flow along rows, weights along columns, partial sums stay in the PE.
- Why, in order of importance:
  1. Simple memory access. Output-stationary needs, per cycle, the inputs of 8 adjacent output pixels for the same k,
     which in CHW layout are 8 consecutive bytes; 8 byte banks give any start address with zero conflicts.
     Weight-stationary needs, per cycle, the inputs of one output pixel for 8 different k, scattered over different
     rows and channels: either 8 read ports, an elaborate bank plan, or an im2col buffer first.
  2. No partial-sum storage. Weight-stationary tiles the K dimension, so partial sums must be written to an int32
     SRAM and read back; output-stationary finishes the K loop inside the PE and partial sums never leave it.
  3. Fits small C_out. LeNet has C_out of 6 and 16. The benefit of keeping weights resident in the array comes from
     large M; on a small network the overhead of K tiling and of swapping weights on every drain does not pay off.
- Rejected: weight-stationary is the TPU's choice and suits large K and N with inputs already in matrix form. It is
  a reasonable comparison experiment after M6, not a first version.
- Cost: weights stream through the array again for every m_tile, so total weight reads are the weight size times
  the number of m_tiles. On-chip SRAM at 8 bytes per cycle is sufficient; with off-chip weights this dataflow would
  have to change. FC utilization is 1/8, but that is the weight-bandwidth ceiling and applies to every dataflow at batch 1.

## ADR-005 CHW layout and 8-byte-bank unaligned access

- Choice: activations in CHW with one byte per element; act_buffer made of 8 byte banks supporting 8-byte reads and
  writes at any byte address.
- Why: ADR-004 needs "the inputs of 8 adjacent output pixels" to be consecutive addresses, and in CHW adjacent
  pixels in a row are consecutive. The FC input vector is then the natural flattening of the previous tensor with no
  reordering. Byte banks plus address rotation is the textbook construction; it is cleaner than a double-word read
  plus shifter and wastes no bandwidth.
- Rejected: NHWC makes channels consecutive and suits channel counts that are multiples of 8; LeNet has C_in of 1,
  3 and 6, so NHWC would waste most lanes.
- Cost: a stride-2 convolution puts 8 lanes into 4 banks, so v1 does not support stride 2 and downsamples with maxpool.

## ADR-006 Drain via shadow registers and a column mux, not a shift chain

- Choice: the PE copies its accumulator into a shadow at `last`; once the last PE has finished, the array reads one
  column per cycle through an 8:1 mux.
- Why: because of the skew, PE completion times form a diagonal wavefront. A shift chain moving with the wavefront
  gets overwritten by later-finishing PEs; one moving against it delivers a value only every second cycle, so the
  drain takes 16 cycles. Shadow plus mux has neither problem and only requires a tile period of at least 8 cycles,
  guaranteed by inserting idle cycles when K_total is below 8.
- Rejected: a shift chain has more local wiring and scales better on large arrays; at 8x8 the mux is entirely acceptable.
- Cost: 32 extra flip-flops per PE and an 8-row by 8:1 by 32-bit mux.

## ADR-007 AXI4-Lite host interface; AXI4 master DMA deferred to M6

- Choice: registers and on-chip memory are accessed through one AXI4-Lite slave; moving data from external memory
  is left to an M6 AXI4 master.
- Why: an accelerator IP in an SoC almost always hangs off an AXI interconnect, and AXI4-Lite is the de facto
  standard for a configuration port. APB would also work but is more typical of low-speed peripherals. Deferring the
  DMA is possible because the v1 on-chip memory holds the whole of LeNet, and loading over AXI4-Lite first lets the
  compute core be verified end to end earlier.
- Rejected: APB is smaller to implement. If ever needed, an APB or SPI to AXI4-Lite bridge can be added in the M6
  `chip_top` without touching the compute core.
- Cost: five AXI4-Lite channels are a little more code than an APB state machine; loading 67 KB of weights word by
  word takes about 17k writes in simulation, which the testbench can shortcut with backdoor loading.

## ADR-008 Descriptor table in the register file, not fetched from memory

- Choice: a flip-flop array of 16 entries x 10 used words, read and written by the host over AXI4-Lite and indexed directly by the sequencer.
- Why: v1 networks have at most 8 layers, so 16 entries suffice; registers avoid a descriptor-fetch state machine
  and a read-port arbiter. 16 x 10 x 32 is about 5k flip-flops, acceptable at this scale.
- Rejected: descriptors in memory with only a pointer register is the more scalable scheme, suited to networks with
  dozens of layers. If deeper networks follow the M6 DMA, switch to "pointer plus prefetch".
- Cost: the layer count limit is fixed by a parameter.

## ADR-009 Single clock domain in v1; all SoC glue goes into chip_top

- Choice: `npu_top` has one clock, one reset, one AXI4-Lite port and one interrupt. Pixel-stream CDC, PLL, clock
  gating, SPI or APB configuration bridge and pads all belong to the M6 `chip_top`.
- Why: the correctness and dataflow of the compute core are the substance of this project; CDC, async FIFOs and
  clock control are skills the author already has and a thin wrapper is enough to show them, so they should not
  consume the first five milestones.
- Rejected: starting with multiple clock domains would expose integration issues earlier but would complicate every unit testbench.
- Cost: at M6 the AXI4-Lite port and the interrupt of `npu_top` need one clock-domain crossing.

## ADR-010 Verilator as the primary simulator, VCS as a secondary check

- Choice: all testbenches stay within the subset Verilator 5 supports; GitHub Actions runs lint, unit and
  integration simulation; VCS results are recorded in `STATUS.md` as additional evidence.
- Why: the intended reader of this repository is anyone who wants to reproduce it, and they do not have VCS. A
  green CI is itself part of the deliverable. The author's VCS access is on a separate machine and cannot be relied on.
- Rejected: UVM and class-based testbenches. Their Verilator support is incomplete and the payoff at this scale is limited.
- Cost: a more procedural testbench style and a weaker coverage toolchain.

## ADR-011 Support both MNIST and CIFAR-10, MNIST first

- Choice: M5 is complete with MNIST; CIFAR-10 is a second descriptor and weight set for the same RTL.
- Why: under LeNet-5 the two differ only in conv1's C_in and padding, so CIFAR-10 costs almost no hardware and
  directly demonstrates that "programmable" is not a slogan. MNIST comes first because it is smaller, easier to
  debug and has better-controlled quantization error.
- Rejected: MNIST alone would leave the three-channel RGB path (C_in = 3) uncovered by a real network.
- Cost: a second set of training and quantization data, and LeNet reaches only about 60% on CIFAR-10, which the
  README must attribute to the network, not the hardware.

## ADR-012 No stride 2 and no per-channel quantization

- Choice: the v1 descriptor reserves a STRIDE_M1 field but requires 0; quantization parameters are per-tensor.
- Why: LeNet needs no stride 2; per-channel quantization improves LeNet's accuracy negligibly while turning mult /
  shift into per-channel tables. Both have explicit extension paths in `architecture.md` section 10, with fields and
  addresses reserved.
- Rejected: more general operator support.
- Cost: a network using stride-2 convolution needs the extension first.

## ADR-013 Branch and commit policy

- Choice: one branch per milestone (`docs/architecture-v1`, `feat/m1-requant-pe`, ...), merged into `main`
  through a pull request with squash merge; `main` always lints and simulates; `docs/STATUS.md` is updated in every PR.
- Why: the PR history is itself a record of the engineering process; small PRs are easier to review later, both for
  the author and for anyone reading the repository.
- Rejected: a long-lived development branch merged in one go.
- Cost: discipline in branch naming and PR descriptions.
