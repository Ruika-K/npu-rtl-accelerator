# Synthesis Results

Open-source flow, run by `.github/workflows/synth.yml` on every push (see `synth/README.md`):
`sv2v` translates the SystemVerilog to Verilog-2005, Yosys synthesizes it, nextpnr-ecp5 places and routes the
ECP5 netlist. These are estimates for sizing and for checking that every construct is synthesizable, not a
tape-out: the SRAMs are the behavioural `sram_1r1w` inferred as block RAM, there are no I/O constraints, and the
default parameters (`ARR_N = 8`, 32 KB activation buffer, 128 KB weight buffer) are used.

## Xilinx 7-series (yosys `synth_xilinx -family xc7`, not flattened)

Numbers are filled in from the CI job summary of the commit named in the table.

| Commit | LUTs | FFs | BRAM36 | DSP48 | Notes |
| --- | --- | --- | --- | --- | --- |
| pending | | | | | |

Per-module breakdown of the largest blocks:

| Module | Instances | LUTs each | FFs each | Notes |
| --- | --- | --- | --- | --- |
| pending | | | | |

## Lattice ECP5-85k (yosys `synth_ecp5` + nextpnr-ecp5, `--freq 100`)

| Commit | Slices | LUT4 | FFs | DP16KD | MULT18X18D | Fmax (clk_i) |
| --- | --- | --- | --- | --- | --- | --- |
| pending | | | | | | |

## Reading the numbers

- The systolic array is 64 `pe` instances: one 8x8 multiplier (one DSP) and a 32-bit accumulator plus a 32-bit
  shadow register each.
- The eight `requant_unit` instances carry a 33x32-bit multiplier each; on an FPGA that is several DSP blocks or a
  large LUT multiplier, and it is the widest arithmetic in the design. A tape-out would size it to the real
  precision need (see `decisions.md` on the TFLite-style multiplier).
- The sequencer's descriptor checks and the engines' address arithmetic contain a few 16x16 and 16x32 multipliers
  that run once per layer or per tile; they could be made sequential if area mattered.
- `npu_regs` holds the 16 x 10-word descriptor table in flip-flops (about 5 k FFs), by design (ADR-008).
