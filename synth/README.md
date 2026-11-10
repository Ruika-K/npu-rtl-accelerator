# Synthesis flow

Open-source only. `sv2v` translates the SystemVerilog to Verilog-2005 (the Yosys
front end does not accept packages in module headers, `return` in functions
or `localparam` in parameter port lists), Yosys synthesizes it, and
nextpnr-ecp5 places and routes the ECP5 netlist for a timing number.

```sh
make synth-xilinx   # sv2v + yosys synth_xilinx (xc7): build/synth/xilinx_stat.txt
make synth-ecp5     # sv2v + yosys synth_ecp5 + nextpnr-ecp5 --85k: build/synth/nextpnr.log
```

- `stat_table.py` turns the non-flattened xc7 report into a per-module table (cells include submodules).
- `nextpnr_summary.py` prints the ECP5 utilisation, the post-route Fmax and the critical path, one line
  per RTL signal on the path. The signal names come from the cell names, which inside a mapped cone can be an
  arbitrary neighbour, so given the synthesized Verilog it also lists the source lines (with their module)
  that the path's cells were built from; the workflow additionally prints the raw path with the placement
  coordinates of every hop.
- `npu_top_ooc.sv` is the out-of-context wrapper used for ECP5 place and route. `npu_top` has 273 input and
  211 output bits, more than any ECP5 package bonds; the wrapper feeds every input from a shift register on
  one pin and XOR-reduces every registered output onto another, so no logic is removed, the clock stays on a
  global network and every path is register to register. It adds 485 flip-flops and a small XOR tree. The
  xc7 resource report runs on `npu_top` itself.
- `ecp5.ys` marks the requant units' layer-constant registers `keep` after `proc`, so that Yosys does not merge
  the eight identical copies into one high-fanout register (ADR-018). It asserts that it found exactly 23
  registers, so a renamed or added register fails the run instead of silently dropping the constraint.

Tools: sv2v (https://github.com/zachjs/sv2v), Yosys 0.33 or newer, nextpnr-ecp5.
The CI workflow `.github/workflows/synth.yml` runs both and publishes the
numbers in the job summary; `docs/synthesis.md` records them. It routes three
placement seeds and stops one after 45 minutes (normal runs take 7 to 12):
router2 has twice failed to converge on a placement of this design.

The SRAMs are the behavioural `sram_1r1w` and are inferred as block RAM; a
real tape-out would replace that module with memory macros.
