# Synthesis flow

Open-source only. `sv2v` translates the SystemVerilog to Verilog-2005 (the Yosys
front end does not accept packages in module headers, `return` in functions
or `localparam` in parameter port lists), Yosys synthesizes it, and
nextpnr-ecp5 places and routes the ECP5 netlist for a timing number.

```sh
make synth-xilinx   # sv2v + yosys synth_xilinx (xc7): build/synth/xilinx_stat.txt
make synth-ecp5     # sv2v + yosys synth_ecp5 + nextpnr-ecp5 --85k: build/synth/nextpnr.log
```

Tools: sv2v (https://github.com/zachjs/sv2v), Yosys 0.33 or newer, nextpnr-ecp5.
The CI workflow `.github/workflows/synth.yml` runs both and publishes the
numbers in the job summary; `docs/synthesis.md` records them.

The SRAMs are the behavioural `sram_1r1w` and are inferred as block RAM; a
real tape-out would replace that module with memory macros.
