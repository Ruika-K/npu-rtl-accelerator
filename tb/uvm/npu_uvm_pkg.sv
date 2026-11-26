`timescale 1ns/1ps

// SystemVerilog UVM environment for npu_top. Written against the common
// subset of UVM 1.2 and IEEE 1800.2 (2017 / 2020).
package npu_uvm_pkg;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "axil_agent.svh"
  `include "axi_slave_mem.svh"
  `include "npu_reg_model.svh"
  `include "npu_vectors.svh"
  `include "npu_scoreboard.svh"
  `include "npu_coverage.svh"
  `include "npu_env.svh"
  `include "npu_seq_lib.svh"
  `include "npu_tests.svh"
endpackage
