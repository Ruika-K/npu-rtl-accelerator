// Tests. Each one builds the same environment and runs one sequence (or, for
// npu_full_test, all of them back to back, which is the one to merge
// coverage from). The register-model tests use the UVM built-in sequences.

class npu_base_test extends uvm_test;
  `uvm_component_utils(npu_base_test)

  npu_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = npu_env::type_id::create("env", this);
  endfunction

  function void end_of_elaboration_phase(uvm_phase phase);
    uvm_root::get().set_timeout(1s, 1);
    // UVM 1800.2-2020 warns whenever the mirror of a volatile field is read,
    // which the predictor's check_on_read does for every read of a register
    // with volatile fields. Those fields are excluded from the compare, so the
    // warning carries no information here.
    uvm_root::get().set_report_id_action("UVM/FLD/GET_MIRRORED_VAL/VOL", UVM_NO_ACTION);
    if (uvm_report_enabled(UVM_HIGH)) uvm_root::get().print_topology();
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    run_body();
    phase.drop_objection(this);
  endtask

  virtual task run_body();
  endtask

  task run_seq(npu_base_seq seq);
    seq.start(env.host.sqr);
  endtask

  function void report_phase(uvm_phase phase);
    uvm_report_server rs = uvm_report_server::get_server();
    int unsigned n = rs.get_severity_count(UVM_ERROR) + rs.get_severity_count(UVM_FATAL);
    if (n == 0) `uvm_info("TEST", {"TEST PASSED: ", get_type_name()}, UVM_NONE)
    else        `uvm_info("TEST", $sformatf("TEST FAILED: %s, %0d errors", get_type_name(), n), UVM_NONE)
  endfunction
endclass

// Every register reads its reset value.
class npu_reg_reset_test extends npu_base_test;
  `uvm_component_utils(npu_reg_reset_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    uvm_reg_hw_reset_seq seq = uvm_reg_hw_reset_seq::type_id::create("seq");
    seq.model = env.regs;
    seq.start(env.host.sqr);
  endtask
endclass

// Every RW bit can be set and cleared on its own, and RO / reserved bits cannot.
class npu_reg_bit_bash_test extends npu_base_test;
  `uvm_component_utils(npu_reg_bit_bash_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    uvm_reg_bit_bash_seq seq = uvm_reg_bit_bash_seq::type_id::create("seq");
    seq.model = env.regs;
    seq.start(env.host.sqr);
  endtask
endclass

// Walking pattern through every word of both buffers.
class npu_mem_walk_test extends npu_base_test;
  `uvm_component_utils(npu_mem_walk_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    uvm_mem_walk_seq seq = uvm_mem_walk_seq::type_id::create("seq");
    seq.model = env.regs;
    seq.start(env.host.sqr);
  endtask
endclass

class npu_layer_test extends npu_base_test;
  `uvm_component_utils(npu_layer_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    npu_layer_seq seq = npu_layer_seq::type_id::create("seq");
    run_seq(seq);
  endtask
endclass

class npu_desc_err_test extends npu_base_test;
  `uvm_component_utils(npu_desc_err_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    npu_desc_err_seq seq = npu_desc_err_seq::type_id::create("seq");
    run_seq(seq);
  endtask
endclass

class npu_dma_test extends npu_base_test;
  `uvm_component_utils(npu_dma_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    npu_dma_seq seq = npu_dma_seq::type_id::create("seq");
    run_seq(seq);
  endtask
endclass

class npu_lenet_test extends npu_base_test;
  `uvm_component_utils(npu_lenet_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  task run_body();
    npu_lenet_seq seq = npu_lenet_seq::type_id::create("seq");
    run_seq(seq);
  endtask
endclass

// Everything in one simulation: reset values, layers, descriptor errors, DMA, the network.
// Fails unless every covergroup bin was hit (functional-coverage closure).
class npu_full_test extends npu_base_test;
  `uvm_component_utils(npu_full_test)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void connect_phase(uvm_phase phase);
    env.cov.require_closure = 1;
  endfunction

  task run_body();
    uvm_reg_hw_reset_seq reset_seq = uvm_reg_hw_reset_seq::type_id::create("reset_seq");
    npu_layer_seq        layer_seq = npu_layer_seq::type_id::create("layer_seq");
    npu_desc_err_seq     err_seq   = npu_desc_err_seq::type_id::create("err_seq");
    npu_dma_seq          dma_seq   = npu_dma_seq::type_id::create("dma_seq");
    npu_lenet_seq        lenet_seq = npu_lenet_seq::type_id::create("lenet_seq");
    reset_seq.model = env.regs;
    reset_seq.start(env.host.sqr);
    run_seq(layer_seq);
    run_seq(err_seq);
    run_seq(dma_seq);
    run_seq(lenet_seq);
  endtask
endclass
