// Environment: AXI4-Lite agent (host), AXI4 slave memory (DMA), register
// model with an explicit predictor, scoreboard and coverage.

class npu_env_cfg extends uvm_object;
  `uvm_object_utils(npu_env_cfg)

  npu_reg_block   regs;
  axi_slave_mem   ext_mem;
  npu_scoreboard  sb;
  virtual irq_if  irq_vif;

  function new(string name = "npu_env_cfg");
    super.new(name);
  endfunction
endclass

class npu_env extends uvm_env;
  `uvm_component_utils(npu_env)

  axil_agent                      host;
  axi_slave_mem                   ext_mem;
  npu_scoreboard                  sb;
  npu_coverage                    cov;
  npu_reg_block                   regs;
  axil_reg_adapter                adapter;
  uvm_reg_predictor #(axil_item)  predictor;
  npu_env_cfg                     cfg;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    host      = axil_agent::type_id::create("host", this);
    ext_mem   = axi_slave_mem::type_id::create("ext_mem", this);
    sb        = npu_scoreboard::type_id::create("sb", this);
    cov       = npu_coverage::type_id::create("cov", this);
    predictor = uvm_reg_predictor #(axil_item)::type_id::create("predictor", this);
    adapter   = axil_reg_adapter::type_id::create("adapter");

    regs = npu_reg_block::type_id::create("regs");
    regs.configure(null, "");
    regs.build();
    regs.lock_model();
    regs.reset();

    cfg = npu_env_cfg::type_id::create("cfg");
    cfg.regs    = regs;
    cfg.ext_mem = ext_mem;
    cfg.sb      = sb;
    if (!uvm_config_db #(virtual irq_if)::get(this, "", "irq_vif", cfg.irq_vif))
      `uvm_fatal("ENV/NOVIF", "no virtual irq_if")
    uvm_config_db #(npu_env_cfg)::set(this, "*", "cfg", cfg);
    cov.regs = regs;
  endfunction

  function void connect_phase(uvm_phase phase);
    regs.default_map.set_sequencer(host.sqr, adapter);
    regs.default_map.set_auto_predict(0);
    regs.default_map.set_check_on_read(1);
    predictor.map     = regs.default_map;
    predictor.adapter = adapter;
    host.mon.ap.connect(predictor.bus_in);
    host.mon.ap.connect(sb.axil_imp);
    host.mon.ap.connect(cov.axil_imp);
    ext_mem.ap.connect(sb.axi_imp);
    ext_mem.ap.connect(cov.axi_imp);
  endfunction
endclass
