// Functional coverage, sampled from the monitors and the register model:
//   cg_axil  - every host access: window x direction x response, misalignment
//   cg_layer - each descriptor of an accepted START: op x kernel x padding, ReLU
//   cg_run   - how runs end (DONE or each ERR_CODE) and the program length
//   cg_dma   - DMA transfers: direction x buffer, outcome
//   cg_burst - DMA bursts on AXI4: direction x length class x 4 KB end

class npu_coverage extends uvm_component;
  `uvm_component_utils(npu_coverage)

  uvm_analysis_imp_axil #(axil_item, npu_coverage)      axil_imp;
  uvm_analysis_imp_axi  #(axi_burst_item, npu_coverage) axi_imp;

  npu_reg_block regs;

  // sampled values
  local int unsigned s_window;      // 0 reg, 1 desc, 2 abuf, 3 wbuf, 4 unmapped
  local bit          s_write, s_slverr, s_misaligned;
  local int unsigned s_op, s_k, s_pad, s_layers;
  local bit          s_relu;
  local int unsigned s_end;         // 0 DONE, else ERR_CODE
  local bit          s_dir, s_sel;
  local int unsigned s_dma_end;     // 0 DONE, else ERR_CODE
  local int unsigned s_beats;
  local bit          s_4k_end;

  covergroup cg_axil;
    window: coverpoint s_window { bins reg_ = {0}; bins desc = {1}; bins abuf = {2}; bins wbuf = {3}; bins none = {4}; }
    dir:    coverpoint s_write;
    resp:   coverpoint s_slverr;
    mis:    coverpoint s_misaligned;
    window_x_dir_x_resp: cross window, dir, resp {
      // Unmapped or misaligned always fail; aligned register accesses never do.
      ignore_bins reg_err = binsof(window.reg_) && binsof(resp) intersect {1};
      ignore_bins desc_err = binsof(window.desc) && binsof(resp) intersect {1};
      ignore_bins none_ok = binsof(window.none) && binsof(resp) intersect {0};
    }
  endgroup

  covergroup cg_layer;
    op:  coverpoint s_op  { bins nop = {0}; bins conv = {1}; bins pool = {2}; bins fc = {3}; bins argmax = {4};
                            bins illegal = {[5:15]}; }
    k:   coverpoint s_k   { bins k[] = {[1:7]}; bins bad = {0, [8:15]}; }
    pad: coverpoint s_pad { bins p[] = {[0:3]}; bins big = {[4:7]}; bins bad = {[8:15]}; }
    relu: coverpoint s_relu;
    conv_k_x_pad: cross op, k, pad {
      ignore_bins not_conv = !binsof(op.conv);
      ignore_bins bad = binsof(k.bad) || binsof(pad.bad);
    }
    op_x_relu: cross op, relu { ignore_bins no_act = !binsof(op.conv) && !binsof(op.fc); }
  endgroup

  covergroup cg_run;
    outcome: coverpoint s_end    { bins done = {0}; bins err[] = {[1:5]}; }
    layers:  coverpoint s_layers { bins one = {1}; bins few = {[2:7]}; bins many = {[8:16]}; }
  endgroup

  covergroup cg_dma;
    dir:     coverpoint s_dir;
    sel:     coverpoint s_sel;
    outcome: coverpoint s_dma_end { bins done = {0}; bins bad_param = {2}; bins bus_err = {3}; }
    dir_x_sel: cross dir, sel;
  endgroup

  covergroup cg_burst;
    dir:   coverpoint s_write;
    beats: coverpoint s_beats { bins single = {1}; bins mid = {[2:15]}; bins full = {16}; }
    at_4k: coverpoint s_4k_end;
    dir_x_beats: cross dir, beats;
    dir_x_4k:    cross dir, at_4k;
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    axil_imp = new("axil_imp", this);
    axi_imp  = new("axi_imp", this);
    cg_axil  = new();
    cg_layer = new();
    cg_run   = new();
    cg_dma   = new();
    cg_burst = new();
  endfunction

  function void write_axil(axil_item t);
    bit [19:0] a = t.addr[19:0];
    s_write      = t.write;
    s_slverr     = (t.resp != 2'b00);
    s_misaligned = (t.addr[1:0] != 2'b00);
    if (a < 20'h00100 || (a >= 20'h00500 && a < 20'h01000)) s_window = 0;
    else if (a < 20'h00500)                                  s_window = 1;
    else if (a >= 20'h10000 && a < 20'h18000)                s_window = 2;
    else if (a >= 20'h20000 && a < 20'h40000)                s_window = 3;
    else                                                     s_window = 4;
    if (s_misaligned) s_window = 4;
    cg_axil.sample();
    if (s_slverr || t.addr[19:0] >= 20'h01000) return;

    // An accepted START: sample the program from the register model's mirror,
    // which the predictor has updated with every descriptor write.
    if (t.write && t.addr[11:0] == 12'h004 && t.strb[0] && t.data[0]) begin
      int unsigned cnt = int'(regs.layer_cnt.get_mirrored_value());
      s_layers = cnt;
      for (int i = 0; i < cnt && i < 16; i++) begin
        s_op   = int'(regs.desc[i].w0.op.get_mirrored_value());
        s_k    = int'(regs.desc[i].w0.k.get_mirrored_value());
        s_pad  = int'(regs.desc[i].w0.pad.get_mirrored_value());
        s_relu = (regs.desc[i].w9.act_min.get_mirrored_value() == regs.desc[i].w9.z_out.get_mirrored_value());
        cg_layer.sample();
      end
    end
    // The first STATUS read that shows the end of a run.
    if (!t.write && t.addr[11:0] == 12'h008 && t.data[2:1] != 2'b00 && !t.data[0]) begin
      s_end = t.data[2] ? int'(t.data[23:16]) : 0;
      cg_run.sample();
    end
    if (!t.write && t.addr[11:0] == 12'h0D0 && t.data[2:1] != 2'b00 && !t.data[0]) begin
      s_dir     = regs.dma_ctrl.dir.get_mirrored_value();
      s_sel     = regs.dma_ctrl.dst_sel.get_mirrored_value();
      s_dma_end = t.data[2] ? int'(t.data[15:8]) : 0;
      cg_dma.sample();
    end
  endfunction

  function void write_axi(axi_burst_item t);
    bit [12:0] end_addr = 13'(t.addr[11:0]) + 13'(8 * (int'(t.len) + 1));
    s_write  = t.write;
    s_beats  = int'(t.len) + 1;
    s_4k_end = (end_addr == 13'h1000);
    cg_burst.sample();
  endfunction

  function void report_phase(uvm_phase phase);
    `uvm_info("COV", $sformatf("axil %.1f%%  layer %.1f%%  run %.1f%%  dma %.1f%%  burst %.1f%%",
              cg_axil.get_inst_coverage(), cg_layer.get_inst_coverage(), cg_run.get_inst_coverage(),
              cg_dma.get_inst_coverage(), cg_burst.get_inst_coverage()), UVM_LOW)
  endfunction
endclass
