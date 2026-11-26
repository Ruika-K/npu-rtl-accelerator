// AXI4-Lite agent: item, driver, monitor, agent. One transaction at a time per
// direction, which is what npu_top's slave accepts (see axil_slave_sva).

class axil_item extends uvm_sequence_item;
  rand bit          write;
  rand bit [31:0]   addr;
  rand bit [31:0]   data;           // write data, or read data once the item completes
  rand bit [3:0]    strb;
  bit [1:0]         resp;           // filled in by the driver / monitor

  // Handshake timing, in cycles, drawn when the item is created (writes may
  // present AW before W, W before AW, or both together). It is not left to
  // randomize(): the items of register and memory accesses have fixed address
  // and data, and a constraint solve per transaction costs more than the
  // transaction itself on simulators that call an external solver.
  int unsigned delay;               // before the first valid, 0..3
  int unsigned aw_delay;            // 0..3
  int unsigned w_delay;             // 0..3
  int unsigned resp_delay;          // before BREADY / RREADY: 0 (60%), 1..3 (30%), 4..12 (10%)

  constraint c_strb { !write -> strb == 4'hF; }

  `uvm_object_utils(axil_item)

  function new(string name = "axil_item");
    int unsigned r;
    super.new(name);
    r = $urandom_range(9);
    delay      = $urandom_range(3);
    aw_delay   = $urandom_range(3);
    w_delay    = $urandom_range(3);
    resp_delay = (r < 6) ? 0 : (r < 9) ? $urandom_range(3, 1) : $urandom_range(12, 4);
  endfunction

  // A fixed access; the handshake timing stays as drawn.
  function void set(bit is_write, bit [31:0] a, bit [31:0] d = '0, bit [3:0] s = 4'hF);
    write = is_write; addr = a; data = is_write ? d : '0; strb = is_write ? s : 4'hF;
  endfunction

  function void do_copy(uvm_object rhs);
    axil_item t;
    super.do_copy(rhs);
    if (!$cast(t, rhs)) `uvm_fatal("AXIL/COPY", "copy from a different type")
    write = t.write; addr = t.addr; data = t.data; strb = t.strb; resp = t.resp;
    delay = t.delay; aw_delay = t.aw_delay; w_delay = t.w_delay; resp_delay = t.resp_delay;
  endfunction

  function string convert2string();
    return $sformatf("%s addr=%h data=%h strb=%h resp=%0d", write ? "WR" : "RD", addr, data, strb, resp);
  endfunction
endclass

typedef uvm_sequencer #(axil_item) axil_sequencer;

class axil_driver extends uvm_driver #(axil_item);
  `uvm_component_utils(axil_driver)

  virtual axil_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db #(virtual axil_if)::get(this, "", "vif", vif))
      `uvm_fatal("AXIL/NOVIF", "no virtual axil_if")
  endfunction

  task run_phase(uvm_phase phase);
    vif.master_idle();
    @(posedge vif.rst_n);
    @(vif.drv_cb);
    forever begin
      seq_item_port.get_next_item(req);
      repeat (req.delay) @(vif.drv_cb);
      if (req.write) do_write(req);
      else           do_read(req);
      seq_item_port.item_done();     // the RAL adapter reads data / resp from req itself
    end
  endtask

  // Each valid is raised right after a clocking event; the handshake is the
  // first later event at which the sampled ready is high.
  task automatic do_write(axil_item t);
    fork
      begin
        repeat (t.aw_delay) @(vif.drv_cb);
        vif.drv_cb.awaddr  <= t.addr;
        vif.drv_cb.awprot  <= 3'b000;
        vif.drv_cb.awvalid <= 1'b1;
        do @(vif.drv_cb); while (!vif.drv_cb.awready);
        vif.drv_cb.awvalid <= 1'b0;
      end
      begin
        repeat (t.w_delay) @(vif.drv_cb);
        vif.drv_cb.wdata  <= t.data;
        vif.drv_cb.wstrb  <= t.strb;
        vif.drv_cb.wvalid <= 1'b1;
        do @(vif.drv_cb); while (!vif.drv_cb.wready);
        vif.drv_cb.wvalid <= 1'b0;
      end
    join
    repeat (t.resp_delay) @(vif.drv_cb);
    vif.drv_cb.bready <= 1'b1;
    do @(vif.drv_cb); while (!vif.drv_cb.bvalid);
    t.resp = vif.drv_cb.bresp;
    vif.drv_cb.bready <= 1'b0;
  endtask

  task automatic do_read(axil_item t);
    vif.drv_cb.araddr  <= t.addr;
    vif.drv_cb.arprot  <= 3'b000;
    vif.drv_cb.arvalid <= 1'b1;
    do @(vif.drv_cb); while (!vif.drv_cb.arready);
    vif.drv_cb.arvalid <= 1'b0;
    repeat (t.resp_delay) @(vif.drv_cb);
    vif.drv_cb.rready <= 1'b1;
    do @(vif.drv_cb); while (!vif.drv_cb.rvalid);
    t.data = vif.drv_cb.rdata;
    t.resp = vif.drv_cb.rresp;
    vif.drv_cb.rready <= 1'b0;
  endtask
endclass

// Publishes one item per completed transaction: writes at the B handshake,
// reads at the R handshake. Works for any AW / W order.
class axil_monitor extends uvm_monitor;
  `uvm_component_utils(axil_monitor)

  virtual axil_if vif;
  uvm_analysis_port #(axil_item) ap;
  int unsigned n_writes, n_reads;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db #(virtual axil_if)::get(this, "", "vif", vif))
      `uvm_fatal("AXIL/NOVIF", "no virtual axil_if")
  endfunction

  task run_phase(uvm_phase phase);
    bit        have_aw, have_w, have_ar;
    bit [31:0] aw_addr, w_data, ar_addr;
    bit [3:0]  w_strb;
    forever begin
      @(vif.mon_cb);
      if (!vif.rst_n) begin
        have_aw = 0; have_w = 0; have_ar = 0;
        continue;
      end
      if (vif.mon_cb.awvalid && vif.mon_cb.awready) begin have_aw = 1; aw_addr = vif.mon_cb.awaddr; end
      if (vif.mon_cb.wvalid && vif.mon_cb.wready) begin
        have_w = 1; w_data = vif.mon_cb.wdata; w_strb = vif.mon_cb.wstrb;
      end
      if (vif.mon_cb.bvalid && vif.mon_cb.bready) begin
        axil_item t = axil_item::type_id::create("wr");
        if (!(have_aw && have_w)) `uvm_error("AXIL/MON", "B handshake without a complete AW and W")
        t.write = 1; t.addr = aw_addr; t.data = w_data; t.strb = w_strb; t.resp = vif.mon_cb.bresp;
        have_aw = 0; have_w = 0;
        n_writes++;
        ap.write(t);
      end
      if (vif.mon_cb.arvalid && vif.mon_cb.arready) begin have_ar = 1; ar_addr = vif.mon_cb.araddr; end
      if (vif.mon_cb.rvalid && vif.mon_cb.rready) begin
        axil_item t = axil_item::type_id::create("rd");
        if (!have_ar) `uvm_error("AXIL/MON", "R handshake without AR")
        t.write = 0; t.addr = ar_addr; t.data = vif.mon_cb.rdata; t.strb = 4'hF; t.resp = vif.mon_cb.rresp;
        have_ar = 0;
        n_reads++;
        ap.write(t);
      end
    end
  endtask

  function void report_phase(uvm_phase phase);
    `uvm_info("AXIL/MON", $sformatf("%0d writes, %0d reads observed", n_writes, n_reads), UVM_LOW)
  endfunction
endclass

class axil_agent extends uvm_agent;
  `uvm_component_utils(axil_agent)

  axil_sequencer sqr;
  axil_driver    drv;
  axil_monitor   mon;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    mon = axil_monitor::type_id::create("mon", this);
    if (get_is_active() == UVM_ACTIVE) begin
      sqr = axil_sequencer::type_id::create("sqr", this);
      drv = axil_driver::type_id::create("drv", this);
    end
  endfunction

  function void connect_phase(uvm_phase phase);
    if (get_is_active() == UVM_ACTIVE) drv.seq_item_port.connect(sqr.seq_item_export);
  endfunction
endclass

// Bridge between the register model and axil_item. Byte enables pass through,
// so partial register writes predict correctly.
class axil_reg_adapter extends uvm_reg_adapter;
  `uvm_object_utils(axil_reg_adapter)

  function new(string name = "axil_reg_adapter");
    super.new(name);
    supports_byte_enable = 1;
    provides_responses   = 0;
  endfunction

  virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
    axil_item t = axil_item::type_id::create("reg_item");
    t.set(rw.kind == UVM_WRITE, 32'(rw.addr), 32'(rw.data), 4'(rw.byte_en));
    return t;
  endfunction

  virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
    axil_item t;
    if (!$cast(t, bus_item)) `uvm_fatal("AXIL/ADAPT", "not an axil_item")
    rw.kind    = t.write ? UVM_WRITE : UVM_READ;
    rw.addr    = t.addr;
    rw.data    = t.data;
    rw.byte_en = t.write ? t.strb : 4'hF;
    rw.n_bits  = 32;
    rw.status  = (t.resp == 2'b00) ? UVM_IS_OK : UVM_NOT_OK;
  endfunction
endclass
