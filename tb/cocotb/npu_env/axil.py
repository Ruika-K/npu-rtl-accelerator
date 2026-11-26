"""AXI4-Lite agent: sequence item, driver, monitor and the agent around them.

Signals are driven right after the falling clock edge and sampled in the
read-only phase of that time step, so a handshake is recognized with exactly the
values the DUT sees at the next rising edge (the same scheme as the SystemVerilog
BFM in tb/common/axil_master_bfm.sv).
"""
import random

from cocotb.triggers import FallingEdge, First, ReadOnly, RisingEdge
from pyuvm import (ConfigDB, uvm_agent, uvm_analysis_port, uvm_driver, uvm_monitor, uvm_sequence_item,
                   uvm_sequencer)

TOGETHER, AW_FIRST, W_FIRST = "together", "aw_first", "w_first"


class AxilItem(uvm_sequence_item):
    """One AXI4-Lite transaction. The driver fills in the response fields."""

    def __init__(self, name="axil_item", write=False, addr=0, data=0, strb=0xF):
        super().__init__(name)
        self.write = write
        self.addr = addr
        self.data = data          # write data, or read data after the transaction
        self.strb = strb
        self.resp = None
        # stimulus knobs
        self.order = TOGETHER     # write: when AW and W become valid
        self.resp_delay = 0       # cycles before BREADY / RREADY
        # what the monitor saw
        self.resp_wait = 0        # cycles BVALID / RVALID waited for the master

    def convert2string(self):
        kind = "W" if self.write else "R"
        return f"{kind} {self.addr:#07x} data={self.data:#010x} strb={self.strb:x} resp={self.resp}"


class AxilDriver(uvm_driver):
    def build_phase(self):
        self.dut = ConfigDB().get(self, "", "DUT")

    def start_of_simulation_phase(self):
        d = self.dut
        for s in (d.s_axil_awvalid_i, d.s_axil_wvalid_i, d.s_axil_bready_i, d.s_axil_arvalid_i,
                  d.s_axil_rready_i, d.s_axil_awaddr_i, d.s_axil_wdata_i, d.s_axil_wstrb_i,
                  d.s_axil_araddr_i, d.s_axil_awprot_i, d.s_axil_arprot_i):
            s.value = 0

    async def run_phase(self):
        while True:
            item = await self.seq_item_port.get_next_item()
            if item.write:
                await self._write(item)
            else:
                await self._read(item)
            self.seq_item_port.item_done()

    async def _response(self, valid, ready, grab, delay):
        d = self.dut
        for _ in range(delay):
            await FallingEdge(d.clk_i)
        ready.value = 1
        while True:
            await ReadOnly()
            if valid.value:
                grab()
                await FallingEdge(d.clk_i)
                ready.value = 0
                return
            await FallingEdge(d.clk_i)

    async def _write(self, item):
        d = self.dut
        d_aw = random.randint(1, 3) if item.order == W_FIRST else 0
        d_w = random.randint(1, 3) if item.order == AW_FIRST else 0
        aw_done = w_done = False
        await FallingEdge(d.clk_i)
        while not (aw_done and w_done):
            if not aw_done and d_aw == 0:
                d.s_axil_awaddr_i.value = item.addr
                d.s_axil_awvalid_i.value = 1
            if not w_done and d_w == 0:
                d.s_axil_wdata_i.value = item.data
                d.s_axil_wstrb_i.value = item.strb
                d.s_axil_wvalid_i.value = 1
            await ReadOnly()
            aw_hs = bool(d.s_axil_awvalid_i.value) and bool(d.s_axil_awready_o.value)
            w_hs = bool(d.s_axil_wvalid_i.value) and bool(d.s_axil_wready_o.value)
            await FallingEdge(d.clk_i)
            if aw_hs:
                aw_done = True
                d.s_axil_awvalid_i.value = 0
            if w_hs:
                w_done = True
                d.s_axil_wvalid_i.value = 0
            d_aw = max(0, d_aw - 1)
            d_w = max(0, d_w - 1)

        def grab():
            item.resp = int(d.s_axil_bresp_o.value)
        await self._response(d.s_axil_bvalid_o, d.s_axil_bready_i, grab, item.resp_delay)

    async def _read(self, item):
        d = self.dut
        await FallingEdge(d.clk_i)
        d.s_axil_araddr_i.value = item.addr
        d.s_axil_arvalid_i.value = 1
        while True:
            await ReadOnly()
            hs = bool(d.s_axil_arready_o.value)
            await FallingEdge(d.clk_i)
            if hs:
                d.s_axil_arvalid_i.value = 0
                break

        def grab():
            item.data = int(d.s_axil_rdata_o.value)
            item.resp = int(d.s_axil_rresp_o.value)
        await self._response(d.s_axil_rvalid_o, d.s_axil_rready_i, grab, item.resp_delay)


class AxilMonitor(uvm_monitor):
    """Passive: rebuilds every completed transaction from the pins and publishes it."""

    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        self.dut = ConfigDB().get(self, "", "DUT")

    async def run_phase(self):
        d = self.dut
        cycle = 0
        wr = None        # write being assembled: dict
        rd = None
        while True:
            idle = wr is None and rd is None
            if idle and not (d.s_axil_awvalid_i.value or d.s_axil_wvalid_i.value or d.s_axil_arvalid_i.value):
                await First(RisingEdge(d.s_axil_awvalid_i), RisingEdge(d.s_axil_wvalid_i),
                            RisingEdge(d.s_axil_arvalid_i))
            else:
                await FallingEdge(d.clk_i)
                cycle += 1
            await ReadOnly()
            awv, wv, arv = (bool(d.s_axil_awvalid_i.value), bool(d.s_axil_wvalid_i.value),
                            bool(d.s_axil_arvalid_i.value))
            if wr is None and (awv or wv):
                wr = {"first_aw": cycle if awv else None, "first_w": cycle if wv else None,
                      "aw": None, "w": None, "wait": 0}
            if wr is not None:
                if awv and wr["first_aw"] is None:
                    wr["first_aw"] = cycle
                if wv and wr["first_w"] is None:
                    wr["first_w"] = cycle
                if awv and d.s_axil_awready_o.value:
                    wr["aw"] = int(d.s_axil_awaddr_i.value)
                if wv and d.s_axil_wready_o.value:
                    wr["w"] = (int(d.s_axil_wdata_i.value), int(d.s_axil_wstrb_i.value))
                if d.s_axil_bvalid_o.value:
                    if d.s_axil_bready_i.value:
                        it = AxilItem("mon_w", True, wr["aw"], wr["w"][0], wr["w"][1])
                        it.resp = int(d.s_axil_bresp_o.value)
                        it.resp_wait = wr["wait"]
                        fa, fw = wr["first_aw"], wr["first_w"]
                        it.order = TOGETHER if fa == fw else (AW_FIRST if fa < fw else W_FIRST)
                        self.ap.write(it)
                        wr = None
                    else:
                        wr["wait"] += 1
            if rd is None and arv:
                rd = {"ar": None, "wait": 0}
            if rd is not None:
                if arv and d.s_axil_arready_o.value:
                    rd["ar"] = int(d.s_axil_araddr_i.value)
                if d.s_axil_rvalid_o.value:
                    if d.s_axil_rready_i.value:
                        it = AxilItem("mon_r", False, rd["ar"], int(d.s_axil_rdata_o.value))
                        it.resp = int(d.s_axil_rresp_o.value)
                        it.resp_wait = rd["wait"]
                        self.ap.write(it)
                        rd = None
                    else:
                        rd["wait"] += 1


class AxilAgent(uvm_agent):
    def build_phase(self):
        self.seqr = uvm_sequencer("seqr", self)
        self.drv = AxilDriver("drv", self)
        self.mon = AxilMonitor("mon", self)

    def connect_phase(self):
        self.drv.seq_item_port.connect(self.seqr.seq_item_export)
