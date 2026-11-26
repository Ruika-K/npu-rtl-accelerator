"""AXI4 slave memory for the DMA master port of npu_top.

A sparse byte memory; a byte never written reads as a fixed function of its
address, so a load of any region has known, non-trivial content. One burst at a
time per direction (the DMA never has two outstanding), random stalls on every
channel, and counters the coverage model reads.
"""
import random

from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


def default_byte(addr):
    return (addr * 131 + (addr >> 8) * 7 + 0x5A) & 0xFF


class AxiMem:
    def __init__(self, dut, stall=0.3, seed=None):
        self.dut = dut
        self.mem = {}
        self.stall = stall
        self.rng = random.Random(seed)
        self.bursts = []          # (is_write, addr, beats) of every accepted burst

    def read_byte(self, addr):
        return self.mem.get(addr, default_byte(addr))

    def read_bytes(self, addr, n):
        return bytes(self.read_byte(addr + i) for i in range(n))

    def write_bytes(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b

    def _go(self):
        return self.rng.random() >= self.stall

    def start(self, start_soon):
        d = self.dut
        for s in (d.m_axi_arready_i, d.m_axi_rvalid_i, d.m_axi_rlast_i, d.m_axi_rresp_i, d.m_axi_rdata_i,
                  d.m_axi_awready_i, d.m_axi_wready_i, d.m_axi_bvalid_i, d.m_axi_bresp_i):
            s.value = 0
        start_soon(self._reads())
        start_soon(self._writes())

    async def _reads(self):
        d = self.dut
        while True:
            if not d.m_axi_arvalid_o.value:
                await RisingEdge(d.m_axi_arvalid_o)
            # address
            while True:
                await FallingEdge(d.clk_i)
                d.m_axi_arready_i.value = 1 if self._go() else 0
                await ReadOnly()
                if d.m_axi_arvalid_o.value and d.m_axi_arready_i.value:
                    addr, beats = int(d.m_axi_araddr_o.value), int(d.m_axi_arlen_o.value) + 1
                    break
            self.bursts.append((False, addr, beats))
            await FallingEdge(d.clk_i)
            d.m_axi_arready_i.value = 0
            # data
            for beat in range(beats):
                while not self._go():
                    await FallingEdge(d.clk_i)
                a = addr + 8 * beat
                d.m_axi_rdata_i.value = int.from_bytes(self.read_bytes(a, 8), "little")
                d.m_axi_rresp_i.value = 0
                d.m_axi_rlast_i.value = 1 if beat == beats - 1 else 0
                d.m_axi_rvalid_i.value = 1
                while True:
                    await ReadOnly()
                    taken = bool(d.m_axi_rready_o.value)
                    await FallingEdge(d.clk_i)
                    if taken:
                        break
                d.m_axi_rvalid_i.value = 0
                d.m_axi_rlast_i.value = 0

    async def _writes(self):
        d = self.dut
        while True:
            if not d.m_axi_awvalid_o.value:
                await RisingEdge(d.m_axi_awvalid_o)
            while True:
                await FallingEdge(d.clk_i)
                d.m_axi_awready_i.value = 1 if self._go() else 0
                await ReadOnly()
                if d.m_axi_awvalid_o.value and d.m_axi_awready_i.value:
                    addr, beats = int(d.m_axi_awaddr_o.value), int(d.m_axi_awlen_o.value) + 1
                    break
            self.bursts.append((True, addr, beats))
            await FallingEdge(d.clk_i)
            d.m_axi_awready_i.value = 0
            beat = 0
            while beat < beats:
                d.m_axi_wready_i.value = 1 if self._go() else 0
                await ReadOnly()
                if d.m_axi_wvalid_o.value and d.m_axi_wready_i.value:
                    data, strb = int(d.m_axi_wdata_o.value), int(d.m_axi_wstrb_o.value)
                    for i in range(8):
                        if strb >> i & 1:
                            self.mem[addr + 8 * beat + i] = (data >> (8 * i)) & 0xFF
                    beat += 1
                await FallingEdge(d.clk_i)
            d.m_axi_wready_i.value = 0
            while not self._go():
                await FallingEdge(d.clk_i)
            d.m_axi_bresp_i.value = 0
            d.m_axi_bvalid_i.value = 1
            while True:
                await ReadOnly()
                taken = bool(d.m_axi_bready_o.value)
                await FallingEdge(d.clk_i)
                if taken:
                    break
            d.m_axi_bvalid_i.value = 0
