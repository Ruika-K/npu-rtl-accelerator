"""Host-level sequences on the AXI4-Lite agent.

HostSeq is the base: single register / memory accesses with randomized bus
behaviour, and the host procedures of docs/programming_model.md built on them
(load a buffer over AXI4-Lite or by DMA, write a program, run it, read results).
The test sequences below compose those procedures with random layers.
"""
import random

import npu_ref as ref
from cocotb.triggers import ClockCycles
from pyuvm import ConfigDB, uvm_sequence

from . import layers as L
from . import regmap as rm
from .axil import AW_FIRST, TOGETHER, W_FIRST, AxilItem

WINDOW = {"abuf": rm.ABUF_BASE, "wbuf": rm.WBUF_BASE}


class HostSeq(uvm_sequence):
    def __init__(self, name="host_seq", rng=None):
        super().__init__(name)
        self.rng = rng or random.Random(1)
        self.dut = ConfigDB().get(None, "", "DUT")
        self.ext = ConfigDB().get(None, "", "EXT_MEM")
        self.ext_next = 0x0010_0000      # next free external region for DMA staging

    # ------------------------------------------------------------ accesses
    def _knobs(self, it):
        r = self.rng.random()
        it.order = TOGETHER if r < 0.6 else AW_FIRST if r < 0.8 else W_FIRST
        it.resp_delay = 0 if self.rng.random() < 0.8 else self.rng.randint(1, 8)

    async def wr(self, addr, data, strb=0xF):
        it = AxilItem("wr", True, addr, data & 0xFFFF_FFFF, strb)
        self._knobs(it)
        await self.start_item(it)
        await self.finish_item(it)
        return it.resp

    async def rd(self, addr):
        it = AxilItem("rd", False, addr)
        self._knobs(it)
        await self.start_item(it)
        await self.finish_item(it)
        return it.data, it.resp

    # ----------------------------------------------------- memory transfers
    async def load_axil(self, buf, off, data):
        """Write bytes at a byte offset of a buffer with 32-bit writes and byte strobes."""
        base = WINDOW[buf]
        a, end = off & ~3, off + len(data)
        while a < end:
            word, strb = 0, 0
            for b in range(4):
                if off <= a + b < end:
                    word |= data[a + b - off] << 8 * b
                    strb |= 1 << b
            await self.wr(base + a, word, strb)
            a += 4

    def stage(self, data, lead, size=None):
        """Reserve an 8-aligned external region of size bytes (default: the data) and place the data
        `lead` bytes into it. Every other region starts just below a 4 KB boundary."""
        size = lead + len(data) if size is None else size
        src = self.ext_next
        if self.rng.random() < 0.5:
            src = ((src | 0xFFF) - 0x2F) & ~7
        self.ext_next = (src + size + 0x1FF) & ~0xFF
        self.ext.write_bytes(src + lead, data)
        return src

    async def dma(self, store, buf, ext, off, n, wait=True):
        await self.wr(rm.DMA_SRC, ext)
        await self.wr(rm.DMA_DST, off)
        await self.wr(rm.DMA_LEN, n)
        await self.wr(rm.DMA_CTRL, (int(buf == "wbuf") << 2) | (int(store) << 1) | 1)
        if wait:
            return await self.wait_dma()

    async def wait_dma(self):
        while True:
            v, _ = await self.rd(rm.DMA_STATUS)
            if not v & 1:
                await self.wr(rm.DMA_STATUS, 0x6)
                return v
            await ClockCycles(self.dut.clk_i, 32)

    async def load(self, buf, off, data):
        """Load over AXI4-Lite or by DMA, at random."""
        if self.rng.random() < 0.5 or not data:
            await self.load_axil(buf, off, data)
            return
        lead = off & 7
        n = (lead + len(data) + 7) & ~7
        src = self.stage(data, lead)
        await self.dma(False, buf, src, off - lead, n)

    async def read_back(self, off, n):
        """Read an act_buffer region over AXI4-Lite or store it by DMA; the scoreboard compares."""
        if self.rng.random() < 0.5:
            for a in range(off & ~3, off + n, 4):
                await self.rd(rm.ABUF_BASE + a)
        else:
            lo, hi = off & ~7, (off + n + 7) & ~7
            await self.dma(True, "abuf", self.stage(b"", 0, hi - lo), lo, hi - lo)

    # --------------------------------------------------------------- runs
    async def write_desc(self, idx, d):
        for i, w in enumerate(ref.desc_pack(d)):
            await self.wr(rm.desc_addr(idx, i), w)

    async def run(self, n_layers):
        await self.wr(rm.LAYER_CNT, n_layers)
        await self.wr(rm.CTRL, 0x1)
        return await self.wait_run()

    async def wait_run(self):
        while True:
            v, _ = await self.rd(rm.STATUS)
            if not v & 1:
                await self.wr(rm.STATUS, 0x6)
                return v
            await ClockCycles(self.dut.clk_i, 64)

    async def results(self):
        await self.rd(rm.RESULT)
        for i in range(16):
            await self.rd(rm.LOGITS + 4 * i)

    async def run_layers(self, lays):
        for lay in lays:
            for buf, off, data in lay["loads"]:
                await self.load(buf, off, data)
        for i, lay in enumerate(lays):
            await self.write_desc(i, lay["desc"])
        status = await self.run(len(lays))
        for lay in lays:
            if lay["out"]:
                await self.read_back(*lay["out"])
        await self.results()
        return status


class RandomLayerSeq(HostSeq):
    """Single random layers, each loaded, run and read back."""

    def __init__(self, name="random_layers", rng=None, n=20):
        super().__init__(name, rng)
        self.n = n

    async def body(self):
        await self.rd(rm.ID)
        for lay in L.corner_layers(self.rng):
            await self.run_layers([lay])
        for _ in range(self.n):
            await self.run_layers([L.random_layer(self.rng)])


class ProgramSeq(HostSeq):
    """Multi-layer programs whose layers feed each other, up to all 16 descriptors."""

    def __init__(self, name="programs", rng=None, lengths=(3, 5, 8, 12, 16)):
        super().__init__(name, rng)
        self.lengths = lengths

    async def body(self):
        for n in self.lengths:
            lays = L.chain(self.rng, n)
            while len(lays) < n:                       # pad with NOPs to the requested length
                lays.append({"desc": ref.make_desc(ref.OP_NOP), "loads": [], "out": None, "shape": None})
            await self.run_layers(lays)


def _break(rng, d):
    """Make one field of a valid descriptor invalid; the model decides which error that is."""
    d = dict(d)
    op = d["op"]
    choices = ["op", "addr_out"]
    if op in (ref.OP_CONV2D, ref.OP_FC):
        choices += ["k", "pad", "shift", "mult", "act", "cin", "wbase", "ktotal"]
        if op == ref.OP_CONV2D:
            choices += ["stride", "outdim"]
    elif op == ref.OP_MAXPOOL2X2:
        choices += ["pool_small", "cin"]
    elif op == ref.OP_ARGMAX:
        choices = ["op", "argmax_c", "argmax_addr"]
    what = rng.choice(choices)
    if what == "op":
        d["op"] = rng.randint(5, 15)
    elif what == "addr_out":
        d["out_base"] = ref.ABUF_BYTES - rng.randint(0, 4)
    elif what == "k":
        d["k"] = rng.choice([0, 8, 15])
    elif what == "pad":
        d["pad"] = rng.randint(8, 15)
    elif what == "shift":
        d["rq_shift"] = rng.choice([0, 63])
    elif what == "mult":
        d["rq_mult"] = rng.choice([0, ref.RQ_MULT_MIN - 1, 0x8000_0000])
    elif what == "act":
        d["act_min"], d["act_max"] = 10, -10
    elif what == "cin":
        d["c_in"] = 0
    elif what == "wbase":
        d["w_base"] |= rng.randint(1, 7)
    elif what == "ktotal":
        d["c_in"] = ref.MAX_K_TOTAL + 1
    elif what == "stride":
        d["stride_m1"] = rng.randint(1, 3)
    elif what == "outdim":
        d["k"], d["pad"], d["in_h"] = 7, 0, rng.randint(1, 6)
    elif what == "pool_small":
        d["in_h"] = 1
    elif what == "argmax_c":
        d["c_in"] = rng.choice([0, 17])
    elif what == "argmax_addr":
        d["in_base"] = ref.ABUF_BYTES - 1
    return d


class ErrorSeq(HostSeq):
    """Invalid descriptors at random positions, bad layer counts, and rejected bus accesses."""

    def __init__(self, name="errors", rng=None, n=24):
        super().__init__(name, rng)
        self.n = n

    async def body(self):
        for i in range(self.n):
            lays = [L.random_layer(self.rng) for _ in range(self.rng.randint(1, 4))]
            bad = self.rng.randrange(len(lays))
            for lay in lays:
                for buf, off, data in lay["loads"]:
                    await self.load(buf, off, data)
            for j, lay in enumerate(lays):
                await self.write_desc(j, _break(self.rng, lay["desc"]) if j == bad else lay["desc"])
            await self.run(len(lays))
            await self.rd(rm.RESULT)
        for cnt in (0, 17, 31):
            await self.run(cnt)
        # bus errors: misaligned, outside every window, memory access during a run
        await self.wr(rm.CTRL + 2, 0)
        await self.rd(rm.STATUS + 1)
        await self.rd(rm.ABUF_BASE + 1)
        await self.wr(0x80000, 0x1234)
        await self.rd(0x50000)
        lay = L.mac_layer(self.rng)
        lay["desc"].update(c_out=20)
        for buf, off, data in lay["loads"]:
            await self.load_axil(buf, off, data)
        await self.write_desc(0, lay["desc"])
        await self.wr(rm.LAYER_CNT, 1)
        await self.wr(rm.CTRL, 0x1)
        await self.rd(rm.ABUF_BASE + 0x100)          # rejected: the run is still going
        await self.wr(rm.ABUF_BASE + 0x100, 0)       # all four rejected
        await self.rd(rm.WBUF_BASE + 0x100)
        await self.wr(rm.WBUF_BASE + 0x100, 0)
        await self.wait_run()
        await self.rd(rm.STATUS)                     # HOST_CONFLICT is set
        await self.wr(rm.STATUS, 0x8)
        await self.rd(rm.STATUS)


class DmaSeq(HostSeq):
    """DMA transfers of random lengths in both directions and to both buffers, and rejected parameters."""

    def __init__(self, name="dma", rng=None, n=16):
        super().__init__(name, rng)
        self.n = n

    async def body(self):
        for _ in range(self.n):
            buf = self.rng.choice(["abuf", "wbuf"])
            n = self.rng.choice([8, self.rng.randrange(16, 129, 8), self.rng.randrange(136, 1025, 8),
                                 self.rng.randrange(1032, 2049, 8)])
            off = self.rng.randrange(0, 0x4000, 8)
            src = self.stage(bytes(self.rng.randint(0, 255) for _ in range(n)), 0)
            await self.dma(False, buf, src, off, n)
            await self.dma(True, buf, self.stage(b"", 0, n), off, n)
            for a in range(off, off + min(n, 32), 4):
                await self.rd(WINDOW[buf] + a)
        for src, off, n in ((0x1004, 0, 64), (0x1000, 4, 64), (0x1000, 0, 12), (0x1000, 0, 0),
                            (0x1000, ref.ABUF_BYTES - 8, 64)):
            await self.dma(False, "abuf", src, off, n)
