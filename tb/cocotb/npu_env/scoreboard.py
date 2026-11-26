"""Scoreboard: compares every monitored AXI4-Lite transaction with NpuModel and
checks DMA stores in the AXI4 memory model once they have finished."""
import cocotb.utils
from pyuvm import ConfigDB, uvm_subscriber

from . import coverage
from . import regmap as rm
from .ref_model import UNKNOWN, NpuModel


class NpuScoreboard(uvm_subscriber):
    def build_phase(self):
        self.ext = ConfigDB().get(self, "", "EXT_MEM")
        self.model = NpuModel(self.ext)
        self.errors = 0
        self.checked = 0
        self.n_runs = self.n_dmas = 0
        self.history = []                 # recent transactions, printed with the first error

    def fail(self, msg):
        self.errors += 1
        if self.errors <= 20:
            self.logger.error(msg)
        if self.errors == 1:
            for h in self.history[-16:]:
                self.logger.error("  history: " + h)

    def expect(self, cond, msg):
        self.checked += 1
        if not cond:
            self.fail(msg)

    # ----------------------------------------------------------------------
    def write(self, item):
        self.history.append(f"{cocotb.utils.get_sim_time('ns'):.0f} ns " + item.convert2string())
        del self.history[:-32]
        coverage.sample_axil(item, rm.window(item.addr))
        if item.write:
            self._on_write(item)
        else:
            self._on_read(item)
        self._sample_new_work()
        self._drain_store_checks()

    def _mem_access_allowed(self, item):
        """Memory window access: True if the data path must be compared."""
        m = self.model
        if m.maybe_busy and item.resp == rm.SLVERR:
            m.host_conflict = 1
            return False
        if m.maybe_busy and item.resp == rm.OKAY:
            m.finish_all()                 # an accepted access proves the run / DMA is over
        self.expect(item.resp == rm.OKAY, f"memory access while idle rejected: {item.convert2string()}")
        return item.resp == rm.OKAY

    def _on_write(self, item):
        exp = self.model.write(item.addr, item.data, item.strb)
        if exp == "mem":
            if self._mem_access_allowed(item):
                self.model.mem_write(item.addr, item.data, item.strb)
            return
        self.expect(item.resp == exp, f"write response {item.resp} expected {exp}: {item.convert2string()}")

    def _on_read(self, item):
        m = self.model
        off = item.addr & 0xFFF
        exp, resp = m.read(item.addr)
        if exp == "mem":
            if self._mem_access_allowed(item):
                want = m.mem_read(item.addr)
                self.expect(item.data == want, f"memory read {item.addr:#07x}: {item.data:#010x}, expected {want:#010x}")
            return
        self.expect(item.resp == resp, f"read response {item.resp} expected {resp}: {item.convert2string()}")
        if resp != rm.OKAY:
            return
        if m.target(item.addr) == "reg" and off == rm.STATUS:
            self._check_status(item.data)
        elif m.target(item.addr) == "reg" and off == rm.DMA_STATUS:
            self._check_dma_status(item.data)
        elif exp is not UNKNOWN:
            self.expect(item.data == exp, f"register {off:#05x}: {item.data:#010x}, expected {exp:#010x}")

    def _check_status(self, v):
        m = self.model
        if m.run_pending is not None and v & 1:
            # still running: DONE / ERR / ERR_CODE were cleared by START
            self.expect((v >> 1) & 3 == 0 and (v >> 16) & 0xFF == 0 and (v >> 3) & 1 == m.host_conflict,
                        f"STATUS while busy: {v:#010x}")
            return
        m.finish_run()
        self.expect(v == m.status_word(), f"STATUS {v:#010x}, expected {m.status_word():#010x}")

    def _check_dma_status(self, v):
        m = self.model
        if m.dma_pending is not None and v & 1:
            self.expect((v >> 1) & 3 == 0, f"DMA_STATUS while busy: {v:#010x}")
            return
        m.finish_dma()
        self.expect(v == m.dma_status_word(), f"DMA_STATUS {v:#010x}, expected {m.dma_status_word():#010x}")

    def _drain_store_checks(self):
        while self.model.store_checks:
            addr, want = self.model.store_checks.pop(0)
            got = self.ext.read_bytes(addr, len(want))
            bad = [i for i in range(len(want)) if got[i] != want[i]]
            self.expect(not bad, f"DMA store to {addr:#x}: {len(bad)} of {len(want)} bytes differ, first at +{bad[0] if bad else 0}")

    def _sample_new_work(self):
        m = self.model
        while self.n_runs < len(m.runs):
            descs, outcome = m.runs[self.n_runs]
            coverage.sample_run(descs, outcome)
            self.n_runs += 1
        while self.n_dmas < len(m.dmas):
            coverage.sample_dma(m.dmas[self.n_dmas])
            self.n_dmas += 1

    def check_phase(self):
        m = self.model
        if m.maybe_busy:
            self.fail("test ended with a run or a DMA whose completion was never observed")
        self.logger.info(f"scoreboard: {self.checked} checks, {self.n_runs} runs, {self.n_dmas} DMA transfers, "
                         f"{self.errors} errors")
        assert self.errors == 0, f"{self.errors} scoreboard errors"
