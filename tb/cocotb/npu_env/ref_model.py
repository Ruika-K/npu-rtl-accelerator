"""Register-level reference model of npu_top, driven by observed bus traffic.

The model keeps shadows of the registers, the descriptor table and both on-chip
buffers, updated from every completed AXI4-Lite write. An accepted START runs the
whole program on the shadows at once with the golden model (model/npu_ref.py); an
accepted DMA_CTRL start moves the bytes between the shadows and the AXI4 memory
model. Each read is answered with what the hardware must return.

Completion is not visible on the bus, so after a START or a DMA start the model is
"maybe busy": a memory access may then be rejected (SLVERR, which sets
HOST_CONFLICT) or succeed, and a STATUS / DMA_STATUS read with BUSY = 0, or a
successful memory access, tells the model the operation has finished.
"""
import npu_ref as ref

from . import regmap as rm

ERR_NONE, ERR_BAD_PARAM, ERR_BAD_LAYER_CNT = ref.ERR_NONE, ref.ERR_BAD_PARAM, ref.ERR_BAD_LAYER_CNT

UNKNOWN = object()      # a read value the model does not predict (performance counters)


def merge(old, data, strb):
    out = old
    for b in range(4):
        if strb >> b & 1:
            out = (out & ~(0xFF << 8 * b)) | (data & (0xFF << 8 * b))
    return out & 0xFFFF_FFFF


class NpuModel:
    def __init__(self, ext_mem):
        self.ext = ext_mem
        self.abuf = bytearray(ref.ABUF_BYTES)
        self.wbuf = bytearray(ref.WBUF_BYTES)
        self.desc = [[0] * rm.DESC_USED_WORDS for _ in range(rm.NUM_DESC)]
        self.irq_en = 0
        self.layer_cnt = 0
        self.done = self.err = self.host_conflict = 0
        self.err_code = 0
        self.cur_layer = 0
        self.result_valid, self.result_idx = 0, 0
        self.logits = [0] * 16
        self.logits_n = 0                 # logits the last ARGMAX read (only those are compared)
        self.run_pending = None           # final register state of a run that may still be going
        self.dma = {"dir": 0, "sel": 0, "src": 0, "dst": 0, "len": 0, "done": 0, "err": 0, "code": 0}
        self.dma_pending = None           # (final state, store check) of a DMA that may still be going
        self.store_checks = []            # (ext address, expected bytes) to compare once a store finished
        self.runs = []                    # predicted runs, for coverage: list of (descs, outcome)
        self.dmas = []                    # predicted DMA transfers, for coverage

    # ------------------------------------------------------------------ decode
    @staticmethod
    def target(addr):
        if addr & 3:
            return None
        if addr < rm.REGS_BYTES:
            return "reg"
        if rm.ABUF_BASE <= addr < rm.ABUF_BASE + ref.ABUF_BYTES:
            return "abuf"
        if rm.WBUF_BASE <= addr < rm.WBUF_BASE + ref.WBUF_BYTES:
            return "wbuf"
        return None

    @property
    def maybe_busy(self):
        return self.run_pending is not None or self.dma_pending is not None

    # ------------------------------------------------------------ completion
    def finish_run(self):
        if self.run_pending is not None:
            for k, v in self.run_pending.items():
                setattr(self, k, v)
            self.run_pending = None

    def finish_dma(self):
        if self.dma_pending is not None:
            final, check = self.dma_pending
            self.dma.update(final)
            if check is not None:
                self.store_checks.append(check)
            self.dma_pending = None

    def finish_all(self):
        self.finish_run()
        self.finish_dma()

    # ------------------------------------------------------------------ runs
    def _start_run(self):
        self.done = self.err = 0
        self.err_code = 0
        self.result_valid = 0
        final = {"done": 0, "err": 0, "err_code": 0}
        descs = []
        cnt = self.layer_cnt
        if cnt == 0 or cnt > rm.NUM_DESC:
            final.update(err=1, err_code=ERR_BAD_LAYER_CNT)
            self.runs.append(([], ("err", ERR_BAD_LAYER_CNT, None)))
            self.run_pending = final
            return
        outcome = ("ok", ERR_NONE, None)
        for i in range(cnt):
            d = ref.desc_unpack(self.desc[i])
            descs.append(d)
            code = ref.desc_check(d)
            if code != ERR_NONE:
                final.update(err=1, err_code=code, cur_layer=i)
                outcome = ("err", code, i)
                break
            res = ref.run_program([d], self.abuf, self.wbuf)
            if d["op"] == ref.OP_ARGMAX:
                final.update(result_valid=1, result_idx=res["result"])
                self.result_valid = 0
                logits = res["logits"] + [0] * (16 - len(res["logits"]))
                final.update(logits=logits, logits_n=len(res["logits"]))
        else:
            final.update(done=1, cur_layer=cnt - 1)
        self.runs.append((descs, outcome))
        self.run_pending = final

    # ------------------------------------------------------------------- DMA
    def _start_dma(self):
        d = self.dma
        d["done"] = d["err"] = 0
        d["code"] = 0
        src, dst, n, store, sel = d["src"], d["dst"], d["len"], d["dir"], d["sel"]
        buf = self.wbuf if sel else self.abuf
        ok = (src % 8 == 0 and dst % 8 == 0 and n % 8 == 0 and n != 0 and dst + n <= len(buf))
        self.dmas.append({"store": store, "sel": sel, "src": src, "len": n, "ok": ok})
        if not ok:
            self.dma_pending = ({"err": 1, "code": ERR_BAD_PARAM}, None)
            return
        check = None
        if store:
            check = (src, bytes(buf[dst:dst + n]))
        else:
            buf[dst:dst + n] = self.ext.read_bytes(src, n)
        self.dma_pending = ({"done": 1}, check)

    # ----------------------------------------------------------------- writes
    def write(self, addr, data, strb):
        """Apply a write; return the expected response, or None if OKAY and SLVERR are both legal."""
        tgt = self.target(addr)
        if tgt is None:
            return rm.SLVERR
        if tgt in ("abuf", "wbuf"):
            return "mem"                      # resolved by the scoreboard with mem_write()
        off = addr & 0xFFF
        if off == rm.CTRL and strb & 1:
            self.irq_en = (data >> 1) & 1
            if data & 1 and not self.maybe_busy:
                self._start_run()
        elif off == rm.STATUS and strb & 1:
            if data & 2:
                self.done = 0
            if data & 4:
                self.err = 0
            if data & 8:
                self.host_conflict = 0
        elif off == rm.LAYER_CNT and strb & 1:
            self.layer_cnt = data & 0x1F
        elif off == rm.DMA_CTRL and strb & 1:
            self.dma["dir"] = (data >> 1) & 1
            self.dma["sel"] = (data >> 2) & 1
            if data & 1 and not self.maybe_busy:
                self._start_dma()
        elif off == rm.DMA_SRC:
            self.dma["src"] = merge(self.dma["src"], data, strb)
        elif off == rm.DMA_DST:
            self.dma["dst"] = merge(self.dma["dst"], data, strb) & 0xFFFFF
        elif off == rm.DMA_LEN:
            self.dma["len"] = merge(self.dma["len"], data, strb) & 0xFFFFF
        elif off == rm.DMA_STATUS and strb & 1:
            if data & 2:
                self.dma["done"] = 0
            if data & 4:
                self.dma["err"] = 0
        elif rm.DESC <= off < rm.DESC + 64 * rm.NUM_DESC:
            idx, word = (off - rm.DESC) // 64, ((off - rm.DESC) % 64) // 4
            if word < rm.DESC_USED_WORDS:
                mask = rm.DESC_WORD_MASK.get(word, 0xFFFF_FFFF)
                self.desc[idx][word] = merge(self.desc[idx][word], data & mask, strb)
        return rm.OKAY

    def mem_write(self, addr, data, strb):
        buf, off = self._buf(addr)
        for b in range(4):
            if strb >> b & 1:
                buf[off + b] = (data >> 8 * b) & 0xFF

    def mem_read(self, addr):
        buf, off = self._buf(addr)
        return int.from_bytes(buf[off:off + 4], "little")

    def _buf(self, addr):
        if addr >= rm.WBUF_BASE:
            return self.wbuf, addr - rm.WBUF_BASE
        return self.abuf, addr - rm.ABUF_BASE

    # ------------------------------------------------------------------ reads
    def read(self, addr):
        """Expected (data, resp) of a register read; data may be UNKNOWN. Memory reads return "mem"."""
        tgt = self.target(addr)
        if tgt is None:
            return 0, rm.SLVERR
        if tgt in ("abuf", "wbuf"):
            return "mem", None
        off = addr & 0xFFF
        if off == rm.ID:
            return rm.NPU_ID, rm.OKAY
        if off == rm.CTRL:
            return self.irq_en << 1, rm.OKAY
        if off == rm.STATUS:
            return None, rm.OKAY              # compared by check_status()
        if off == rm.LAYER_CNT:
            return self.layer_cnt, rm.OKAY
        if off == rm.RESULT:
            if self.run_pending is not None:
                return UNKNOWN, rm.OKAY
            return (self.result_valid << 8) | self.result_idx, rm.OKAY
        if rm.LOGITS <= off < rm.LOGITS + 64:
            i = (off - rm.LOGITS) // 4
            if self.run_pending is not None or i >= self.logits_n:
                return UNKNOWN, rm.OKAY
            return self.logits[i] & 0xFFFF_FFFF, rm.OKAY
        if off in (rm.PERF_TOTAL, rm.PERF_ARRAY_BUSY) or rm.PERF_LAYER <= off < rm.PERF_LAYER + 64:
            return UNKNOWN, rm.OKAY
        if off == rm.DMA_CTRL:
            return (self.dma["sel"] << 2) | (self.dma["dir"] << 1), rm.OKAY
        if off == rm.DMA_SRC:
            return self.dma["src"], rm.OKAY
        if off == rm.DMA_DST:
            return self.dma["dst"], rm.OKAY
        if off == rm.DMA_LEN:
            return self.dma["len"], rm.OKAY
        if off == rm.DMA_STATUS:
            return None, rm.OKAY              # compared by check_dma_status()
        if rm.DESC <= off < rm.DESC + 64 * rm.NUM_DESC:
            idx, word = (off - rm.DESC) // 64, ((off - rm.DESC) % 64) // 4
            return (self.desc[idx][word] if word < rm.DESC_USED_WORDS else 0), rm.OKAY
        return 0, rm.OKAY

    def status_word(self):
        return ((self.err_code << 16) | (self.cur_layer << 8) | (self.host_conflict << 3) | (self.err << 2)
                | (self.done << 1))

    def dma_status_word(self):
        return (self.dma["code"] << 8) | (self.dma["err"] << 2) | (self.dma["done"] << 1)
