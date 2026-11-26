"""Functional coverage of the npu_top environment (cocotb-coverage).

Sampled by the scoreboard from what the reference model predicts and the monitor
observes, so a bin counts only for stimulus that actually reached the DUT.
closure() lists every bin that was never hit; the last test fails on it.
"""
import npu_ref as ref
from cocotb_coverage.coverage import CoverCross, CoverPoint, coverage_db

OPS = [ref.OP_NOP, ref.OP_CONV2D, ref.OP_MAXPOOL2X2, ref.OP_FC, ref.OP_ARGMAX]
ERRS = [ref.ERR_BAD_OP, ref.ERR_BAD_PARAM, ref.ERR_BAD_ADDR, ref.ERR_BAD_LAYER_CNT, ref.ERR_BAD_OUT_DIM]


def _len_bin(n):
    return "1" if n == 1 else "2-4" if n <= 4 else "5-8" if n <= 8 else "9-16"


def _shift_bin(s):
    return "1-15" if s < 16 else "16-31" if s < 32 else "32-47" if s < 48 else "48-62"


def _wait_bin(w):
    return "0" if w == 0 else "1-3" if w <= 3 else "4+"


# ----------------------------------------------------------------- programs
@CoverPoint("npu.program.length", xf=lambda descs, outcome: _len_bin(max(1, len(descs))),
            bins=["1", "2-4", "5-8", "9-16"])
@CoverPoint("npu.program.outcome", xf=lambda descs, outcome: outcome[0], bins=["ok", "err"])
def _sample_program(descs, outcome):
    pass


@CoverPoint("npu.layer.op", xf=lambda d, result: d["op"], bins=OPS)
@CoverPoint("npu.layer.result", xf=lambda d, result: result, bins=["ok", "err"])
@CoverCross("npu.layer.op_x_result", items=["npu.layer.op", "npu.layer.result"],
            ign_bins=[(ref.OP_NOP, "err")])
def _sample_layer(d, result):
    pass


@CoverPoint("npu.error.code", xf=lambda code, at_first: code, bins=ERRS)
@CoverPoint("npu.error.position", xf=lambda code, at_first: "first" if at_first else "later",
            bins=["first", "later"])
def _sample_error(code, at_first):
    pass


def _conv(d):
    fc = d["op"] == ref.OP_FC
    k = 1 if fc else d["k"]
    return fc, k, d["c_in"] * k * k


@CoverPoint("npu.conv.kernel", xf=lambda d: d["k"], bins=list(range(1, 8)))
@CoverPoint("npu.conv.pad", xf=lambda d: min(d["pad"], 3), bins=[0, 1, 2, 3])
def _sample_conv2d(d):
    pass


@CoverPoint("npu.mac.k_total", xf=lambda d: "<8" if _conv(d)[2] < 8 else "8" if _conv(d)[2] == 8 else ">8",
            bins=["<8", "8", ">8"])
@CoverPoint("npu.mac.c_out", xf=lambda d: "partial tile" if d["c_out"] % 8 else "full tiles",
            bins=["partial tile", "full tiles"])
@CoverPoint("npu.mac.n_tiles", xf=lambda d: "1" if d["c_out"] <= 8 else "2+", bins=["1", "2+"])
@CoverPoint("npu.mac.relu", xf=lambda d: "relu" if d["act_min"] > -128 else "full range", bins=["relu", "full range"])
@CoverPoint("npu.mac.shift", xf=lambda d: _shift_bin(d["rq_shift"]), bins=["1-15", "16-31", "32-47", "48-62"])
@CoverPoint("npu.mac.z_in", xf=lambda d: "zero" if d["z_in"] == 0 else "nonzero", bins=["zero", "nonzero"])
def _sample_mac(d):
    pass


@CoverPoint("npu.pool.odd", xf=lambda d: (d["in_h"] % 2, d["in_w"] % 2), bins=[(0, 0), (0, 1), (1, 0), (1, 1)])
def _sample_pool(d):
    pass


@CoverPoint("npu.argmax.c_in", xf=lambda d: "1" if d["c_in"] == 1 else "16" if d["c_in"] == 16 else "2-15",
            bins=["1", "2-15", "16"])
def _sample_argmax(d):
    pass


def sample_run(descs, outcome):
    status, code, at = outcome
    _sample_program(descs, outcome)
    if status == "err" and code == ref.ERR_BAD_LAYER_CNT:
        _sample_error(code, True)
        return
    for i, d in enumerate(descs):
        failed = status == "err" and i == at
        _sample_layer(d, "err" if failed else "ok")
        if failed:
            _sample_error(code, i == 0)
            break
        if d["op"] == ref.OP_CONV2D:
            _sample_conv2d(d)
        if d["op"] in (ref.OP_CONV2D, ref.OP_FC):
            _sample_mac(d)
        elif d["op"] == ref.OP_MAXPOOL2X2:
            _sample_pool(d)
        elif d["op"] == ref.OP_ARGMAX:
            _sample_argmax(d)


# --------------------------------------------------------------------- DMA
def _dma_len_bin(n):
    return "8" if n == 8 else "16-128" if n <= 128 else "136-1024" if n <= 1024 else ">1024"


@CoverPoint("dma.direction", xf=lambda t: ("store" if t["store"] else "load", "wbuf" if t["sel"] else "abuf"),
            bins=[("load", "abuf"), ("load", "wbuf"), ("store", "abuf"), ("store", "wbuf")])
@CoverPoint("dma.length", xf=lambda t: _dma_len_bin(t["len"]), bins=["8", "16-128", "136-1024", ">1024"])
@CoverPoint("dma.crosses_4k", xf=lambda t: (t["src"] % 4096) + t["len"] > 4096, bins=[True, False])
@CoverPoint("dma.params", xf=lambda t: "ok" if t["ok"] else "rejected", bins=["ok", "rejected"])
def sample_dma(t):
    pass


# --------------------------------------------------------------- AXI4-Lite
@CoverPoint("axil.write.order", xf=lambda it, tgt: it.order if it.write else None,
            bins=["together", "aw_first", "w_first"])
@CoverPoint("axil.kind", xf=lambda it, tgt: "write" if it.write else "read", bins=["write", "read"])
@CoverPoint("axil.target", xf=lambda it, tgt: tgt, bins=["reg", "abuf", "wbuf", "none"])
@CoverPoint("axil.aligned", xf=lambda it, tgt: "aligned" if it.addr % 4 == 0 else "misaligned",
            bins=["aligned", "misaligned"])
@CoverPoint("axil.resp", xf=lambda it, tgt: "OKAY" if it.resp == 0 else "SLVERR", bins=["OKAY", "SLVERR"])
@CoverPoint("axil.resp_wait", xf=lambda it, tgt: _wait_bin(it.resp_wait), bins=["0", "1-3", "4+"])
@CoverCross("axil.kind_x_target_x_resp", items=["axil.kind", "axil.target", "axil.resp"],
            ign_bins=[("write", "none", "OKAY"), ("read", "none", "OKAY")])
def sample_axil(it, tgt):
    pass


# ----------------------------------------------------------------- closure
def closure():
    """(total bins, hit bins, [names of points or crosses with a missed bin and the missed bins])."""
    total = hit = 0
    missed = []
    for name in sorted(coverage_db):
        item = coverage_db[name]
        if not isinstance(item, (CoverPoint, CoverCross)):
            continue
        total += item.size
        hit += item.coverage
        if item.coverage < item.size:
            holes = [b for b, n in item.detailed_coverage.items() if n == 0]
            missed.append((name, holes))
    return total, hit, missed
