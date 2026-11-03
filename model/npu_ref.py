"""Integer golden model for the NPU v1.

Every function here uses only Python integers, so the arithmetic is exact and
is the single source of truth that the RTL must match bit for bit
(docs/architecture.md section 3, docs/verification.md section 1).

M1 scope: requantization and the PE multiply-accumulate.
Later milestones add conv2d / fc, maxpool, argmax and the full network.
"""

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1

RQ_MULT_MIN = 2 ** 30          # rq_mult is normalised into [2^30, 2^31)
RQ_MULT_MAX = 2 ** 31 - 1
RQ_SHIFT_MIN, RQ_SHIFT_MAX = 1, 62


def _check_int8(name, v):
    if not INT8_MIN <= v <= INT8_MAX:
        raise ValueError(f"{name}={v} is not int8")


def _check_int32(name, v):
    if not INT32_MIN <= v <= INT32_MAX:
        raise ValueError(f"{name}={v} is not int32")


def requant(acc, bias, mult, shift, z_out, act_min=INT8_MIN, act_max=INT8_MAX):
    """int32 accumulator -> int8 activation (architecture.md section 3.3).

    sum     = acc + bias
    prod    = sum * mult
    rounded = (prod + 2^(shift-1)) >> shift      (arithmetic shift, round half up)
    q       = clamp(rounded + z_out, act_min, act_max)

    shift == 0 is outside the documented range; it is defined as "no rounding,
    no shift" so that the RTL and this model agree even on illegal input.
    """
    _check_int32("acc", acc)
    _check_int32("bias", bias)
    _check_int32("mult", mult)
    if not 0 <= shift <= 63:
        raise ValueError(f"shift={shift} out of range")
    _check_int8("z_out", z_out)
    _check_int8("act_min", act_min)
    _check_int8("act_max", act_max)

    s = acc + bias
    prod = s * mult
    if shift == 0:
        rounded = prod
    else:
        rounded = (prod + (1 << (shift - 1))) >> shift   # Python >> floors: arithmetic shift
    v = rounded + z_out
    if v > act_max:
        v = act_max
    elif v < act_min:
        v = act_min
    return v


def quantize_multiplier(m0):
    """Real multiplier m0 > 0 -> (mult, shift) with m0 ~= mult / 2^shift.

    mult is normalised into [2^30, 2^31) so the RTL multiplier keeps full
    precision; shift must land in 1..62 (checked).
    """
    import math

    if m0 <= 0:
        raise ValueError("multiplier must be positive")
    frac, exp = math.frexp(m0)            # m0 = frac * 2^exp, frac in [0.5, 1)
    mult = round(frac * (1 << 31))        # in [2^30, 2^31]
    shift = 31 - exp
    if mult == 1 << 31:                   # rounding overflow, renormalise
        mult >>= 1
        shift -= 1
    if not RQ_SHIFT_MIN <= shift <= RQ_SHIFT_MAX:
        raise ValueError(f"multiplier {m0} needs shift {shift}, outside 1..62")
    return mult, shift


def pe_mac(acc, a, w, first):
    """One PE step: int8 x int8 product accumulated into int32 (architecture.md 5.2)."""
    _check_int8("a", a)
    _check_int8("w", w)
    prod = a * w
    return prod if first else acc + prod


def pe_tile(a_seq, w_seq):
    """Dot product of one tile as the PE computes it; the value parked in shadow."""
    if len(a_seq) != len(w_seq) or not a_seq:
        raise ValueError("a_seq and w_seq must be non-empty and equal length")
    acc = 0
    for k, (a, w) in enumerate(zip(a_seq, w_seq)):
        acc = pe_mac(acc, a, w, k == 0)
    _check_int32("acc", acc)
    return acc


def matmul_int(a, w):
    """Out[M][N] = In[M][K] x W[K][N] with exact integer arithmetic.

    This is what the systolic array computes tile by tile (architecture.md 5.1):
    each output element is one PE's accumulator after K beats. Inputs are lists
    of lists of int8; the result is checked to fit int32.
    """
    m_rows = len(a)
    k_len = len(a[0]) if m_rows else 0
    if any(len(row) != k_len for row in a):
        raise ValueError("ragged In matrix")
    if len(w) != k_len:
        raise ValueError("K mismatch between In and W")
    n_cols = len(w[0]) if k_len else 0
    if any(len(row) != n_cols for row in w):
        raise ValueError("ragged W matrix")
    for row in a:
        for v in row:
            _check_int8("a", v)
    for row in w:
        for v in row:
            _check_int8("w", v)
    out = []
    for m in range(m_rows):
        out_row = []
        for n in range(n_cols):
            acc = 0
            for k in range(k_len):
                acc += a[m][k] * w[k][n]
            _check_int32("acc", acc)
            out_row.append(acc)
        out.append(out_row)
    return out


# ---------------------------------------------------------------------------
# Descriptors (programming_model.md section 3) and the sequencer's checks
# (programming_model.md section 5, npu_sequencer.sv check_desc)
# ---------------------------------------------------------------------------

ARR_N = 8
ABUF_BYTES = 32768
WBUF_BYTES = 131072
NUM_DESC = 16
MAX_KERNEL = 7
MAX_PAD = 7
MAX_K_TOTAL = 4096
MAX_ARGMAX_C = 16

OP_NOP, OP_CONV2D, OP_MAXPOOL2X2, OP_FC, OP_ARGMAX = 0, 1, 2, 3, 4
ERR_NONE, ERR_BAD_OP, ERR_BAD_PARAM, ERR_BAD_ADDR, ERR_BAD_LAYER_CNT, ERR_BAD_OUT_DIM = 0, 1, 2, 3, 4, 5

DESC_FIELDS = ("op", "k", "pad", "stride_m1", "in_h", "in_w", "c_in", "c_out", "in_base", "out_base",
               "w_base", "b_base", "rq_mult", "rq_shift", "z_in", "z_out", "act_min", "act_max")


def _u(v, bits):
    return v & ((1 << bits) - 1)


def _s8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v


def desc_pack(d):
    """Descriptor dict -> the 10 stored 32-bit words W0..W9 (reserved bits zero)."""
    w = [0] * 10
    w[0] = _u(d["op"], 4) | (_u(d["k"], 4) << 4) | (_u(d["pad"], 4) << 8) | (_u(d["stride_m1"], 2) << 12)
    w[1] = _u(d["in_h"], 16) | (_u(d["in_w"], 16) << 16)
    w[2] = _u(d["c_in"], 16) | (_u(d["c_out"], 16) << 16)
    w[3] = _u(d["in_base"], 32)
    w[4] = _u(d["out_base"], 32)
    w[5] = _u(d["w_base"], 32)
    w[6] = _u(d["b_base"], 32)
    w[7] = _u(d["rq_mult"], 32)
    w[8] = _u(d["rq_shift"], 6)
    w[9] = _u(d["z_in"], 8) | (_u(d["z_out"], 8) << 8) | (_u(d["act_min"], 8) << 16) | (_u(d["act_max"], 8) << 24)
    return w


def desc_unpack(w):
    """10 words -> descriptor dict, the way npu_pkg::desc_from_words decodes them."""
    return {
        "op": w[0] & 0xF, "k": (w[0] >> 4) & 0xF, "pad": (w[0] >> 8) & 0xF, "stride_m1": (w[0] >> 12) & 0x3,
        "in_h": w[1] & 0xFFFF, "in_w": (w[1] >> 16) & 0xFFFF,
        "c_in": w[2] & 0xFFFF, "c_out": (w[2] >> 16) & 0xFFFF,
        "in_base": w[3], "out_base": w[4], "w_base": w[5], "b_base": w[6],
        "rq_mult": w[7], "rq_shift": w[8] & 0x3F,
        "z_in": _s8(w[9]), "z_out": _s8(w[9] >> 8), "act_min": _s8(w[9] >> 16), "act_max": _s8(w[9] >> 24),
    }


def pad_to_arr(c):
    return (c + ARR_N - 1) // ARR_N * ARR_N


def desc_check(d):
    """Return the error code the sequencer reports for descriptor dict d.

    Same checks in the same order as npu_sequencer.sv check_desc:
    ERR_BAD_OP, then ERR_BAD_PARAM, then ERR_BAD_OUT_DIM, then ERR_BAD_ADDR.
    """
    op = d["op"]
    if op == OP_NOP:
        return ERR_NONE
    if op not in (OP_CONV2D, OP_FC, OP_MAXPOOL2X2, OP_ARGMAX):
        return ERR_BAD_OP

    if op == OP_ARGMAX:
        if d["c_in"] == 0 or d["c_in"] > MAX_ARGMAX_C:
            return ERR_BAD_PARAM
        if d["in_base"] + d["c_in"] > ABUF_BYTES:
            return ERR_BAD_ADDR
        return ERR_NONE

    if op == OP_MAXPOOL2X2:
        if d["c_in"] == 0:
            return ERR_BAD_PARAM
        if d["in_h"] < 2 or d["in_w"] < 2:
            return ERR_BAD_OUT_DIM
        oh, ow = d["in_h"] >> 1, d["in_w"] >> 1
        if d["in_base"] + d["in_h"] * d["in_w"] * d["c_in"] > ABUF_BYTES:
            return ERR_BAD_ADDR
        if d["out_base"] + oh * ow * d["c_in"] > ABUF_BYTES:
            return ERR_BAD_ADDR
        return ERR_NONE

    # CONV2D and FC
    fc = op == OP_FC
    k = 1 if fc else d["k"]
    pad = 0 if fc else d["pad"]
    in_h = 1 if fc else d["in_h"]
    in_w = 1 if fc else d["in_w"]
    if k == 0 or k > MAX_KERNEL:
        return ERR_BAD_PARAM
    if pad > MAX_PAD:
        return ERR_BAD_PARAM
    if op == OP_CONV2D and d["stride_m1"] != 0:
        return ERR_BAD_PARAM
    if d["c_in"] == 0 or d["c_out"] == 0:
        return ERR_BAD_PARAM
    k_total = d["c_in"] * k * k
    if k_total > MAX_K_TOTAL:
        return ERR_BAD_PARAM
    if d["rq_shift"] == 0 or d["rq_shift"] == 63:
        return ERR_BAD_PARAM
    if not (RQ_MULT_MIN <= d["rq_mult"] <= RQ_MULT_MAX):
        return ERR_BAD_PARAM
    if d["act_min"] > d["act_max"]:
        return ERR_BAD_PARAM
    if in_h + 2 * pad < k or in_w + 2 * pad < k:
        return ERR_BAD_OUT_DIM
    oh = in_h + 2 * pad - k + 1
    ow = in_w + 2 * pad - k + 1
    n_pad = pad_to_arr(d["c_out"])
    if d["w_base"] % 8 != 0 or d["b_base"] % 8 != 0:
        return ERR_BAD_ADDR
    if d["in_base"] + in_h * in_w * d["c_in"] > ABUF_BYTES:
        return ERR_BAD_ADDR
    if d["out_base"] + oh * ow * d["c_out"] > ABUF_BYTES:
        return ERR_BAD_ADDR
    if d["w_base"] + k_total * n_pad > WBUF_BYTES:
        return ERR_BAD_ADDR
    if d["b_base"] + n_pad * 4 > WBUF_BYTES:
        return ERR_BAD_ADDR
    return ERR_NONE


# ---------------------------------------------------------------------------
# Layers (architecture.md sections 3, 5, 6) and memory images (section 4)
# Tensors are nested lists in CHW order: x[c][h][w], int8 values.
# ---------------------------------------------------------------------------

def to_u8(v):
    """int8 -> the byte stored in memory (two's complement)."""
    _check_int8("v", v)
    return v & 0xFF


def from_u8(b):
    return _s8(b)


def chw_bytes(x):
    """CHW tensor -> list of bytes in memory order (address = c*H*W + h*W + w)."""
    out = []
    for plane in x:
        for row in plane:
            for v in row:
                out.append(to_u8(v))
    return out


def bytes_to_chw(data, c, h, w):
    assert len(data) >= c * h * w
    it = iter(data)
    return [[[from_u8(next(it)) for _ in range(w)] for _ in range(h)] for _ in range(c)]


def pack_weights(w, n=ARR_N):
    """w[cout][cin][k][k] -> bytes of W[K_total][N_pad] (k order ci, kh, kw; N padded to n with zeros)."""
    cout = len(w)
    cin = len(w[0])
    k = len(w[0][0])
    n_pad = (cout + n - 1) // n * n
    out = []
    for ci in range(cin):
        for kh in range(k):
            for kw in range(k):
                for co in range(n_pad):
                    out.append(to_u8(w[co][ci][kh][kw]) if co < cout else 0)
    return out


def pack_bias(bias, cout, n=ARR_N):
    """bias[cout] int32 -> little-endian bytes, padded to a multiple of n entries."""
    n_pad = (cout + n - 1) // n * n
    out = []
    for co in range(n_pad):
        v = bias[co] if co < cout else 0
        _check_int32("bias", v)
        v &= 0xFFFFFFFF
        out += [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]
    return out


def conv2d_layer(x, w, bias, k, pad, z_in, mult, shift, z_out, act_min, act_max):
    """CONV2D exactly as the hardware computes it.

    x[cin][h][w] int8, w[cout][cin][k][k] int8, bias[cout] int32 (already
    holds the folded zero-point correction). Padding positions read z_in.
    Returns out[cout][oh][ow] int8 after requantization.
    """
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout = len(w)
    oh = h + 2 * pad - k + 1
    ow = wd + 2 * pad - k + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("empty output")
    out = []
    for co in range(cout):
        plane = []
        for y in range(oh):
            row = []
            for xx in range(ow):
                acc = 0
                for ci in range(cin):
                    for kh in range(k):
                        ih = y + kh - pad
                        for kw in range(k):
                            iw = xx + kw - pad
                            v = x[ci][ih][iw] if (0 <= ih < h and 0 <= iw < wd) else z_in
                            acc += v * w[co][ci][kh][kw]
                _check_int32("acc", acc)
                row.append(requant(acc, bias[co], mult, shift, z_out, act_min, act_max))
            plane.append(row)
        out.append(plane)
    return out


def fc_layer(x_vec, w_mat, bias, mult, shift, z_out, act_min, act_max):
    """FC as the 1x1 CONV2D the hardware runs: x_vec[cin], w_mat[cout][cin]."""
    x = [[[v]] for v in x_vec]
    w = [[[[wv]] for wv in row] for row in w_mat]
    out = conv2d_layer(x, w, bias, 1, 0, 0, mult, shift, z_out, act_min, act_max)
    return [plane[0][0] for plane in out]


def maxpool2x2(x):
    """2x2 stride-2 max pooling per channel; output size rounds down."""
    out = []
    for plane in x:
        h, w = len(plane), len(plane[0])
        oh, ow = h // 2, w // 2
        out.append([[max(plane[2 * y][2 * xx], plane[2 * y][2 * xx + 1],
                         plane[2 * y + 1][2 * xx], plane[2 * y + 1][2 * xx + 1])
                     for xx in range(ow)] for y in range(oh)])
    return out


def argmax(logits):
    """Index of the maximum; ties go to the lowest index."""
    best, best_i = logits[0], 0
    for i, v in enumerate(logits):
        if v > best:
            best, best_i = v, i
    return best_i


# ---------------------------------------------------------------------------
# Programs: a descriptor list executed against the two memories, exactly as the
# hardware does (programming_model.md sections 3 to 6)
# ---------------------------------------------------------------------------

def _read_u8(mem, base, n):
    return [mem[base + i] for i in range(n)]


def _write_u8(mem, base, data):
    for i, b in enumerate(data):
        mem[base + i] = b & 0xFF


def unpack_weights(data, cout, cin, k, n=ARR_N):
    """Inverse of pack_weights: bytes of W[K_total][N_pad] -> w[cout][cin][k][k]."""
    n_pad = (cout + n - 1) // n * n
    w = [[[[0] * k for _ in range(k)] for _ in range(cin)] for _ in range(cout)]
    i = 0
    for ci in range(cin):
        for kh in range(k):
            for kw in range(k):
                for co in range(n_pad):
                    if co < cout:
                        w[co][ci][kh][kw] = from_u8(data[i])
                    i += 1
    return w


def unpack_bias(data, cout):
    out = []
    for co in range(cout):
        v = data[4 * co] | (data[4 * co + 1] << 8) | (data[4 * co + 2] << 16) | (data[4 * co + 3] << 24)
        out.append(v - (1 << 32) if v >= (1 << 31) else v)
    return out


def run_program(descs, abuf, wbuf):
    """Execute descriptor dicts in order against bytearrays abuf / wbuf.

    Returns {"result": argmax index or None, "logits": [...], "layers": [(out_base, nbytes), ...]}.
    The memories are modified in place, like the hardware's act_buffer.
    """
    result, logits, layers = None, [], []
    for d in descs:
        code = desc_check(d)
        if code != ERR_NONE:
            raise ValueError(f"descriptor rejected with error code {code}: {d}")
        op = d["op"]
        if op == OP_NOP:
            layers.append((0, 0))
        elif op in (OP_CONV2D, OP_FC):
            fc = op == OP_FC
            k = 1 if fc else d["k"]
            pad = 0 if fc else d["pad"]
            h = 1 if fc else d["in_h"]
            w = 1 if fc else d["in_w"]
            cin, cout = d["c_in"], d["c_out"]
            k_total = cin * k * k
            n_pad = pad_to_arr(cout)
            x = bytes_to_chw(_read_u8(abuf, d["in_base"], cin * h * w), cin, h, w)
            wt = unpack_weights(_read_u8(wbuf, d["w_base"], k_total * n_pad), cout, cin, k)
            bias = unpack_bias(_read_u8(wbuf, d["b_base"], n_pad * 4), cout)
            out = conv2d_layer(x, wt, bias, k, pad, d["z_in"], d["rq_mult"], d["rq_shift"], d["z_out"],
                               d["act_min"], d["act_max"])
            data = chw_bytes(out)
            _write_u8(abuf, d["out_base"], data)
            layers.append((d["out_base"], len(data)))
        elif op == OP_MAXPOOL2X2:
            c, h, w = d["c_in"], d["in_h"], d["in_w"]
            x = bytes_to_chw(_read_u8(abuf, d["in_base"], c * h * w), c, h, w)
            data = chw_bytes(maxpool2x2(x))
            _write_u8(abuf, d["out_base"], data)
            layers.append((d["out_base"], len(data)))
        elif op == OP_ARGMAX:
            raw = _read_u8(abuf, d["in_base"], d["c_in"])
            logits = [from_u8(b) for b in raw]
            result = argmax(logits)
            layers.append((0, 0))
    return {"result": result, "logits": logits, "layers": layers}


def make_desc(op, **f):
    """Descriptor dict with every field present (unused ones zero)."""
    d = {name: 0 for name in DESC_FIELDS}
    d.update({"act_min": -128, "act_max": 127})
    d["op"] = op
    d.update(f)
    return d
