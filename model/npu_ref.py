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
