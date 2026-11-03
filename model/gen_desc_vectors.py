#!/usr/bin/env python3
"""Generate descriptor-check vectors from the golden model.

Output: one descriptor per line, the 10 stored words W0..W9 as unsigned
decimals followed by the expected sequencer error code:
    w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 err_code
Consumed by tb/unit/npu_sequencer_tb.sv via +desc_vectors=<path>.
"""
import argparse
import random

from npu_ref import (ABUF_BYTES, OP_ARGMAX, OP_CONV2D, OP_FC, OP_MAXPOOL2X2, OP_NOP, RQ_MULT_MAX, RQ_MULT_MIN,
                     WBUF_BYTES, desc_check, desc_pack, desc_unpack)


def valid_base(rng):
    """A descriptor that passes every check, LeNet-like."""
    op = rng.choice((OP_CONV2D, OP_CONV2D, OP_FC, OP_MAXPOOL2X2, OP_ARGMAX, OP_NOP))
    d = {
        "op": op, "k": rng.choice((1, 3, 5, 7)), "pad": rng.randint(0, 3), "stride_m1": 0,
        "in_h": rng.randint(2, 32), "in_w": rng.randint(2, 32), "c_in": rng.randint(1, 16), "c_out": rng.randint(1, 32),
        "in_base": rng.randrange(0, 0x2000, 4), "out_base": rng.randrange(0x2000, 0x4000, 4),
        "w_base": rng.randrange(0, 0x8000, 8), "b_base": rng.randrange(0x8000, 0xA000, 8),
        "rq_mult": rng.randint(RQ_MULT_MIN, RQ_MULT_MAX), "rq_shift": rng.randint(1, 62),
        "z_in": rng.randint(-128, 127), "z_out": rng.randint(-128, 127), "act_min": -128, "act_max": 127,
    }
    if op == OP_FC:
        d["c_in"] = rng.randint(1, 512)
        d["c_out"] = rng.randint(1, 128)
    if op == OP_ARGMAX:
        d["c_in"] = rng.randint(1, 16)
    return d


def mutate(rng, d):
    """Break one thing, or several, in a valid descriptor."""
    for _ in range(rng.choice((1, 1, 1, 2, 3))):
        field = rng.choice(("op", "k", "pad", "stride_m1", "in_h", "in_w", "c_in", "c_out", "in_base", "out_base",
                            "w_base", "b_base", "rq_mult", "rq_shift", "act_min", "act_max"))
        if field == "op":
            d["op"] = rng.choice((5, 9, 15))
        elif field == "k":
            d["k"] = rng.choice((0, 8, 9, 15))
        elif field == "pad":
            d["pad"] = rng.choice((8, 15))
        elif field == "stride_m1":
            d["stride_m1"] = rng.choice((1, 2, 3))
        elif field in ("in_h", "in_w"):
            d[field] = rng.choice((0, 1, 2, 3, 4, 65535))
        elif field in ("c_in", "c_out"):
            d[field] = rng.choice((0, 0, 17, 100, 4097, 65535))
        elif field in ("in_base", "out_base"):
            d[field] = rng.choice((ABUF_BYTES - 4, ABUF_BYTES, ABUF_BYTES + 4, 0xFFFF_FFF0))
        elif field == "w_base":
            d[field] = rng.choice((4, 12, WBUF_BYTES - 8, WBUF_BYTES, 0xFFFF_FFF8))
        elif field == "b_base":
            d[field] = rng.choice((2, 4, 6, WBUF_BYTES - 8, WBUF_BYTES, 0xFFFF_FFF8))
        elif field == "rq_mult":
            d[field] = rng.choice((0, 1, RQ_MULT_MIN - 1, RQ_MULT_MAX + 1, 0xFFFF_FFFF))
        elif field == "rq_shift":
            d[field] = rng.choice((0, 63))
        elif field == "act_min":
            d["act_min"] = 100
            d["act_max"] = -100
        elif field == "act_max":
            d["act_max"] = rng.randint(-128, 127)
            d["act_min"] = rng.randint(d["act_max"] + 1, 127) if d["act_max"] < 127 else 127
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hist = {}
    with open(args.output, "w", encoding="ascii") as f:
        for i in range(args.n):
            d = valid_base(rng)
            if rng.random() < 0.6:
                d = mutate(rng, d)
            words = desc_pack(d)
            code = desc_check(desc_unpack(words))   # check what the hardware will actually see
            hist[code] = hist.get(code, 0) + 1
            f.write(" ".join(str(v) for v in words) + f" {code}\n")
    print(f"wrote {args.n} descriptors to {args.output}; codes: " +
          ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))


if __name__ == "__main__":
    main()
