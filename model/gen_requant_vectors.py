#!/usr/bin/env python3
"""Generate requant test vectors from the golden model.

Output: one vector per line, eight signed decimals:
    acc bias mult shift z_out act_min act_max expected
Consumed by tb/unit/requant_unit_tb.sv via +vectors=<path>.

The distribution mirrors the testbench's own random generator but is produced
independently, so the Python model, the SystemVerilog reference function and
the RTL are all checked against each other.
"""
import argparse
import random

from npu_ref import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN, RQ_MULT_MAX, RQ_MULT_MIN, requant

DIRECTED = [
    # acc, bias, mult, shift, z_out, act_min, act_max
    (0, 0, 1 << 30, 31, 0, -128, 127),
    (3, 0, 1 << 30, 31, 0, -128, 127),
    (-3, 0, 1 << 30, 31, 0, -128, 127),
    (1, 0, 1 << 30, 31, 0, -128, 127),
    (-1, 0, 1 << 30, 31, 0, -128, 127),
    (INT32_MAX, INT32_MAX, RQ_MULT_MAX, 1, 0, -128, 127),
    (INT32_MIN, INT32_MIN, RQ_MULT_MAX, 1, 0, -128, 127),
    (INT32_MAX, INT32_MIN, RQ_MULT_MAX, 1, 0, -128, 127),
    (INT32_MAX, 0, RQ_MULT_MAX, 62, 0, -128, 127),
    (INT32_MIN, 0, RQ_MULT_MAX, 62, 0, -128, 127),
    (0, 0, RQ_MULT_MAX, 0, 7, -128, 127),
    (-20, 0, 1 << 30, 31, 3, 3, 127),
]


def rand_acc(rng):
    mode = rng.randrange(4)
    if mode == 0:
        return rng.randint(INT32_MIN, INT32_MAX)
    if mode == 1:
        return rng.randint(-(1 << 19), (1 << 19) - 1)
    if mode == 2:
        return rng.randint(-2048, 2047)
    return rng.choice((INT32_MIN, INT32_MAX))


def rand_mult(rng):
    mode = rng.randrange(8)
    if mode == 0:
        return RQ_MULT_MIN
    if mode == 1:
        return RQ_MULT_MAX
    return rng.randint(RQ_MULT_MIN, RQ_MULT_MAX)


def rand_shift(rng):
    mode = rng.randrange(4)
    if mode == 0:
        return rng.randint(1, 62)
    if mode == 1:
        return rng.randint(56, 62)
    return rng.randint(24, 44)


def rand_vector(rng):
    acc = rand_acc(rng)
    bias = rand_acc(rng) if rng.randrange(2) == 0 else 0
    mult = rand_mult(rng)
    shift = rand_shift(rng)
    z_out = rng.randint(INT8_MIN, INT8_MAX)
    mode = rng.randrange(3)
    if mode == 0:
        act_min, act_max = INT8_MIN, INT8_MAX
    elif mode == 1:
        act_min, act_max = z_out, INT8_MAX
    else:
        a, b = rng.randint(INT8_MIN, INT8_MAX), rng.randint(INT8_MIN, INT8_MAX)
        act_min, act_max = min(a, b), max(a, b)
    return (acc, bias, mult, shift, z_out, act_min, act_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100000, help="number of random vectors")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.output, "w", encoding="ascii") as f:
        for vec in DIRECTED:
            f.write(" ".join(str(v) for v in (*vec, requant(*vec))) + "\n")
        for _ in range(args.n):
            vec = rand_vector(rng)
            f.write(" ".join(str(v) for v in (*vec, requant(*vec))) + "\n")
    print(f"wrote {len(DIRECTED) + args.n} vectors to {args.output}")


if __name__ == "__main__":
    main()
