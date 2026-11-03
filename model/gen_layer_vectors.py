#!/usr/bin/env python3
"""Generate single-layer test cases for the engine and top-level testbenches.

Each case is a descriptor plus the memory images it needs and the expected
output bytes. Text format, repeated per case:

    CASE <op> <k> <pad> <in_h> <in_w> <c_in> <c_out> <in_base> <out_base> <w_base> <b_base> <rq_mult> <rq_shift> <z_in> <z_out> <act_min> <act_max>
    IN  <n> <n bytes>        placed at act_buffer[in_base]
    W   <n> <n bytes>        placed at weight_buffer[w_base]   (absent for MAXPOOL / ARGMAX)
    B   <n> <n bytes>        placed at weight_buffer[b_base]   (absent for MAXPOOL / ARGMAX)
    OUT <n> <n bytes>        expected at act_buffer[out_base]  (ARGMAX: 1 byte, the index)

Bytes are unsigned decimals; z_in / z_out / act_min / act_max are signed decimals.
"""
import argparse
import random

from npu_ref import (ARR_N, OP_ARGMAX, OP_CONV2D, OP_FC, OP_MAXPOOL2X2, RQ_MULT_MAX, RQ_MULT_MIN, argmax,
                     chw_bytes, conv2d_layer, maxpool2x2, pack_bias, pack_weights, quantize_multiplier)


def rand_tensor(rng, c, h, w, lo=-128, hi=127):
    return [[[rng.randint(lo, hi) for _ in range(w)] for _ in range(h)] for _ in range(c)]


def rand_weights(rng, cout, cin, k, lo=-128, hi=127):
    return [[[[rng.randint(lo, hi) for _ in range(k)] for _ in range(k)] for _ in range(cin)] for _ in range(cout)]


def emit(f, case, sections):
    f.write("CASE " + " ".join(str(v) for v in case) + "\n")
    for tag, data in sections:
        f.write(f"{tag} {len(data)} " + " ".join(str(b) for b in data) + "\n")


def conv_case(rng, op, in_h, in_w, c_in, c_out, k, pad, relu, small):
    lo, hi = (-8, 7) if small else (-128, 127)
    x = rand_tensor(rng, c_in, in_h, in_w, lo, hi)
    w = rand_weights(rng, c_out, c_in, k, lo, hi)
    k_total = c_in * k * k
    # Bias of the same magnitude as a typical accumulator so that requant is exercised in range.
    bias = [rng.randint(-k_total * 64, k_total * 64) for _ in range(c_out)]
    z_in = rng.randint(-128, 127)
    z_out = rng.randint(-128, 127)
    # Choose the multiplier so that the accumulator range maps roughly onto int8.
    acc_scale = max(1.0, k_total * 40.0)
    mult, shift = quantize_multiplier(rng.uniform(0.5, 2.0) * 128.0 / acc_scale)
    if relu:
        act_min, act_max = z_out, 127
    else:
        act_min, act_max = -128, 127
    out = conv2d_layer(x, w, bias, k, pad, z_in, mult, shift, z_out, act_min, act_max)
    oh = in_h + 2 * pad - k + 1
    ow = in_w + 2 * pad - k + 1
    n_pad = (c_out + ARR_N - 1) // ARR_N * ARR_N
    in_bytes = c_in * in_h * in_w
    out_bytes = c_out * oh * ow
    in_base = rng.randrange(0, 0x1000, 1)
    out_base = 0x4000 + rng.randrange(0, 0x800, 1)
    w_base = rng.randrange(0, 0x4000, 8)
    b_base = 0x18000 + rng.randrange(0, 0x1000, 8)
    assert in_base + in_bytes <= 0x4000 and out_base + out_bytes <= 0x8000
    assert w_base + k_total * n_pad <= 0x18000
    desc = (op, k if op == OP_CONV2D else 1, pad if op == OP_CONV2D else 0,
            in_h if op == OP_CONV2D else 1, in_w if op == OP_CONV2D else 1,
            c_in, c_out, in_base, out_base, w_base, b_base, mult, shift, z_in, z_out, act_min, act_max)
    return desc, [("IN", chw_bytes(x)), ("W", pack_weights(w)), ("B", pack_bias(bias, c_out)), ("OUT", chw_bytes(out))]


def pool_case(rng, in_h, in_w, c_in):
    x = rand_tensor(rng, c_in, in_h, in_w)
    out = maxpool2x2(x)
    in_base = rng.randrange(0, 0x1000, 1)
    out_base = 0x4000 + rng.randrange(0, 0x800, 1)
    desc = (OP_MAXPOOL2X2, 0, 0, in_h, in_w, c_in, 0, in_base, out_base, 0, 0, 0, 0, 0, 0, 0, 0)
    return desc, [("IN", chw_bytes(x)), ("OUT", chw_bytes(out))]


def argmax_case(rng, c_in, tie):
    logits = [rng.randint(-128, 127) for _ in range(c_in)]
    if tie and c_in > 1:
        i, j = rng.sample(range(c_in), 2)
        logits[i] = logits[j] = 127
    in_base = rng.randrange(0, 0x1000, 1)
    desc = (OP_ARGMAX, 0, 0, 0, 0, c_in, 0, in_base, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return desc, [("IN", [v & 0xFF for v in logits]), ("OUT", [argmax(logits)])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="number of random cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kind", choices=("conv", "pool", "argmax", "all"), default="all")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_cases = 0
    with open(args.output, "w", encoding="ascii") as f:
        if args.kind in ("conv", "all"):
            # Directed: every kernel size, padding, partial tiles in every dimension, FC.
            for (in_h, in_w, c_in, c_out, k, pad) in [
                (1, 1, 1, 1, 1, 0), (8, 8, 1, 8, 1, 0), (5, 5, 1, 6, 5, 2), (9, 9, 3, 6, 5, 0), (10, 10, 6, 16, 5, 0),
                (6, 7, 2, 9, 3, 1), (4, 12, 1, 1, 3, 2), (7, 7, 8, 8, 1, 0), (3, 3, 1, 17, 3, 1), (12, 12, 3, 4, 7, 3),
            ]:
                d, s = conv_case(rng, OP_CONV2D, in_h, in_w, c_in, c_out, k, pad, rng.randint(0, 1), False)
                emit(f, d, s); n_cases += 1
            for (c_in, c_out) in [(1, 1), (8, 8), (400, 120), (120, 84), (84, 10), (17, 3)]:
                d, s = conv_case(rng, OP_FC, 1, 1, c_in, c_out, 1, 0, rng.randint(0, 1), False)
                emit(f, d, s); n_cases += 1
            for _ in range(args.n):
                if rng.random() < 0.25:
                    c_in, c_out = rng.randint(1, 300), rng.randint(1, 40)
                    d, s = conv_case(rng, OP_FC, 1, 1, c_in, c_out, 1, 0, rng.randint(0, 1), rng.random() < 0.3)
                else:
                    k = rng.choice((1, 3, 5, 7))
                    pad = rng.randint(0, min(3, k - 1 + 1))
                    in_h = rng.randint(max(1, k - 2 * pad), 14)
                    in_w = rng.randint(max(1, k - 2 * pad), 14)
                    c_in = rng.randint(1, 6)
                    c_out = rng.randint(1, 20)
                    d, s = conv_case(rng, OP_CONV2D, in_h, in_w, c_in, c_out, k, pad, rng.randint(0, 1), rng.random() < 0.3)
                emit(f, d, s); n_cases += 1
        if args.kind in ("pool", "all"):
            for (in_h, in_w, c_in) in [(2, 2, 1), (28, 28, 6), (10, 10, 16), (5, 7, 3), (3, 2, 2), (17, 9, 1)]:
                d, s = pool_case(rng, in_h, in_w, c_in)
                emit(f, d, s); n_cases += 1
            for _ in range(args.n // 3):
                d, s = pool_case(rng, rng.randint(2, 20), rng.randint(2, 20), rng.randint(1, 8))
                emit(f, d, s); n_cases += 1
        if args.kind in ("argmax", "all"):
            for c_in in (1, 2, 8, 9, 10, 16):
                d, s = argmax_case(rng, c_in, False)
                emit(f, d, s); n_cases += 1
            for _ in range(args.n // 3):
                d, s = argmax_case(rng, rng.randint(1, 16), rng.random() < 0.5)
                emit(f, d, s); n_cases += 1
    print(f"wrote {n_cases} layer cases to {args.output}")


if __name__ == "__main__":
    main()
