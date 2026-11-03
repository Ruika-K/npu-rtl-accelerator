#!/usr/bin/env python3
"""Generate integer GEMM test vectors from the golden model.

Output format, repeated per case:
    M K N
    M lines of K int8 values        (In)
    K lines of N int8 values        (W)
    M lines of N int32 values       (Out = In x W)
Consumed by tb/unit/systolic_array_tb.sv via +gemm_vectors=<path>.
"""
import argparse
import random

from npu_ref import INT8_MAX, INT8_MIN, matmul_int

DIRECTED = [
    # (M, K, N)
    (1, 1, 1),
    (8, 1, 8),
    (8, 8, 8),
    (8, 7, 8),
    (8, 9, 8),
    (13, 25, 6),
    (5, 3, 3),
    (16, 2, 16),
    (9, 64, 9),
]


def rand_matrix(rng, rows, cols, mode):
    if mode == 0:
        return [[rng.randint(INT8_MIN, INT8_MAX) for _ in range(cols)] for _ in range(rows)]
    if mode == 1:
        return [[INT8_MIN for _ in range(cols)] for _ in range(rows)]
    if mode == 2:
        return [[rng.choice((INT8_MIN, INT8_MAX)) for _ in range(cols)] for _ in range(rows)]
    return [[rng.randint(-3, 3) for _ in range(cols)] for _ in range(rows)]


def emit(f, a, w, out):
    m, k, n = len(a), len(w), len(w[0])
    f.write(f"{m} {k} {n}\n")
    for row in a:
        f.write(" ".join(str(v) for v in row) + "\n")
    for row in w:
        f.write(" ".join(str(v) for v in row) + "\n")
    for row in out:
        f.write(" ".join(str(v) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="number of random cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-m", type=int, default=24)
    ap.add_argument("--max-k", type=int, default=64)
    ap.add_argument("--max-n", type=int, default=16)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases = 0
    with open(args.output, "w", encoding="ascii") as f:
        for m, k, n in DIRECTED:
            a = rand_matrix(rng, m, k, 0)
            w = rand_matrix(rng, k, n, 0)
            emit(f, a, w, matmul_int(a, w))
            cases += 1
        for _ in range(args.n):
            m = rng.randint(1, args.max_m)
            k = rng.randint(1, args.max_k)
            n = rng.randint(1, args.max_n)
            mode_a = rng.choice((0, 0, 0, 1, 2, 3))
            mode_w = rng.choice((0, 0, 0, 1, 2, 3))
            a = rand_matrix(rng, m, k, mode_a)
            w = rand_matrix(rng, k, n, mode_w)
            emit(f, a, w, matmul_int(a, w))
            cases += 1
    print(f"wrote {cases} GEMM cases to {args.output}")


if __name__ == "__main__":
    main()
