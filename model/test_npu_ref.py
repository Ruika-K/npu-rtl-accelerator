#!/usr/bin/env python3
"""Hand-computed checks for model/npu_ref.py. Run: python3 model/test_npu_ref.py"""
from npu_ref import INT32_MAX, INT32_MIN, pe_tile, quantize_multiplier, requant

HALF = 1 << 30   # with shift 31 this is an exact x0.5 multiplier

CASES = [
    # (args, expected)
    ((0, 0, HALF, 31, 0, -128, 127), 0),
    ((3, 0, HALF, 31, 0, -128, 127), 2),      # 1.5 -> 2 (half up)
    ((-3, 0, HALF, 31, 0, -128, 127), -1),    # -1.5 -> -1 (half up, toward +inf)
    ((1, 0, HALF, 31, 0, -128, 127), 1),
    ((-1, 0, HALF, 31, 0, -128, 127), 0),
    ((5, 0, HALF, 31, 0, -128, 127), 3),
    ((4, 0, HALF, 31, 0, -128, 127), 2),
    ((200, 0, HALF, 31, 0, -128, 127), 100),
    ((0, 0, HALF, 31, -128, -128, 127), -128),
    ((60, 0, HALF, 31, 100, -128, 127), 127),  # 30 + 100 clamps
    ((-20, 0, HALF, 31, 3, 3, 127), 3),        # ReLU: -10 + 3 -> z_out
    ((20, 0, HALF, 31, 3, 3, 50), 13),
    ((200, 0, HALF, 31, 3, 3, 50), 50),
    ((10, 12, HALF, 31, 0, -128, 127), 11),
    ((INT32_MAX, INT32_MAX, INT32_MAX, 1, 0, -128, 127), 127),
    ((INT32_MIN, INT32_MIN, INT32_MAX, 1, 0, -128, 127), -128),
    ((INT32_MAX, INT32_MIN, INT32_MAX, 1, 0, -128, 127), -128),  # sum = -1
    ((INT32_MAX, 0, INT32_MAX, 62, 0, -128, 127), 1),
    ((INT32_MAX, 0, HALF, 62, 0, -128, 127), 0),
    ((0, 0, INT32_MAX, 0, 7, -128, 127), 7),
    ((1, 0, HALF, 0, 0, -128, 127), 127),
]


def main():
    for args, expected in CASES:
        got = requant(*args)
        assert got == expected, f"requant{args} = {got}, expected {expected}"

    assert pe_tile([-128] * 48, [-128] * 48) == 48 * 16384
    assert pe_tile([-128] * 48, [127] * 48) == 48 * -16256
    assert pe_tile([3], [-7]) == -21

    for m0 in (0.5, 0.25, 1e-3, 0.0123, 0.999, 1.0, 7.5):
        mult, shift = quantize_multiplier(m0)
        assert (1 << 30) <= mult < (1 << 31), (m0, mult)
        assert abs(mult / (1 << shift) - m0) <= m0 * 2 ** -30, (m0, mult, shift)

    print("PASS: test_npu_ref")


if __name__ == "__main__":
    main()
