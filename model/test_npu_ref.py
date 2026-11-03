#!/usr/bin/env python3
"""Hand-computed checks for model/npu_ref.py. Run: python3 model/test_npu_ref.py"""
from npu_ref import (ERR_BAD_ADDR, ERR_BAD_OUT_DIM, ERR_BAD_PARAM, ERR_NONE, INT32_MAX, INT32_MIN, OP_CONV2D, OP_FC,
                     argmax, bytes_to_chw, chw_bytes, conv2d_layer, desc_check, desc_pack, desc_unpack, fc_layer,
                     matmul_int, maxpool2x2, pack_bias, pack_weights, pe_tile, quantize_multiplier, requant)

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

    assert matmul_int([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
    assert matmul_int([[-128] * 512], [[-128]] * 512) == [[512 * 16384]]
    assert matmul_int([[127, -128]], [[-128, 127], [127, -128]]) == [[-32512, 32513]]

    conv1 = {"op": OP_CONV2D, "k": 5, "pad": 2, "stride_m1": 0, "in_h": 28, "in_w": 28, "c_in": 1, "c_out": 6,
             "in_base": 0x0000, "out_base": 0x0400, "w_base": 0x00000, "b_base": 0x00100,
             "rq_mult": 1 << 30, "rq_shift": 31, "z_in": -128, "z_out": -128, "act_min": -128, "act_max": 127}
    assert desc_unpack(desc_pack(conv1)) == conv1
    assert desc_check(conv1) == ERR_NONE
    assert desc_check({**conv1, "k": 8}) == ERR_BAD_PARAM
    assert desc_check({**conv1, "in_h": 1, "pad": 0}) == ERR_BAD_OUT_DIM
    assert desc_check({**conv1, "w_base": 4}) == ERR_BAD_ADDR
    assert desc_check({**conv1, "out_base": 0x8000 - 4703}) == ERR_BAD_ADDR
    assert desc_check({**conv1, "out_base": 0x8000 - 4704}) == ERR_NONE
    fc = {**conv1, "op": OP_FC, "in_h": 1, "in_w": 1, "c_in": 400, "c_out": 120, "k": 1, "pad": 0,
          "w_base": 0x01000, "b_base": 0x0CC00}
    assert desc_check(fc) == ERR_NONE
    assert desc_check({**fc, "k": 9, "pad": 15, "in_h": 7}) == ERR_NONE   # FC ignores k, pad, in_h

    # conv2d: 1x1 kernel on a 1x1 image is a dot product; padding reads z_in
    x = [[[3]], [[-4]]]
    w = [[[[2]], [[5]]], [[[1]], [[1]]]]
    out = conv2d_layer(x, w, [0, 0], 1, 0, 0, 1 << 30, 31, 0, -128, 127)
    assert out == [[[-7]], [[0]]], out                       # (6-20)*0.5 = -7; (3-4)*0.5 = -0.5 rounds half up to 0
    x = [[[1, 2], [3, 4]]]
    w = [[[[1, 0], [0, 0]]]]
    out = conv2d_layer(x, w, [0], 2, 1, 9, 1 << 30, 31, 0, -128, 127)   # z_in = 9 fills the border
    assert out[0] == [[5, 5, 5], [5, 1, 1], [5, 2, 2]], out   # x0.5 then round half up
    assert maxpool2x2([[[1, 5, 3], [2, 4, 9], [7, 7, 7]]]) == [[[5]]]
    assert argmax([3, 9, 9, 1]) == 1
    assert fc_layer([1, 2], [[3, 4], [-1, 0]], [0, 0], 1 << 30, 31, 0, -128, 127) == [6, 0]   # 5.5 -> 6, -0.5 -> 0
    t = [[[1, -2], [3, -4]], [[5, 6], [-7, 8]]]
    assert bytes_to_chw(chw_bytes(t), 2, 2, 2) == t
    assert pack_weights([[[[1]]]]) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert pack_bias([-1], 1)[:8] == [255, 255, 255, 255, 0, 0, 0, 0]

    print("PASS: test_npu_ref")


if __name__ == "__main__":
    main()
