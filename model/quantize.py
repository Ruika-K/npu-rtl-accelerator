#!/usr/bin/env python3
"""Post-training quantization and export of a trained LeNet-5.

Input (from train_lenet.py, under model/out/<net>/):
    fp32.npz   conv1.w conv1.b conv2.w conv2.b fc1.w fc1.b fc2.w fc2.b fc3.w fc3.b   (float32)
    calib.npz  x: uint8 [n, cin, h, w]                                            calibration images
    test.npz   x: uint8 [n, cin, h, w], y: int64 [n]                              test images

Output (committed, under model/data/<net>/):
    fp32.npz        the checkpoint as float16, so this script can be re-run without training
    net.json        layouts, quantization parameters, descriptors, accuracy numbers
    program.txt     what the testbench loads: descriptors, weight-buffer image, images, expected results
                    and the per-layer memory traces of the first images (see write_program)

Quantization follows docs/architecture.md section 3: int8 asymmetric activations,
int8 symmetric per-tensor weights, int32 biases with the zero point folded in,
mult / shift from quantize_multiplier. Every expected value in program.txt comes
from the integer model (npu_ref.run_program), never from the float model.
"""
import argparse
import json
import os

import numpy as np

from npu_ref import (ARR_N, OP_ARGMAX, OP_CONV2D, OP_FC, OP_MAXPOOL2X2, chw_bytes, desc_pack, make_desc,
                     pack_bias, pack_weights, quantize_multiplier, run_program)

NETS = {
    "mnist": {"cin": 1, "size": 28, "conv1_pad": 2},
    "cifar10": {"cin": 3, "size": 32, "conv1_pad": 0},
}
LAYER_NAMES = ["conv1", "pool1", "conv2", "pool2", "fc1", "fc2", "fc3", "argmax"]
TRACE_IMAGES = 2    # images whose every layer output is written to program.txt


# ---------------------------------------------------------------------------
# float32 reference forward (numpy), only for calibration and accuracy reporting
# ---------------------------------------------------------------------------
def conv2d_f32(x, w, b, pad):
    # x [n, cin, h, w], w [cout, cin, k, k]
    n, cin, h, wd = x.shape
    cout, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = h + 2 * pad - k + 1, wd + 2 * pad - k + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for kh in range(k):
        for kw in range(k):
            out += np.einsum("nchw,oc->nohw", xp[:, :, kh:kh + oh, kw:kw + ow], w[:, :, kh, kw])
    return out + b[None, :, None, None]


def maxpool_f32(x):
    n, c, h, w = x.shape
    x = x[:, :, : h // 2 * 2, : w // 2 * 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def forward_f32(p, x, conv1_pad):
    """Returns the list of layer outputs in LAYER_NAMES order (argmax as class ids)."""
    outs = []
    a = np.maximum(conv2d_f32(x, p["conv1.w"], p["conv1.b"], conv1_pad), 0); outs.append(a)
    a = maxpool_f32(a); outs.append(a)
    a = np.maximum(conv2d_f32(a, p["conv2.w"], p["conv2.b"], 0), 0); outs.append(a)
    a = maxpool_f32(a); outs.append(a)
    a = a.reshape(a.shape[0], -1)
    a = np.maximum(a @ p["fc1.w"].T + p["fc1.b"], 0); outs.append(a)
    a = np.maximum(a @ p["fc2.w"].T + p["fc2.b"], 0); outs.append(a)
    a = a @ p["fc3.w"].T + p["fc3.b"]; outs.append(a)
    outs.append(a.argmax(axis=1))
    return outs


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def act_params(vals):
    """TFLite-style asymmetric int8 parameters covering [min(0, min), max(0, max)]."""
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    if hi == lo:
        hi = lo + 1.0
    scale = (hi - lo) / 255.0
    zp = int(round(-128 - lo / scale))
    zp = max(-128, min(127, zp))
    return scale, zp


def quant_weights(w):
    s = float(np.abs(w).max()) / 127.0
    if s == 0.0:
        s = 1.0
    q = np.clip(np.round(w / s), -127, 127).astype(np.int64)
    return s, q


def quantize_net(p, calib, conv1_pad):
    """Returns per-layer dicts with int8 weights, folded biases, requant parameters."""
    x = calib.astype(np.float32) / 255.0
    outs = forward_f32(p, x, conv1_pad)
    layers = []
    s_in, z_in = 1.0 / 255.0, -128          # input: q = pixel - 128
    for i, name in enumerate(LAYER_NAMES):
        if name.startswith("conv") or name.startswith("fc"):
            w, b = p[name + ".w"], p[name + ".b"]
            s_w, wq = quant_weights(w)
            s_out, z_out = act_params(outs[i])
            relu = name != "fc3"
            if relu:
                z_out = -128                  # post-ReLU range starts at 0
                s_out = max(0.0, float(outs[i].max())) / 255.0 or 1.0 / 255.0
            m0 = s_in * s_w / s_out
            mult, shift = quantize_multiplier(m0)
            bias_q = np.round(b / (s_in * s_w)).astype(np.int64)
            w2 = wq.reshape(wq.shape[0], -1)
            folded = bias_q - z_in * w2.sum(axis=1)
            assert np.all(np.abs(folded) < 2 ** 31), "folded bias out of int32"
            layers.append({
                "name": name, "op": OP_FC if name.startswith("fc") else OP_CONV2D,
                "w_q": wq, "bias": folded.tolist(), "s_w": s_w, "s_in": s_in, "z_in": z_in,
                "s_out": s_out, "z_out": z_out, "mult": mult, "shift": shift, "m0": m0,
                "act_min": z_out if relu else -128, "act_max": 127,
            })
            s_in, z_in = s_out, z_out
        elif name.startswith("pool"):
            layers.append({"name": name, "op": OP_MAXPOOL2X2, "s_in": s_in, "z_in": z_in})
        else:
            layers.append({"name": name, "op": OP_ARGMAX, "s_in": s_in, "z_in": z_in})
    return layers


# ---------------------------------------------------------------------------
# Memory layout and descriptors (programming_model.md section 6)
# ---------------------------------------------------------------------------
def align(v, a):
    return (v + a - 1) // a * a


def build_program(layers, cin, size, conv1_pad):
    """Returns (descriptor dicts, weight-buffer image bytes, layout info)."""
    # activation regions, each starting on a 0x100 boundary
    shapes = []
    c, h, w = cin, size, size
    abuf_cursor = 0
    in_region = (abuf_cursor, c * h * w)
    abuf_cursor = align(abuf_cursor + c * h * w, 0x100)
    regions = [in_region]
    for lay in layers:
        if lay["op"] == OP_CONV2D:
            k = lay["w_q"].shape[2]
            pad = conv1_pad if lay["name"] == "conv1" else 0
            c, h, w = lay["w_q"].shape[0], h + 2 * pad - k + 1, w + 2 * pad - k + 1
        elif lay["op"] == OP_MAXPOOL2X2:
            h, w = h // 2, w // 2
        elif lay["op"] == OP_FC:
            c, h, w = lay["w_q"].shape[0], 1, 1
        else:
            shapes.append((c, h, w))
            continue
        shapes.append((c, h, w))
        regions.append((abuf_cursor, c * h * w))
        abuf_cursor = align(abuf_cursor + c * h * w, 0x100)
    assert abuf_cursor <= 0x8000, "activations do not fit"

    # weight buffer image
    wbuf = bytearray()
    w_off = {}
    for lay in layers:
        if lay["op"] in (OP_CONV2D, OP_FC):
            wq = lay["w_q"]
            cout = wq.shape[0]
            w4 = wq.reshape(cout, -1, *([1, 1] if lay["op"] == OP_FC else [])) if lay["op"] == OP_FC else wq
            if lay["op"] == OP_FC:
                w4 = wq.reshape(cout, wq.shape[1], 1, 1)
            wb = pack_weights(w4.tolist())
            bb = pack_bias(lay["bias"], cout)
            w_base = align(len(wbuf), 0x100)
            wbuf += bytes(w_base - len(wbuf)) + bytes(wb)
            b_base = align(len(wbuf), 0x100)
            wbuf += bytes(b_base - len(wbuf)) + bytes(bb)
            w_off[lay["name"]] = (w_base, b_base, len(wb), len(bb))
    assert len(wbuf) <= 0x20000, "weights do not fit"

    # descriptors
    descs = []
    c, h, w = cin, size, size
    region_idx = 0
    for lay in layers:
        in_base = regions[region_idx][0]
        if lay["op"] == OP_CONV2D:
            wq = lay["w_q"]
            k = wq.shape[2]
            pad = conv1_pad if lay["name"] == "conv1" else 0
            region_idx += 1
            descs.append(make_desc(OP_CONV2D, k=k, pad=pad, in_h=h, in_w=w, c_in=c, c_out=wq.shape[0],
                                   in_base=in_base, out_base=regions[region_idx][0],
                                   w_base=w_off[lay["name"]][0], b_base=w_off[lay["name"]][1],
                                   rq_mult=lay["mult"], rq_shift=lay["shift"], z_in=lay["z_in"], z_out=lay["z_out"],
                                   act_min=lay["act_min"], act_max=lay["act_max"]))
            c, h, w = wq.shape[0], h + 2 * pad - k + 1, w + 2 * pad - k + 1
        elif lay["op"] == OP_MAXPOOL2X2:
            region_idx += 1
            descs.append(make_desc(OP_MAXPOOL2X2, in_h=h, in_w=w, c_in=c, in_base=in_base, out_base=regions[region_idx][0]))
            h, w = h // 2, w // 2
        elif lay["op"] == OP_FC:
            wq = lay["w_q"]
            region_idx += 1
            descs.append(make_desc(OP_FC, k=1, in_h=1, in_w=1, c_in=c * h * w, c_out=wq.shape[0],
                                   in_base=in_base, out_base=regions[region_idx][0],
                                   w_base=w_off[lay["name"]][0], b_base=w_off[lay["name"]][1],
                                   rq_mult=lay["mult"], rq_shift=lay["shift"], z_in=lay["z_in"], z_out=lay["z_out"],
                                   act_min=lay["act_min"], act_max=lay["act_max"]))
            c, h, w = wq.shape[0], 1, 1
        else:
            descs.append(make_desc(OP_ARGMAX, c_in=c, in_base=in_base))
    layout = {"act_regions": regions, "weights": w_off, "wbuf_bytes": len(wbuf), "shapes": shapes}
    return descs, bytes(wbuf), layout


def hexstr(data):
    return "".join(f"{b:02x}" for b in data)


def write_program(path, descs, wbuf, images, labels, cin, size, n_trace):
    """program.txt for the testbench; returns the integer model's predictions."""
    preds = []
    words = []
    padded = wbuf + bytes((-len(wbuf)) % 4)
    for i in range(0, len(padded), 4):
        words.append(padded[i] | (padded[i + 1] << 8) | (padded[i + 2] << 16) | (padded[i + 3] << 24))
    in_base = descs[0]["in_base"]
    with open(path, "w", encoding="ascii") as f:
        f.write(f"LAYERS {len(descs)}\n")
        for i, d in enumerate(descs):
            f.write(f"DESC {i} " + " ".join(str(w) for w in desc_pack(d)) + "\n")
        f.write(f"WORDS {len(words)}\n")
        for w in words:
            f.write(f"{w:08x}\n")
        for idx, (img, label) in enumerate(zip(images, labels)):
            pix = [int(v) for v in img.reshape(-1)]
            q = [(v - 128) & 0xFF for v in pix]          # int8 input, zero point -128
            abuf = bytearray(0x8000)
            abuf[in_base:in_base + len(q)] = bytes(q)
            wb = bytearray(0x20000)
            wb[: len(wbuf)] = wbuf
            res = run_program(descs, abuf, wb)
            preds.append(res["result"])
            f.write(f"IMAGE {idx} {int(label)} {res['result']} {in_base} {len(q)} {hexstr(q)}\n")
            if idx < n_trace:
                for li, (base, n) in enumerate(res["layers"]):
                    if n:
                        f.write(f"TRACE {idx} {li} {base} {n} {hexstr(abuf[base:base + n])}\n")
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", choices=sorted(NETS), required=True)
    ap.add_argument("--src", default=None, help="directory with fp32.npz, calib.npz, test.npz (default model/out/<net>)")
    ap.add_argument("--out", default=None, help="output directory (default model/data/<net>)")
    ap.add_argument("--images", type=int, default=100, help="test images written to program.txt")
    ap.add_argument("--report", type=int, default=1000, help="test images used for the accuracy numbers (pure Python, about 0.15 s per image)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = args.src or os.path.join(here, "out", args.net)
    out = args.out or os.path.join(here, "data", args.net)
    os.makedirs(out, exist_ok=True)
    cfg = NETS[args.net]

    fp32_path = os.path.join(src, "fp32.npz")
    if not os.path.exists(fp32_path):
        fp32_path = os.path.join(out, "fp32.npz")       # committed float16 copy
    p = {k: v.astype(np.float32) for k, v in np.load(fp32_path).items()}
    calib = np.load(os.path.join(src, "calib.npz"))["x"]
    test = np.load(os.path.join(src, "test.npz"))
    tx, ty = test["x"], test["y"]

    layers = quantize_net(p, calib, cfg["conv1_pad"])
    descs, wbuf, layout = build_program(layers, cfg["cin"], cfg["size"], cfg["conv1_pad"])

    # accuracy of the float model and of the integer program on the report set
    n_rep = min(args.report, len(tx))
    f32_pred = forward_f32(p, tx[:n_rep].astype(np.float32) / 255.0, cfg["conv1_pad"])[-1]
    f32_acc = float((f32_pred == ty[:n_rep]).mean())
    int_preds = []
    for img in tx[:n_rep]:
        q = [(int(v) - 128) & 0xFF for v in img.reshape(-1)]
        abuf = bytearray(0x8000); abuf[descs[0]["in_base"]:descs[0]["in_base"] + len(q)] = bytes(q)
        wb = bytearray(0x20000); wb[: len(wbuf)] = wbuf
        int_preds.append(run_program(descs, abuf, wb)["result"])
    int_acc = float((np.array(int_preds) == ty[:n_rep]).mean())

    n_img = min(args.images, len(tx))
    preds = write_program(os.path.join(out, "program.txt"), descs, wbuf, tx[:n_img], ty[:n_img],
                          cfg["cin"], cfg["size"], TRACE_IMAGES)
    np.savez_compressed(os.path.join(out, "fp32.npz"), **{k: v.astype(np.float16) for k, v in p.items()})

    info = {
        "net": args.net, "input": {"cin": cfg["cin"], "size": cfg["size"], "scale": 1 / 255, "zero_point": -128},
        "layers": [{k: v for k, v in lay.items() if k not in ("w_q", "bias")} for lay in layers],
        "descriptors": [desc_pack(d) for d in descs],
        "layout": {"act_regions": layout["act_regions"], "weights": layout["weights"],
                   "wbuf_bytes": layout["wbuf_bytes"], "shapes": layout["shapes"]},
        "accuracy": {"report_images": n_rep, "float32": f32_acc, "int8_program": int_acc,
                     "program_images": n_img, "int8_program_on_program_images": float(np.mean(np.array(preds) == ty[:n_img]))},
    }
    with open(os.path.join(out, "net.json"), "w", encoding="ascii") as f:
        json.dump(info, f, indent=1)
    print(f"{args.net}: float32 accuracy {f32_acc:.4f}, int8 program accuracy {int_acc:.4f} on {n_rep} images; "
          f"weights {layout['wbuf_bytes']} bytes; program.txt with {n_img} images written to {out}")


if __name__ == "__main__":
    main()
