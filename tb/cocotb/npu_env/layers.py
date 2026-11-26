"""Random single layers and multi-layer programs that the golden model accepts.

A layer is a dict: desc (descriptor dict), loads (list of (buffer, offset,
bytes)) and out (offset, length) of its output in act_buffer. Sizes are kept
small so that loading them over AXI4-Lite stays fast in simulation.
"""
import npu_ref as ref


def _tensor(rng, n, lo=-128, hi=127):
    return bytes(rng.randint(lo, hi) & 0xFF for _ in range(n))


def _requant(rng, k_total, relu):
    mult, shift = ref.quantize_multiplier(rng.uniform(0.5, 2.0) * 128.0 / max(1.0, k_total * 40.0))
    if rng.random() < 0.15:                 # outside the usual range: a large or a small shift
        shift = rng.choice([rng.randint(1, 15), rng.randint(48, 62)])
    z_out = rng.randint(-128, 127)
    act_min, act_max = (z_out, 127) if relu else (-128, 127)
    return mult, shift, z_out, act_min, act_max


def mac_layer(rng, fc=False, in_base=None, out_base=None, shape=None, **fix):
    """CONV2D or FC with its input, weights and biases. fix overrides c_in, h, w, k, pad, c_out."""
    if fc:
        c_in = shape[0] * shape[1] * shape[2] if shape else fix.get("c_in", rng.randint(1, 64))
        k, pad, h, w = 1, 0, 1, 1
    else:
        c_in, h, w = shape if shape else (fix.get("c_in", rng.randint(1, 4)), fix.get("h", rng.randint(1, 10)),
                                          fix.get("w", rng.randint(1, 10)))
        k = fix.get("k", rng.randint(1, 7))
        while c_in * k * k > 128 and "k" not in fix:
            k -= 1
        pad = fix.get("pad", rng.randint(0, 3))
        pad = max(pad, (k - min(h, w) + 1) // 2)          # output at least 1 x 1
        if pad > 7:
            k, pad = 1, 0
    c_out = fix.get("c_out", rng.randint(1, 20))
    k_total = c_in * k * k
    mult, shift, z_out, act_min, act_max = _requant(rng, k_total, rng.random() < 0.5)
    n_pad = ref.pad_to_arr(c_out)
    oh, ow = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    in_base = rng.randrange(0, 0x1000) if in_base is None else in_base
    out_base = 0x4000 + rng.randrange(0, 0x800) if out_base is None else out_base
    w_base = rng.randrange(0, 0x8000, 8)
    b_base = 0x18000 + rng.randrange(0, 0x1000, 8)
    weights = [[[[rng.randint(-128, 127) for _ in range(k)] for _ in range(k)] for _ in range(c_in)]
               for _ in range(c_out)]
    bias = [rng.randint(-k_total * 64, k_total * 64) for _ in range(c_out)]
    desc = ref.make_desc(ref.OP_FC if fc else ref.OP_CONV2D, k=k, pad=pad, in_h=h, in_w=w, c_in=c_in, c_out=c_out,
                         in_base=in_base, out_base=out_base, w_base=w_base, b_base=b_base, rq_mult=mult,
                         rq_shift=shift, z_in=rng.choice([0, rng.randint(-128, 127)]), z_out=z_out,
                         act_min=act_min, act_max=act_max)
    loads = [("wbuf", w_base, bytes(ref.pack_weights(weights))), ("wbuf", b_base, bytes(ref.pack_bias(bias, c_out)))]
    if shape is None:
        loads.append(("abuf", in_base, _tensor(rng, c_in * h * w)))
    assert ref.desc_check(desc) == ref.ERR_NONE, desc
    shape_out = (c_out, 1, 1) if fc else (c_out, oh, ow)
    return {"desc": desc, "loads": loads, "out": (out_base, c_out * oh * ow), "shape": shape_out}


def pool_layer(rng, in_base=None, out_base=None, shape=None, h=None, w=None):
    c, h, w = shape if shape else (rng.randint(1, 6), h or rng.randint(2, 13), w or rng.randint(2, 13))
    in_base = rng.randrange(0, 0x1000) if in_base is None else in_base
    out_base = 0x4000 + rng.randrange(0, 0x800) if out_base is None else out_base
    desc = ref.make_desc(ref.OP_MAXPOOL2X2, in_h=h, in_w=w, c_in=c, in_base=in_base, out_base=out_base)
    loads = [] if shape else [("abuf", in_base, _tensor(rng, c * h * w))]
    assert ref.desc_check(desc) == ref.ERR_NONE, desc
    return {"desc": desc, "loads": loads, "out": (out_base, c * (h // 2) * (w // 2)), "shape": (c, h // 2, w // 2)}


def argmax_layer(rng, in_base=None, n=None, tie=False):
    n = n or rng.choice([1, 16, rng.randint(2, 15)])
    in_base = rng.randrange(0, 0x1000) if in_base is None else in_base
    logits = [rng.randint(-128, 127) for _ in range(n)]
    if tie and n > 1:
        i, j = rng.sample(range(n), 2)
        logits[i] = logits[j] = max(logits)
    desc = ref.make_desc(ref.OP_ARGMAX, c_in=n, in_base=in_base)
    return {"desc": desc, "loads": [("abuf", in_base, bytes(v & 0xFF for v in logits))], "out": None, "shape": None}


def corner_layers(rng):
    """Directed layers for the coverage corners random stimulus reaches only by luck."""
    lays = [mac_layer(rng, k=k, c_in=1 if k > 4 else 2, h=8, w=8, pad=rng.randint(0, 3)) for k in range(1, 8)]
    lays += [mac_layer(rng, k=1, c_in=2, pad=p, h=5, w=6) for p in range(4)]
    lays += [mac_layer(rng, fc=True, c_in=8, c_out=8), mac_layer(rng, k=2, c_in=2, c_out=16, h=4, w=4),
             mac_layer(rng, fc=True, c_in=3, c_out=13)]
    lays += [pool_layer(rng, h=h, w=w) for h, w in ((4, 6), (4, 7), (5, 6), (5, 7))]
    lays += [argmax_layer(rng, n=n) for n in (1, 16, 9)]
    return lays


def random_layer(rng):
    kind = rng.choices(["conv", "fc", "pool", "argmax"], [4, 2, 2, 1])[0]
    if kind == "conv":
        return mac_layer(rng)
    if kind == "fc":
        return mac_layer(rng, fc=True)
    if kind == "pool":
        return pool_layer(rng)
    return argmax_layer(rng, tie=rng.random() < 0.3)


def chain(rng, n_layers):
    """A program whose layers feed each other, ping-ponging between two act_buffer regions."""
    regions = [0x0000, 0x4000]
    first = mac_layer(rng, in_base=regions[0], out_base=regions[1])
    layers = [first]
    shape = first["shape"]
    for i in range(1, n_layers):
        src, dst = regions[i % 2], regions[(i + 1) % 2]
        c, h, w = shape
        last = i == n_layers - 1
        if last and c * h * w <= 16:
            layers.append({"desc": ref.make_desc(ref.OP_ARGMAX, c_in=c * h * w, in_base=src), "loads": [],
                           "out": None, "shape": None})
            break
        if h >= 2 and w >= 2 and rng.random() < 0.4:
            lay = pool_layer(rng, src, dst, shape)
        elif c * h * w <= 64 and rng.random() < 0.5:
            lay = mac_layer(rng, fc=True, in_base=src, out_base=dst, shape=shape)
        elif c <= 4:
            lay = mac_layer(rng, in_base=src, out_base=dst, shape=shape)
        else:
            lay = {"desc": ref.make_desc(ref.OP_NOP), "loads": [], "out": layers[-1]["out"], "shape": shape}
        layers.append(lay)
        if lay["desc"]["op"] != ref.OP_NOP:
            shape = lay["shape"]
        else:
            regions.reverse()      # a NOP does not move the data: keep the next layer reading it
    return layers
