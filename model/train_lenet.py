#!/usr/bin/env python3
"""Train float32 LeNet-5 for MNIST or CIFAR-10 with numpy only, and dump the
checkpoint, a calibration set and a test subset for quantize.py.

Optional: the committed model/data/<net>/ files already contain everything the
simulations need. Run this only to regenerate them from scratch.

Network (docs/programming_model.md section 6):
    conv1 5x5 (pad 2 for MNIST, 0 for CIFAR-10) -> ReLU -> maxpool 2x2
    conv2 5x5 -> ReLU -> maxpool 2x2 -> fc 400-120 -> ReLU -> fc 120-84 -> ReLU -> fc 84-10
Inputs are pixel / 255 with no further normalization, so that the integer
input is pixel - 128 with scale 1/255 and zero point -128.

Datasets: MNIST is fetched as mnist.npz from the Keras mirror; CIFAR-10 is
fetched as cifar-10-python.tar.gz from the University of Toronto, or taken from
--data if already present (download it by hand if the network is restricted).
"""
import argparse
import os
import pickle
import tarfile
import urllib.request

import numpy as np

NETS = {"mnist": (1, 28, 2), "cifar10": (3, 32, 0)}   # cin, size, conv1 pad
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mnist(data_dir):
    path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(path):
        print("downloading", MNIST_URL)
        urllib.request.urlretrieve(MNIST_URL, path)
    d = np.load(path)
    return d["x_train"][:, None], d["y_train"].astype(np.int64), d["x_test"][:, None], d["y_test"].astype(np.int64)


def load_cifar10(data_dir):
    path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(path):
        print("downloading", CIFAR_URL)
        urllib.request.urlretrieve(CIFAR_URL, path)
    xs, ys, xt, yt = [], [], None, None
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name.startswith("data_batch_") or name == "test_batch":
                d = pickle.load(tf.extractfile(m), encoding="bytes")
                x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
                y = np.array(d[b"labels"], dtype=np.int64)
                if name == "test_batch":
                    xt, yt = x, y
                else:
                    xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), xt, yt


# ---------------------------------------------------------------------------
# Layers with backward passes (batch-first, float32)
# ---------------------------------------------------------------------------
def im2col(x, k, pad):
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))   # n, c, oh, ow, k, k
    cols = win.transpose(0, 1, 4, 5, 2, 3).reshape(n, c * k * k, oh * ow)
    return np.ascontiguousarray(cols), oh, ow


def col2im(dcols, x_shape, k, pad, oh, ow):
    n, c, h, w = x_shape
    d = dcols.reshape(n, c, k, k, oh, ow)
    dxp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=np.float32)
    for kh in range(k):
        for kw in range(k):
            dxp[:, :, kh:kh + oh, kw:kw + ow] += d[:, :, kh, kw]
    return dxp[:, :, pad:pad + h, pad:pad + w]


class Conv:
    def __init__(self, cin, cout, k, pad, rng):
        self.k, self.pad = k, pad
        self.w = (rng.standard_normal((cout, cin * k * k)) * np.sqrt(2.0 / (cin * k * k))).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
        self.params = [self.w, self.b]
        self.dw, self.db = np.zeros_like(self.w), np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape
        self.cols, self.oh, self.ow = im2col(x, self.k, self.pad)
        out = np.einsum("ok,nkp->nop", self.w, self.cols) + self.b[None, :, None]
        return out.reshape(x.shape[0], -1, self.oh, self.ow)

    def backward(self, dout):
        n = dout.shape[0]
        d = dout.reshape(n, dout.shape[1], -1)
        self.dw = np.einsum("nop,nkp->ok", d, self.cols)
        self.db = d.sum(axis=(0, 2))
        dcols = np.einsum("ok,nop->nkp", self.w, d)
        return col2im(dcols, self.x_shape, self.k, self.pad, self.oh, self.ow)

    def grads(self):
        return [self.dw, self.db]


class FC:
    def __init__(self, cin, cout, rng):
        self.w = (rng.standard_normal((cout, cin)) * np.sqrt(2.0 / cin)).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
        self.params = [self.w, self.b]
        self.dw, self.db = np.zeros_like(self.w), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.w.T + self.b

    def backward(self, dout):
        self.dw = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.w

    def grads(self):
        return [self.dw, self.db]


class ReLU:
    params = []

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2:
    params = []

    def forward(self, x):
        n, c, h, w = x.shape
        self.x_shape = x.shape
        xr = x[:, :, : h // 2 * 2, : w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2)
        out = xr.max(axis=(3, 5))
        self.mask = (xr == out[:, :, :, None, :, None])
        return out

    def backward(self, dout):
        n, c, h, w = self.x_shape
        d = self.mask * dout[:, :, :, None, :, None]
        dx = np.zeros(self.x_shape, dtype=np.float32)
        dx[:, :, : h // 2 * 2, : w // 2 * 2] = d.reshape(n, c, h // 2 * 2, w // 2 * 2)
        return dx


class Flatten:
    params = []

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class LeNet5:
    def __init__(self, cin, pad, rng):
        self.conv1, self.conv2 = Conv(cin, 6, 5, pad, rng), Conv(6, 16, 5, 0, rng)
        self.fc1, self.fc2, self.fc3 = FC(400, 120, rng), FC(120, 84, rng), FC(84, 10, rng)
        self.layers = [self.conv1, ReLU(), MaxPool2(), self.conv2, ReLU(), MaxPool2(), Flatten(),
                       self.fc1, ReLU(), self.fc2, ReLU(), self.fc3]
        self.named = {"conv1": self.conv1, "conv2": self.conv2, "fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3}

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def param_list(self):
        ps, gs = [], []
        for layer in self.layers:
            if layer.params:
                ps += layer.params
                gs += layer.grads()
        return ps, gs


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    return loss, (dlogits / len(y)).astype(np.float32)


def evaluate(model, x, y, batch=500):
    correct = 0
    for i in range(0, len(x), batch):
        logits = model.forward(x[i:i + batch].astype(np.float32) / 255.0)
        correct += int((logits.argmax(axis=1) == y[i:i + batch]).sum())
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", choices=sorted(NETS), required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default=None, help="dataset directory (default model/out/datasets)")
    ap.add_argument("--out", default=None, help="output directory (default model/out/<net>)")
    ap.add_argument("--calib", type=int, default=500)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cin, size, pad = NETS[args.net]
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(here, "out", "datasets")
    out = args.out or os.path.join(here, "out", args.net)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    epochs = args.epochs or (4 if args.net == "mnist" else 20)

    xtr, ytr, xte, yte = load_mnist(data_dir) if args.net == "mnist" else load_cifar10(data_dir)
    model = LeNet5(cin, pad, rng)
    params, _ = model.param_list()
    opt = Adam(params, lr=args.lr)
    steps = len(xtr) // args.batch
    for ep in range(epochs):
        if ep and ep % max(1, epochs // 3) == 0:
            opt.lr *= 0.5
        perm = rng.permutation(len(xtr))
        loss_sum = 0.0
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            x = xtr[idx].astype(np.float32) / 255.0
            logits = model.forward(x)
            loss, dlogits = softmax_ce(logits, ytr[idx])
            model.backward(dlogits)
            params, grads = model.param_list()
            opt.step(params, grads)
            loss_sum += loss
        print(f"epoch {ep + 1}/{epochs}: loss {loss_sum / steps:.4f}, test accuracy {evaluate(model, xte, yte):.4f}", flush=True)

    ck = {}
    for name, layer in model.named.items():
        w = layer.w
        if name.startswith("conv"):
            w = w.reshape(w.shape[0], -1, 5, 5)
        ck[name + ".w"] = w.astype(np.float32)
        ck[name + ".b"] = layer.b.astype(np.float32)
    np.savez(os.path.join(out, "fp32.npz"), **ck)
    cidx = rng.permutation(len(xtr))[: args.calib]
    np.savez_compressed(os.path.join(out, "calib.npz"), x=xtr[cidx].astype(np.uint8))
    np.savez_compressed(os.path.join(out, "test.npz"), x=xte[: args.test].astype(np.uint8), y=yte[: args.test])
    print(f"wrote fp32.npz, calib.npz ({args.calib} images), test.npz ({min(args.test, len(xte))} images) to {out}")


if __name__ == "__main__":
    main()
