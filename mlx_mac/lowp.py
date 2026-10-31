"""Simulated low-precision training for the activation x format study (Mac, MLX).

Every hidden Linear (qkv, o, fc, proj) becomes a QLinear: all three GEMMs quantize their operands along the contraction
axis, as low-precision hardware does (fake quantization: values rounded to the format, the GEMM in fp32, so the numerics
match the format but there is no speedup):
  fprop  y  = Q_x(x) @ Q_w(W)^T                      (contraction over d_in)
  dgrad  dx = Q_g(dy) @ Q_w(W^T)^T                   (over d_out)
  wgrad  dW = Q_g(dy^T) @ Q_x(x^T)^T                 (over tokens; optional random Hadamard transform on both sides)
Gradients use stochastic rounding (SR) where the recipe says so. Embedding, head, attention and norms stay fp32 (the
papers keep the first/last layers in high precision too).

Recipes (arm flag "+q=<recipe>"):
  fp8     e4m3 x, w; e5m2 g; per-tensor current amax scale
  int8    int8 per row (= per token for x/g, per output channel for W), symmetric; SR on g (TPU / AQT-style)
  mxfp4   e2m1 in 32-blocks with a power-of-two (E8M0) scale; SR on g
  nvfp4   e2m1 in 16-blocks with an e4m3 scale under a per-tensor scale; SR on g
  nvfp4h  nvfp4 + a 16-point random Hadamard transform on the wgrad operands (NVIDIA's recipe, arXiv 2509.25149)
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _round(a, sr):
    if sr:
        return mx.floor(a + mx.random.uniform(shape=a.shape, dtype=a.dtype))
    return mx.round(a)


def _float_q(a, mant, emin, amaxv, sr):
    """Round |a| to a float format with `mant` mantissa bits, min normal exponent emin, max value amaxv."""
    m = mx.abs(a)
    e = mx.maximum(mx.floor(mx.log2(mx.maximum(m, 2.0 ** (emin - mant - 4)))), float(emin))
    step = mx.power(2.0, e - mant)
    return mx.sign(a) * mx.minimum(_round(m / step, sr) * step, amaxv)


def e4m3(a, sr=False):
    return _float_q(a, 3, -6, 448.0, sr)


def e5m2(a, sr=False):
    return _float_q(a, 2, -14, 57344.0, sr)


def e2m1(a, sr=False):
    return _float_q(a, 1, 0, 6.0, sr)


def _blocks(t, k):
    return t.reshape(*t.shape[:-1], t.shape[-1] // k, k)


def quant(t, fmt, sr=False):
    """Fake-quantize t along its last axis (the contraction axis of the GEMM it feeds)."""
    if fmt == "none":
        return t
    if fmt in ("e4m3", "e5m2"):
        hi = 448.0 if fmt == "e4m3" else 57344.0
        s = mx.maximum(mx.max(mx.abs(t)), 1e-30) / hi
        return (e4m3 if fmt == "e4m3" else e5m2)(t / s, sr) * s
    if fmt == "int8":
        s = mx.maximum(mx.max(mx.abs(t), axis=-1, keepdims=True), 1e-30) / 127.0
        return mx.clip(_round(t / s, sr), -127, 127) * s
    if fmt == "mxfp4":
        b = _blocks(t, 32)
        amax = mx.maximum(mx.max(mx.abs(b), axis=-1, keepdims=True), 1e-30)
        s = mx.power(2.0, mx.ceil(mx.log2(amax / 6.0)))
        return (e2m1(b / s, sr) * s).reshape(t.shape)
    if fmt == "nvfp4":
        b = _blocks(t, 16)
        sb = mx.maximum(mx.max(mx.abs(b), axis=-1, keepdims=True), 1e-30) / 6.0
        g = mx.maximum(mx.max(sb), 1e-30) / 448.0
        s = mx.maximum(e4m3(sb / g) * g, 1e-30)
        return (e2m1(b / s, sr) * s).reshape(t.shape)
    raise ValueError(fmt)


RECIPES = {   # (x / w format, gradient format, SR on gradients, Hadamard on wgrad)
    "none": ("none", "none", False, False),
    "fp8": ("e4m3", "e5m2", False, False),
    "int8": ("int8", "int8", True, False),
    "mxfp4": ("mxfp4", "mxfp4", True, False),
    "nvfp4": ("nvfp4", "nvfp4", True, False),
    "nvfp4h": ("nvfp4", "nvfp4", True, True),
}


def _hadamard16(seed=0):
    h = np.array([[1.0]])
    while h.shape[0] < 16:
        h = np.block([[h, h], [h, -h]])
    d = np.random.default_rng(seed).choice([-1.0, 1.0], 16)
    return mx.array((h * d[None, :] / 4.0).astype(np.float32))   # orthogonal: R @ R.T = I


R16 = _hadamard16()


def _rht(t):
    return (_blocks(t, 16) @ R16.T).reshape(t.shape)   # the same R on both wgrad operands cancels: (gR)(xR)^T = g x^T


def make_qmm(recipe):
    fxw, fg, sr, had = RECIPES[recipe]

    @mx.custom_function
    def qmm(x, w):
        return quant(x, fxw) @ quant(w, fxw).T

    @qmm.vjp
    def qmm_vjp(primals, cotangent, output):
        x, w = primals
        g = cotangent
        dx = quant(g, fg, sr) @ quant(w.T, fxw).T
        g2 = g.reshape(-1, g.shape[-1]).T     # (d_out, tokens)
        x2 = x.reshape(-1, x.shape[-1]).T     # (d_in, tokens)
        if had:
            g2, x2 = _rht(g2), _rht(x2)
        dw = quant(g2, fg, sr) @ quant(x2, fxw).T
        return dx, dw

    return qmm


class QLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear: same `weight` parameter (so the init and the optimizer paths are unchanged)."""

    def __init__(self, lin: nn.Linear, recipe: str):
        super().__init__()
        self.weight = lin.weight
        self.recipe = recipe
        self._qmm = make_qmm(recipe)

    def __call__(self, x):
        return self._qmm(x, self.weight)


def uses_random(recipe):
    return RECIPES[recipe][2]
