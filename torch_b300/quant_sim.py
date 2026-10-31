"""Fake-quantized low-precision GEMMs in PyTorch: a port of anvil_r/mlx_gpt/lowp.py (same formulas, same Hadamard signs),
so the Mac, TPU and GPU simulations agree bit for bit in round-to-nearest mode. Runs on any device. On a B300 it is
the cross-check of real NVFP4 (Transformer Engine) against the simulation, and the only path for MXFP4 / INT8 here.

All three GEMMs of a Linear quantize their operands along the contraction axis, as hardware does:
  fprop  y  = Q_x(x) @ Q_w(W)^T          dgrad  dx = Q_g(dy) @ Q_w(W^T)^T          wgrad  dW = Q_g(dy^T) @ Q_x(x^T)^T
"""
from __future__ import annotations

import math

import numpy as np
import torch

E2M1_MAX, E4M3_MAX, E5M2_MAX = 6.0, 448.0, 57344.0


def _round(a, sr):
    return torch.floor(a + torch.rand_like(a)) if sr else torch.round(a)


def _float_q(a, mant, emin, amaxv, sr):
    m = a.abs()
    e = torch.clamp(torch.floor(torch.log2(torch.clamp(m, min=2.0 ** (emin - mant - 4)))), min=float(emin))
    step = torch.pow(2.0, e - mant)
    return torch.sign(a) * torch.clamp(_round(m / step, sr) * step, max=amaxv)


def e4m3(a, sr=False):
    return _float_q(a, 3, -6, E4M3_MAX, sr)


def e5m2(a, sr=False):
    return _float_q(a, 2, -14, E5M2_MAX, sr)


def e2m1(a, sr=False):
    return _float_q(a, 1, 0, E2M1_MAX, sr)


def _blocks(t, k):
    return t.reshape(*t.shape[:-1], t.shape[-1] // k, k)


def quant(t, fmt, sr=False):
    """Fake-quantize t along its last axis. Formats: none, e4m3, e5m2 (per tensor), int8 (per row), mxfp4, nvfp4."""
    if fmt == "none":
        return t
    if fmt in ("e4m3", "e5m2"):
        hi = E4M3_MAX if fmt == "e4m3" else E5M2_MAX
        s = torch.clamp(t.abs().max(), min=1e-30) / hi
        return (e4m3 if fmt == "e4m3" else e5m2)(t / s, sr) * s
    if fmt == "int8":
        s = torch.clamp(t.abs().amax(dim=-1, keepdim=True), min=1e-30) / 127.0
        return torch.clamp(_round(t / s, sr), -127, 127) * s
    if fmt == "mxfp4":   # OCP MX: 32-blocks, power-of-two (E8M0) scale rounded up so nothing saturates
        b = _blocks(t, 32)
        amax = torch.clamp(b.abs().amax(dim=-1, keepdim=True), min=1e-30)
        s = torch.pow(2.0, torch.ceil(torch.log2(amax / E2M1_MAX)))
        return (e2m1(b / s, sr) * s).reshape(t.shape)
    if fmt == "nvfp4":   # 16-blocks, E4M3 block scale under a per-tensor fp32 scale (NVIDIA's two-level scheme)
        b = _blocks(t, 16)
        sb = torch.clamp(b.abs().amax(dim=-1, keepdim=True), min=1e-30) / E2M1_MAX
        g = torch.clamp(sb.max(), min=1e-30) / E4M3_MAX
        s = torch.clamp(e4m3(sb / g) * g, min=1e-30)
        return (e2m1(b / s, sr) * s).reshape(t.shape)
    raise ValueError(fmt)


def hadamard16(seed=0):
    h = np.array([[1.0]])
    while h.shape[0] < 16:
        h = np.block([[h, h], [h, -h]])
    d = np.random.default_rng(seed).choice([-1.0, 1.0], 16)   # identical signs to lowp.py's R16
    return torch.tensor(h * d[None, :] / 4.0, dtype=torch.float32)


_R16 = {}


def _rht(t):
    key = t.device
    if key not in _R16:
        _R16[key] = hadamard16().to(t.device)
    return (_blocks(t, 16) @ _R16[key].T).reshape(t.shape)


class _QMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, fxw, fg, sr, had):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32, w32 = x.float(), w.float()
            ctx.save_for_backward(x32, w32)
            ctx.cfg = (fxw, fg, sr, had)
            ctx.dtypes = (x.dtype, w.dtype)
            return quant(x32, fxw) @ quant(w32, fxw).T

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        fxw, fg, sr, had = ctx.cfg
        with torch.autocast(device_type=g.device.type, enabled=False):
            g = g.float()
            dx = quant(g, fg, sr) @ quant(w.T.contiguous(), fxw).T
            g2 = g.reshape(-1, g.shape[-1]).T.contiguous()     # (d_out, tokens)
            x2 = x.reshape(-1, x.shape[-1]).T.contiguous()     # (d_in, tokens)
            if had:
                g2, x2 = _rht(g2), _rht(x2)
            dw = quant(g2, fg, sr) @ quant(x2, fxw).T
        return dx.to(ctx.dtypes[0]), dw.to(ctx.dtypes[1]), None, None, None, None


class SimLinear(torch.nn.Module):
    """Bias-free Linear whose three GEMMs use the simulated recipe (config.SIM_RECIPES)."""

    def __init__(self, d_in, d_out, recipe_tuple):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_in))
        self.recipe = recipe_tuple

    def forward(self, x):
        return _QMM.apply(x, self.weight, *self.recipe)
