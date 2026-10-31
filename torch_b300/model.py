"""The study's GPT in PyTorch, structurally identical to anvil_r/mlx_gpt/train_mlx.py (the Mac grid):
pre-norm blocks (RMSNorm without weights), QK-norm + RoPE, causal SDPA, zero-init output projections, learned residual
scales, softcapped logits; the MLP activation and the hidden-matrix Linear type (recipe) are the factors under study.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import HEAD_STD, NORM_EPS, ROPE_BASE, SOFTCAP, XIELU, hidden_for  # noqa: E402


def rms(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + NORM_EPS).to(x.dtype)


def _inv_softplus(y):
    return math.log(math.expm1(y))


def activation(act, h, xielu_raw=None):
    if act == "relu2":
        return F.relu(h).square()
    if act == "lrelu2":
        return F.leaky_relu(h, 0.5).square()
    if act == "xielu":
        ap = F.softplus(xielu_raw[0])
        an = XIELU["beta"] + F.softplus(xielu_raw[1])
        b = XIELU["beta"]
        return torch.where(h > 0, ap * h * h + b * h, (torch.expm1(torch.clamp(h, max=XIELU["eps"])) - h) * an + b * h)
    u, v = h.chunk(2, dim=-1)
    if act == "swiglu":
        return F.silu(u) * v
    if act == "s2glu_r2":
        return 2.0 * torch.sigmoid(u) * F.relu(v).square()
    if act == "eglu_r2n":
        return (torch.exp(torch.tanh(u)) - 0.36787944) * 0.42545906 * F.relu(v).square()
    if act == "eml":
        return torch.exp(torch.tanh(u)) - torch.log1p(F.relu(v))
    raise ValueError(act)


class Rotary(nn.Module):
    def __init__(self, hd, seq):
        super().__init__()
        inv = ROPE_BASE ** (-torch.arange(0, hd // 2, dtype=torch.float32) * 2 / hd)
        f = torch.outer(torch.arange(seq, dtype=torch.float32), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x):   # x (B, H, T, hd); non-interleaved halves, as mx.fast.rope(traditional=False)
        T, h = x.shape[-2], x.shape[-1] // 2
        c, s = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Block(nn.Module):
    def __init__(self, d, n_head, act, make_linear):
        super().__init__()
        self.n_head, self.act = n_head, act
        self.qkv = make_linear(d, 3 * d)
        self.o = make_linear(d, d)
        fc_out, proj_in = hidden_for(act, d)
        self.fc = make_linear(d, fc_out)
        self.proj = make_linear(proj_in, d)
        self.lam = nn.Parameter(torch.ones(2))
        if act == "xielu":
            self.xielu_raw = nn.Parameter(torch.tensor([_inv_softplus(XIELU["alpha_p"]),
                                                        _inv_softplus(XIELU["alpha_n"] - XIELU["beta"])]))

    def forward(self, x, rotary):
        B, T, D = x.shape
        H, hd = self.n_head, D // self.n_head
        h = rms(x)
        q, k, v = self.qkv(h).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = rotary(rms(q)), rotary(rms(k))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd ** -0.5)
        x = x + self.lam[0] * self.o(a.transpose(1, 2).reshape(B, T, D))
        g = activation(self.act, self.fc(rms(x)), getattr(self, "xielu_raw", None))
        return x + self.lam[1] * self.proj(g)


class GPT(nn.Module):
    def __init__(self, cfg, act, make_linear):
        super().__init__()
        d, self.vocab = cfg["d"], cfg["vocab"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList([Block(d, cfg["heads"], act, make_linear) for _ in range(cfg["layers"])])
        self.head = nn.Linear(d, cfg["vocab"], bias=False)
        self.rotary = Rotary(d // cfg["heads"], cfg["seq"])

    def forward(self, idx):
        x = rms(self.embed(idx))
        for b in self.blocks:
            x = b(x, self.rotary)
        logits = self.head(rms(x)).float()
        return SOFTCAP * torch.tanh(logits / SOFTCAP)


@torch.no_grad()
def init_weights(model: GPT, seed: int):
    """Same init whatever the Linear type (nn / Transformer Engine / simulated): MLX's defaults, in a fixed order."""
    g = torch.Generator().manual_seed(seed)
    d = model.embed.weight.shape[1]
    model.embed.weight.copy_(torch.randn(model.embed.weight.shape, generator=g) * d ** -0.5)
    for b in model.blocks:
        for name in ("qkv", "o", "fc", "proj"):
            w = getattr(b, name).weight
            if name in ("o", "proj"):
                w.zero_()
            else:
                bound = w.shape[1] ** -0.5
                w.copy_((torch.rand(w.shape, generator=g) * 2 - 1) * bound)
    model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=g) * HEAD_STD)


def hidden_matrices(model: GPT):
    """The parameters Muon updates: the blocks' 2-D weights."""
    return [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
