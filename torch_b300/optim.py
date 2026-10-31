"""Muon and Adam with mlx.optimizers semantics (as in the Mac grid): Muon = momentum 0.95, Nesterov, quintic Newton-Schulz
(3.4445, -4.7750, 2.0315) x 5 on X / ||X||_F in fp32, step lr * sqrt(max(1, rows / cols)); Adam without bias correction.

Same-shape matrices are stacked and orthogonalized in one batched Newton-Schulz, and Adam uses foreach ops: identical math
per matrix, ~4x fewer kernel launches (the B300 host CPU, not the GPU, limits the small model's step time)."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import ADAM, MUON  # noqa: E402


@torch.no_grad()
def newton_schulz(G):
    """G: (..., rows, cols); each trailing matrix is normalized by its own Frobenius norm."""
    X = G.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.mT
    X = X / (torch.linalg.matrix_norm(X, keepdim=True) + 1e-7)
    a, b, c = MUON["ns_coeffs"]
    for _ in range(MUON["ns_steps"]):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


class MuonAdam:
    def __init__(self, muon_params, adam_params):
        groups = {}
        for p in muon_params:
            groups.setdefault(tuple(p.shape), []).append(p)
        self.muon = [(ps, torch.zeros((len(ps),) + shape, device=ps[0].device, dtype=ps[0].dtype))
                     for shape, ps in groups.items()]
        self.adam_p = list(adam_params)
        self.adam_m = [torch.zeros_like(p) for p in self.adam_p]
        self.adam_v = [torch.zeros_like(p) for p in self.adam_p]

    @torch.no_grad()
    def step(self, lr_muon, lr_adam):
        mu = MUON["momentum"]
        for ps, v in self.muon:
            g = torch.stack([p.grad for p in ps])
            v.mul_(mu).add_(g, alpha=1 - mu)
            u = g * (1 - mu) + v * mu if MUON["nesterov"] else v
            scale = max(1.0, ps[0].shape[0] / ps[0].shape[1]) ** 0.5
            upd = newton_schulz(u).to(ps[0].dtype)
            torch._foreach_add_(ps, list(upd.unbind(0)), alpha=-lr_muon * scale)
        if self.adam_p:
            b1, b2, eps = ADAM["b1"], ADAM["b2"], ADAM["eps"]
            gs = [p.grad for p in self.adam_p]
            torch._foreach_mul_(self.adam_m, b1)
            torch._foreach_add_(self.adam_m, gs, alpha=1 - b1)
            torch._foreach_mul_(self.adam_v, b2)
            torch._foreach_addcmul_(self.adam_v, gs, gs, value=1 - b2)
            den = torch._foreach_sqrt(self.adam_v)
            torch._foreach_add_(den, eps)
            torch._foreach_addcdiv_(self.adam_p, self.adam_m, den, value=-lr_adam)

    def zero_grad(self):
        for ps, _ in self.muon:
            for p in ps:
                p.grad = None
        for p in self.adam_p:
            p.grad = None
