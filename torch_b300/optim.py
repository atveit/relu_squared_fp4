"""Muon and Adam with mlx.optimizers semantics (as in the Mac grid): Muon = momentum 0.95, Nesterov, quintic Newton-Schulz
(3.4445, -4.7750, 2.0315) x 5 on X / ||X||_F in fp32, step lr * sqrt(max(1, rows / cols)); Adam without bias correction."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import ADAM, MUON  # noqa: E402


@torch.no_grad()
def newton_schulz(G):
    X = G.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    a, b, c = MUON["ns_coeffs"]
    for _ in range(MUON["ns_steps"]):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if tall else X


class MuonAdam:
    def __init__(self, muon_params, adam_params):
        self.muon = [(p, torch.zeros_like(p)) for p in muon_params]
        self.adam = [(p, torch.zeros_like(p), torch.zeros_like(p)) for p in adam_params]

    @torch.no_grad()
    def step(self, lr_muon, lr_adam):
        mu = MUON["momentum"]
        for p, v in self.muon:
            g = p.grad
            v.mul_(mu).add_(g, alpha=1 - mu)
            u = g * (1 - mu) + v * mu if MUON["nesterov"] else v
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            p.add_(newton_schulz(u).to(p.dtype), alpha=-lr_muon * scale)
        b1, b2, eps = ADAM["b1"], ADAM["b2"], ADAM["eps"]
        for p, m, v in self.adam:
            g = p.grad
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_adam)

    def zero_grad(self):
        for p, *_ in self.muon + self.adam:
            p.grad = None
