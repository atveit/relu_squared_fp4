"""Steady-state step time of the trainer, split into forward+backward and optimizer (median over --iters after warm-up).

  python bench_step.py --config tiny --act relu2 --recipes bf16,nvfp4,sim_nvfp4
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import train as TR  # noqa: E402
from common.config import CONFIGS  # noqa: E402
from model import GPT, hidden_matrices, init_weights  # noqa: E402
from optim import MuonAdam  # noqa: E402


def bench(config, act, recipe, iters, warm):
    dev = torch.device("cuda")
    cfg = dict(CONFIGS[config])
    model = GPT(cfg, act, TR.make_linear_factory(recipe))
    init_weights(model, 1)
    model.to(dev)
    mp = hidden_matrices(model)
    ids = {id(p) for p in mp}
    opt = MuonAdam(mp, [p for p in model.parameters() if id(p) not in ids])
    prec = TR.Precision(recipe, dev)
    x = torch.randint(0, 50257, (cfg["batch"], cfg["seq"]), device=dev)
    y = torch.randint(0, 50257, (cfg["batch"], cfg["seq"]), device=dev)
    tf, to = [], []
    for i in range(warm + iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with prec():
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.step(1e-3, 1e-4)
        opt.zero_grad()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= warm:
            tf.append(1000 * (t1 - t0))
            to.append(1000 * (t2 - t1))
    fb, op = statistics.median(tf), statistics.median(to)
    print(f"{config:<6} {act:<7} {recipe:<11} fwd+bwd {fb:7.1f} ms  opt {op:6.1f} ms  step {fb + op:7.1f} ms  "
          f"tok/s {cfg['batch'] * cfg['seq'] / (fb + op) * 1000:,.0f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--act", default="relu2")
    ap.add_argument("--recipes", default="bf16,nvfp4")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warm", type=int, default=8)
    g = ap.parse_args()
    for r in g.recipes.split(","):
        bench(g.config, g.act, r, g.iters, g.warm)
