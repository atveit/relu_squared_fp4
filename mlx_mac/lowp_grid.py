"""Activation x number-format grid in simulated low-precision training (lowp.py): does the MLP activation decide whether
FP8 / INT8 / FP4 training works?

  d256 L4, N = 600, seeds 1-2, 2 runs at a time (stage1.run_jobs); results/lowp_A.jsonl
  python3 -m mlx_mac.lowp_grid [run|report]

Report: (1) precision penalty = val(arm, recipe) - val(arm, fp32), paired per seed, per activation and recipe; (2) each
activation vs ReLU^2 within the same recipe (does any format change the ranking?).
"""
from __future__ import annotations

import statistics as st
import sys

from .stage1 import SMALL, load, run_jobs

ACTS = ["relu2_metal", "lrelu2", "swiglu", "xielu", "s2glu_r2", "eglu_r2n", "eml_metal"]
# Stress arm: AdamW on every matrix (Muon itself suppresses outliers, arXiv 2506.19697, which may hide activation effects)
STRESS_ACTS, STRESS_RECIPES = ["relu2_metal", "lrelu2", "swiglu"], ["none", "int8", "nvfp4"]
RECIPES = ["none", "fp8", "int8", "mxfp4", "nvfp4", "nvfp4h"]
SEEDS = [1, 2]
N = 600


def arm(act, recipe, opt=None):
    a = act if recipe == "none" else f"{act}+q={recipe}"
    return a if opt is None else f"{a}@opt={opt}"


def run():
    jobs = [(arm(a, r), s, N) for r in RECIPES for s in SEEDS for a in ACTS]
    run_jobs(jobs, "lowp_A", SMALL)
    run_jobs([(arm(a, r, "adamw"), s, N) for r in STRESS_RECIPES for s in SEEDS for a in STRESS_ACTS], "lowp_S", SMALL)


def report():
    res = load()
    v = lambda a, r, s: res.get((arm(a, r), s, N, 256))
    print(f"== Precision penalty: val(recipe) - val(fp32), same activation, paired, N={N} [mnat, mean over seeds]")
    print(f"   {'activation':<14}" + "".join(f"{r:>10}" for r in RECIPES[1:]))
    for a in ACTS:
        cells = []
        for r in RECIPES[1:]:
            d = [1000 * (v(a, r, s) - v(a, "none", s)) for s in SEEDS if v(a, r, s) is not None and v(a, "none", s) is not None]
            cells.append(f"{st.mean(d):+10.1f}" if d else f"{'—':>10}")
        print(f"   {a:<14}" + "".join(cells))
    print(f"\n== Each activation vs ReLU^2 within the same recipe [mnat; + = worse than ReLU^2]")
    print(f"   {'activation':<14}" + "".join(f"{r:>10}" for r in RECIPES))
    for a in ACTS[1:]:
        cells = []
        for r in RECIPES:
            d = [1000 * (v(a, r, s) - v("relu2_metal", r, s)) for s in SEEDS
                 if v(a, r, s) is not None and v("relu2_metal", r, s) is not None]
            cells.append(f"{st.mean(d):+10.1f}" if d else f"{'—':>10}")
        print(f"   {a:<14}" + "".join(cells))


def report_stress():
    res = load()
    v = lambda a, r, s: res.get((arm(a, r, "adamw"), s, N, 256))
    print(f"\n== AdamW stress arm: precision penalty vs the same activation in fp32 (AdamW) [mnat]")
    print(f"   {'activation':<14}" + "".join(f"{r:>10}" for r in STRESS_RECIPES[1:]))
    for a in STRESS_ACTS:
        cells = []
        for r in STRESS_RECIPES[1:]:
            d = [1000 * (v(a, r, s) - v(a, "none", s)) for s in SEEDS if v(a, r, s) is not None and v(a, "none", s) is not None]
            cells.append(f"{st.mean(d):+10.1f}" if d else f"{'—':>10}")
        print(f"   {a:<14}" + "".join(cells))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    else:
        report()
        report_stress()
