"""The B300 grid: every (activation, recipe, seed) of a plan once, skipping runs already in the results file. One GPU,
sequential, seed-major (seed 1 of every arm finishes before seed 2 starts, so a cut-short grid still has full coverage).

  python grid.py --plan tiny_core --log-dir ../results/torch_b300/logs        # real NVFP4 / MXFP8 / FP8 vs simulation
  python grid.py --plan scale --max-minutes 25                                 # GPT-2-small shape, stop launching at 25 min
  python grid.py --plan tiny_core --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common.config import ACTIVATIONS  # noqa: E402
from common import data as D  # noqa: E402
import train as TR  # noqa: E402

PLANS = {
    # Phase A: does real NVFP4 behave like the simulation, per activation; and what do its three tricks (Hadamard on
    # wgrad, stochastic rounding, 2D weight blocks) buy? Same model / steps / seeds as the Mac grid (lowp_A).
    "tiny_core": dict(config="tiny", steps=600, seeds=[1, 2], acts=["relu2", "lrelu2", "swiglu"],
                      recipes=["bf16", "nvfp4", "nvfp4_nrht", "nvfp4_bare", "mxfp8", "fp8_cs", "sim_nvfp4"]),
    # Phase A, matched pair: TE NVFP4 with 1D blocks (RHT on the wgrad operands + SR on gradients, TE's defaults
    # otherwise) vs the simulator with the same three choices (sim_nvfp4h). The cleanest real-vs-simulated comparison.
    "tiny_match": dict(config="tiny", steps=600, seeds=[1, 2], acts=["relu2", "lrelu2", "swiglu"],
                       recipes=["nvfp4_1d", "sim_nvfp4h"]),
    # Phase A': the other Mac activations under the key recipes (if time allows).
    "tiny_more": dict(config="tiny", steps=600, seeds=[1, 2], acts=["xielu", "eglu_r2n", "s2glu_r2", "eml"],
                      recipes=["bf16", "nvfp4", "sim_nvfp4"]),
    # Phase B: GPT-2-small shape, where FP4 GEMMs are big enough to be fast and activation outliers can grow.
    "scale": dict(config="gpt2s", steps=1000, seeds=[1], acts=["relu2", "lrelu2", "swiglu"],
                  recipes=["bf16", "nvfp4", "mxfp8"]),
    # The original full plan (7 activations x 7 recipes x 2 seeds, ~100 runs): too long for one B300 hour.
    "small": dict(config="tiny", steps=600, seeds=[1, 2], acts=list(ACTIVATIONS),
                  recipes=["bf16", "fp8_cs", "fp8_ds", "nvfp4", "nvfp4_bare", "sim_nvfp4", "sim_mxfp4"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="tiny_core", choices=list(PLANS))
    ap.add_argument("--lr-mult", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override the plan's steps")
    ap.add_argument("--out", default=None, help="default: ../results/torch_b300_<plan>.jsonl")
    ap.add_argument("--log-dir", default=None, help="one <plan>_<run_id>.log per run (full output + settings)")
    ap.add_argument("--max-minutes", type=float, default=None, help="do not start a new run after this many minutes")
    ap.add_argument("--seeds", default=None, help="override, e.g. 1 or 1,2")
    ap.add_argument("--acts", default=None, help="override, comma-separated")
    ap.add_argument("--recipes", default=None, help="override, comma-separated")
    ap.add_argument("--dry-run", action="store_true")
    g = ap.parse_args()
    plan = dict(PLANS[g.plan])
    if g.steps:
        plan["steps"] = g.steps
    if g.seeds:
        plan["seeds"] = [int(s) for s in g.seeds.split(",")]
    for k in ("acts", "recipes"):
        if getattr(g, k):
            plan[k] = getattr(g, k).split(",")
    base = TR.parser().parse_args([])
    out = g.out or os.path.join(os.path.dirname(HERE), "results", f"torch_b300_{g.plan}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["act"], r["recipe"], r["seed"], r["n"]) for r in map(json.loads, open(out))}
    jobs = [(a, r, s) for s in plan["seeds"] for r in plan["recipes"] for a in plan["acts"]
            if (a, r, s, plan["steps"]) not in done]
    print(f"plan {g.plan}: {len(jobs)} runs to do -> {out}", flush=True)
    if g.dry_run:
        for j in jobs:
            print("  ", j)
        return
    D.ensure_data(base.data_dir)
    t0, fails = time.time(), 0
    for i, (act, recipe, seed) in enumerate(jobs):
        if g.max_minutes and (time.time() - t0) / 60 > g.max_minutes:
            print(f"time budget {g.max_minutes} min reached: {len(jobs) - i} runs left", flush=True)
            break
        a = TR.parser().parse_args(["--act", act, "--recipe", recipe, "--seed", str(seed), "--steps", str(plan["steps"]),
                                    "--config", plan["config"], "--out", out]
                                   + (["--lr-mult", str(g.lr_mult)] if g.lr_mult else []))
        rec = TR.run_logged(a, g.log_dir, prefix=f"{g.plan}_") if g.log_dir else TR.run(a)
        fails += rec is None
        print(f"[{(time.time() - t0) / 60:5.1f} min] {i + 1}/{len(jobs)} done, {fails} failed", flush=True)


if __name__ == "__main__":
    main()
