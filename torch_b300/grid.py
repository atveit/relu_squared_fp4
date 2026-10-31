"""The B300 grid: every (activation, recipe, seed) once, skipping runs already in the results file. One GPU, sequential.

  python grid.py --plan small     # tiny model: real NVFP4 / FP8 vs the simulation, 7 activations (~30-45 min on a B300)
  python grid.py --plan scale     # GPT-2-small shape: 4 activations x bf16 / fp8 / nvfp4, longer (~$40-90 on a B300)
  python grid.py --plan small --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common.config import ACTIVATIONS  # noqa: E402
from common import data as D  # noqa: E402
import train as TR  # noqa: E402

PLANS = {
    # Real-hardware check of the Mac findings: does real NVFP4 behave like the simulation, per activation?
    "small": dict(config="tiny", steps=600, seeds=[1, 2], acts=list(ACTIVATIONS),
                  recipes=["bf16", "fp8_cs", "fp8_ds", "nvfp4", "nvfp4_bare", "sim_nvfp4", "sim_mxfp4"]),
    # Scale-up where FP4 speed and outliers become real; fewer arms, longer runs.
    "scale": dict(config="gpt2s", steps=2000, seeds=[1, 2], acts=["relu2", "lrelu2", "swiglu", "eglu_r2n"],
                  recipes=["bf16", "fp8_ds", "nvfp4", "nvfp4_bare"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="small", choices=list(PLANS))
    ap.add_argument("--lr-mult", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    g = ap.parse_args()
    plan = PLANS[g.plan]
    base = TR.parser().parse_args([])
    out = base.out.replace("torch_runs.jsonl", f"torch_{g.plan}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["act"], r["recipe"], r["seed"], r["n"]) for r in map(json.loads, open(out))}
    jobs = [(a, r, s) for s in plan["seeds"] for r in plan["recipes"] for a in plan["acts"]
            if (a, r, s, plan["steps"]) not in done]
    print(f"plan {g.plan}: {len(jobs)} runs to do -> {out}", flush=True)
    if g.dry_run:
        for j in jobs[:10]:
            print("  ", j)
        return
    D.ensure_data(base.data_dir)
    for act, recipe, seed in jobs:
        a = TR.parser().parse_args(["--act", act, "--recipe", recipe, "--seed", str(seed), "--steps", str(plan["steps"]),
                                    "--config", plan["config"], "--out", out]
                                   + (["--lr-mult", str(g.lr_mult)] if g.lr_mult else []))
        TR.run(a)


if __name__ == "__main__":
    main()
