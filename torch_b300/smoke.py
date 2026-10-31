"""Smoke test of the PyTorch trainer: every (activation, recipe) for a few steps, in one process, one log per run.

  python smoke.py --acts relu2,lrelu2,swiglu --recipes bf16,sim_fp8,fp8_cs --steps 20 --log-dir ../results/logs
Prints a table (loss at the first / last step, val, step time, FAIL / n/a) and appends the records to --out.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import train as TR  # noqa: E402


def te_available(recipe):
    if recipe not in TR.TE_RECIPES:
        return True, ""
    try:
        import transformer_engine.pytorch as te
    except Exception as e:   # no TE installed
        return False, f"no transformer_engine ({type(e).__name__})"
    fn = te.is_nvfp4_available if recipe.startswith("nvfp4") else (
        te.is_mxfp8_available if recipe == "mxfp8" else te.is_fp8_available)
    ok, why = fn(return_reason=True)
    return ok, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="relu2,lrelu2,swiglu")
    ap.add_argument("--recipes", default=",".join(TR.RECIPES))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--val-tokens", type=int, default=1 << 16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", required=True)
    g = ap.parse_args()
    rows = []
    for recipe in g.recipes.split(","):
        ok, why = te_available(recipe)
        for act in g.acts.split(","):
            if not ok:
                rows.append((act, recipe, "n/a", why))
                print(f"SKIP {act} {recipe}: {why}", flush=True)
                continue
            a = TR.parser().parse_args(["--act", act, "--recipe", recipe, "--steps", str(g.steps), "--config", g.config,
                                        "--val-tokens", str(g.val_tokens), "--out", g.out])
            rec = TR.run_logged(a, g.log_dir, prefix="smoke_")
            if rec is None:
                rows.append((act, recipe, "FAIL", "see log"))
                continue
            l0, l1 = rec["losses"][0][1], rec["losses"][-1][1]
            good = all(math.isfinite(x) for x in (l0, l1, rec["val_loss"])) and l1 < l0 - 0.5
            rows.append((act, recipe, "OK" if good else "CHECK",
                         f"loss {l0:.3f}->{l1:.3f} val {rec['val_loss']:.4f} hp {rec['val_loss_hp_eval']:.4f} "
                         f"step {rec['step_ms']:.1f}ms first {rec['first_step_s']:.1f}s"))
    print("\n== smoke summary")
    for r in rows:
        print(f"  {r[0]:<8} {r[1]:<11} {r[2]:<5} {r[3]}")
    n_fail = sum(r[2] in ("FAIL", "CHECK") for r in rows)
    print(f"== {len(rows)} runs, {n_fail} FAIL/CHECK, {sum(r[2] == 'n/a' for r in rows)} n/a", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
