"""Summarize the B300 grid: precision penalty per activation and recipe (val(recipe) - val(bf16), paired per seed, mnat),
and each activation vs ReLU^2 within a recipe.

  python report.py ../results/torch_b300/torch_b300_tiny_core.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict


def main(paths):
    res = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"(missing {p})")
            continue
        for r in map(json.loads, open(p)):
            res[(r["config"], r["act"], r["recipe"], r["seed"], r["n"])] = r
    groups = defaultdict(set)
    for c, a, rc, s, n in res:
        groups[(c, n)].add((a, rc, s))
    for (c, n), keys in sorted(groups.items()):
        acts = sorted({k[0] for k in keys}, key=lambda a: (a != "relu2", a))
        recs = [r for r in ["bf16", "fp8_cs", "fp8_ds", "mxfp8", "nvfp4", "nvfp4_nrht", "nvfp4_nsr", "nvfp4_1d", "nvfp4_bare",
                            "sim_fp8", "sim_int8", "sim_mxfp4", "sim_nvfp4", "sim_nvfp4h"] if any(k[1] == r for k in keys)]
        seeds = sorted({k[2] for k in keys})
        v = lambda a, r, s: res.get((c, a, r, s, n), {}).get("val_loss")
        print(f"\n== {c}, N={n}, seeds {seeds}: val loss (mean over seeds)")
        print(f"   {'act':<9}" + "".join(f"{r:>11}" for r in recs))
        for a in acts:
            cells = [[v(a, r, s) for s in seeds if v(a, r, s) is not None] for r in recs]
            print(f"   {a:<9}" + "".join(f"{st.mean(x):11.4f}" if x else f"{'-':>11}" for x in cells))
        print(f"== precision penalty vs bf16, same activation, paired [mnat]  (n = seeds)")
        print(f"   {'act':<9}" + "".join(f"{r:>11}" for r in recs[1:]))
        for a in acts:
            cells = []
            for r in recs[1:]:
                d = [1000 * (v(a, r, s) - v(a, "bf16", s)) for s in seeds if v(a, r, s) is not None and v(a, "bf16", s) is not None]
                cells.append(f"{st.mean(d):+8.1f} ({len(d)})" if d else f"{'-':>11}")
            print(f"   {a:<9}" + "".join(f"{x:>11}" for x in cells))
        if "relu2" in acts:
            print(f"== each activation vs relu2, same recipe, paired [mnat; + = worse than relu2]")
            print(f"   {'act':<9}" + "".join(f"{r:>11}" for r in recs))
            for a in acts[1:]:
                cells = []
                for r in recs:
                    d = [1000 * (v(a, r, s) - v("relu2", r, s)) for s in seeds if v(a, r, s) is not None and v("relu2", r, s) is not None]
                    cells.append(f"{st.mean(d):+8.1f}" if d else "-")
                print(f"   {a:<9}" + "".join(f"{x:>11}" for x in cells))
        sm = defaultdict(list)
        for (cc, a, r, s, nn), rec in res.items():
            if (cc, nn) == (c, n):
                sm[r].append(rec["step_ms"])
        print("== step_ms (median; concurrent lanes, so only roughly comparable): " +
              ", ".join(f"{r} {st.median(sm[r]):.0f}" for r in recs if sm[r]))


if __name__ == "__main__":
    main(sys.argv[1:])
