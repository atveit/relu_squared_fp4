"""Stage 1 of the EML isolation ladder (Mac, $0): can a gated EML-family MLP beat ReLU^2, and does the win hold?

Pre-registered pass (2026-10-31, before any stage-1 run): a gated variant beats ReLU^2, each at its own best LR,
by >= 5 mnat paired over 5 seeds with t > 3, at N = 300 and at N = 600; N = 1200 and d = 384 must not reverse it.
Only a pass justifies stage 2 (single H100, bf16 #393 with the activation swapped).

Phases (one train_mlx subprocess per run, 2 at a time: +13% throughput vs 1, 3 at a time only +5%, smoke-tested):
  A  screen   d256 L4 N300, seeds 1-3: CANDIDATES + lines of stage1_extra_arms.txt, paired vs relu2_metal
  L  own LR   relu2_metal and the top 3 (LR-matched) of A at x1.5, x2, x3 (x1 from A), seeds 1-2; an edge winner is
             extended once (x0.7 or x4)   [amended 08:40, before selection: was top 2 at x0.7 / x1.4]
  B  horizon  relu2 and the top 3, each at its best LR: N300 and N600 seeds 1-5, N1200 seeds 1-3
  C  width    d384 L6 H6, N600 seeds 1-3: relu2 and the top 1

  python3 -m mlx_mac.stage1 all       # resumable: runs already in results/*.jsonl are skipped
  python3 -m mlx_mac.stage1 report
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
BASE = "relu2_metal"
CANDIDATES = ["eglu_r2", "eglu0_r2", "s2glu_r2", "swiglu", "reglu"]
EXTRA = os.path.join(HERE, "stage1_extra_arms.txt")
SUMMARY = os.path.join(HERE, "results", "stage1_summary.json")
SMALL = dict(d=256, layers=4, heads=4)
WIDE = dict(d=384, layers=6, heads=6)
WORKERS = 2


def load():
    """(arm, seed, n, d) -> val_loss over every results file (records without cfg are the d=256 grids)."""
    res = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*.jsonl"))):
        if os.path.basename(f).startswith("smoke"):
            continue
        for line in open(f):
            r = json.loads(line)
            d = r.get("cfg", {}).get("d", 256)
            res[(r["arm"], r["seed"], r["n"], d)] = r["val_loss"]
    return res


def with_lr(arm, lr):
    return arm if lr == 1.0 else f"{arm}@lr{lr:g}"


def paired(res, arm, base, n, d, seeds=None):
    common = sorted(s for (a, s, nn, dd) in res if a == arm and nn == n and dd == d and (base, s, n, d) in res)
    if seeds is not None:
        common = [s for s in common if s in seeds]
    return common, [1000 * (res[(arm, s, n, d)] - res[(base, s, n, d)]) for s in common]


def stats(d):
    if not d:
        return float("nan"), float("nan"), float("nan")
    m = st.mean(d)
    if len(d) < 2:
        return m, float("nan"), float("nan")
    se = st.stdev(d) / len(d) ** 0.5
    return m, se, (m / se if se > 0 else float("nan"))


def run_jobs(jobs, tag, cfg):
    """jobs: [(arm, seed, n)]; skips those already in the results; runs WORKERS subprocesses at a time."""
    res = load()
    todo = [j for j in jobs if (j[0], j[1], j[2], cfg["d"]) not in res]
    todo.sort(key=lambda j: -j[2])   # longest first
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {len(todo)} runs to do ({len(jobs) - len(todo)} already done)", flush=True)

    def one(job):
        arm, seed, n = job
        cmd = [sys.executable, "-u", "-m", "mlx_mac.train_mlx", "--tag", tag, "--arms", arm, "--seeds", str(seed),
               "--ns", str(n), "--d", str(cfg["d"]), "--layers", str(cfg["layers"]), "--heads", str(cfg["heads"])]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out = [l for l in (p.stdout + p.stderr).splitlines() if "Warning" not in l and l.strip()]
        if p.returncode != 0:
            print(f"   FAILED {job}: " + " | ".join(out[-5:]), flush=True)
        else:
            print("   " + (out[-1] if out else f"{job} (no output)"), flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))


def extra_arms():
    if not os.path.exists(EXTRA):
        return []
    return [l.split("#")[0].strip() for l in open(EXTRA) if l.split("#")[0].strip()]


def screen():
    seen = set()
    while True:   # re-read the extra-arms file after each pass, so arms added mid-run are screened too
        arms = [a for a in CANDIDATES + extra_arms() if a not in seen]
        if not arms:
            return sorted(seen)
        seen.update(arms)
        run_jobs([(a, s, 300) for s in (1, 2, 3) for a in [BASE] + arms], "stage1_A", SMALL)


def rank(res, arms):
    """LR-matched arms only: an '@lr' arm vs ReLU^2 at x1 is confounded (amended 08:40, before selection: the dual-space
    'eml_metal@lr1.25' looked -13.8 mnat better, yet at a matched x1.25 it is +33 worse)."""
    rows = []
    for a in (a for a in arms if "@lr" not in a):
        seeds, d = paired(res, a, BASE, 300, 256, seeds=[1, 2, 3])
        if d:
            rows.append((st.mean(d), a))
    return [a for _, a in sorted(rows)]


def best_lr(res, arm, lrs, seeds=(1, 2)):
    def mean_val(lr):
        v = [res.get((with_lr(arm, lr), s, 300, 256)) for s in seeds]
        return st.mean(v) if all(x is not None for x in v) else float("inf")
    return min(lrs, key=mean_val)


LRS = (1.0, 1.5, 2.0, 3.0)   # amended 08:40: x1.25 already gives ReLU^2 -45 mnat, so the default LR is too low for all arms


def lr_phase(top):
    arms = [BASE] + top
    run_jobs([(with_lr(a, lr), s, 300) for a in arms for lr in LRS if lr != 1.0 for s in (1, 2)], "stage1_L", SMALL)
    best = {}
    for a in arms:
        grid = list(LRS)
        b = best_lr(load(), a, grid)
        if b in (grid[0], grid[-1]):   # extend once past the edge
            ext = 0.7 if b == grid[0] else 4.0
            run_jobs([(with_lr(a, ext), s, 300) for s in (1, 2)], "stage1_L", SMALL)
            b = best_lr(load(), a, sorted(grid + [ext]))
        best[a] = b
    return best


def horizon(base_arm, cand_arms):
    arms = [base_arm] + cand_arms
    run_jobs([(a, s, 300) for s in range(1, 6) for a in arms], "stage1_B", SMALL)
    run_jobs([(a, s, 600) for s in range(1, 6) for a in arms], "stage1_B", SMALL)
    run_jobs([(a, s, 1200) for s in range(1, 4) for a in arms], "stage1_B", SMALL)


def width(base_arm, cand_arm):
    run_jobs([(a, s, 600) for s in range(1, 4) for a in (base_arm, cand_arm)], "stage1_C", WIDE)


def report(summary=None):
    res = load()
    summary = summary or (json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {})
    print("\n== A: screen, d256 L4 N300, paired vs relu2_metal (lr x1), all common seeds [mnat; + = worse than ReLU^2]")
    arms = sorted({a for (a, s, n, d) in res if n == 300 and d == 256 and a != BASE
                   and not any(f"@lr{x}" in a and "@opt" not in a for x in ("0.5", "0.7", "1.4", "2"))})   # LR-sweep points are listed in L
    rows = []
    for a in arms:
        seeds, dl = paired(res, a, BASE, 300, 256, seeds=[1, 2, 3, 4, 5])
        m, se, t = stats(dl)
        rows.append((m, a, seeds, dl, se))
    for m, a, seeds, dl, se in sorted(rows):
        print(f"   {a:<34} n={len(dl)}  {m:+7.1f} (se {se:4.1f})   " + " ".join(f"{x:+.1f}" for x in dl))
    best = summary.get("best_lr", {})
    if best:
        print("\n== L: own best LR multiplier (mean val, seeds 1-2, N300): " + ", ".join(f"{a} x{lr:g}" for a, lr in best.items()))
    base_arm = summary.get("base_arm")
    if base_arm:
        print(f"\n== B/C: each candidate at its best LR vs {base_arm} (ReLU^2 at its best LR)")
        for cand in summary.get("cand_arms", []):
            for n, d in ((300, 256), (600, 256), (1200, 256), (600, 384)):
                seeds, dl = paired(res, cand, base_arm, n, d)
                if dl:
                    m, se, t = stats(dl)
                    print(f"   {cand:<26} N={n:<5} d={d}  n={len(dl)}  {m:+7.1f} (se {se:4.1f}, t {t:+5.1f})   "
                          + " ".join(f"{x:+.1f}" for x in dl))
        verdict = {}
        for cand in summary.get("cand_arms", []):
            ok = True
            for n in (300, 600):
                _, dl = paired(res, cand, base_arm, n, 256)
                m, se, t = stats(dl)
                ok &= len(dl) >= 5 and m <= -5 and t < -3
            for n, d in ((1200, 256), (600, 384)):
                _, dl = paired(res, cand, base_arm, n, d)
                if dl:
                    ok &= st.mean(dl) < 0
            verdict[cand] = "PASS" if ok else "fail"
        print("\n== Pre-registered verdict (<= -5 mnat, t < -3, 5 seeds at N300 and N600; N1200 and d384 not reversed): "
              + ", ".join(f"{c}: {v}" for c, v in verdict.items()))
        return verdict


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "report":
        report()
        return
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    arms = screen()
    top = rank(load(), arms)[:3]   # amended 08:40: top 3, so skip-gate arms can't crowd out the best MLP activation
    summary["screen_rank"] = rank(load(), arms)
    summary["top"] = top
    print(f"{time.strftime('%H:%M:%S')} screen rank: {summary['screen_rank']}", flush=True)
    best = lr_phase(top)
    summary["best_lr"] = best
    base_arm = with_lr(BASE, best[BASE])
    cand_arms = [with_lr(a, best[a]) for a in top]
    summary.update(base_arm=base_arm, cand_arms=cand_arms)
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(f"{time.strftime('%H:%M:%S')} best LR {best}; horizon arms {base_arm} vs {cand_arms}", flush=True)
    horizon(base_arm, cand_arms)
    width(base_arm, cand_arms[0])
    summary["verdict"] = report(summary)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(f"{time.strftime('%H:%M:%S')} stage 1 done", flush=True)


if __name__ == "__main__":
    main()
