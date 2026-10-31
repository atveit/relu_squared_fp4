"""One training run of the ReLU^2 / FP4 study in PyTorch: real low precision via NVIDIA Transformer Engine on Blackwell
(B300 / B200), or the simulated recipes on any device. Appends one JSON line per run.

  python train.py --act relu2 --recipe nvfp4 --seed 1 --steps 600            # real NVFP4 on a B300
  python train.py --act relu2 --recipe sim_nvfp4 --seed 1 --steps 600        # simulated NVFP4, same numerics as the Mac
  python train.py --act relu2 --recipe bf16 --device cpu --steps 5 --batch 4 # smoke test

Recipes (hidden Linears only: qkv, o, fc, proj; embedding, head, attention and norms stay bf16 / fp32):
  bf16          plain bf16 GEMMs (the baseline)
  fp8_cs        TE Float8CurrentScaling (per-tensor current amax)
  fp8_ds        TE DelayedScaling(amax_history_len=1, margin=0): last step's amax, like modded-nanogpt's FP8 MLP
  mxfp8         TE MXFP8BlockScaling (32-blocks, E8M0 scales)
  nvfp4         TE NVFP4BlockScaling defaults (16-blocks + E4M3 scales; 2D weight blocks; SR on gradients; RHT)
  nvfp4_nrht / nvfp4_nsr / nvfp4_1d / nvfp4_bare   ablations: no Hadamard / no SR / 1D weight blocks / none of the three
  sim_fp8 / sim_int8 / sim_mxfp4 / sim_nvfp4 / sim_nvfp4h   simulated (quant_sim.py), any device
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common.config import ADAM, CONFIGS, DEFAULT_LR_MULT, MUON, SCHEDULE_MODES, SIM_RECIPES, lr_at  # noqa: E402
from common import data as D  # noqa: E402
from model import GPT, hidden_matrices, init_weights  # noqa: E402
from optim import MuonAdam  # noqa: E402
from quant_sim import SimLinear  # noqa: E402

TE_RECIPES = ["fp8_cs", "fp8_ds", "mxfp8", "nvfp4", "nvfp4_nrht", "nvfp4_nsr", "nvfp4_1d", "nvfp4_bare"]
RECIPES = ["bf16"] + TE_RECIPES + list(SIM_RECIPES)


def te_recipe(name):
    from transformer_engine.common import recipe as R
    if name == "fp8_cs":
        return R.Float8CurrentScaling()
    if name == "fp8_ds":
        return R.DelayedScaling(margin=0, amax_history_len=1, amax_compute_algo="max")
    if name == "mxfp8":
        return R.MXFP8BlockScaling()
    flags = {"nvfp4": {}, "nvfp4_nrht": dict(disable_rht=True), "nvfp4_nsr": dict(disable_stochastic_rounding=True),
             "nvfp4_1d": dict(disable_2d_quantization=True),
             "nvfp4_bare": dict(disable_rht=True, disable_stochastic_rounding=True, disable_2d_quantization=True)}[name]
    return R.NVFP4BlockScaling(**flags)


def make_linear_factory(recipe):
    if recipe in SIM_RECIPES:
        return lambda i, o: SimLinear(i, o, SIM_RECIPES[recipe])
    if recipe in TE_RECIPES:
        import transformer_engine.pytorch as te

        class TELinear(te.Linear):
            """te.Linear fed bf16 activations, like the bf16 baseline's nn.Linear under autocast. TE quantizes its input
            in the dtype it arrives in (our residual stream is fp32), and NVFP4's Hadamard kernel only accepts bf16
            ("RHT is only supported for bfloat16 input"), so every TE recipe gets the same bf16 input."""

            def forward(self, inp, *a, **k):
                return super().forward(inp.to(torch.bfloat16) if inp.is_cuda else inp, *a, **k)

        return lambda i, o: TELinear(i, o, bias=False, params_dtype=torch.float32)
    return lambda i, o: nn.Linear(i, o, bias=False)


class Precision:
    """Context for forward passes: bf16 autocast (CUDA / CPU), plus Transformer Engine's autocast for TE recipes."""

    def __init__(self, recipe, device):
        self.recipe, self.device = recipe, device
        self.te_recipe = te_recipe(recipe) if recipe in TE_RECIPES else None

    def __call__(self, quantize=True):
        import contextlib
        stack = contextlib.ExitStack()
        if self.device.type == "cuda":   # CPU / MPS smoke tests run in fp32
            stack.enter_context(torch.autocast(device_type=self.device.type, dtype=torch.bfloat16))
        if self.te_recipe is not None and quantize:
            import transformer_engine.pytorch as te
            stack.enter_context(te.autocast(enabled=True, recipe=self.te_recipe))
        return stack


def evaluate(model, val, seq, device, prec, quantize=True):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in D.val_batches(val, seq):
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            with prec(quantize):
                logits = model(x)
            tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).long()).item()
            n += 1
    model.train()
    return tot / n


def run(args):
    device = torch.device(args.device)
    cfg = dict(CONFIGS[args.config])
    if args.batch:
        cfg["batch"] = args.batch
    B, T = cfg["batch"], cfg["seq"]
    if args.recipe in TE_RECIPES:
        import transformer_engine.pytorch as te
        ok = te.is_nvfp4_available() if args.recipe.startswith("nvfp4") else (
            te.is_mxfp8_available() if args.recipe == "mxfp8" else te.is_fp8_available())
        if not ok:
            raise RuntimeError(f"recipe {args.recipe} is not available on this GPU / Transformer Engine build")
    torch.manual_seed(args.seed)
    model = GPT(cfg, args.act, make_linear_factory(args.recipe))
    init_weights(model, args.seed)
    model.to(device)
    muon_p = hidden_matrices(model)
    ids = {id(p) for p in muon_p}
    opt = MuonAdam(muon_p, [p for p in model.parameters() if id(p) not in ids])
    prec = Precision(args.recipe, device)
    train, val = D.load(args.data_dir, args.steps, B, T, args.val_tokens)
    losses, t_first = [], None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(args.steps):
        x, y = D.train_batch(train, s, B, T)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with prec():
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).long())
        loss.backward()
        opt.step(lr_at(s, args.steps, MUON["lr"] * args.lr_mult, args.schedule),
                 lr_at(s, args.steps, ADAM["lr"] * args.lr_mult, args.schedule))
        opt.zero_grad()
        if s == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
        if s % 50 == 0 or s == args.steps - 1:
            losses.append((s, loss.item()))
            if args.verbose:
                print(f"  step {s} loss {loss.item():.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    v = evaluate(model, val, T, device, prec, quantize=True)
    v_hp = evaluate(model, val, T, device, prec, quantize=False) if args.recipe in TE_RECIPES else v
    rec = dict(run_id=run_id(args), platform=args.platform, framework="torch", device=torch.cuda.get_device_name() if device.type == "cuda"
               else platform.processor() or device.type, act=args.act, recipe=args.recipe, seed=args.seed, n=args.steps,
               config=args.config, cfg=cfg, lr_mult=args.lr_mult, lr_schedule=args.schedule, val_loss=v, val_loss_hp_eval=v_hp,
               step_ms=round(1000 * train_s / max(1, args.steps - 1), 2), first_step_s=round(t_first or 0, 2),
               tokens_per_s=round((args.steps - 1) * B * T / train_s), losses=losses,
               params=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
               te=_te_version(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(f"{rec['time']} {args.act:<9} {args.recipe:<11} seed={args.seed} N={args.steps} val={v:.4f} "
          f"(hp-eval {v_hp:.4f}) step={rec['step_ms']:.0f}ms tok/s={rec['tokens_per_s']:,}", flush=True)
    return rec


def _te_version():
    try:
        import transformer_engine
        return transformer_engine.__version__
    except Exception:
        return None


def run_id(args):
    return (f"{args.config}_{args.act}_{args.recipe}_s{args.seed}_n{args.steps}"
            + (f"_lr{args.lr_mult:g}" if args.lr_mult != 1.0 else "")
            + (f"_{args.schedule}" if args.schedule != "intended" else ""))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def run_logged(args, log_dir, prefix=""):
    """run(args) with its full stdout / stderr and settings in <log_dir>/<prefix><run_id>.log; returns the record, or
    None on an exception (the traceback goes to the log; the caller decides whether to continue)."""
    import traceback
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, prefix + run_id(args) + ".log")
    with open(path, "a") as fh:
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old[0], fh), _Tee(old[1], fh)
        try:
            print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} run {run_id(args)}\nargs {json.dumps(vars(args))}\n"
                  f"torch {torch.__version__} cuda {torch.version.cuda} te {_te_version()} "
                  f"device {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}", flush=True)
            rec = run(args)
            print(f"=== record {json.dumps(rec)}", flush=True)
            return rec
        except Exception:
            print(f"=== FAILED\n{traceback.format_exc()}", flush=True)
            return None
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            sys.stdout, sys.stderr = old


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="relu2")
    ap.add_argument("--recipe", default="bf16", choices=RECIPES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--config", default="tiny", choices=list(CONFIGS))
    ap.add_argument("--val-tokens", type=int, default=1 << 19, help="smaller for smoke tests")
    ap.add_argument("--batch", type=int, default=0, help="override the config's batch (smoke tests)")
    ap.add_argument("--lr-mult", type=float, default=DEFAULT_LR_MULT)
    ap.add_argument("--schedule", default="intended", choices=SCHEDULE_MODES,
                    help="common.config.lr_mult mode; 'intended' unless reproducing the Mac grids' uint64 bug")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", default=os.environ.get("FP4_DATA", os.path.join(os.path.dirname(HERE), "data")))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "torch_runs.jsonl"))
    ap.add_argument("--platform", default=os.environ.get("FP4_PLATFORM", "b300"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log-dir", default=None, help="also write the run's full output + settings to <dir>/<run_id>.log")
    return ap


if __name__ == "__main__":
    a = parser().parse_args()
    D.ensure_data(a.data_dir)
    if a.log_dir:
        sys.exit(0 if run_logged(a, a.log_dir) is not None else 1)
    run(a)
