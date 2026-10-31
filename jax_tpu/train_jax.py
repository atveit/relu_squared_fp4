"""The ReLU^2 / FP4 study in JAX (Flax NNX) for TPUs. One file, so a Colab notebook can %%writefile it.

  python train_jax.py --act relu2 --recipe bf16 --seed 1 --steps 600
  python train_jax.py --act relu2 --recipe sim_nvfp4 --seed 1 --steps 600       # simulated NVFP4 (any TPU)
  python train_jax.py --grid small                                              # the TPU grid, skipping done runs

Recipes (hidden Linears only: qkv, o, fc, proj):
  bf16                    bf16 matmuls, fp32 master weights (the baseline)
  int8                    Qwix quantized training (QtRule): int8 weights, activations and gradients, stochastic rounding
                          on the gradients. Native int8 MXUs on TPU v5e / v6e (2x bf16)
  fp8                     Qwix QT with float8_e4m3fn forward and float8_e5m2 gradients. Native on TPU7x (Ironwood); emulated on
                          older TPUs (numerics only)
  sim_fp8 / sim_int8 / sim_mxfp4 / sim_nvfp4 / sim_nvfp4h
                          fake quantization with exactly the numerics of the Mac (lowp.py) and PyTorch (quant_sim.py) sims.
                          No TPU has FP4 yet (TPU 8t is expected in 2027), so FP4 on TPU is simulated: the TPU's value
                          here is running the simulated grid at GPT-2-small scale for free.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common.config import (ACTIVATIONS, ADAM, CONFIGS, DEFAULT_LR_MULT, HEAD_STD, MUON, NORM_EPS, ROPE_BASE,  # noqa: E402
                           SIM_RECIPES, SOFTCAP, XIELU, hidden_for, lr_at)
from common import data as D  # noqa: E402

RECIPES = ["bf16", "int8", "fp8"] + list(SIM_RECIPES)
QWIX_RECIPES = {"int8": ("int8", "int8"), "fp8": ("float8_e4m3fn", "float8_e5m2")}


# ----------------------------------------------------------------------------------------------- simulated formats
def _round(a, key):
    return jnp.floor(a + jax.random.uniform(key, a.shape, a.dtype)) if key is not None else jnp.round(a)


def _float_q(a, mant, emin, amaxv, key):
    m = jnp.abs(a)
    e = jnp.maximum(jnp.floor(jnp.log2(jnp.maximum(m, 2.0 ** (emin - mant - 4)))), float(emin))
    step = jnp.exp2(e - mant)
    return jnp.sign(a) * jnp.minimum(_round(m / step, key) * step, amaxv)


def _blocks(t, k):
    return t.reshape(*t.shape[:-1], t.shape[-1] // k, k)


def quant(t, fmt, key=None):
    """Fake-quantize t along its last axis; key != None means stochastic rounding. Same as lowp.py / quant_sim.py."""
    if fmt == "none":
        return t
    if fmt in ("e4m3", "e5m2"):
        hi = 448.0 if fmt == "e4m3" else 57344.0
        s = jnp.maximum(jnp.max(jnp.abs(t)), 1e-30) / hi
        q = _float_q(t / s, 3, -6, 448.0, key) if fmt == "e4m3" else _float_q(t / s, 2, -14, 57344.0, key)
        return q * s
    if fmt == "int8":
        s = jnp.maximum(jnp.max(jnp.abs(t), axis=-1, keepdims=True), 1e-30) / 127.0
        return jnp.clip(_round(t / s, key), -127, 127) * s
    if fmt == "mxfp4":
        b = _blocks(t, 32)
        amax = jnp.maximum(jnp.max(jnp.abs(b), axis=-1, keepdims=True), 1e-30)
        s = jnp.exp2(jnp.ceil(jnp.log2(amax / 6.0)))
        return (_float_q(b / s, 1, 0, 6.0, key) * s).reshape(t.shape)
    if fmt == "nvfp4":
        b = _blocks(t, 16)
        sb = jnp.maximum(jnp.max(jnp.abs(b), axis=-1, keepdims=True), 1e-30) / 6.0
        g = jnp.maximum(jnp.max(sb), 1e-30) / 448.0
        s = jnp.maximum(_float_q(sb / g, 3, -6, 448.0, None) * g, 1e-30)
        return (_float_q(b / s, 1, 0, 6.0, key) * s).reshape(t.shape)
    raise ValueError(fmt)


def _hadamard16(seed=0):
    h = np.array([[1.0]])
    while h.shape[0] < 16:
        h = np.block([[h, h], [h, -h]])
    d = np.random.default_rng(seed).choice([-1.0, 1.0], 16)   # identical signs to lowp.py / quant_sim.py
    return jnp.asarray((h * d[None, :] / 4.0).astype(np.float32))


R16 = _hadamard16()


def _rht(t):
    return (_blocks(t, 16) @ R16.T).reshape(t.shape)


def make_qmm(recipe):
    fxw, fg, sr, had = SIM_RECIPES[recipe]

    @jax.custom_vjp
    def qmm(x, w, key):            # x (..., d_in), w (d_in, d_out) as in nnx.Linear; GEMMs in fp32
        return quant(x.astype(jnp.float32), fxw) @ quant(w.T.astype(jnp.float32), fxw).T

    def fwd(x, w, key):
        return qmm(x, w, key), (x.astype(jnp.float32), w.astype(jnp.float32), key, x.dtype, w.dtype)

    def bwd(res, g):
        x, w, key, xdt, wdt = res
        k1, k2 = jax.random.split(key) if sr else (None, None)
        g = g.astype(jnp.float32)
        dx = quant(g, fg, k1) @ quant(w, fxw).T                       # contraction over d_out: w rows = d_in
        g2 = g.reshape(-1, g.shape[-1]).T                              # (d_out, tokens)
        x2 = x.reshape(-1, x.shape[-1]).T                              # (d_in, tokens)
        if had:
            g2, x2 = _rht(g2), _rht(x2)
        dw = (quant(g2, fg, k2) @ quant(x2, fxw).T).T                  # (d_in, d_out)
        return dx.astype(xdt), dw.astype(wdt), None

    qmm.defvjp(fwd, bwd)
    return qmm


# ----------------------------------------------------------------------------------------------- model
def rms(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), -1, keepdims=True) + NORM_EPS).astype(x.dtype)


class Linear(nnx.Module):
    """Bias-free Linear, kernel (d_in, d_out), MLX/PyTorch default init U(+-1/sqrt(d_in)) or zeros; optional sim recipe."""

    def __init__(self, d_in, d_out, rngs, zero=False, recipe="bf16"):
        b = d_in ** -0.5
        k = jnp.zeros((d_in, d_out)) if zero else jax.random.uniform(rngs.params(), (d_in, d_out), minval=-b, maxval=b)
        self.kernel = nnx.Param(k)
        self.recipe = recipe
        self.qmm = make_qmm(recipe) if recipe in SIM_RECIPES else None
        self.sr_rngs = rngs if recipe in SIM_RECIPES else None

    def __call__(self, x):
        if self.qmm is not None:
            return self.qmm(x, self.kernel.value, self.sr_rngs.sr())
        if self.recipe in QWIX_RECIPES:   # Qwix intercepts this dot_general and quantizes both operands itself
            return jnp.dot(x.astype(jnp.float32), self.kernel.value)
        return x.astype(jnp.bfloat16) @ self.kernel.value.astype(jnp.bfloat16)


def activation(act, h, xielu_raw=None):
    h = h.astype(jnp.float32)
    if act == "relu2":
        return jnp.square(jax.nn.relu(h))
    if act == "lrelu2":
        return jnp.square(jnp.where(h > 0, h, 0.5 * h))
    if act == "xielu":
        ap = jax.nn.softplus(xielu_raw[0])
        an = XIELU["beta"] + jax.nn.softplus(xielu_raw[1])
        b = XIELU["beta"]
        return jnp.where(h > 0, ap * h * h + b * h, (jnp.expm1(jnp.minimum(h, XIELU["eps"])) - h) * an + b * h)
    u, v = jnp.split(h, 2, axis=-1)
    if act == "swiglu":
        return jax.nn.silu(u) * v
    if act == "s2glu_r2":
        return 2.0 * jax.nn.sigmoid(u) * jnp.square(jax.nn.relu(v))
    if act == "eglu_r2n":
        return (jnp.exp(jnp.tanh(u)) - 0.36787944) * 0.42545906 * jnp.square(jax.nn.relu(v))
    if act == "eml":
        return jnp.exp(jnp.tanh(u)) - jnp.log1p(jax.nn.relu(v))
    raise ValueError(act)


def rope(x, seq):   # x (B, T, H, hd); non-interleaved halves, as mx.fast.rope(traditional=False)
    hd = x.shape[-1]
    inv = ROPE_BASE ** (-jnp.arange(0, hd // 2, dtype=jnp.float32) * 2 / hd)
    f = jnp.outer(jnp.arange(x.shape[1], dtype=jnp.float32), inv)[None, :, None, :]
    c, s = jnp.cos(f).astype(x.dtype), jnp.sin(f).astype(x.dtype)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class Block(nnx.Module):
    def __init__(self, d, n_head, act, recipe, rngs):
        self.n_head, self.act = n_head, act
        lin = partial(Linear, rngs=rngs, recipe=recipe)
        self.qkv, self.o = lin(d, 3 * d), lin(d, d, zero=True)
        fc_out, proj_in = hidden_for(act, d)
        self.fc, self.proj = lin(d, fc_out), lin(proj_in, d, zero=True)
        self.lam = nnx.Param(jnp.ones((2,)))
        if act == "xielu":
            self.xielu_raw = nnx.Param(jnp.array([math.log(math.expm1(XIELU["alpha_p"])),
                                                  math.log(math.expm1(XIELU["alpha_n"] - XIELU["beta"]))]))

    def __call__(self, x):
        B, T, Dm = x.shape
        H, hd = self.n_head, Dm // self.n_head
        h = rms(x)
        q, k, v = jnp.split(self.qkv(h).reshape(B, T, 3, H, hd), 3, axis=2)
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        q, k = rope(rms(q), T), rope(rms(k), T)
        a = jax.nn.dot_product_attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                                         scale=hd ** -0.5, is_causal=True)
        x = x + self.lam.value[0] * self.o(a.reshape(B, T, Dm)).astype(jnp.float32)
        g = activation(self.act, self.fc(rms(x)), self.xielu_raw.value if self.act == "xielu" else None)
        return x + self.lam.value[1] * self.proj(g).astype(jnp.float32)


class GPT(nnx.Module):
    def __init__(self, cfg, act, recipe, rngs):
        d = cfg["d"]
        self.embed = nnx.Param(jax.random.normal(rngs.params(), (cfg["vocab"], d)) * d ** -0.5)
        self.blocks = nnx.List([Block(d, cfg["heads"], act, recipe, rngs) for _ in range(cfg["layers"])])
        self.head = nnx.Param(jax.random.normal(rngs.params(), (d, cfg["vocab"])) * HEAD_STD)

    def __call__(self, idx):
        x = rms(self.embed.value[idx])
        for b in self.blocks:
            x = b(x)
        logits = (rms(x).astype(jnp.bfloat16) @ self.head.value.astype(jnp.bfloat16)).astype(jnp.float32)
        return SOFTCAP * jnp.tanh(logits / SOFTCAP)


# ----------------------------------------------------------------------------------------------- optimizer (MLX semantics)
def newton_schulz(G):
    X = G.astype(jnp.float32)
    tall = X.shape[0] > X.shape[1]
    X = X.T if tall else X
    X = X / (jnp.linalg.norm(X) + 1e-7)
    a, b, c = MUON["ns_coeffs"]
    for _ in range(MUON["ns_steps"]):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if tall else X


def muon(lr_fn):
    """Muon on (d_in, d_out) kernels: scale sqrt(max(1, d_out / d_in)), i.e. rows / cols of the (out, in) weight."""
    mu = MUON["momentum"]

    def init(params):
        return dict(v=jax.tree.map(jnp.zeros_like, params), step=jnp.zeros([], jnp.int32))

    def update(grads, state, params=None):
        lr = lr_fn(state["step"])
        v = jax.tree.map(lambda v, g: mu * v + (1 - mu) * g, state["v"], grads)
        def upd(g, vv):
            u = g * (1 - mu) + vv * mu if MUON["nesterov"] else vv
            return -lr * max(1.0, g.shape[1] / g.shape[0]) ** 0.5 * newton_schulz(u)
        return jax.tree.map(upd, grads, v), dict(v=v, step=state["step"] + 1)

    return optax.GradientTransformation(init, update)


def adam_nobc(lr_fn):
    b1, b2, eps = ADAM["b1"], ADAM["b2"], ADAM["eps"]

    def init(params):
        z = jax.tree.map(jnp.zeros_like, params)
        return dict(m=z, v=jax.tree.map(jnp.zeros_like, params), step=jnp.zeros([], jnp.int32))

    def update(grads, state, params=None):
        lr = lr_fn(state["step"])
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
        return jax.tree.map(lambda m, v: -lr * m / (jnp.sqrt(v) + eps), m, v), dict(m=m, v=v, step=state["step"] + 1)

    return optax.GradientTransformation(init, update)


def schedule(base, n):
    cd = int(n * 0.6)
    return lambda s: base * (1.0 - 0.9 * jnp.clip((s - cd) / max(1, n - cd), 0.0, 1.0))


def label_params(model):
    """'muon' for the blocks' 2-D kernels, 'adam' for everything else."""
    state = nnx.state(model, nnx.Param)
    def lab(path, x):
        keys = [getattr(p, "key", getattr(p, "idx", p)) for p in path]
        return "muon" if (keys and keys[0] == "blocks" and x.ndim == 2) else "adam"
    return jax.tree_util.tree_map_with_path(lab, state)


# ----------------------------------------------------------------------------------------------- training
def build(args, cfg):
    rngs = nnx.Rngs(params=args.seed, sr=1000 + args.seed)
    model = GPT(cfg, args.act, args.recipe, rngs)
    if args.recipe in QWIX_RECIPES:
        import qwix
        wq, gq = QWIX_RECIPES[args.recipe]
        rules = [qwix.QtRule(module_path=r".*blocks.*(qkv|o|fc|proj).*", weight_qtype=wq, act_qtype=wq, bwd_qtype=gq,
                             bwd_stochastic_rounding="uniform" if args.recipe == "int8" else None)]
        x0 = jnp.zeros((2, cfg["seq"]), jnp.int32)
        model = qwix.quantize_model(model, qwix.QtProvider(rules), x0, rngs=nnx.Rngs(stochastic_rounding=0))
        model.set_attributes(qwix_rngs=nnx.Rngs(stochastic_rounding=args.seed))
    n = args.steps
    tx = optax.multi_transform({"muon": muon(schedule(MUON["lr"] * args.lr_mult, n)),
                                "adam": adam_nobc(schedule(ADAM["lr"] * args.lr_mult, n))}, label_params(model))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return model, opt


def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model, opt, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    opt.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    return loss_fn(model, x, y)


def run(args):
    cfg = dict(CONFIGS[args.config])
    if args.batch:
        cfg["batch"] = args.batch
    B, T = cfg["batch"], cfg["seq"]
    model, opt = build(args, cfg)
    train, val = D.load(args.data_dir, args.steps, B, T, args.val_tokens)
    losses, t0, t_first = [], time.perf_counter(), None
    for s in range(args.steps):
        x, y = D.train_batch(train, s, B, T)
        loss = train_step(model, opt, jnp.asarray(x), jnp.asarray(y))
        if s == 0:
            loss.block_until_ready()
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
        if s % 50 == 0 or s == args.steps - 1:
            losses.append((s, float(loss)))
            if args.verbose:
                print(f"  step {s} loss {float(loss):.4f}", flush=True)
    jax.block_until_ready(loss)
    train_s = time.perf_counter() - t0
    vals = [float(eval_step(model, jnp.asarray(x), jnp.asarray(y))) for x, y in D.val_batches(val, T)]
    v = sum(vals) / len(vals)
    dev = jax.devices()[0]
    rec = dict(platform=args.platform, framework="jax", device=getattr(dev, "device_kind", str(dev)), act=args.act,
               recipe=args.recipe, seed=args.seed, n=args.steps, config=args.config, cfg=cfg, lr_mult=args.lr_mult,
               val_loss=v, step_ms=round(1000 * train_s / max(1, args.steps - 1), 2), first_step_s=round(t_first, 2),
               tokens_per_s=round((args.steps - 1) * B * T / train_s), losses=losses,
               params=sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param))),
               jax=jax.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(f"{rec['time']} {args.act:<9} {args.recipe:<11} seed={args.seed} N={args.steps} val={v:.4f} "
          f"step={rec['step_ms']:.0f}ms tok/s={rec['tokens_per_s']:,} [{rec['device']}]", flush=True)
    return rec


GRIDS = {
    # Same arms as the Mac grid (tiny model), now with REAL int8 next to the simulated formats.
    "small": dict(config="tiny", steps=600, seeds=[1, 2], acts=list(ACTIVATIONS),
                  recipes=["bf16", "int8", "sim_fp8", "sim_int8", "sim_nvfp4", "sim_nvfp4h", "sim_mxfp4"]),
    # GPT-2-small shape, simulated FP4 at a scale the Mac can't reach.
    "scale": dict(config="gpt2s", steps=2000, seeds=[1], acts=["relu2", "lrelu2", "swiglu", "eglu_r2n"],
                  recipes=["bf16", "int8", "sim_nvfp4", "sim_nvfp4h"]),
}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="relu2", choices=list(ACTIVATIONS))
    ap.add_argument("--recipe", default="bf16", choices=RECIPES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--config", default="tiny", choices=list(CONFIGS))
    ap.add_argument("--val-tokens", type=int, default=1 << 19, help="smaller for smoke tests")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr-mult", type=float, default=DEFAULT_LR_MULT)
    ap.add_argument("--data-dir", default=os.environ.get("FP4_DATA", os.path.join(os.path.dirname(HERE), "data")))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "jax_runs.jsonl"))
    ap.add_argument("--platform", default=os.environ.get("FP4_PLATFORM", "tpu"))
    ap.add_argument("--grid", choices=list(GRIDS))
    ap.add_argument("--verbose", action="store_true")
    return ap


def main():
    args = parser().parse_args()
    D.ensure_data(args.data_dir)
    if not args.grid:
        run(args)
        return
    g = GRIDS[args.grid]
    out = args.out.replace("jax_runs.jsonl", f"jax_{args.grid}.jsonl")
    done = {(r["act"], r["recipe"], r["seed"], r["n"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    jobs = [(a, r, s) for s in g["seeds"] for r in g["recipes"] for a in g["acts"] if (a, r, s, g["steps"]) not in done]
    print(f"grid {args.grid}: {len(jobs)} runs to do -> {out}", flush=True)
    for act, recipe, seed in jobs:
        a = parser().parse_args(["--act", act, "--recipe", recipe, "--seed", str(seed), "--steps", str(g["steps"]),
                                 "--config", g["config"], "--out", out, "--data-dir", args.data_dir,
                                 "--lr-mult", str(args.lr_mult), "--platform", args.platform])
        run(a)


if __name__ == "__main__":
    main()
