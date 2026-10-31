"""The ReLU^2 / FP4 study in JAX (Flax NNX + optax + Qwix) for TPUs. Mirrors the Mac trainer (mlx_mac/train_mlx.py and
mlx_mac/lowp.py) and shares common/config.py and common/data.py with it: same model, init scheme, optimizer (Muon on the
blocks' 2-D matrices, Adam without bias correction elsewhere), LR schedule and token stream.

  python jax_tpu/train_jax.py --act relu2 --recipe bf16 --seed 1 --steps 600
  python jax_tpu/train_jax.py --act relu2 --recipe sim_nvfp4 --seed 1 --steps 600   # simulated NVFP4 (any device)
  python jax_tpu/train_jax.py --grid small                                          # the TPU grid; skips finished runs
  JAX_PLATFORMS=cpu python jax_tpu/train_jax.py --grid smoke --data-dir <dir>       # CPU smoke test: every act x recipe

LR schedule: --schedule intended (default; common.config.lr_mult) or mlx_uint64_bug, an exact replica of what the Mac grids
ran before 2026-10-31 (0.1x LR for the first 60% of each run: MLX's uint64 step wrapped around; jax_tpu/LEARNINGS.md).
Parity: from identical weights this trainer matches mlx_mac/train_mlx.py step for step (every activation, every sim format).

Recipes (the format applies to the blocks' hidden Linears qkv, o, fc, proj; embedding, norms, activations, residual stream
and optimizer are fp32 in every recipe):
  fp32     fp32 GEMMs everywhere, with precision HIGHEST so a TPU doesn't round fp32 operands to bf16. = the Mac's baseline
           (recipe "none" in mlx_mac/lowp_grid.py)
  bf16     bf16 GEMM operands, fp32 accumulation, for the hidden Linears, attention and head: the TPU-native baseline
  int8     Qwix quantized training (QtRule): int8 weights, activations and gradients, elementwise stochastic rounding on the
           gradients. Real int8 MXU GEMMs on TPU v5e / v6e. Attention and head bf16, so pair it with the bf16 baseline
  fp8      Qwix QT, e4m3 forward / e5m2 gradients (native only on TPU7x, emulated elsewhere). Not in the grids
  sim_fp8 / sim_int8 / sim_mxfp4 / sim_nvfp4 / sim_nvfp4h
           fake quantization with the numerics of mlx_mac/lowp.py: operands rounded to the format, GEMMs in fp32 (HIGHEST).
           Attention and head fp32, as on the Mac, so pair them with the fp32 baseline. No TPU has FP4 hardware: the FP4
           arms are simulated on every TPU (and on every other device)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common.config import (ACTIVATIONS, ADAM, CONFIGS, DEFAULT_LR_MULT, HEAD_STD, MUON, NORM_EPS, ROPE_BASE,  # noqa: E402
                           SCHEDULE_MODES, SIM_RECIPES, SOFTCAP, VAL_TOKENS, XIELU, hidden_for, lr_mult)
from common import data as D  # noqa: E402

F32, BF16 = jnp.float32, jnp.bfloat16
HI = jax.lax.Precision.HIGHEST
QWIX_RECIPES = {"int8": ("int8", "int8"), "fp8": ("float8_e4m3fn", "float8_e5m2")}
RECIPES = ["fp32", "bf16"] + list(QWIX_RECIPES) + list(SIM_RECIPES)


def hp_of(recipe: str) -> str:
    """Operand precision of the GEMMs outside the format under test (attention, head): bf16 for the TPU-native recipes,
    fp32 where the Mac used fp32 (the fp32 baseline and the simulated formats)."""
    return "bf16" if recipe == "bf16" or recipe in QWIX_RECIPES else "fp32"


def mm(a, b, hp: str):
    """a @ b (b is 2-D) with operands in hp and fp32 accumulation and output."""
    dt = BF16 if hp == "bf16" else F32
    return jnp.matmul(a.astype(dt), b.astype(dt), precision=None if hp == "bf16" else HI, preferred_element_type=F32)


# ----------------------------------------------------------------------------------------------- simulated formats
# Ports of mlx_mac/lowp.py. Exponents come from frexp and powers of two from the bit pattern, so the grid is exact on any
# backend (a TPU's log2 / exp2 are approximations).
def _pow2(e):
    """2.0 ** e for integer-valued e in [-126, 127], exactly."""
    e = jnp.clip(e, -126, 127).astype(jnp.int32)
    return jax.lax.bitcast_convert_type((e + 127) << 23, F32)


def _floor_log2(m):   # m > 0 normal: m = f * 2^e with f in [0.5, 1)
    return jnp.frexp(m)[1] - 1


def _ceil_log2(m):
    f, e = jnp.frexp(m)
    return e - (f == 0.5).astype(e.dtype)


def _round(a, key):
    return jnp.floor(a + jax.random.uniform(key, a.shape, a.dtype)) if key is not None else jnp.round(a)


def _float_q(a, mant, emin, amaxv, key):
    """Round |a| to a float format with `mant` mantissa bits, min normal exponent emin, max value amaxv (lowp._float_q)."""
    m = jnp.abs(a)
    e = jnp.maximum(_floor_log2(jnp.maximum(m, 2.0 ** (emin - mant - 4))), emin)
    return jnp.sign(a) * jnp.minimum(_round(m * _pow2(mant - e), key) * _pow2(e - mant), amaxv)


def e4m3(a, key=None):
    return _float_q(a, 3, -6, 448.0, key)


def e5m2(a, key=None):
    return _float_q(a, 2, -14, 57344.0, key)


def e2m1(a, key=None):
    return _float_q(a, 1, 0, 6.0, key)


def _blocks(t, k):
    return t.reshape(*t.shape[:-1], t.shape[-1] // k, k)


def quant(t, fmt, key=None):
    """Fake-quantize t along its last axis (the GEMM's contraction axis); key != None = stochastic rounding."""
    if fmt == "none":
        return t
    if fmt in ("e4m3", "e5m2"):
        hi = 448.0 if fmt == "e4m3" else 57344.0
        s = jnp.maximum(jnp.max(jnp.abs(t)), 1e-30) / hi
        return (e4m3 if fmt == "e4m3" else e5m2)(t / s, key) * s
    if fmt == "int8":
        s = jnp.maximum(jnp.max(jnp.abs(t), axis=-1, keepdims=True), 1e-30) / 127.0
        return jnp.clip(_round(t / s, key), -127, 127) * s
    if fmt == "mxfp4":
        b = _blocks(t, 32)
        amax = jnp.maximum(jnp.max(jnp.abs(b), axis=-1, keepdims=True), 1e-30)
        s = _pow2(_ceil_log2(amax / 6.0))                      # OCP round-up E8M0 scale, as lowp.py
        return (e2m1(b / s, key) * s).reshape(t.shape)
    if fmt == "nvfp4":
        b = _blocks(t, 16)
        sb = jnp.maximum(jnp.max(jnp.abs(b), axis=-1, keepdims=True), 1e-30) / 6.0
        g = jnp.maximum(jnp.max(sb), 1e-30) / 448.0
        s = jnp.maximum(e4m3(sb / g) * g, 1e-30)               # e4m3 block scale under an fp32 tensor scale
        return (e2m1(b / s, key) * s).reshape(t.shape)
    raise ValueError(fmt)


def _hadamard16(seed=0):
    h = np.array([[1.0]])
    while h.shape[0] < 16:
        h = np.block([[h, h], [h, -h]])
    d = np.random.default_rng(seed).choice([-1.0, 1.0], 16)   # the same signs as lowp.py / quant_sim.py
    return (h * d[None, :] / 4.0).astype(np.float32)          # orthogonal


R16 = _hadamard16()


def _rht(t):
    return jnp.matmul(_blocks(t, 16), jnp.asarray(R16.T), precision=HI).reshape(t.shape)


def _hmm(a, b):
    return jnp.matmul(a, b, precision=HI)


def make_qmm(recipe):
    """y = Q(x) @ Q(W) with a custom VJP that quantizes all three GEMMs along their contraction axes (lowp.make_qmm).
    x (..., d_in) fp32, w (d_in, d_out) fp32 (nnx layout = the transpose of MLX's (out, in) weight), key: SR key."""
    fxw, fg, sr, had = SIM_RECIPES[recipe]

    @jax.custom_vjp
    def qmm(x, w, key):
        return _hmm(quant(x, fxw), quant(w.T, fxw).T)                 # fprop: contraction over d_in

    def fwd(x, w, key):
        return qmm(x, w, key), (x, w, key)

    def bwd(res, g):
        x, w, key = res
        k1, k2 = jax.random.split(key) if sr else (None, None)
        g = g.astype(F32)
        dx = _hmm(quant(g, fg, k1), quant(w, fxw).T)                  # dgrad: contraction over d_out
        g2 = g.reshape(-1, g.shape[-1]).T                              # (d_out, tokens)
        x2 = x.reshape(-1, x.shape[-1]).T                              # (d_in, tokens)
        if had:
            g2, x2 = _rht(g2), _rht(x2)                                # 16-point RHT on the wgrad operands only
        dw = _hmm(quant(g2, fg, k2), quant(x2, fxw).T).T               # wgrad: contraction over tokens -> (d_in, d_out)
        return dx, dw, None

    qmm.defvjp(fwd, bwd)
    return qmm


QMM = {r: make_qmm(r) for r in SIM_RECIPES}


# ----------------------------------------------------------------------------------------------- model
def rms(x):
    x = x.astype(F32)
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + NORM_EPS)


class Linear(nnx.Module):
    """Bias-free Linear, kernel (d_in, d_out), MLX default init U(+-1/sqrt(d_in)) or zeros; the recipe picks the GEMM."""

    def __init__(self, d_in, d_out, recipe, lid, rngs, zero=False):
        b = d_in ** -0.5
        k = jnp.zeros((d_in, d_out), F32) if zero else jax.random.uniform(rngs.params(), (d_in, d_out), F32, -b, b)
        self.kernel = nnx.Param(k)
        self.recipe, self.lid = recipe, lid

    def __call__(self, x, key):
        w = self.kernel[...]
        if self.recipe in SIM_RECIPES:
            return QMM[self.recipe](x.astype(F32), w, jax.random.fold_in(key, self.lid))
        if self.recipe in QWIX_RECIPES:
            # Qwix intercepts this dot_general (rule on the module path blocks/<i>/<name>) and quantizes x, w and dy.
            return jax.lax.dot_general(x.astype(F32), w, (((x.ndim - 1,), (0,)), ((), ())))
        return mm(x, w, self.recipe)                                   # "fp32" or "bf16"


def activation(act, h, xielu_raw=None):
    h = h.astype(F32)
    if act == "relu2":
        return jnp.square(jax.nn.relu(h))
    if act == "lrelu2":
        return jnp.square(jnp.where(h > 0, h, 0.5 * h))
    if act == "xielu":   # mlx_mac/eml_kernels._xielu_t: a_p = softplus(raw_p), a_n = beta + softplus(raw_n)
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


def rope(x):   # x (B, T, H, hd); non-interleaved halves, as mx.fast.rope(traditional=False, base=10000)
    hd = x.shape[-1]
    inv = ROPE_BASE ** (-jnp.arange(0, hd // 2, dtype=F32) * 2 / hd)
    f = jnp.outer(jnp.arange(x.shape[1], dtype=F32), inv)[None, :, None, :]
    c, s = jnp.cos(f), jnp.sin(f)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def attention(q, k, v, hp):
    """Causal softmax attention, (B, T, H, hd) in and out; operands in hp, scores and softmax in fp32."""
    dt, prec = (BF16, None) if hp == "bf16" else (F32, HI)
    T = q.shape[1]
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(dt), k.astype(dt), precision=prec, preferred_element_type=F32)
    s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s * q.shape[-1] ** -0.5, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", p.astype(dt), v.astype(dt), precision=prec, preferred_element_type=F32)


class Block(nnx.Module):
    def __init__(self, d, n_head, act, recipe, layer, rngs):
        self.n_head, self.act, self.hp = n_head, act, hp_of(recipe)
        lin = partial(Linear, recipe=recipe, rngs=rngs)
        fc_out, proj_in = hidden_for(act, d)
        self.qkv = lin(d, 3 * d, lid=4 * layer)
        self.o = lin(d, d, lid=4 * layer + 1, zero=True)               # zero-init projections (modded-nanogpt)
        self.fc = lin(d, fc_out, lid=4 * layer + 2)
        self.proj = lin(proj_in, d, lid=4 * layer + 3, zero=True)
        self.lam = nnx.Param(jnp.ones((2,), F32))
        if act == "xielu":   # trainable per layer (1-d, so Adam), alpha_p = alpha_n = 0.8 at init
            self.xielu_raw = nnx.Param(jnp.array([math.log(math.expm1(XIELU["alpha_p"])),
                                                  math.log(math.expm1(XIELU["alpha_n"] - XIELU["beta"]))], F32))

    def __call__(self, x, key):
        B, T, Dm = x.shape
        H, hd = self.n_head, Dm // self.n_head
        qkv = self.qkv(rms(x), key).reshape(B, T, 3, H, hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        a = attention(rope(rms(q)), rope(rms(k)), v, self.hp)
        x = x + self.lam[0] * self.o(a.reshape(B, T, Dm), key)
        g = activation(self.act, self.fc(rms(x), key), self.xielu_raw[...] if self.act == "xielu" else None)
        return x + self.lam[1] * self.proj(g, key)


class GPT(nnx.Module):
    def __init__(self, cfg, act, recipe, rngs, remat=False):
        d = cfg["d"]
        self.embed = nnx.Param(jax.random.normal(rngs.params(), (cfg["vocab"], d), F32) * d ** -0.5)   # nn.Embedding
        self.blocks = nnx.List([Block(d, cfg["heads"], act, recipe, i, rngs) for i in range(cfg["layers"])])
        self.head = nnx.Param(jax.random.normal(rngs.params(), (d, cfg["vocab"]), F32) * HEAD_STD)
        self.hp, self.remat = hp_of(recipe), remat

    def __call__(self, idx, key):
        x = rms(self.embed[...][idx])
        for b in self.blocks:
            x = nnx.remat(lambda blk, x_, k_: blk(x_, k_))(b, x, key) if self.remat else b(x, key)
        logits = mm(rms(x), self.head[...], self.hp)
        return SOFTCAP * jnp.tanh(logits / SOFTCAP)


def xent(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits.astype(F32), y).mean()


# ----------------------------------------------------------------------------------------------- optimizer (MLX semantics)
def lr_schedule(base, n, mode="intended"):
    """step -> LR, from common.config.lr_mult (a lookup table, so every trainer uses the same numbers). mode
    "mlx_uint64_bug" replicates the Mac grids run before 2026-10-31 (0.1x LR until the cooldown; jax_tpu/LEARNINGS.md)."""
    table = jnp.asarray([base * lr_mult(s, n, mode) for s in range(n)], F32)
    return lambda step: table[jnp.clip(step, 0, n - 1)]


def newton_schulz(G):
    """mlx.optimizers.Muon._zeropower_via_newtonschulz5, in fp32 (HIGHEST): iterate on the wide orientation."""
    a, b, c = MUON["ns_coeffs"]
    X = G.astype(F32)
    tall = X.shape[0] > X.shape[1]
    X = X.T if tall else X
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(MUON["ns_steps"]):
        A = jnp.matmul(X, X.T, precision=HI)
        X = a * X + jnp.matmul(b * A + c * jnp.matmul(A, A, precision=HI), X, precision=HI)
    return X.T if tall else X


def muon(lr_fn):
    """mlx.optimizers.Muon (weight_decay 0, Nesterov). Kernels are (d_in, d_out) = MLX's weight transposed, so MLX's
    lr scale sqrt(max(1, rows / cols)) of the (out, in) weight is sqrt(max(1, d_out / d_in)) here."""
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
    """mlx.optimizers.AdamW(weight_decay=0): Adam without bias correction."""
    b1, b2, eps = ADAM["b1"], ADAM["b2"], ADAM["eps"]

    def init(params):
        return dict(m=jax.tree.map(jnp.zeros_like, params), v=jax.tree.map(jnp.zeros_like, params),
                    step=jnp.zeros([], jnp.int32))

    def update(grads, state, params=None):
        lr = lr_fn(state["step"])
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
        return jax.tree.map(lambda m, v: -lr * m / (jnp.sqrt(v) + eps), m, v), dict(m=m, v=v, step=state["step"] + 1)

    return optax.GradientTransformation(init, update)


def label_params(params):
    """'muon' for the blocks' 2-D kernels, 'adam' for everything else (train_mlx.make_optimizer's is_hidden)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, x: "muon" if (path[0].key == "blocks" and x.ndim == 2) else "adam", params)


def make_tx(params, n, mult, schedule="intended"):
    return optax.multi_transform({"muon": muon(lr_schedule(MUON["lr"] * mult, n, schedule)),
                                  "adam": adam_nobc(lr_schedule(ADAM["lr"] * mult, n, schedule))},
                                 label_params(params))


# ----------------------------------------------------------------------------------------------- training
def build_model(cfg, act, recipe, seed, remat=False):
    model = GPT(cfg, act, recipe, nnx.Rngs(params=seed), remat=remat)
    provider = None
    if recipe in QWIX_RECIPES:
        import qwix
        wq, gq = QWIX_RECIPES[recipe]
        rule = qwix.QtRule(module_path=r"blocks/\d+/(qkv|o|fc|proj)", weight_qtype=wq, act_qtype=wq, bwd_qtype=gq,
                           # elementwise noise (Qwix's default (0,) shares one draw per batch row), like the sims
                           bwd_stochastic_rounding="uniform" if recipe == "int8" else None,
                           channelwise_noise_axes=(0, 1, 2))
        provider = qwix.QtProvider([rule])
        x0 = jnp.zeros((1, 64), jnp.int32)
        model = qwix.quantize_model(model, provider, x0, jax.random.key(0), rngs=nnx.Rngs(0))
    return model, provider


def make_fns(graphdef, recipe, tx, accum):
    def loss_fn(params, rest, x, y, key):
        model = nnx.merge(graphdef, params, rest)
        if recipe in QWIX_RECIPES:   # Qwix draws its SR keys from this stream (flax_util.make_rng)
            model.set_attributes(qwix_rngs=nnx.Rngs(stochastic_rounding=key))
        return xent(model(x, key), y)

    vg = jax.value_and_grad(loss_fn)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, rest, x, y, base_key, step):
        key = jax.random.fold_in(base_key, step)
        if accum == 1:
            loss, grads = vg(params, rest, x, y, key)
        else:   # micro-batches of B / accum sequences; the mean of equal-size means = the full-batch mean
            xs, ys = x.reshape(accum, -1, x.shape[-1]), y.reshape(accum, -1, y.shape[-1])

            def body(carry, i):
                l, g = vg(params, rest, xs[i], ys[i], jax.random.fold_in(key, i))
                return (carry[0] + l, jax.tree.map(jnp.add, carry[1], g)), None
            (loss, grads), _ = jax.lax.scan(body, (jnp.zeros([], F32), jax.tree.map(jnp.zeros_like, params)),
                                            jnp.arange(accum))
            loss, grads = loss / accum, jax.tree.map(lambda g: g / accum, grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_step(params, rest, x, y):
        return loss_fn(params, rest, x, y, jax.random.key(0))   # forward only: no SR draws matter

    return train_step, eval_step


def device_tag():
    """'v5e', 'v6e', 'cpu', ... from the first device's kind (TPU v5 lite = v5e, TPU v6 lite = v6e)."""
    k = getattr(jax.devices()[0], "device_kind", jax.default_backend()).lower()
    for pat, tag in (("v5 lite", "v5e"), ("v5e", "v5e"), ("v6 lite", "v6e"), ("v6e", "v6e"), ("v5p", "v5p"),
                     ("v5", "v5p"), ("v4", "v4"), ("v3", "v3"), ("v2", "v2"), ("7x", "tpu7x")):
        if pat in k:
            return tag
    return jax.default_backend()


# Activation memory per sequence of one train step (GiB), measured from the CPU compile's memory analysis
# (jax_tpu/LEARNINGS.md); TPU XLA usually needs less. Used to pick micro-batches so a step fits the chip.
GIB_PER_SEQ = {("tiny", False): 0.56, ("tiny", True): 0.3, ("gpt2s", False): 3.4, ("gpt2s", True): 1.0}
GIB_FIXED = {"tiny": 1.2, "gpt2s": 6.0}   # params, grads, Adam / Muon state


def auto_accum(config, B, remat):
    """Smallest power-of-two micro-batch count whose estimated step memory fits in 75% of the device's memory."""
    try:
        limit = jax.devices()[0].memory_stats()["bytes_limit"] / 2 ** 30
    except Exception:   # CPU: no limit reported
        return 1
    per = GIB_PER_SEQ.get((config, remat), 1.0)
    accum = 1
    while accum < B and B % (2 * accum) == 0 and per * B / accum + GIB_FIXED.get(config, 2.0) > 0.75 * limit:
        accum *= 2
    return accum


def run(args):
    cfg = dict(CONFIGS[args.config])
    if args.batch:
        cfg["batch"] = args.batch
    B, T = cfg["batch"], cfg["seq"]
    remat = bool(args.remat) if args.remat is not None else args.config != "tiny"
    accum = args.accum or auto_accum(args.config, B, remat)
    assert B % accum == 0, (B, accum)
    model, _ = build_model(cfg, args.act, args.recipe, args.seed, remat)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    params = nnx.to_pure_dict(params)
    tx = make_tx(params, args.steps, args.lr_mult, args.schedule)
    opt_state = tx.init(params)
    train_step, eval_step = make_fns(graphdef, args.recipe, tx, accum)
    base_key = jax.random.key(1000 + args.seed)
    train, val = D.load(args.data_dir, args.steps, B, T, args.val_tokens)
    losses, t0, t_first, loss = [], time.perf_counter(), None, None
    for s in range(args.steps):
        x, y = D.train_batch(train, s, B, T)
        params, opt_state, loss = train_step(params, opt_state, rest, jnp.asarray(x), jnp.asarray(y), base_key,
                                             jnp.asarray(s, jnp.int32))
        if s == 0:
            loss.block_until_ready()
            t_first = time.perf_counter() - t0   # includes compilation
            t0 = time.perf_counter()
        if s % args.log_every == 0 or s == args.steps - 1:
            losses.append((s, float(loss)))
            if args.verbose:
                print(f"  step {s} loss {losses[-1][1]:.4f}", flush=True)
    jax.block_until_ready(loss)
    train_s = time.perf_counter() - t0
    t_eval = time.perf_counter()
    vals = [float(eval_step(params, rest, jnp.asarray(x), jnp.asarray(y))) for x, y in D.val_batches(val, T)]
    v = sum(vals) / len(vals)
    t_eval = time.perf_counter() - t_eval
    dev = jax.devices()[0]
    nsteps = max(1, args.steps - 1)
    rec = dict(platform=args.platform or jax.default_backend(), framework="jax", device=getattr(dev, "device_kind", str(dev)),
               device_tag=device_tag(), n_devices=jax.device_count(), act=args.act, recipe=args.recipe, hp=hp_of(args.recipe),
               seed=args.seed, n=args.steps, config=args.config, cfg=cfg, accum=accum, remat=remat, lr_mult=args.lr_mult,
               schedule=args.schedule,
               val_loss=v, val_tokens=len(vals) * 16 * T, step_ms=round(1000 * train_s / nsteps, 2),
               first_step_s=round(t_first, 2), eval_s=round(t_eval, 2), tokens_per_s=round(nsteps * B * T / max(train_s, 1e-9)), losses=losses,
               params=sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params)),
               jax=jax.__version__, flax=_ver("flax"), qwix=_ver("qwix"), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(f"{rec['time']} {args.act:<9} {args.recipe:<11} seed={args.seed} N={args.steps} val={v:.4f} "
          f"step={rec['step_ms']:.0f}ms first={rec['first_step_s']:.0f}s tok/s={rec['tokens_per_s']:,} [{rec['device']}]",
          flush=True)
    return rec


def _ver(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


GRIDS = {
    # CPU / first-contact check: every activation x recipe, a few steps on a small batch.
    "smoke": dict(config="tiny", steps=4, seeds=[1], acts=list(ACTIVATIONS), recipes=RECIPES, batch=4,
                  val_tokens=1 << 14),
    # The notebook's 20-step TPU check: one activation, the four kinds of recipe, the full batch; gives step times.
    "tpu20": dict(config="tiny", steps=20, seeds=[1], acts=["relu2"], recipes=["fp32", "bf16", "int8", "sim_nvfp4"],
                  val_tokens=1 << 16),
    # The Mac grid's arms (tiny model, N = 600, seeds 1-2), with REAL int8 next to the simulated formats. fp32 is the
    # Mac's baseline (pair the sim_* arms with it); bf16 is the TPU-native baseline (pair int8 with it).
    "small": dict(config="tiny", steps=600, seeds=[1, 2], acts=list(ACTIVATIONS),
                  recipes=["fp32", "bf16", "int8", "sim_fp8", "sim_int8", "sim_nvfp4", "sim_nvfp4h", "sim_mxfp4"]),
    # Cross-check with the Mac grid as it ran (mlx_mac/results/lowp_A.jsonl, made before the LR-schedule fix): the same
    # arms with the MLX uint64 schedule bug replicated. fp32 has the Mac's exact semantics; bf16 is the TPU baseline.
    "xcheck": dict(config="tiny", steps=600, seeds=[1, 2], acts=["relu2", "lrelu2"], recipes=["fp32", "bf16", "sim_nvfp4"],
                   schedule="mlx_uint64_bug"),
    # GPT-2-small shape: simulated FP4 at a scale the Mac can't reach (per-block remat; --accum on a 16 GB v5e).
    "scale": dict(config="gpt2s", steps=2000, seeds=[1], acts=["relu2", "lrelu2", "swiglu", "eglu_r2n"],
                  recipes=["bf16", "int8", "fp32", "sim_nvfp4", "sim_nvfp4h"]),
}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", default="relu2", choices=list(ACTIVATIONS))
    ap.add_argument("--recipe", default="bf16", choices=RECIPES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--config", default="tiny", choices=list(CONFIGS))
    ap.add_argument("--val-tokens", type=int, default=VAL_TOKENS, help="smaller for smoke tests")
    ap.add_argument("--batch", type=int, default=0, help="sequences per step (0 = the config's)")
    ap.add_argument("--accum", type=int, default=0, help="micro-batches per step (same math, less memory; 0 = 1)")
    ap.add_argument("--remat", type=int, default=None, help="per-block rematerialization 0/1 (default: on except tiny)")
    ap.add_argument("--lr-mult", type=float, default=DEFAULT_LR_MULT)
    ap.add_argument("--schedule", default="intended", choices=SCHEDULE_MODES,
                    help="intended = common.config.lr_mult; mlx_uint64_bug = what the Mac grids ran before 2026-10-31 "
                         "(0.1x LR until the cooldown), to compare with those results")
    ap.add_argument("--data-dir", default=os.environ.get("FP4_DATA", os.path.join(ROOT, "data")))
    ap.add_argument("--out", default=None, help="jsonl to append to (default results/jax_runs.jsonl, or "
                                                "results/jax_<grid>.jsonl with --grid)")
    ap.add_argument("--platform", default=os.environ.get("FP4_PLATFORM"), help="label (default: the JAX backend)")
    ap.add_argument("--grid", choices=list(GRIDS))
    ap.add_argument("--acts", default=None, help="comma list: restrict the grid's activations")
    ap.add_argument("--recipes", default=None, help="comma list: restrict the grid's recipes")
    ap.add_argument("--seeds", default=None, help="comma list: override the grid's seeds")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--cache-dir", default=os.environ.get("JAX_CACHE_DIR", os.path.expanduser("~/.cache/relu2_fp4_jax")),
                    help="persistent XLA compilation cache ('' = off); seed 2 reuses seed 1's compile")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main():
    args = parser().parse_args()
    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    D.ensure_data(args.data_dir)
    if not args.grid:
        args.out = args.out or os.path.join(ROOT, "results", "jax_runs.jsonl")
        run(args)
        return
    g = GRIDS[args.grid]
    out = args.out or os.path.join(ROOT, "results", f"jax_{args.grid}.jsonl")
    acts = args.acts.split(",") if args.acts else g["acts"]
    recipes = args.recipes.split(",") if args.recipes else g["recipes"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else g["seeds"]
    steps = g["steps"] if args.steps == parser().get_default("steps") else args.steps
    schedule = g.get("schedule", args.schedule)
    dev = getattr(jax.devices()[0], "device_kind", jax.default_backend())
    done = set()
    if os.path.exists(out):   # done per device kind, so one file can hold a v5e and a v6e pass
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["act"], r["recipe"], r["seed"], r["n"], r["config"], r.get("device"), r.get("schedule")))
            except (json.JSONDecodeError, KeyError):
                pass
    jobs = [(a, r, s) for s in seeds for r in recipes for a in acts
            if (a, r, s, steps, g["config"], dev, schedule) not in done]
    print(f"grid {args.grid} ({schedule} LR schedule) on {dev}: {len(jobs)} runs to do ({len(done)} records already in {out})", flush=True)
    failed = []
    for act, recipe, seed in jobs:
        argv = ["--act", act, "--recipe", recipe, "--seed", str(seed), "--steps", str(steps), "--config", g["config"],
                "--out", out, "--data-dir", args.data_dir, "--lr-mult", str(args.lr_mult), "--schedule", schedule, "--log-every", str(args.log_every),
                "--val-tokens", str(g.get("val_tokens", args.val_tokens)), "--cache-dir", ""]
        if args.batch or g.get("batch"):
            argv += ["--batch", str(args.batch or g["batch"])]
        if args.accum:
            argv += ["--accum", str(args.accum)]
        if args.remat is not None:
            argv += ["--remat", str(args.remat)]
        if args.platform:
            argv += ["--platform", args.platform]
        if args.verbose:
            argv += ["--verbose"]
        try:
            try:
                run(parser().parse_args(argv))
            except Exception as e:   # out of device memory: retry with twice the micro-batches (same math)
                if "RESOURCE_EXHAUSTED" not in str(e) and "out of memory" not in str(e).lower():
                    raise
                a = parser().parse_args(argv)
                a.accum = 2 * (a.accum or auto_accum(a.config, a.batch or CONFIGS[a.config]["batch"],
                                                     bool(a.remat) if a.remat is not None else a.config != "tiny"))
                print(f"  out of memory: retrying with --accum {a.accum}", flush=True)
                run(a)
        except Exception as e:   # keep the grid going; the traceback goes next to the results
            failed.append((act, recipe, seed))
            print(f"FAILED {act} {recipe} seed={seed}: {type(e).__name__}: {e}", flush=True)
            with open(out.replace(".jsonl", "_errors.log"), "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {dev} {act} {recipe} seed={seed}\n{traceback.format_exc()}\n")
    print(f"grid {args.grid}: {len(jobs) - len(failed)} done, {len(failed)} failed {failed if failed else ''}", flush=True)


if __name__ == "__main__":
    main()
