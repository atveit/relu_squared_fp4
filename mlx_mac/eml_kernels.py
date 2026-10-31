"""The "Dual-Space EML" MLP gate and the ReLU^2 baseline, in MLX: a reference, an mx.compile version (how mlx-lm
writes swiglu) and a fused Metal kernel with a custom backward (how mlx-lm writes its kernels/ and tuner/losses.py).

The gate reads the packed c_fc output h = [u | v] (last dim 2H) and returns (last dim H):
    eml(u, v) = exp(tanh(u)) - log1p(relu(v)) - offset
offset = 0 is the proposal as written (it outputs 1 at zero input); offset = 1 centres it (0 at zero input).
Gradients: d/du = exp(tanh u) * (1 - tanh(u)^2); d/dv = -1 / (1 + v) for v > 0, else 0.
"""
from functools import partial

import math

import mlx.core as mx
import mlx.nn as nn


# ----------------------------------------------------------------------------------------------- reference / compiled
def eml_ref(h: mx.array, offset: float = 0.0) -> mx.array:
    u, v = mx.split(h, 2, axis=-1)
    return mx.exp(mx.tanh(u)) - mx.log1p(mx.maximum(v, 0.0)) - offset


@partial(mx.compile, shapeless=True)
def _eml_uv(u: mx.array, v: mx.array, offset: mx.array) -> mx.array:
    return mx.exp(mx.tanh(u)) - mx.log1p(mx.maximum(v, 0.0)) - offset


def eml_compiled(h: mx.array, offset: mx.array) -> mx.array:
    # Split outside the shapeless compile (it cannot trace split), as mlx-lm's swiglu(gate, x) does.
    H = h.shape[-1] // 2
    return _eml_uv(h[..., :H], h[..., H:], offset)


@partial(mx.compile, shapeless=True)
def relu2_compiled(x: mx.array) -> mx.array:
    return mx.square(nn.relu(x))


# ----------------------------------------------------------------------------------------------- fused Metal kernel
# One thread per output element: read u and v from the packed row, fp32 math, one rounding on store.
_FWD_SRC = """
    // 2-D grid: x = column block of E, y = row; no per-element div/mod. H is a multiple of E.
    uint c0 = thread_position_in_grid.x * E;
    uint r = thread_position_in_grid.y;
    if (c0 >= H) return;
    const device T* hu = h + r * (2 * H) + c0;
    const device T* hv = hu + H;
    device T* o = out + r * H + c0;
    #pragma unroll
    for (int j = 0; j < E; j++) {
        float u = static_cast<float>(hu[j]);
        float v = static_cast<float>(hv[j]);
        o[j] = static_cast<T>(metal::exp(metal::precise::tanh(u)) - metal::log(1.0f + metal::max(v, 0.0f)) - offset);
    }
"""

# Backward: dh = [g * exp(t) * (1 - t^2) | -g * (v > 0) / (1 + v)], with t = tanh(u).
_BWD_SRC = """
    uint c0 = thread_position_in_grid.x * E;
    uint r = thread_position_in_grid.y;
    if (c0 >= H) return;
    const device T* hu = h + r * (2 * H) + c0;
    const device T* hv = hu + H;
    const device T* gr = g + r * H + c0;
    device T* du = dh + r * (2 * H) + c0;
    device T* dv = du + H;
    #pragma unroll
    for (int j = 0; j < E; j++) {
        float gj = static_cast<float>(gr[j]);
        float t = metal::precise::tanh(static_cast<float>(hu[j]));
        float v = static_cast<float>(hv[j]);
        du[j] = static_cast<T>(gj * metal::exp(t) * (1.0f - t * t));
        dv[j] = static_cast<T>(v > 0.0f ? -gj / (1.0f + v) : 0.0f);
    }
"""

_fwd_kernel = mx.fast.metal_kernel(name="eml_gate_fwd", input_names=["h", "offset"], output_names=["out"],
                                   source=_FWD_SRC, ensure_row_contiguous=True)
_bwd_kernel = mx.fast.metal_kernel(name="eml_gate_bwd", input_names=["h", "g"], output_names=["dh"],
                                   source=_BWD_SRC, ensure_row_contiguous=True)
_TG, _E = 256, 2   # defaults from smoke.py S3 (E=2, TG=256 best at d=768; E=4 within 4%)
_CFG = {"tg": _TG, "e": _E}


def set_metal_config(tg: int = None, e: int = None):
    """For smoke-test sweeps: threadgroup width and elements per thread of the EML Metal kernels."""
    if tg:
        _CFG["tg"] = tg
    if e:
        _CFG["e"] = e


def _rows(h):
    return h.size // h.shape[-1]


@mx.custom_function
def eml_metal(h: mx.array, offset: mx.array) -> mx.array:
    H = h.shape[-1] // 2
    E, TG = _CFG["e"], _CFG["tg"]
    assert H % E == 0, "eml_metal: H must be a multiple of E"
    rows = _rows(h)
    out = _fwd_kernel(inputs=[h.reshape(rows, 2 * H), offset], template=[("T", h.dtype), ("H", H), ("E", E)],
                      grid=(H // E, rows, 1), threadgroup=(min(TG, H // E), 1, 1),
                      output_shapes=[(rows, H)], output_dtypes=[h.dtype])[0]
    return out.reshape(*h.shape[:-1], H)


@eml_metal.vjp
def _eml_metal_vjp(primals, cotangent, output):
    h, offset = primals
    H = h.shape[-1] // 2
    E, TG = _CFG["e"], _CFG["tg"]
    rows = _rows(h)
    dh = _bwd_kernel(inputs=[h.reshape(rows, 2 * H), cotangent.reshape(rows, H)], template=[("T", h.dtype), ("H", H), ("E", E)],
                     grid=(H // E, rows, 1), threadgroup=(min(TG, H // E), 1, 1),
                     output_shapes=[(rows, 2 * H)], output_dtypes=[h.dtype])[0]
    return dh.reshape(h.shape), mx.zeros_like(offset)


# ReLU^2 with the same kernel style, so the baseline is as optimized as the EML gate (fair comparison).
_R2_FWD = """
    uint c0 = thread_position_in_grid.x * E;
    uint r = thread_position_in_grid.y;
    if (c0 >= N) return;
    #pragma unroll
    for (int j = 0; j < E; j++) {
        float a = metal::max(static_cast<float>(x[r * N + c0 + j]), 0.0f);
        out[r * N + c0 + j] = static_cast<T>(a * a);
    }
"""
_R2_BWD = """
    uint c0 = thread_position_in_grid.x * E;
    uint r = thread_position_in_grid.y;
    if (c0 >= N) return;
    #pragma unroll
    for (int j = 0; j < E; j++) {
        float a = metal::max(static_cast<float>(x[r * N + c0 + j]), 0.0f);
        dx[r * N + c0 + j] = static_cast<T>(2.0f * a * static_cast<float>(g[r * N + c0 + j]));
    }
"""
_r2_fwd = mx.fast.metal_kernel(name="relu2_fwd", input_names=["x"], output_names=["out"], source=_R2_FWD,
                               ensure_row_contiguous=True)
_r2_bwd = mx.fast.metal_kernel(name="relu2_bwd", input_names=["x", "g"], output_names=["dx"], source=_R2_BWD,
                               ensure_row_contiguous=True)


@mx.custom_function
def relu2_metal(x: mx.array) -> mx.array:
    N = x.shape[-1]
    E, TG = _CFG["e"], _CFG["tg"]
    rows = _rows(x)
    return _r2_fwd(inputs=[x.reshape(rows, N)], template=[("T", x.dtype), ("N", N), ("E", E)],
                   grid=(N // E, rows, 1), threadgroup=(min(TG, N // E), 1, 1),
                   output_shapes=[(rows, N)], output_dtypes=[x.dtype])[0].reshape(x.shape)


@relu2_metal.vjp
def _relu2_metal_vjp(primals, cotangent, output):
    # One input: MLX passes the array itself (not a tuple) and expects one gradient back.
    x = primals[0] if isinstance(primals, (tuple, list)) else primals
    N = x.shape[-1]
    E, TG = _CFG["e"], _CFG["tg"]
    rows = _rows(x)
    dx = _r2_bwd(inputs=[x.reshape(rows, N), cotangent.reshape(rows, N)], template=[("T", x.dtype), ("N", N), ("E", E)],
                 grid=(N // E, rows, 1), threadgroup=(min(TG, N // E), 1, 1),
                 output_shapes=[(rows, N)], output_dtypes=[x.dtype])[0].reshape(x.shape)
    return (dx,) if isinstance(primals, (tuple, list)) else dx


# ---------------------------------------------------------------- variants for the quality hypotheses (compiled versions)
@partial(mx.compile, shapeless=True)
def _eglu(u, v):            # H1: multiplicative gate instead of additive merge
    return mx.exp(mx.tanh(u)) * v


@partial(mx.compile, shapeless=True)
def _eglu_r2(u, v):         # H1: bounded exp gate modulating a ReLU^2 channel
    return mx.exp(mx.tanh(u)) * mx.square(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _eglu0_r2(u, v):        # stage 1: the EML gate shifted to (0, 3.72), 1 at u = 0, so it can switch a channel off
    return (mx.exp(mx.tanh(u)) - 0.36787944) * 1.58197671 * mx.square(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _s2glu_r2(u, v):        # stage 1 control: the same gated ReLU^2 with a plain 2*sigmoid gate (1 at u = 0, like #393)
    return 2.0 * mx.sigmoid(u) * mx.square(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _eglu_r2n(u, v):        # catalogue #3: gate (e^tanh u - 1/e)/(e - 1/e) in (0, 1); output never exceeds relu(v)^2 (FP8-safe)
    return (mx.exp(mx.tanh(u)) - 0.36787944) * 0.42545906 * mx.square(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _relu2glu(u, v):        # catalogue #4 (calibrator): relu(u)^2 * v, a ReLU^2 gate on a linear channel
    return mx.square(mx.maximum(u, 0.0)) * v


@partial(mx.compile, shapeless=True)
def _eml_mul(u, v):         # H1 diagnostic: the proposal's two branches, multiplied instead of subtracted
    return mx.exp(mx.tanh(u)) * mx.log1p(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _eml_sep(a, b):       # H1 alone: the same two functions, kept as separate channels (no merge)
    return mx.concatenate([mx.exp(mx.tanh(a)) - 1.0, -mx.log1p(mx.maximum(b, 0.0))], axis=-1)


@partial(mx.compile, shapeless=True)
def _logr2(x):              # outlier compression of ReLU^2 (the FP8 crest-factor motivation)
    return mx.log1p(mx.square(mx.maximum(x, 0.0)))


@partial(mx.compile, shapeless=True)
def _sp2_b4(x):             # sharpened softplus^2: (log1p(e^{4x}) / 4)^2. Monotone, >= 0, gradient for x < 0, and the
    return mx.square(mx.logaddexp(4.0 * x, 0.0) / 4.0)   # gradient is recoverable from post alone (no sign mask)


@partial(mx.compile, shapeless=True)
def _elu2_e25(x):           # "ELU-square": (x + e)^2 for x > 0, e^2 exp(2x / e) for x < 0 (e = 0.25; C1 at 0). The branch is
    e = 0.25                 # post < e^2, so the backward needs no mask; gradient 2 sqrt(post) or (2 / e) post
    return mx.where(x > 0, mx.square(x + e), (e * e) * mx.exp(2.0 * mx.minimum(x, 0.0) / e))


@partial(mx.compile, shapeless=True)
def _ssq_a50(x):            # signed square, leaky: x^2 for x > 0, -(0.5 x)^2 for x < 0. Post carries the sign, so a fused
    return mx.where(x > 0, mx.square(x), -0.25 * mx.square(x))   # FP8 kernel needs no sign mask (see lrelu2_record)


@partial(mx.compile, shapeless=True)
def _ssq_a25(x):
    return mx.where(x > 0, mx.square(x), -0.0625 * mx.square(x))


@partial(mx.compile, shapeless=True)
def _lrelu2_a25(x):         # LeakyReLU(0.25)^2: slope sweep for the #393 record attempt
    return mx.square(mx.where(x > 0, x, 0.25 * x))


@partial(mx.compile, shapeless=True)
def _lrelu2_a75(x):         # LeakyReLU(0.75)^2
    return mx.square(mx.where(x > 0, x, 0.75 * x))


@partial(mx.compile, shapeless=True)
def _lrelu2(x):             # LeakyReLU(0.5)^2: every OpenAI parameter-golf record since 2026-03-23 (PRs #2135, #2014)
    return mx.square(mx.where(x > 0, x, 0.5 * x))


@partial(mx.compile, shapeless=True)
def _xielu_t(x, ap_raw, an_raw):   # xIELU (arXiv 2411.13010) as in mlx-lm: a_p = softplus(raw), a_n = beta + softplus(raw)
    ap, an, beta = mx.logaddexp(ap_raw, 0.0), 0.5 + mx.logaddexp(an_raw, 0.0), 0.5
    return mx.where(x > 0, ap * mx.square(x) + beta * x, (mx.expm1(mx.minimum(x, -1e-6)) - x) * an + beta * x)


# Raw values for a_p = a_n = 0.8 at init (mlx-lm stores the inverse softplus). The Block makes them trainable per layer;
# the earlier fixed version used a_p = 1.171, a_n = 1.671 (softplus applied twice) and is retired.
XIELU_RAW = (math.log(math.expm1(0.8)), math.log(math.expm1(0.3)))


def _xielu(x):
    return _xielu_t(x, mx.array(XIELU_RAW[0]), mx.array(XIELU_RAW[1]))


# Fixes for the additive EML's specific weaknesses (still additive, packed 2H -> H):
@partial(mx.compile, shapeless=True)
def _eml_t(u, v):           # saturation: exp(3 tanh(u/3)) saturates later (range ~0.05..20)
    return mx.exp(3.0 * mx.tanh(u / 3.0)) - mx.log1p(mx.maximum(v, 0.0))


@partial(mx.compile, shapeless=True)
def _eml_sym(u, v):         # weak v-branch: signed log (symlog) keeps the compression, restores sign and gradient
    return mx.exp(mx.tanh(u)) - mx.sign(v) * mx.log1p(mx.abs(v))


@partial(mx.compile, shapeless=True)
def _eml_ts(u, v):          # both fixes
    return mx.exp(3.0 * mx.tanh(u / 3.0)) - mx.sign(v) * mx.log1p(mx.abs(v))


# Reference gated units (Shazeer 2020, "GLU Variants Improve Transformer"): calibrate where the EML gates land.
@partial(mx.compile, shapeless=True)
def _swiglu(u, v):
    return u * mx.sigmoid(u) * v


@partial(mx.compile, shapeless=True)
def _reglu(u, v):
    return mx.maximum(u, 0.0) * v


@partial(mx.compile, shapeless=True)
def _bilinear(u, v):
    return u * v


def _halves(fn):
    def g(h):
        H = h.shape[-1] // 2
        return fn(h[..., :H], h[..., H:])
    return g


VARIANTS = {"eglu": _halves(_eglu), "eglu_r2": _halves(_eglu_r2), "eml_sep": _halves(_eml_sep),
            "eglu0_r2": _halves(_eglu0_r2), "s2glu_r2": _halves(_s2glu_r2),
            "eglu_r2n": _halves(_eglu_r2n), "relu2glu": _halves(_relu2glu), "eml_mul": _halves(_eml_mul),
            "logr2": _logr2, "xielu": _xielu, "lrelu2": _lrelu2, "lrelu2_a25": _lrelu2_a25, "ssq_a50": _ssq_a50, "sp2_b4": _sp2_b4, "elu2_e25": _elu2_e25, "ssq_a25": _ssq_a25, "lrelu2_a75": _lrelu2_a75,
            "swiglu": _halves(_swiglu), "reglu": _halves(_reglu), "bilinear": _halves(_bilinear),
            "eml_t": _halves(_eml_t), "eml_sym": _halves(_eml_sym), "eml_ts": _halves(_eml_ts)}
PACKED = {"eglu", "eglu_r2", "eglu0_r2", "s2glu_r2", "eglu_r2n", "relu2glu", "eml_mul", "swiglu", "reglu", "bilinear", "eml_t", "eml_sym", "eml_ts"}   # c_fc -> 2H, gate -> H (like eml); the others keep ReLU^2's 4d -> 4d shapes


def make_gate(kind: str):
    """'relu2' -> (fn, hidden multiplier 4); 'eml'/'eml_c' (+ '_metal'/'_compiled') -> packed gate (hidden 8/3, so 2 of 3
    hidden blocks feed the gate and the parameter count matches ReLU^2's 4x MLP)."""
    if kind == "relu2":
        return relu2_compiled
    if kind == "relu2_metal":
        return relu2_metal
    if kind in VARIANTS:
        return VARIANTS[kind]  # checked before the eml_* parser below
    # kinds: eml[_metal|_compiled|_ref] (offset 0, as proposed) and eml_c[_metal|_compiled|_ref] (centred, offset 1).
    # Parse the parts: "eml_compiled".startswith("eml_c") is True, so a prefix test would pick the wrong offset.
    parts = kind.split("_")
    assert parts[0] == "eml", kind
    offset = mx.array(1.0 if len(parts) > 1 and parts[1] == "c" else 0.0)
    if kind.endswith("_metal"):
        return lambda h: eml_metal(h, offset)
    if kind.endswith("_ref"):
        return lambda h: eml_ref(h, float(offset.item()))
    return lambda h: eml_compiled(h, offset)
