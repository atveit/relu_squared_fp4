"""End-to-end MLX/Metal GPT training for the EML vs ReLU^2 test (Apple GPU).

Patterns from mlx / mlx-lm (external/): mx.fast.rms_norm / rope / scaled_dot_product_attention (causal), activations as
fused Metal kernels with custom VJPs (eml_kernels.py, like mlx_lm/kernels + tuner/losses.py), one compiled train step
with model + optimizer state as compile inputs/outputs (mlx_lm/tuner/trainer.py), Muon for hidden matrices and AdamW
for embedding / head / scalars (mlx.optimizers.MultiOptimizer).

Same data stream as anvil_r/train_small.py: fineweb_train_000001.bin, contiguous chunks, TRAIN seed = init only;
val = the first 512k tokens of the val shard. One JSON line per (arm, seed, N) in mlx_mac/results/<tag>.jsonl.

  python3 -m mlx_mac.train_mlx --smoke                       # pipeline smoke tests
  python3 -m mlx_mac.train_mlx --tag eml1 --arms relu2_metal,eml_metal,eml_c_metal --seeds 1-5 --ns 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .eml_kernels import make_gate

DATA = os.environ.get("FP4_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
HERE = os.path.dirname(__file__)
TOKC = 8   # channels of the normed block input that the per-token residual gates read (#393's gates read 6-12)


def hidden_for(kind: str, d: int) -> tuple[int, int]:
    """(c_fc out features, c_proj in features). ReLU^2: 4d, 4d. EML gate: 2*He, He with He ~ 8d/3 (matched params)."""
    from .eml_kernels import PACKED, VARIANTS
    kind = kind.split("+")[0]
    if kind not in PACKED and (kind.startswith("relu2") or kind in VARIANTS):   # "relu2glu" is packed: no prefix test
        return 4 * d, 4 * d
    he = (8 * d // 3 + 63) // 64 * 64
    return 2 * he, he


class Block(nn.Module):
    def __init__(self, d, n_head, gate_kind):
        super().__init__()
        # gate_kind = "<gate>[+emlskip][+bias][+subnorm]": +emlskip = residual scales as exp(tanh a) - log1p(relu b) (the
        # proposal's "log-space multiplicative skip"; a = b = 0 gives 1); +bias = learnable bias after the gate;
        # +subnorm = RMSNorm of the gated hidden vector before c_proj (NormFormer / BitNet subln).
        # Residual-scale modes: +emlskip (as proposed; relu damping branch, dies), +emlskip2 (softplus damping, never dead),
        # +emltok (emlskip2 computed per token from the first TOKC channels of the normed block input, "damp outliers"),
        # +sigtok (control: per-token 2*sigmoid gate from the same channels). All give lambda = 1 at init.
        gate_kind, *flags = gate_kind.split("+")
        self.emlskip, self.use_bias, self.subnorm = "emlskip" in flags, "bias" in flags, "subnorm" in flags
        self.skipmode = next((f for f in flags if f in ("emlskip", "emlskip2", "emltok", "sigtok")), None)
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        fc_out, proj_in = hidden_for(gate_kind, d)
        self.fc = nn.Linear(d, fc_out, bias=False)
        self.proj = nn.Linear(proj_in, d, bias=False)
        self.proj.weight = mx.zeros_like(self.proj.weight)   # zero-init projections (modded-nanogpt)
        self.o.weight = mx.zeros_like(self.o.weight)
        qrec = next((f[2:] for f in flags if f.startswith("q=")), None)   # +q=<recipe>: simulated low precision (lowp.py)
        if qrec:
            from .lowp import QLinear
            self.qkv, self.o, self.fc, self.proj = (QLinear(m, qrec) for m in (self.qkv, self.o, self.fc, self.proj))
        if self.emlskip:
            # b starts at 0.01, not 0: relu passes no gradient at exactly 0, so b = 0 would leave the damping branch
            # -log1p(relu b) dead forever (seen in the smoke test). a is set so that lambda = 1 at init.
            b0 = 0.01
            self.skip_b = mx.full((2,), b0)
            self.skip_a = mx.full((2,), math.atanh(math.log(1.0 + math.log1p(b0))))
        elif self.skipmode in ("emlskip2", "emltok"):
            # b = 0 as proposed, but softplus(0) = ln 2 > 0 keeps the damping branch live; a is set so lambda = 1.
            self.skip_b = mx.zeros((2,))
            self.skip_a = mx.full((2,), math.atanh(math.log1p(math.log1p(math.log(2.0)))))
            if self.skipmode == "emltok":
                self.tok_wa = mx.zeros((TOKC * 2,))   # 1-d, so AdamW (like #393's gate weights), reshaped at use
                self.tok_wb = mx.zeros((TOKC * 2,))
        elif self.skipmode == "sigtok":
            self.skip_a = mx.zeros((2,))
            self.tok_wa = mx.zeros((TOKC * 2,))
        else:
            self.lam = mx.ones((2,))
        if self.use_bias:
            self.gbias = mx.zeros((proj_in,))
        self.gate = make_gate(gate_kind)
        if gate_kind == "xielu":   # trainable per-layer alphas, as in the paper and mlx-lm (AdamW: 1-d parameter)
            from .eml_kernels import XIELU_RAW, _xielu_t
            self.xielu_raw = mx.array(XIELU_RAW)
            self.gate = lambda h: _xielu_t(h, self.xielu_raw[0], self.xielu_raw[1])

    def _lams(self, h):
        """(attention scale, MLP scale): scalars, or (B, T, 1) per-token gates for +emltok / +sigtok."""
        m = self.skipmode
        if m is None:
            return self.lam[0], self.lam[1]
        if m in ("emlskip", "emlskip2"):
            damp = mx.maximum(self.skip_b, 0.0) if m == "emlskip" else mx.logaddexp(self.skip_b, 0.0)
            lam = mx.exp(mx.tanh(self.skip_a)) - mx.log1p(damp)
            return lam[0], lam[1]
        f = h[..., :TOKC]
        za = self.skip_a + f @ self.tok_wa.reshape(TOKC, 2)
        if m == "emltok":
            zb = self.skip_b + f @ self.tok_wb.reshape(TOKC, 2)
            lam = mx.exp(mx.tanh(za)) - mx.log1p(mx.logaddexp(zb, 0.0))
        else:
            lam = 2.0 * mx.sigmoid(za)
        return lam[..., 0:1], lam[..., 1:2]

    def __call__(self, x):
        B, T, D = x.shape
        H = self.n_head
        hd = D // H
        h = mx.fast.rms_norm(x, None, 1e-6)
        q, k, v = mx.split(self.qkv(h).reshape(B, T, 3, H, hd).transpose(2, 0, 3, 1, 4), 3, axis=0)
        q, k, v = q[0], k[0], v[0]
        q = mx.fast.rope(mx.fast.rms_norm(q, None, 1e-6), hd, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(mx.fast.rms_norm(k, None, 1e-6), hd, traditional=False, base=10000.0, scale=1.0, offset=0)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask="causal")
        la, lm = self._lams(h)
        x = x + la * self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        g = self.gate(self.fc(mx.fast.rms_norm(x, None, 1e-6)))
        if self.use_bias:
            g = g + self.gbias
        if self.subnorm:
            g = mx.fast.rms_norm(g, None, 1e-6)
        x = x + lm * self.proj(g)
        return x


class GPT(nn.Module):
    def __init__(self, vocab=50304, d=256, n_layer=4, n_head=4, gate_kind="relu2_metal", softcap=15.0):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = [Block(d, n_head, gate_kind) for _ in range(n_layer)]
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = mx.random.normal(self.head.weight.shape) * 0.005
        self.softcap = softcap

    def __call__(self, idx):
        x = mx.fast.rms_norm(self.embed(idx), None, 1e-6)
        for b in self.blocks:
            x = b(x)
        logits = self.head(mx.fast.rms_norm(x, None, 1e-6))
        return self.softcap * mx.tanh(logits / self.softcap)


def loss_fn(model, x, y):
    logits = model(x).astype(mx.float32)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


# "intended" (constant, then linear decay to floor over the last cooldown_frac) or "mlx_bug": the schedule every run
# before 2026-10-31 actually got. mlx.optimizers passes `step` as a uint64 array, so `step - cd` wrapped around below cd,
# t clipped to 1 and the LR sat at floor x base (0.1x) for the first 60% of the run, then jumped to 1x at step cd and
# decayed to 0.1x (= common.config.lr_mult(..., "mlx_uint64_bug")). Set by --schedule; stored in every result.
SCHEDULE_MODE = "intended"


def lr_schedule(base, n, cooldown_frac=0.4, floor=0.1, mode=None):
    cd = int(n * (1 - cooldown_frac))
    mode = mode or SCHEDULE_MODE
    assert mode in ("intended", "mlx_bug"), mode

    def f(step):
        s = step if mode == "mlx_bug" else mx.array(step).astype(mx.float32)   # the fix: signed arithmetic
        t = mx.clip((s - cd) / max(1, n - cd), 0.0, 1.0)
        return base * (1.0 - (1.0 - floor) * t)
    return f


def make_optimizer(n, muon_lr=0.02, adam_lr=0.008):
    muon = optim.Muon(learning_rate=lr_schedule(muon_lr, n), momentum=0.95, weight_decay=0.0, nesterov=True)
    adam = optim.AdamW(learning_rate=lr_schedule(adam_lr, n), betas=[0.9, 0.95], weight_decay=0.0)
    is_hidden = lambda path, w: w.ndim == 2 and ("blocks" in path) and not path.endswith("lam")
    return optim.MultiOptimizer([muon, adam], [is_hidden])


def tokens(path, n):
    hdr = np.fromfile(path, dtype=np.int32, count=256)
    return np.fromfile(path, dtype=np.uint16, count=min(n, int(hdr[2])), offset=1024)


def parse_arm(arm):
    """'<gate>[@lr<x>][@opt=<name>]', e.g. 'eml_metal@lr1.25', 'eml_metal@opt=muon_r1', 'relu2_metal@opt=adamw@lr0.5'."""
    kind, *opts = arm.split("@")
    lr_mult, opt_name = 1.0, None
    for o in opts:
        if o.startswith("lr"):
            lr_mult = float(o[2:])
        elif o.startswith("opt="):
            opt_name = o[4:]
    return kind, lr_mult, opt_name


def build(arm, seed, n, args):
    kind, lr_mult, opt_name = parse_arm(arm)
    mx.random.seed(seed)
    model = GPT(d=args.d, n_layer=args.layers, n_head=args.heads, gate_kind=kind)
    mx.eval(model.parameters())
    if opt_name is None:   # the original path (MLX Muon + AdamW), unchanged for earlier grids
        opt = make_optimizer(n, muon_lr=0.02 * lr_mult, adam_lr=0.008 * lr_mult)
    else:
        from .opt_variants import make as make_opt
        fc_out, _ = hidden_for(kind, args.d)
        opt = make_opt(n, opt_name, lr_schedule, 0.02 * lr_mult, 0.008 * lr_mult,
                       packed_fc_rows=fc_out if fc_out != 4 * args.d else None)
    state = [model.state, opt.state]
    if "+q=" in kind:   # stochastic rounding inside the compiled step needs the random state as a compile input/output
        state.append(mx.random.state)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        opt.update(model, grads)
        return loss
    return model, opt, state, step


def batch(train, s, B, T):
    off = s * B * T
    c = mx.array(train[off:off + B * T + 1].astype(np.int32))
    return c[:-1].reshape(B, T), c[1:].reshape(B, T)


def evaluate(model, val, B, T):
    tot, nb = 0.0, 0
    for i in range(0, len(val) - B * T - 1, B * T):
        c = mx.array(val[i:i + B * T + 1].astype(np.int32))
        tot += loss_fn(model, c[:-1].reshape(B, T), c[1:].reshape(B, T)).item()
        nb += 1
    return tot / nb


def run(arm, seed, n, args, train, val):
    model, opt, state, step = build(arm, seed, n, args)
    t_first = None
    losses = []
    t0 = time.perf_counter()
    for s in range(n):
        x, y = batch(train, s, args.batch, args.seq)
        loss = step(x, y)
        mx.eval(state, loss)
        if s == 0:
            t_first = time.perf_counter() - t0   # includes compile
            t0 = time.perf_counter()
        if s % 50 == 0 or s == n - 1:
            losses.append((s, loss.item()))
    train_s = time.perf_counter() - t0
    vl = evaluate(model, val, 16, args.seq)
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    return dict(arm=arm, seed=seed, n=n, schedule=SCHEDULE_MODE, val_loss=vl, step_ms=round(1000 * train_s / max(1, n - 1), 2),
                first_step_s=round(t_first, 2), tokens_per_s=round((n - 1) * args.batch * args.seq / train_s),
                params=nparams, losses=losses, cfg=dict(d=args.d, layers=args.layers, batch=args.batch, seq=args.seq))


def smoke(args, train, val):
    """Pipeline smoke tests: compile with custom Metal VJPs, loss falls, step-time split, determinism, fwd/bwd parity."""
    print("[P1] compiled step runs with each activation's Metal kernel; loss falls over 20 steps")
    for arm in ("relu2_metal", "eml_metal", "eml_c_metal", "relu2", "eml_compiled"):
        model, opt, state, step = build(arm, 1, 20, args)
        l = []
        for s in range(20):
            x, y = batch(train, s, args.batch, args.seq)
            loss = step(x, y)
            mx.eval(state, loss)
            l.append(loss.item())
        print(f"   {arm:<13} loss {l[0]:.3f} -> {l[-1]:.3f}  {'OK' if l[-1] < l[0] - 0.5 and all(map(math.isfinite, l)) else 'FAIL'}")

    print("[P2] Metal vs compiled activation give the same training trajectory (fp32, 10 steps)")
    for a, b in (("relu2_metal", "relu2"), ("eml_metal", "eml_compiled")):
        traj = {}
        for arm in (a, b):
            model, opt, state, step = build(arm, 3, 10, args)
            out = []
            for s in range(10):
                x, y = batch(train, s, args.batch, args.seq)
                loss = step(x, y)
                mx.eval(state, loss)
                out.append(loss.item())
            traj[arm] = out
        diff = max(abs(p - q) for p, q in zip(traj[a], traj[b]))
        print(f"   {a} vs {b}: max |loss diff| over 10 steps {diff:.2e}  {'OK' if diff < 2e-2 else 'CHECK'}")

    print("[P3] determinism: same seed twice (relu2_metal, 10 steps)")
    outs = []
    for _ in range(2):
        model, opt, state, step = build("relu2_metal", 7, 10, args)
        for s in range(10):
            x, y = batch(train, s, args.batch, args.seq)
            loss = step(x, y)
            mx.eval(state, loss)
        outs.append(loss.item())
    print(f"   final losses {outs[0]:.6f} vs {outs[1]:.6f}  {'bit-identical' if outs[0] == outs[1] else f'differ by {abs(outs[0]-outs[1]):.2e}'}")

    print("[P4] step-time split (median of 15 steps after warmup): full step vs fwd+bwd only vs fwd only")
    for arm in ("relu2_metal", "eml_metal"):
        model, opt, state, step = build(arm, 1, 100, args)
        x, y = batch(train, 0, args.batch, args.seq)
        # Compile with the model state as an implicit input (passing the module as an argument flattens it to a dict).
        vg_raw = nn.value_and_grad(model, loss_fn)
        def fb(a, b):  # loss + a scalar reduction of every gradient: forces the full backward, returns only scalars
            l, g = vg_raw(model, a, b)
            return l, sum(mx.sum(v * v) for _, v in tree_flatten(g))
        vg = mx.compile(fb, inputs=[model.state], outputs=[model.state])  # value_and_grad writes params back
        fw = mx.compile(lambda a, b: loss_fn(model, a, b), inputs=[model.state])

        def t_of(fn, k=15):
            for _ in range(3):
                mx.eval(fn())
            ts = []
            for _ in range(k):
                t = time.perf_counter()
                mx.eval(fn())
                ts.append(1000 * (time.perf_counter() - t))
            return sorted(ts)[k // 2]
        t_full = t_of(lambda: (step(x, y), state))
        t_fb = t_of(lambda: vg(x, y))
        t_f = t_of(lambda: fw(x, y))
        print(f"   {arm:<12} full {t_full:6.1f} ms | fwd+bwd {t_fb:6.1f} ms | fwd {t_f:6.1f} ms | optimizer ~{t_full - t_fb:5.1f} ms "
              f"| tokens/s {args.batch * args.seq / t_full * 1000:,.0f}")


def parse_list(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--arms", default="relu2_metal,eml_metal,eml_c_metal")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--ns", default="300")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-tokens", type=int, default=1 << 19)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--schedule", default="intended", choices=["intended", "mlx_bug"],
                    help="mlx_bug reproduces the runs made before 2026-10-31 (0.1x LR until the cooldown)")
    args = ap.parse_args()
    global SCHEDULE_MODE
    SCHEDULE_MODE = args.schedule
    ns, seeds, arms = parse_list(args.ns), parse_list(args.seeds), args.arms.split(",")
    need = max(ns + [120]) * args.batch * args.seq + 1
    train = tokens(f"{DATA}/fineweb_train_000001.bin", need)
    val = tokens(f"{DATA}/fineweb_val_000000.bin", args.val_tokens + 1)
    if args.smoke:
        smoke(args, train, val)
        return
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{args.tag}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["arm"], r["seed"], r["n"]) for r in map(json.loads, open(out))
                if r.get("schedule", "mlx_bug") == SCHEDULE_MODE}   # records without the field predate the fix
    for n in ns:
        for seed in seeds:
            for arm in arms:
                if (arm, seed, n) in done:
                    continue
                rec = run(arm, seed, n, args, train, val)
                with open(out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{time.strftime('%H:%M:%S')} {arm:<12} seed={seed:<3} N={n:<4} val={rec['val_loss']:.4f} "
                      f"step={rec['step_ms']:.0f}ms tok/s={rec['tokens_per_s']:,} params={rec['params']:,}", flush=True)


if __name__ == "__main__":
    main()
