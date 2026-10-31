# jax_tpu: learnings (2026-10-31)

## 1. Headline: the MLX LR schedule had a uint64 underflow; every Mac run before the fix got ~1/3 of the intended LR
- **Where:** `lr_schedule()` in `mlx_mac/train_mlx.py`, identical to the original in nanogpt `anvil_r/mlx_gpt/train_mlx.py`:
  ```python
  cd = int(n * (1 - cooldown_frac))
  def f(step):
      t = mx.clip((step - cd) / max(1, n - cd), 0.0, 1.0)
      return base * (1.0 - (1.0 - floor) * t)
  ```
- **Why it fails:** `mlx.optimizers.Optimizer` stores the step as `mx.array(0, mx.uint64)` (mlx 0.31.2, `mlx/optimizers/optimizers.py` line 17) and passes it to the schedule.
  - For step < cd, `step - cd` wraps around to ≈ 1.8e19. So t clips to 1 and the LR is `floor × base` = **0.1× base**.
  - Example: uint64 0 − 360 gives 18446744073709551256.
- **What the Mac runs actually got, measured inside the compiled MLX step:**

  | steps (N = 600) | Muon LR | Adam LR | multiplier |
  |---|---|---|---|
  | 0 … 359 (the first 60%) | 0.002 | 0.0008 | 0.1 |
  | 360 | 0.02 | 0.008 | 1.0 (jumps up) |
  | 361 … 599 | linear decay | linear decay | 0.996 → 0.104 |

  - Intended instead: 1.0 for steps 0–359, then the same decay.
  - Same pattern for any N: N = 10, 50, 300, 600 and 2000 all checked.
  - `lr_schedule(1, 600)` returns 0.1 at steps 0, 100 and 359, 1.0 at 360, 0.55 at 480, and 0.1037 at 599.
- **Size of the effect:** the mean LR multiplier over a run is 0.6 × 0.1 + 0.4 × 0.55 = **0.28**, against 0.82 intended, i.e. about 34% of the intended LR integral. The run is then mostly a short high-LR cooldown.
  - This likely explains stage 1's "the default LR is too low for 300-step runs (×1.25 helped ReLU² by −45 mnat)".
- **Affected:** every run made with `train_mlx.build()` before the fix. That is lowp_A, lowp_S, the stage1 grids, the EML grids and the `@opt=` variants (they receive the same `lr_schedule`).
  - Their *paired* comparisons (same schedule on both sides) remain internally valid.
  - Absolute losses, and any "LR is too low" conclusion, reflect the bug.
- **How it was found:** the JAX port learned much faster than MLX from step 1 on. With weights transferred from MLX into JAX:
  - the forward pass, every gradient and the first optimizer update matched to 5e-6;
  - JAX's loss evaluated on MLX's own post-step-1 weights reproduced JAX's step-1 loss exactly;
  - printing `opt.state["learning_rate"]` inside the MLX step then showed 0.002 / 0.0008.
- **Fixes (in this repo):**
  - `common/config.py`: `lr_mult(step, n, mode)` with modes `"intended"` and `"mlx_uint64_bug"` (an exact replica), and `lr_at(step, n, base, mode)`.
  - `jax_tpu/train_jax.py`:
    - `--schedule {intended, mlx_uint64_bug}`, default **intended**;
    - the LR is a lookup table built from `lr_mult`, so every trainer uses the same numbers;
    - every record stores `schedule`.
  - `mlx_mac/train_mlx.py`: casts the step to float32 before the subtraction.
    - `--schedule mlx_bug` keeps the old arithmetic, to reproduce the old results.
    - Records store `schedule`; the resume check treats records without the field as `mlx_bug`.
    - Checked: both modes match `common.config.lr_mult` to 7e-8 at N = 10, 50 and 600.
  - **Not fixed here:** the nanogpt copy (`anvil_r/mlx_gpt/`, the coordinator handles it). Also, `mlx_mac/stage1.py`'s `load()` / `run_jobs()` dedupe on (arm, seed, n, d) without the schedule, so re-run the Mac grid under a new tag or results file.

## 2. What was broken in train_jax.py, and the fixes
1. **"pytree structure error … Param vs DynamicJaxprTracer":**
   - Cause: `label_params()` mapped over an `nnx.State` whose leaves sit inside `nnx.Param` nodes, so the optax label tree kept `Param` wrappers. Meanwhile `nnx.Optimizer` handed optax raw arrays.
   - Fix: the train step is now a pure `jax.jit` over `nnx.split(model, nnx.Param, ...)`:
     - params are kept as a pure dict (`nnx.to_pure_dict`);
     - `nnx.merge(graphdef, params, rest)` runs inside the loss;
     - `optax.multi_transform` gets a pure-dict label tree, and parameters and optimizer state are donated.
2. **"Argument 'float32' of type numpy Float32DType is not a valid JAX type":**
   - Cause: the sim recipes' `custom_vjp` stored `x.dtype` / `w.dtype` in its residuals.
   - Fix: the residuals are now `(x, w, key)` in fp32; the key's cotangent is `None`.
3. **Stochastic-rounding keys:**
   - Before: a stateful `nnx.Rngs` was called inside the custom VJP.
   - Now: an explicit per-step key `fold_in(key(1000+seed), step)`, folded with a per-Linear id. Micro-batches fold in their index.
4. **Qwix int8:**
   - The rule is `QtRule(module_path=r"blocks/\d+/(qkv|o|fc|proj)", weight/act int8, bwd int8, bwd_stochastic_rounding="uniform", channelwise_noise_axes=(0, 1, 2))`.
   - `qwix_rngs` is set inside the traced loss from the step key: `model.set_attributes(qwix_rngs=nnx.Rngs(stochastic_rounding=key))` after `nnx.merge`. That avoids carrying RNG state through the jit.
   - The Linear calls `jax.lax.dot_general` on the raw kernel, so Qwix's `find_param` identifies the weight.
   - Verified: the lowered step has **48 int8 × int8 → int32 `dot_general`s**, i.e. 16 Linears × (fwd, dgrad, wgrad); attention and head stay bf16. Qwix fp8 lowers to f8E4M3FN dots.
   - Qwix's default `channelwise_noise_axes=(0,)` would draw **one SR noise value per batch row**, broadcast over T and d. It is set to elementwise noise here, like the simulators.
5. **Precision on TPU:** a TPU runs fp32 matmuls as single-pass bf16 by default. So:
   - the `sim_*` GEMMs, Muon's Newton-Schulz, the RHT and the "fp32" recipe use `precision=HIGHEST`;
   - the `bf16` recipe casts its operands to bf16 and accumulates in fp32 (`preferred_element_type`).
6. **Exact format grids:** `_float_q` takes exponents from `jnp.frexp` and builds powers of two from the bit pattern, instead of `log2` / `exp2`, which are approximations on TPU. MXFP4's E8M0 scale uses an exact ceil-log2.
7. **Recipe semantics** (on the Mac, embedding, head, attention and norms are fp32):
   - `fp32` = the Mac's baseline ("none").
   - `sim_*`: attention and head fp32, like the Mac. Pair them with fp32.
   - `bf16` and `int8` / `fp8` (Qwix): attention and head bf16, the TPU-native setting. Pair int8 with bf16.
   - The original grid had no fp32 arm; I added one.
8. **Attention:** explicit causal einsum attention, so the precision can be controlled; `jax.nn.dot_product_attention` has no precision argument.
9. **Memory:**
   - Per-block `nnx.remat` (default on for gpt2s).
   - `--accum` micro-batches, with the same math (mean of equal-size means).
   - Automatic micro-batching from `memory_stats()["bytes_limit"]`.
   - The grid retries a run with 2× `--accum` on RESOURCE_EXHAUSTED.
   - CPU compile `memory_analysis()` temp sizes:
     - tiny, B=32: bf16 15.5 GiB, sim_nvfp4 17.7 GiB, bf16 with accum 2: 8.0 GiB;
     - gpt2s, B=4: 13.7 GiB without remat, 3.5 GiB with remat.
   - So on a 16 GB v5e, tiny probably needs accum 2 (auto-selected); a 32 GB v6e runs accum 1.
10. **Grid plumbing:**
    - Resume is keyed by (act, recipe, seed, n, config, device, schedule), so v5e and v6e passes can share one file.
    - A failed run is logged to `<out>_errors.log`, and the grid continues.
    - A persistent XLA compilation cache means seed 2 skips compilation.
    - Each record also has `device_tag`, `hp`, `accum`, `remat`, `eval_s` and the package versions.

## 3. Parity with MLX (Mac CPU for JAX; MLX on the Mac GPU)
- **Exact weight transfer.** MLX init weights are copied into JAX, then 10 steps at B = 8 with the `mlx_uint64_bug` schedule.
  - Step-0 loss: 10.827296 in both.
  - Gradients: relative difference ≤ 9e-6.
  - Weights after 10 steps: relative difference ≤ 1e-5.
  - Losses agree to 4 decimals at all 10 steps for relu2, lrelu2, swiglu, xielu, s2glu_r2, eglu_r2n and eml (fp32).
  - Same for relu2 under sim_int8, sim_fp8, sim_mxfp4, sim_nvfp4 and sim_nvfp4h, and swiglu under sim_nvfp4. The only differences, ≤ 1e-4, come from the random SR draws.
- **Independent inits**, 50 steps at B = 8, `mlx_uint64_bug` schedule, val on 64k tokens:

  | | seed 1 | seed 2 |
  |---|---|---|
  | MLX relu2 fp32 | 7.7164 | 7.6910 |
  | JAX relu2 fp32 | 7.7165 | 7.6979 |
  | JAX relu2 bf16 | 7.7166 | 7.6980 |
  | JAX relu2 int8 (Qwix) | 7.7161 | 7.6977 |
  | MLX relu2 nvfp4 | 7.7237 | 7.6956 |
  | JAX relu2 sim_nvfp4 | 7.7211 | 7.7014 |

  The trajectories overlap within seed noise.
- **Before the schedule replica:** JAX (with the intended schedule) reached val 7.383 vs MLX 7.716 at the same 50 steps. That gap is the bug.

## 4. CPU smoke test: every activation × recipe (4 steps, B = 4, val on 16k tokens; all finite, step 0 = 10.8256)
| act | fp32 | bf16 | int8 | fp8 | sim_fp8 | sim_int8 | sim_mxfp4 | sim_nvfp4 | sim_nvfp4h |
|---|---|---|---|---|---|---|---|---|---|
| relu2 | 10.2336 | 10.2336 | 10.2339 | 10.2341 | 10.2345 | 10.2337 | 10.2434 | 10.2347 | 10.2345 |
| lrelu2 | 10.2068 | 10.2068 | 10.2072 | 10.2071 | 10.2078 | 10.2070 | 10.2176 | 10.2073 | 10.2095 |
| swiglu | 10.3608 | 10.3609 | 10.3609 | 10.3609 | 10.3612 | 10.3608 | 10.3636 | 10.3620 | 10.3620 |
| xielu | 10.2118 | 10.2119 | 10.2120 | 10.2126 | 10.2129 | 10.2119 | 10.2198 | 10.2127 | 10.2126 |
| s2glu_r2 | 10.2514 | 10.2514 | 10.2517 | 10.2520 | 10.2521 | 10.2516 | 10.2622 | 10.2534 | 10.2535 |
| eglu_r2n | 10.3417 | 10.3417 | 10.3419 | 10.3419 | 10.3420 | 10.3418 | 10.3463 | 10.3426 | 10.3429 |
| eml | 9.8440 | 9.8441 | 9.8442 | 9.8450 | 9.8463 | 9.8440 | 9.8583 | 9.8505 | 9.8515 |

- These ran with the old default (intended schedule) before the `--schedule` flag existed. The code paths are unchanged.
- Parameter counts match MLX: 28,901,384 for relu2 / lrelu2 (+8 for xielu); 28,966,920 for the packed activations.

## 5. Qwix int8 vs sim_int8: real differences in the recipe, not only the hardware
- Qwix reuses the forward-quantized residuals in the backward pass and folds their scales into the gradient before quantizing it.
  - dgrad = int8(g · s_w) @ w_q;
  - wgrad = int8(g · s_x) @ x_q.
- sim_int8 (lowp.py) instead re-quantizes x and W along each GEMM's own contraction axis.
- Qwix uses one noise function, with the same key, for both dgrad and wgrad of a layer.
- Both are "int8 everywhere with SR on gradients". The int8 − bf16 vs sim_int8 − fp32 comparison therefore measures the recipe plus the hardware, not the hardware alone.

## 6. Open risks on TPU (can't be tested from the Mac)
- **Install:** the notebook keeps Colab's preinstalled jax / jaxlib / libtpu and pins flax / optax / qwix against them. If that doesn't resolve, it installs jax[tpu]==0.12.0.1, which needs a restart. An older preinstalled jax (< 0.11.1) forces the fallback.
- **Memory:** the CPU memory analysis suggests tiny at B = 32 is ≈ 16–18 GiB without micro-batching. auto_accum picks accum = 2 on v5e, and the OOM retry is the backstop. TPU XLA usually needs less, so it may run faster than estimated.
- **Speed:** fp32 / HIGHEST matmuls are ~3–6× slower than bf16 on the MXU, and the sim recipes add elementwise quantize work. The head GEMM (16k tokens × 256 × 50,304) dominates. The notebook's tpu20 cell measures the step times and prints the grid estimate.
- **Unverified on TPU:**
  - Qwix's int8 path on a real MXU, including `preferred_element_type` and SR noise generation cost;
  - frexp / bitcast lowering;
  - nnx.remat combined with Qwix interception (traced fine on CPU).
- **Device kind strings** ("TPU v5 lite", "TPU v6 lite") are stored raw in `device`; `device_tag` maps them to v5e / v6e.
