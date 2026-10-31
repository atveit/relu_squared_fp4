# torch_b300: learnings (b300-prep agent, 2026-10-31)

Newest facts first within each section. Times are UTC.

## Environment and install (verified)
- **Torch:** `torch==2.10.0+cu130` from `https://download.pytorch.org/whl/cu130`.
  - Its arch list is `sm_75 … sm_120` with no `sm_103`. It still runs on the B300 (CC 10.3), because `sm_100` SASS is binary-compatible within major 10. A bf16 matmul and all 14 recipes ran.
  - The pods' driver is 595.91.07 (CUDA 13.0 toolkit in the image), so use cu130 wheels. The cu128 venv on the H100 (`venv_train`) would need a CUDA 12 nvcc to build TE: torch's cpp_extension rejects a major-version mismatch with the image's nvcc 13.0.
- **Transformer Engine 2.20.2** (latest on PyPI, 2026-10-05) has `NVFP4BlockScaling` with `disable_rht`, `disable_stochastic_rounding` and `disable_2d_quantization`.
  - Its defaults (printed): forward input RHT on (for the wgrad copy), weights 2D 16×16 blocks, gradients RHT + SR.
  - The API matches `train.py`: `te.autocast(enabled, recipe)` and `te.is_{fp8,mxfp8,nvfp4}_available(return_reason)`.
- **Prebuilt wheels:** the GitHub release wheels only cover NGC torch builds (`torch26.06`–`26.09`) and `torch2.8.0+cu129`. For vanilla torch 2.10 the PyTorch extension `transformer_engine_torch` is an **sdist**.
  - It is a C++-only `CppExtension` (33 `.cpp` files, 0 `.cu`), so it needs only the CUDA headers, not nvcc kernels.
  - It must be built with `--no-build-isolation`: otherwise pip builds against a freshly downloaded torch.
- **Install recipe** (this is `install_te.sh`):
  ```bash
  uv venv venv --python 3.12 && . venv/bin/activate
  uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
  uv pip install numpy setuptools wheel ninja pybind11 packaging pip
  uv pip install "transformer_engine_cu13==2.20.2" "nvidia-cudnn-frontend>=1.29.0" einops onnxscript onnx pydantic nvdlfw-inspect
  MAX_JOBS=64 NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation "transformer_engine_torch==2.20.2" "transformer_engine==2.20.2"
  ```
- **Install time:**

  | Pod | Total | torch | TE core wheel (276 MB) | TE torch extension build |
  |---|---|---|---|---|
  | H100 | **90 s** | 33 s | 4 s | 53 s |
  | B300 (Xeon 6740E E-cores) | **198 s** | 33 s | 16 s | 149 s; `pybind.cpp` alone takes ~2 min single-threaded |

  The data (2 shards, 400 MB, curl from HF) takes 9 s, in parallel.
- **Not tried: the NGC container** (`nvcr.io/nvidia/pytorch:26.09-py3`, with TE preinstalled). It would mean a ~20 GB image pull at pod start, which is likely slower than the 3.3 min pip route, and its torch is NGC's build rather than 2.10.
- **TE availability on Hopper** (H100, TE 2.20.2): FP8 delayed and current scaling work. MXFP8 and NVFP4 refuse with "Device compute capability 10.0 or higher required". On B300, `is_fp8 / mxfp8 / fp8_block_scaling / nvfp4_available` are all True.

## Bugs found and fixed
1. **TE NVFP4 crashed on fp32 input** (error log E33).
   - Error: `RHT is only supported for bfloat16 input, got dtype enum value 4`.
   - Cause: TE quantizes a Linear's input in whatever dtype it arrives in, and does not cast under `torch.autocast` first. The study keeps the residual stream in fp32, like the MLX reference.
   - Fix: `TELinear(te.Linear)` casts its input to bf16, for every TE recipe. This matches the bf16 baseline, whose `nn.Linear` gets a bf16 input from autocast.
   - Only the recipes with RHT failed (`nvfp4`, `nvfp4_nsr`, `nvfp4_1d`). MXFP8, FP8, `nvfp4_nrht` and `nvfp4_bare` had run, quantizing from fp32.
2. **`run()` raised `SystemExit`** for an unavailable TE recipe. That killed a whole smoke / grid loop. It is now a `RuntimeError`, which `run_logged` catches and logs.
3. **The optimizer was launch-bound on the B300's slow host CPU:** 16 matrices, each with its own Newton–Schulz loop.
   - Same-shape matrices are now stacked into one batched NS, and Adam uses `torch._foreach_*`.
   - The result is **bitwise identical** to the old per-matrix code after 5 steps on CPU (`max |old - new| = 0`).
   - The optimizer now takes 17 ms of a 75 ms tiny bf16 step.

No other semantic deviations were found against `common/config.py` / the MLX reference (activations, xIELU raw init, packed hidden size, Muon scale and Nesterov, Adam without bias correction, init, softcap, QK-norm, RoPE halves).

## Parity with MLX (re-checked on the Mac CPU)
- `quant_sim.quant` vs `lowp.quant` on 32k values with an extreme dynamic range, zeros and 1e-30: **0 mismatches** for e4m3, e5m2, int8, mxfp4 and nvfp4. The Hadamard-16 is identical.
- 3-GEMM VJP in round-to-nearest mode, all six recipes: `y` and `dW` are bitwise identical. `dx` differs by ≤ 5e-6, from the fp32 GEMM summation order (3-D @ 2-D), not from the quantizers.
- **Cross-GPU:** the 20-step smoke val losses on H100 and B300 agree to ≤ 0.4 mnat for bf16 / FP8 / sim_fp8 / sim_int8 (relu2 bf16 is 7.3472 on both). SR recipes differ by up to 5 mnat (random draws).

## LR schedule (coordinator note 16:0x)
- **16:10:** `train.py` now has `--schedule {intended, mlx_uint64_bug}` (default intended) and calls the shared `common.config.lr_at(..., mode)` / `lr_mult` that jax-tpu added. Records store `lr_schedule`, and `run_id` gets a `_mlx_uint64_bug` suffix when that mode is used. The tiny_core and scale runs started before 16:10 used the old `lr_at`, which is the same intended formula, bit for bit.
- torch uses `common.config.lr_at`, the **intended** schedule: constant, then linear decay over the last 40% to 0.1×. Every record now stores `lr_schedule: "intended"`.
- The Mac `lowp_A` grid ran with the uint64 bug (0.1× LR for the first 60%; error log E32), so its *absolute* losses are not comparable. Paired deltas within a platform still are.

## Step times (eager PyTorch, no torch.compile / CUDA graphs)

| Config | Recipe | H100 (train.py, 100 steps) | B300 alone (`bench_step.py`, median: fwd+bwd + opt) |
|---|---|---|---|
| tiny (d256 L4, 16k tok) | bf16 | 44 ms | 75 ms (59 + 17) |
| tiny | fp8_cs | 48 ms | — |
| tiny | mxfp8 | n/a | 99 ms |
| tiny | nvfp4 | n/a | 109 ms |
| tiny | sim_nvfp4 | 97 ms | 216 ms |
| gpt2s (d768 L12, 32k tok) | bf16 | 277 ms | 175 ms (143 + 31) |
| gpt2s | fp8_cs | 278 ms | 158 ms |
| gpt2s | mxfp8 | n/a | 185 ms |
| gpt2s | nvfp4 | n/a | 240 ms |
| gpt2s | sim_nvfp4 | 1281 ms | — |

- **The B300 is slower than the H100 on the tiny model.** The host CPU is a Xeon 6740E with E-cores, and the run is launch-bound.
- **Fix: several grid processes at once.** With 2 concurrent processes, per-process tiny bf16 steps stayed at 80–86 ms (vs 73–84 ms alone), so throughput almost doubles. With 3 (two tiny lanes plus gpt2s), nvfp4 tiny took 135 ms.
- **Contention cuts both ways.** The gpt2s lane is GPU-bound. Next to two tiny lanes, its bf16 step went from 175 to 409 ms: without MPS the GPU time-slices between processes, it does not overlap their kernels. So 3 lanes help the tiny runs a lot but stretch the gpt2s lane by about 2.3×. Total work still finishes sooner than with serial lanes. Measured: 3 lanes, about 7 tiny runs per lane in 8.5 min, plus 1 gpt2s run in 7.3 min.
- **NVFP4 is not faster than bf16 at GPT-2-small size in eager mode** (240 vs 175 ms). The quantize / RHT kernels plus host overhead outweigh the FP4 GEMM gain at d = 768. FP8 current scaling is 10% faster. Make no speed claims from this study without CUDA graphs and a larger d.

## Grid run on the B300 (pod 2ego4dckx8oek6, billing from 15:41)
- **Smoke 15:50–15:56:** 42/42 OK (3 activations × 14 recipes, 20 steps). Records are in `results/torch_b300/torch_b300_smoke.jsonl`, logs in `logs/smoke_*.log`.
  - The smoke records predate the `lr_schedule` field, but they used the same intended `lr_at`.
  - The first attempt (fp32 input to TE, E33) is kept in `logs/attempt1_fp32_input/`.
- **Grid from 16:01, three lanes:** `tiny_core` seed 1, `tiny_core` seed 2, and `scale` (gpt2s, 1000 steps). Chained after the tiny lanes: `tiny_match`, then `tiny_more`, with no new run after 16:58.
- **16:16–16:20 PAUSED** (Amund's call: focus on the 8× experiment). The coordinator killed the grid and stopped the pod.
  - Done: tiny_core 24/42, scale 1/9.
  - Killed mid-run, logs only: scale lrelu2 bf16, and tiny_match relu2 nvfp4_1d s1 / s2, which the chain started when the tiny lanes died.
  - At 16:16:52 I had also started a runbook `setup` test in `/root/rb_test`. I stopped it at 16:17:06, before it reached the GPU, and deleted the directory.
  - Final pull 16:17:34: pod 102 log files + 3 jsonl. On the Mac: the same 102, plus 4 copies of top-level pod files in `logs/env/`; the jsonl md5s match.
- **Lesson:** a chain that waits on a lane's PID must also check *why* the lane ended. When the lanes were killed, `chain_more.sh` took that as "finished" and started the next plan. In `run_b300.sh`, a lane stops when grid.py exits with rc ≥ 128 (killed); grid.py itself exits 0 after a normal plan, even with failed runs.
- **First results:** see README_B300.md.
  - Real NVFP4 costs relu2 / lrelu2 / swiglu about the same, +30–40 mnat at N = 600.
  - The Hadamard does not help at this horizon. bare ≤ default.
  - LeakyReLU² is best in every recipe. ReLU² gets no FP4 advantage.
  - The gpt2s bf16 relu2 baseline at N = 1000 is 4.4812.
