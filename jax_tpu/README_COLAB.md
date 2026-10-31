# Running the TPU grid on Colab (TPU v5e-1 / v6e-1)

The notebook [`jax_tpu/colab_tpu.ipynb`](colab_tpu.ipynb) runs the study's JAX trainer (`jax_tpu/train_jax.py`) on a Colab TPU. It trains the same small GPT, with the same optimizer and the same tokens, as the Mac MLX runs.
- **int8 is real:** Qwix quantized training, int8 × int8 → int32 on the MXU.
- **FP8 and FP4 are simulated:** TPU v5e and v6e have no FP4 or FP8 hardware, so those arms fake-quantize with the Mac simulator's numerics.
- **Each result records the chip** in its `device` field (`TPU v5 lite` = v5e, `TPU v6 lite` = v6e) and in `device_tag`.

## Steps
1. **Open the notebook**, by either route:
   - the "Open in Colab" badge in the repo README;
   - `https://colab.research.google.com/github/atveit/relu_squared_fp4/blob/main/jax_tpu/colab_tpu.ipynb`.
2. **Pick the runtime:** *Runtime → Change runtime type → TPU v5e-1* (or *TPU v6e-1*), then *Connect*.
3. **Steps 1–2 (runtime check, install):**
   - Step 1 should print `TPU v5 lite` (or `TPU v6 lite`).
   - Step 2 keeps Colab's preinstalled JAX and pins `flax 0.12.12`, `optax 0.2.8`, `qwix 0.1.8`.
   - If it says *Restart the runtime*, use *Runtime → Restart session* and continue from step 3.
4. **Step 3 (get the code):**
   - Default: `git clone https://github.com/atveit/relu_squared_fp4`.
   - If the clone fails, use fallback A: upload `colab_bundle.zip`, which is made on the Mac and holds `common/`, `jax_tpu/` and the Mac results.
   - Or fallback B: put the zip in `MyDrive/relu_squared_fp4/` on Drive.
5. **Step 4 (Drive):** allow access.
   - Results are written to `results/` in the clone and copied to `MyDrive/relu_squared_fp4_results/` after every chunk.
   - A new session copies them back first, so nothing is lost when Colab disconnects.
6. **Step 5 (20-step smoke run, ~3–6 min):**
   - It runs ReLU² × fp32 / bf16 / int8 / sim_nvfp4 with the full batch. The first run also downloads the FineWeb shards (~400 MB).
   - Check that the loss starts at ≈ 10.83 and falls.
   - The next cell prints step times and a **runtime estimate for the small grid on this chip**.
7. **Step 5b (optional, about 20–40 min):** a cross-check against the Mac grid. It reruns 12 arms with the old MLX LR schedule replicated and prints TPU − Mac per arm.
   - fp32 should agree within seed noise (a few mnat).
   - bf16 can differ slightly more, because of its bf16 GEMMs.
8. **Step 6 (the small grid, 112 runs):** 7 activations × 8 recipes × 2 seeds, 600 steps each.
   - It runs the core arms first (ReLU², LeakyReLU², SwiGLU × fp32 / bf16 / int8 / sim_int8 / sim_nvfp4, 30 runs), then the rest one recipe at a time.
   - **Resumable:** after a disconnect, re-run steps 2, 3 and 4, then step 6. Finished runs are skipped.
   - The *Quick look* cell prints val losses and the paired precision penalties: sim_* vs fp32, and int8 vs bf16.
9. **Step 7 (save and download):** downloads `jax_small.jsonl`, `jax_tpu20.jsonl`, `jax_xcheck.jsonl` and any `*_errors.log`. Copies are also on Drive.
10. **Step 8 (optional):** the GPT-2-small "scale" grid (20 runs × 2,000 steps). Only start it with many hours to spare, preferably on v6e.

## Expected runtimes
Rough figures, before any measurement; the notebook's step 5 prints a measured estimate.

| | TPU v5e-1 (16 GB) | TPU v6e-1 (32 GB) |
|---|---|---|
| install + clone + data | ~5 min | ~5 min |
| 20-step smoke run | ~3–6 min | ~2–4 min |
| cross-check (12 runs, optional) | ~20–40 min | ~10–20 min |
| small grid, core chunk (30 runs) | ~1–1.5 h | ~0.5–1 h |
| small grid, all 112 runs | ~3–6 h | ~1.5–3 h |
| scale grid (20 runs, optional) | 20+ h | ~8–15 h |

- **Per-run cost:** bf16 and int8 runs are the fastest. fp32 and the sim_* arms use full-precision (HIGHEST) GEMMs, which are about 3–6× slower on the MXU, plus quantization work, so they take about twice as long per run.
- **Compilation:** each new arm compiles for ~0.5–1 min; seed 2 reuses seed 1's compile from the cache.
- **Memory:** on a 16 GB v5e the trainer splits the 32-sequence step into 2 micro-batches automatically, with the same math. If a run still runs out of memory, the grid retries it with twice the micro-batches.

## Running on both v5e and v6e
- Run the whole notebook once on each runtime type. Both passes write to the same `results/jax_small.jsonl` (via the Drive copy).
- Every record carries `device` and `device_tag`, and the grid only skips runs already finished **on the current chip**. A v6e pass after a v5e pass therefore runs the full grid again, on v6e.
- If you only have time for one chip, prefer the v5e: it is the one the plan's real-int8 question was scoped for. A v6e pass adds a second int8 data point and is faster.

## Sending the results back
1. Download, or copy from `MyDrive/relu_squared_fp4_results/`:
   - `jax_small.jsonl`
   - `jax_tpu20.jsonl`
   - `jax_xcheck.jsonl`, if you ran step 5b
   - `jax_small_errors.log`, if it exists
   - `jax_scale.jsonl`, if you ran step 8
2. Put them in `results/` of the repo on the Mac (`/Users/amundtveit/relu_squared_fp4/results/`). Alternatively, give the Drive folder path and they will be pulled from there.

## Notes
- **LR schedule:**
  - The grids use the intended schedule (`common.config.lr_mult(..., "intended")`): constant, then a linear decay over the last 40% to 0.1×.
  - The Mac grid in `mlx_mac/results/` predates the fix of an MLX uint64 schedule bug: 0.1× LR for the first 60% of every run. Step 5b replicates that schedule (`--schedule mlx_uint64_bug`) only for the cross-check. Details are in `jax_tpu/LEARNINGS.md`.
- **Baselines:**
  - `fp32` reproduces the Mac's baseline exactly: fp32 everywhere, HIGHEST precision on TPU.
  - `bf16` is the TPU-native baseline (bf16 hidden GEMMs, attention and head).
  - Compare the sim_* arms with fp32, and real int8 with bf16.
- **Command line:** the trainer also runs without the notebook:
  ```bash
  python jax_tpu/train_jax.py --act relu2 --recipe int8 --steps 600 --verbose
  python jax_tpu/train_jax.py --grid small --acts relu2,swiglu --recipes bf16,int8
  JAX_PLATFORMS=cpu python jax_tpu/train_jax.py --grid smoke --data-dir <fineweb dir>   # CPU smoke test, ~5 min
  ```
