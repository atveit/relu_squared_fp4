# Running the PyTorch / B300 part of the study

This runs real NVFP4, MXFP8 and FP8 training (NVIDIA Transformer Engine) on one B300, next to the bf16 baseline and the simulated NVFP4 that the Mac grid used. One script does it: `run_b300.sh`.

## Status (2026-10-31 16:20 UTC): paused after 20 min of grid
Amund paused the B300 to focus on the 8× experiment. The coordinator killed the grid and stopped pod `2ego4dckx8oek6`. Everything is in `results/torch_b300/`.

| Plan | Done | Left | What is done |
|---|---|---|---|
| smoke | 42 / 42 | 0 | relu2, lrelu2, swiglu × all 14 recipes, 20 steps: all OK |
| `tiny_core` | 24 / 42 | 18 | bf16, nvfp4, nvfp4_nrht, nvfp4_bare × 3 acts × seeds 1, 2. Left: mxfp8, fp8_cs, sim_nvfp4 |
| `scale` | 1 / 9 | 8 | relu2 bf16 (val 4.4812). gpt2s lrelu2 bf16 was killed mid-run |
| `tiny_match` | 0 / 12 | 12 | relu2 nvfp4_1d seeds 1 and 2 had just started and were killed (logs only, no records) |
| `tiny_more` | 0 / 24 | 24 | |

**To resume:**
1. Rent the pod and copy the repo **including** `results/torch_b300/*.jsonl` (step 2 below).
2. Run `bash run_b300.sh` (setup → smoke → grid).
   - `grid` first copies `results/torch_b300/torch_b300_{tiny_core,tiny_match,tiny_more,scale}.jsonl` from the repo into `$WORK/results/`.
   - `grid.py` then skips every `(act, recipe, seed, n)` already recorded. A killed run has no record, so it is redone from step 0.
   - The 62 remaining runs take about 60–65 min in 3 lanes. The scale lane is the critical path: 8 gpt2s runs, about 7–8 min each next to the tiny lanes, about 3.5–4 min alone.
3. **Single plans by hand:** `python grid.py --plan tiny_core --log-dir ../results/torch_b300/logs` (default `--out` is the repo file, so it resumes too). Use `--dry-run` to list what is left.

## What it measures
| Plan | Model | Arms | Runs | Question |
|---|---|---|---|---|
| `tiny_core` | tiny (d256 L4, 29M, 600 steps, 9.8M tokens) | relu2, lrelu2, swiglu × bf16, nvfp4, nvfp4_nrht, nvfp4_bare, mxfp8, fp8_cs, sim_nvfp4 × seeds 1, 2 | 42 | Does real NVFP4 match the simulator, per activation? What does the Hadamard (and SR + 2D blocks) buy, per activation? |
| `scale` | gpt2s (d768 L12, 162M, 1000 steps, 33M tokens) | relu2, lrelu2, swiglu × bf16, nvfp4, mxfp8 × seed 1 | 9 | Does the activation ranking under NVFP4 hold at GPT-2-small width? |
| `tiny_match` | tiny | relu2, lrelu2, swiglu × nvfp4_1d, sim_nvfp4h × seeds 1, 2 | 12 | Matched real-vs-simulated pair: both use 1D 16-blocks, SR on gradients and a 16-point Hadamard on the wgrad operands |
| `tiny_more` | tiny | xielu, eglu_r2n, s2glu_r2, eml × bf16, nvfp4, sim_nvfp4 × seeds 1, 2 | 24 | The other Mac activations, if time allows (time-capped) |

- **Settings:** all runs use the intended LR schedule (`lr_schedule: "intended"` in each record) and the shared `common/config.py` settings.
- **Precision:** hidden Linears run in the recipe's format. Embedding, head, attention and norms run in bf16, and the residual stream in fp32.

## Steps
1. **Rent** 1× B300 on RunPod: image `runpod/base` (CUDA 13.0 toolkit, Python 3.12), ≥ 50 GB disk. Amund's go is needed for any rental.
2. **Copy the repo** to the pod (or let the script `git clone` it):
   `rsync -az --exclude .git --exclude data --exclude results/torch_b300/logs -e "ssh -p PORT -i KEY" relu_squared_fp4/ root@HOST:/root/agents/b300_prep/study/`
   (Keep `results/torch_b300/*.jsonl`: `grid` resumes from them.)
3. **Run it:** `ssh ... 'cd /root/agents/b300_prep/study/torch_b300 && setsid -f bash run_b300.sh > /root/agents/b300_prep/runbook.out 2>&1 < /dev/null'`
   - It stops after `setup` if NVFP4 is not available, and after `smoke` if any smoke run fails.
   - Watch progress with `bash run_b300.sh status`.
4. **Pull the logs every ~15 min and at the end:** `rsync -az ... root@HOST:/root/agents/b300_prep/{logs,results} relu_squared_fp4/results/torch_b300/`. Compare the file counts with the pod.
5. **Terminate the pod.**

## Expected runtime and cost (measured 2026-10-31 on pod `2ego4dckx8oek6`)
| Phase | Time | Notes |
|---|---|---|
| Pod boot + copying the repo | ~3 min | |
| `setup` | ~3.5 min | torch 33 s; TE core wheel 16 s; TE PyTorch extension build 149 s (the slow E-core host CPU; 53 s on the H100 pod's CPU); data 9 s in parallel |
| `smoke` | ~1.5 min | 10 runs × 20 steps |
| `grid` | 60 min (cap `GRID_MINUTES`; no lane starts a run after it, so the last run ends ≤ ~8 min later) | 3 concurrent lanes; see LEARNINGS.md for the step times |
| Pull + terminate | ~2 min | |
| **Total** | **~70–78 min** | **≈ $10.50–11.70 at $8.99/h (secure), ≈ $8.10–9.00 at $6.94/h (community)** |

**Why 3 lanes:**
- The tiny model is limited by the host CPU (eager PyTorch, many small kernels). On the B300 host (Xeon 6740E) a tiny bf16 step takes 75 ms, against 44 ms on the H100 pod.
- With 2–3 grid processes sharing the GPU, each process's tiny step time barely changes, so throughput almost doubles.
- The GPU-bound gpt2s lane is the exception: without MPS the processes time-slice the GPU, so next to two tiny lanes its step goes from 175 to 409 ms.
- **Single-lane estimate:** about 27 min per tiny_core seed and about 32 min for scale, so ~90 min for tiny_core + scale.

## Outputs
- `results/torch_b300_<plan>.jsonl` on the pod, which is `results/torch_b300/torch_b300_<plan>.jsonl` in the repo: one record per run. Fields: `run_id`, `act`, `recipe`, `seed`, `n`, `config`, `val_loss`, `val_loss_hp_eval` (TE recipes evaluated without quantization), `step_ms`, `tokens_per_s`, `losses` every 50 steps, `lr_schedule`, `torch`, `te`, `device`.
- `logs/<plan>_<run_id>.log`: each run's full output, with its arguments and versions at the top and the record (or the traceback) at the end.
- `logs/env/`: `nvidia-smi -q`, `pip freeze`, `versions.txt` (torch / TE / CC / recipe availability), `install_te.log`. `logs/gpu_dmon.log` has power, clocks and utilization every 5 s.

## Versions that work
- **Software:** torch 2.10.0+cu130 and Transformer Engine 2.20.2 (`transformer_engine_cu13` wheel, plus the `transformer_engine_torch` sdist built with `--no-build-isolation`).
- **Driver:** 595.91.07.
- **sm_103:** torch's cu130 wheels contain no sm_103 code, but the sm_100 code runs on the B300 (CC 10.3).
- **TE recipes:**
  - On Hopper (H100) only FP8 (delayed and current scaling) is available.
  - MXFP8 and NVFP4 need CC ≥ 10.0.

## First results (B300, tiny, N = 600, seeds 1–2; `python report.py ../results/torch_b300/torch_b300_tiny_core.jsonl`)

Precision penalty = val(recipe) − val(bf16), same activation, paired per seed, mean of 2 seeds [mnat]. The seed-to-seed spread of a penalty is about ±10 mnat.

| Activation | bf16 val | nvfp4 (TE default) | nvfp4_nrht (no Hadamard) | nvfp4_bare (no RHT, SR or 2D) |
|---|---|---|---|---|
| relu2 | 5.4357 | +39.7 | +40.9 | +32.6 |
| lrelu2 | 5.4261 | +33.8 | +38.2 | +33.2 |
| swiglu | 5.4311 | +35.1 | +35.9 | +29.7 |

- **At this scale, real NVFP4 costs every activation about the same:** 30–40 mnat (~0.7%).
- **ReLU² does not pay less.** LeakyReLU² is best in bf16 and under every NVFP4 variant (−9 to −16 mnat vs ReLU²).
- **The random Hadamard transform gives no measurable benefit** in 600-step runs (nvfp4 vs nvfp4_nrht: −1 to −4 mnat).
- **The bare variant is slightly better** (−1 to −7 mnat vs the default; RTN gradients have lower variance over a short horizon).
- **Most of the penalty comes from training, not from the quantized forward:** evaluating the same weights without quantization (`val_loss_hp_eval`) recovers only ~6 mnat.
- **Still missing:** the simulated arm (sim_nvfp4, tiny_match), FP8 / MXFP8, and the gpt2s scale-up.
