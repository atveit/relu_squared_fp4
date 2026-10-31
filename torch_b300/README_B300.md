# Running the PyTorch / B300 part of the study

This runs real NVFP4, MXFP8 and FP8 training (NVIDIA Transformer Engine) on one B300, next to the bf16 baseline and the simulated NVFP4 that the Mac grid used. One script does it: `run_b300.sh`.

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
   `rsync -az --exclude .git --exclude results --exclude data -e "ssh -p PORT -i KEY" relu_squared_fp4/ root@HOST:/root/agents/b300_prep/study/`
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
| `grid` | 60 min (cap `GRID_MINUTES`) | 3 concurrent lanes; see LEARNINGS.md for the step times |
| Pull + terminate | ~2 min | |
| **Total** | **~70 min** | **≈ $10.50 at $8.99/h (secure), ≈ $8.10 at $6.94/h (community)** |

**Why 3 lanes:**
- The tiny model is limited by the host CPU (eager PyTorch, many small kernels). On the B300 host (Xeon 6740E) a tiny bf16 step takes 75 ms, against 44 ms on the H100 pod.
- With 2–3 grid processes sharing the GPU, each process's step time barely changes, so throughput almost doubles.
- **Single-lane estimate:** about 27 min per tiny_core seed and about 32 min for scale, so ~90 min for tiny_core + scale.

## Outputs
- `results/torch_b300_<plan>.jsonl`: one record per run. Fields: `run_id`, `act`, `recipe`, `seed`, `n`, `config`, `val_loss`, `val_loss_hp_eval` (TE recipes evaluated without quantization), `step_ms`, `tokens_per_s`, `losses` every 50 steps, `lr_schedule`, `torch`, `te`, `device`.
- `logs/<plan>_<run_id>.log`: each run's full output, with its arguments and versions at the top and the record (or the traceback) at the end.
- `logs/env/`: `nvidia-smi -q`, `pip freeze`, `versions.txt` (torch / TE / CC / recipe availability), `install_te.log`. `logs/gpu_dmon.log` has power, clocks and utilization every 5 s.

## Versions that work
- **Software:** torch 2.10.0+cu130 and Transformer Engine 2.20.2 (`transformer_engine_cu13` wheel, plus the `transformer_engine_torch` sdist built with `--no-build-isolation`).
- **Driver:** 595.91.07.
- **sm_103:** torch's cu130 wheels contain no sm_103 code, but the sm_100 code runs on the B300 (CC 10.3).
- **TE recipes:**
  - On Hopper (H100) only FP8 (delayed and current scaling) is available.
  - MXFP8 and NVFP4 need CC ≥ 10.0.
