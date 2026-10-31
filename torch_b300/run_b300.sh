#!/bin/bash
# One-shot B300 runbook for the ReLU^2 / FP4 study (PyTorch + Transformer Engine). On a fresh RunPod B300 pod
# (image runpod/base with the CUDA 13.0 toolkit, Python 3.12, driver >= 580):
#
#   bash run_b300.sh            # setup -> smoke -> grid  (~65 min, see README_B300.md)
#   bash run_b300.sh setup      # venv (torch 2.10.0+cu130) + Transformer Engine 2.20.2 + data   (~3.5 min)
#   bash run_b300.sh smoke      # NVFP4 / MXFP8 / FP8 / bf16 / simulated NVFP4, 20 steps each   (~1.5 min)
#   bash run_b300.sh grid       # 3 concurrent lanes: tiny_core s1 | tiny_core s2 | scale, then tiny_match, tiny_more   (~60 min)
#   bash run_b300.sh status     # what is running, what is done
#
# Everything lands under $WORK: logs/<plan>_<run_id>.log (full output + settings per run), logs/env/ (nvidia-smi -q,
# pip freeze, versions), logs/gpu_dmon.log (power / clocks every 5 s), results/torch_b300_<plan>.jsonl (one record per run).
# Pull them to the Mac between phases:  rsync -az -e "ssh -p PORT -i KEY" root@HOST:$WORK/{logs,results} <repo>/results/torch_b300/
set -uo pipefail

WORK=${WORK:-/root/agents/b300_prep}
REPO=${REPO:-$WORK/study}                     # the relu_squared_fp4 checkout (rsync it here, or it is cloned)
REPO_URL=${REPO_URL:-https://github.com/atveit/relu_squared_fp4.git}
TORCH_VER=${TORCH_VER:-2.10.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
TE_VER=${TE_VER:-2.20.2}
GRID_MINUTES=${GRID_MINUTES:-60}              # no lane starts a new run after this many minutes of grid time
LOGS=$WORK/logs RES=$WORK/results DATA=$WORK/data VENV=$WORK/venv
export FP4_DATA=$DATA FP4_PLATFORM=${FP4_PLATFORM:-b300}
mkdir -p "$LOGS/env" "$RES" "$DATA"
ts() { date -u +%H:%M:%S; }
say() { echo "[$(ts)] $*" | tee -a "$LOGS/runbook.log"; }

setup() {
  say "setup: GPU"; nvidia-smi -q > "$LOGS/env/nvidia-smi-q_start.txt"; nvidia-smi | tee "$LOGS/env/nvidia-smi_start.txt"
  pgrep -f "[n]vidia-smi dmon" > /dev/null || setsid -f nvidia-smi dmon -s pucm -d 5 -o TD > "$LOGS/gpu_dmon.log" 2>&1 < /dev/null
  lscpu > "$LOGS/env/lscpu.txt"; uname -a > "$LOGS/env/uname.txt"
  if [ ! -f "$REPO/torch_b300/train.py" ]; then say "cloning $REPO_URL"; git clone --depth 1 "$REPO_URL" "$REPO" || exit 1; fi
  say "setup: data (2 FineWeb shards, 400 MB) in the background"
  ( t=$(date +%s); cd "$DATA" && for f in fineweb_train_000001.bin fineweb_val_000000.bin; do
      [ -f $f ] || { curl -sfL -o $f.part https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/$f && mv $f.part $f; }
    done; echo "data $(( $(date +%s) - t ))s" ) > "$LOGS/data_download.log" 2>&1 &
  local dpid=$!
  command -v uv > /dev/null || pip install -q uv
  ( set -e; t0=$(date +%s)
    [ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12 -q
    . "$VENV/bin/activate"
    uv pip install -q "torch==$TORCH_VER" --index-url "$TORCH_INDEX"; t1=$(date +%s); echo "TIME torch $((t1 - t0))s"
    uv pip install -q numpy setuptools wheel ninja pybind11 packaging pip
    uv pip install -q "transformer_engine_cu13==$TE_VER" "nvidia-cudnn-frontend>=1.29.0" einops onnxscript onnx pydantic nvdlfw-inspect
    t2=$(date +%s); echo "TIME te_core $((t2 - t1))s"
    # The PyTorch extension is an sdist (C++ only); build it against the torch above, not an isolated one.
    MAX_JOBS=64 NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation "transformer_engine_torch==$TE_VER" "transformer_engine==$TE_VER"
    t3=$(date +%s); echo "TIME te_torch_build $((t3 - t2))s total $((t3 - t0))s"
  ) 2>&1 | tee "$LOGS/install_te.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "INSTALL FAILED (see logs/install_te.log)"; exit 1; }
  wait $dpid; cat "$LOGS/data_download.log"
  . "$VENV/bin/activate"
  pip freeze > "$LOGS/env/pip_freeze.txt"
  python - << 'EOF' 2>&1 | tee "$LOGS/env/versions.txt"
import torch, transformer_engine as T, transformer_engine.pytorch as te
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "te", T.__version__)
print("device", torch.cuda.get_device_name(), "cc", torch.cuda.get_device_capability(), "torch archs", torch.cuda.get_arch_list())
for n in ("is_fp8_available", "is_mxfp8_available", "is_fp8_block_scaling_available", "is_nvfp4_available"):
    print(n, getattr(te, n)(return_reason=True))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", bool(torch.isfinite(x @ x).all()))
EOF
  grep -q "is_nvfp4_available (True" "$LOGS/env/versions.txt" || { say "NVFP4 NOT AVAILABLE: stop here (wrong GPU / TE build)"; exit 1; }
  say "setup done"
}

smoke() {
  . "$VENV/bin/activate"; cd "$REPO/torch_b300"
  say "smoke: relu2 / swiglu x nvfp4 / mxfp8 / fp8_cs / bf16 / sim_nvfp4, 20 steps"
  python smoke.py --acts relu2,swiglu --recipes nvfp4,mxfp8,fp8_cs,bf16,sim_nvfp4 --steps 20 \
    --out "$RES/torch_b300_smoke.jsonl" --log-dir "$LOGS" 2>&1 | tee "$LOGS/smoke_all.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "SMOKE FAILED: fix before the grid (logs/smoke_*.log)"; exit 1; }
  say "smoke OK"
}

lane() {   # lane <name> <plan[:seeds]>...  : runs the plans one after another, all under the shared grid deadline
  local name=$1; shift
  . "$VENV/bin/activate"; cd "$REPO/torch_b300"
  for spec in "$@"; do
    local plan=${spec%%:*} seeds=""; [ "$spec" != "$plan" ] && seeds="--seeds ${spec#*:}"
    local left=$(( (DEADLINE - $(date +%s)) / 60 ))
    [ $left -le 0 ] && { say "lane $name: deadline reached before $spec"; break; }
    say "lane $name: $spec (max $left min)"
    python grid.py --plan $plan $seeds --max-minutes $left --out "$RES/torch_b300_$plan.jsonl" --log-dir "$LOGS" \
      >> "$LOGS/grid_lane_$name.log" 2>&1
    say "lane $name: $spec exit $?"
  done
}

grid() {
  # The tiny model is launch-bound on the B300's E-core host CPU, so 3 processes share the GPU (per-process step time
  # barely changes with 2; ~+25% with 3). Seed-major lanes: a cut-short grid still has every arm at seed 1.
  DEADLINE=$(( $(date +%s) + GRID_MINUTES * 60 ))
  say "grid: 3 lanes, deadline in $GRID_MINUTES min"
  lane A tiny_core:1 tiny_match:1 tiny_more:1 &
  lane B tiny_core:2 tiny_match:2 tiny_more:2 &
  lane C scale &
  wait
  say "grid done: $(cat $RES/torch_b300_*.jsonl 2>/dev/null | wc -l) records; failures: $(grep -l '=== FAILED' $LOGS/*.log 2>/dev/null | wc -l) logs"
}

status() {
  pgrep -fa "[g]rid.py" | cut -c1-160
  for f in "$RES"/torch_b300_*.jsonl; do [ -f "$f" ] && echo "$(wc -l < "$f") $(basename "$f")"; done
  tail -n 3 "$LOGS"/grid_lane_*.log 2>/dev/null
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader
}

case "${1:-all}" in
  setup) setup ;;
  smoke) smoke ;;
  grid) grid ;;
  status) status ;;
  all) setup && smoke && grid ;;
  *) echo "usage: $0 [all|setup|smoke|grid|status]"; exit 2 ;;
esac
