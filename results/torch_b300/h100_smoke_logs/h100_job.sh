#!/bin/bash
# H100 smoke + timing for the ReLU^2/FP4 study (runs under gpu_fifo). torch 2.10.0+cu130, TE 2.20.2.
cd /root/agents/b300_prep/study/torch_b300
. /root/agents/b300_prep/venv/bin/activate
export FP4_DATA=/root/agents/integration/data/fineweb10B FP4_PLATFORM=h100
L=/root/agents/b300_prep/logs/h100; R=/root/agents/b300_prep/results
echo "[$(date -u +%T)] start"; nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm --format=csv,noheader
pip freeze > $L/pip_freeze.txt; nvidia-smi -q > $L/nvidia-smi-q.txt
python smoke.py --acts relu2,lrelu2,swiglu --recipes bf16,sim_fp8,sim_int8,sim_mxfp4,sim_nvfp4,sim_nvfp4h,fp8_cs,fp8_ds,mxfp8,nvfp4 \
  --steps 20 --out $R/h100_smoke.jsonl --log-dir $L 2>&1 | tee $L/smoke_all.log
echo "[$(date -u +%T)] timing"
for spec in "tiny bf16 100" "tiny sim_nvfp4 60" "tiny fp8_cs 100" "gpt2s bf16 40" "gpt2s fp8_cs 40" "gpt2s sim_nvfp4 20"; do
  set -- $spec
  python train.py --act relu2 --recipe $2 --config $1 --steps $3 --val-tokens 65536 --out $R/h100_timing.jsonl --log-dir $L
done 2>&1 | tee $L/timing_all.log
echo "[$(date -u +%T)] done"
