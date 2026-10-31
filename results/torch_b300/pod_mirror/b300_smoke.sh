#!/bin/bash
cd /root/b300/study/torch_b300
. /root/b300/venv/bin/activate
export FP4_DATA=/root/b300/data FP4_PLATFORM=b300
L=/root/b300/logs; R=/root/b300/results
echo "[$(date -u +%T)] smoke start"
python smoke.py --acts relu2,lrelu2,swiglu --recipes nvfp4,mxfp8,fp8_cs,fp8_ds,bf16,nvfp4_nrht,nvfp4_nsr,nvfp4_1d,nvfp4_bare,sim_nvfp4,sim_nvfp4h,sim_mxfp4,sim_fp8,sim_int8 \
  --steps 20 --out $R/torch_b300_smoke.jsonl --log-dir $L 2>&1 | tee $L/smoke_all.log
echo "[$(date -u +%T)] smoke done"
