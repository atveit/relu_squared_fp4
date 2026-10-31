#!/bin/bash
# usage: grid_phase.sh <plan> [grid.py args...]
cd /root/b300/study/torch_b300
. /root/b300/venv/bin/activate
export FP4_DATA=/root/b300/data FP4_PLATFORM=b300
plan=$1; shift
echo "[$(date -u +%T)] grid $plan $* start"
python grid.py --plan $plan --out /root/b300/results/torch_b300_$plan.jsonl --log-dir /root/b300/logs "$@"
echo "[$(date -u +%T)] grid $plan $* done rc=$?"
