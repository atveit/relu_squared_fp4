#!/bin/bash
# chain_more.sh <pid to wait for> <seed>: after that lane finishes, run tiny_match then tiny_more for <seed>;
# no new run starts after 16:58 UTC
pid=$1; seed=$2
while kill -0 $pid 2>/dev/null; do sleep 5; done
deadline=$(date -u -d "2026-10-31 16:58:00" +%s)
for plan in tiny_match tiny_more; do
  left=$(( (deadline - $(date +%s)) / 60 ))
  echo "[$(date -u +%T)] lane seed $seed: $plan with $left min"
  [ $left -gt 0 ] && /root/b300/grid_phase.sh $plan --seeds $seed --max-minutes $left
done
