#!/bin/bash
# Per-segment true-SM-occupancy probe: drive workload at steady state while nsys
# samples GPU hardware "SMs Active [%]" (the real GPU-work metric).
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling; BU=http://localhost:8000; M=qwen3-omni-colocated-h200
mkdir -p $D/results
probe(){ local name=$1 C=$2 N=$3 MODS=$4 DEV=$5; shift 5
  echo "### PROBE $name C=$C N=$N MODS=$MODS DEV=$DEV ###"
  python $D/run_workload.py --base-url $BU --model $M --concurrency $C --num-requests $N \
    --modalities "$MODS" --max-tokens 32 --ignore-eos "$@" --label sm_$name --out $D/results/sm_$name.json >$D/logs/wl_$name.log 2>&1 &
  local WL=$!
  sleep 5   # ramp to steady state
  nsys profile --gpu-metrics-devices=$DEV --gpu-metrics-frequency=1000 \
    -o $D/results/sm_$name -f true -d 18 sleep 18 >/dev/null 2>&1
  wait $WL 2>/dev/null
  nsys export --type sqlite --force-overwrite true -o $D/results/sm_$name.sqlite $D/results/sm_$name.nsys-rep >/dev/null 2>&1
  echo "done $name"
}
probe "$@"
echo "SM_PROBE_DONE"
