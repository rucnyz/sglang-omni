#!/bin/bash
# Disaggregated campaign: thinker on GPU0, talker+code2wav on GPU1.
# Goal = the clean "wasted whole GPU" number: GPU0 (thinker alone) utilization
# for speech output. NVML samples BOTH GPUs.
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling
BU=http://localhost:8000
M=qwen3-omni  # disagg server reports model id "qwen3-omni" (default name)
mkdir -p $D/results $D/findings
ns(){ python -c 'import time;print(time.time_ns())'; }

# discover served model id
MID=$(curl -s $BU/v1/models | python -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3-omni)
echo "model id = $MID"

echo "=== warmup ==="
python $D/run_workload.py --base-url $BU --model "$MID" --concurrency 1 --num-requests 3 --label dwarm --out $D/results/dwarm.json 2>&1 | tail -3

seg(){ local name=$1 C=$2 N=$3
  echo "SEG=$name SINCE=$(ns) C=$C N=$N MODS=text,audio MT=256"
  python $D/nvml_sampler.py --out $D/results/gpu_${name}.csv --interval-ms 50 --gpus 0,1 &
  local SP=$!
  python $D/run_workload.py --base-url $BU --model "$MID" --concurrency $C --num-requests $N \
    --modalities "text,audio" --max-tokens 256 --label $name --out $D/results/${name}.json 2>&1 | tail -3
  kill $SP 2>/dev/null; sleep 1
  echo "SEG=$name UNTIL=$(ns)"
}

echo "########## disagg speech sweep ##########"
seg disagg_c1 1 6
seg disagg_c4 4 16
seg disagg_c8 8 24
echo "CAMPAIGN_DISAGG_DONE"
