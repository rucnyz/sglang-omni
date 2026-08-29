#!/bin/bash
# Hardened campaign: ignore_eos (constant per-request work), pynvml NVML,
# coordinator admission events captured (four-way decomposition revived),
# + a mixed-traffic segment. Larger N for stability.
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling; BU=http://localhost:8000; M=qwen3-omni-colocated-h200
mkdir -p $D/results $D/findings; ns(){ python -c 'import time;print(time.time_ns())'; }

# seg <name> <C> <N> <mods> <mt> <extra run_workload args...>
seg(){ local name=$1 C=$2 N=$3 MODS=$4 MT=$5; shift 5
  echo "SEG=$name SINCE=$(ns) C=$C N=$N MODS=$MODS MT=$MT"
  python $D/nvml_sampler.py --out $D/results/gpu_${name}.csv --interval-ms 50 --gpus 0 &
  local SP=$!
  python $D/run_workload.py --base-url $BU --model $M --concurrency $C --num-requests $N \
    --modalities "$MODS" --max-tokens $MT --ignore-eos "$@" --label $name --out $D/results/${name}.json 2>&1 | tail -4
  kill $SP 2>/dev/null; sleep 1
  echo "SEG=$name UNTIL=$(ns)"
}
echo "=== warmup ==="
python $D/run_workload.py --base-url $BU --model $M --concurrency 2 --num-requests 4 --ignore-eos --max-tokens 32 --label hwarm --out $D/results/hwarm.json 2>&1 | tail -2

echo "########## hardened speech sweep (fixed work, mt=128) ##########"
seg hard_c1  1  8  "text,audio" 32
seg hard_c4  4  24 "text,audio" 32
seg hard_c8  8  32 "text,audio" 32
echo "########## understanding (text-only, fixed work) ##########"
seg hard_text_c1 1 8  "text" 32
seg hard_text_c8 8 32 "text" 32
echo "########## MIXED traffic (50% speech / 50% text) @ C=8 ##########"
seg hard_mix_c8 8 32 "text,audio" 32 --speech-frac 0.5
echo "CAMPAIGN_HARD_DONE"
