#!/bin/bash
# Comprehensive co-location measurement campaign. Each measured segment is
# bracketed by machine-parseable SEG markers + its own NVML sample so the
# offline analyzer can window to it (--since-ns/--until-ns).
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling
BU=http://localhost:8000
M=qwen3-omni-colocated-h200
mkdir -p $D/results $D/findings
ns(){ python -c 'import time;print(time.time_ns())'; }

seg(){  # seg <name> <concurrency> <nreq> <modalities> <max_tokens>
  local name=$1 C=$2 N=$3 MODS=$4 MT=$5
  echo "SEG=$name SINCE=$(ns) C=$C N=$N MODS=$MODS MT=$MT"
  python $D/nvml_sampler.py --out $D/results/gpu_${name}.csv --interval-ms 50 --gpus 0 &
  local SP=$!
  python $D/run_workload.py --base-url $BU --model $M --concurrency $C --num-requests $N \
    --modalities "$MODS" --max-tokens $MT --label $name --out $D/results/${name}.json 2>&1 | tail -3
  kill $SP 2>/dev/null; sleep 1
  echo "SEG=$name UNTIL=$(ns)"
}

echo "########## A1: concurrency sweep, text->speech ##########"
seg sweep_c1  1  6  "text,audio" 256
seg sweep_c2  2  8  "text,audio" 256
seg sweep_c4  4  16 "text,audio" 256
seg sweep_c8  8  24 "text,audio" 256
seg sweep_c16 16 32 "text,audio" 256

echo "########## A2: understanding (text-only output) ##########"
seg text_c1 1 6  "text" 256
seg text_c4 4 16 "text" 256
seg text_c8 8 24 "text" 256

echo "########## A3: audio-length variation @ C=4 ##########"
seg audio_short 4 16 "text,audio" 64
seg audio_long  4 12 "text,audio" 512

echo "CAMPAIGN_COLOC_DONE"
