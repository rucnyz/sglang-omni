#!/bin/bash
# Hardening v2: (A) low-load think-time → genuine IDLE (validate four-way),
# (B) concurrency sweep x3 repeats → CIs on GPU%/duty/throughput.
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling; BU=http://localhost:8000; M=qwen3-omni-colocated-h200
mkdir -p $D/results $D/findings; ns(){ python -c 'import time;print(time.time_ns())'; }
seg(){ local name=$1 C=$2 N=$3 MT=$4; shift 4
  echo "SEG=$name SINCE=$(ns) C=$C N=$N MODS=text,audio MT=$MT"
  python $D/nvml_sampler.py --out $D/results/gpu_${name}.csv --interval-ms 50 --gpus 4 &
  local SP=$!
  python $D/run_workload.py --base-url $BU --model $M --concurrency $C --num-requests $N \
    --modalities "text,audio" --max-tokens $MT --ignore-eos "$@" --label $name --out $D/results/${name}.json 2>&1 | tail -3
  kill $SP 2>/dev/null; sleep 1
  echo "SEG=$name UNTIL=$(ns)"
}
echo "=== warmup ==="
python $D/run_workload.py --base-url $BU --model $M --concurrency 2 --num-requests 4 --ignore-eos --max-tokens 32 --label h2warm 2>&1 | tail -1

echo "########## (A) LOW-LOAD think-time 3s @ C=1 → expect IDLE>0 ##########"
seg lowload_think3 1 8 32 --think-time 3

echo "########## (B) concurrency sweep x3 repeats (CIs) ##########"
for C in 1 2 4 8 16; do
  N=$(( C<4 ? 8 : C*3 ))
  for r in 1 2 3; do seg sweep2_c${C}_r${r} $C $N 32; done
done
echo "CAMPAIGN_HARD2_DONE"
