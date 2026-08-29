#!/bin/bash
cd /workspace/sglang-omni; D=dev/pipeline-coscheduling
for C in 4 8; do
  N=$((C*4))
  echo "=== SWEEP C=$C N=$N ==="
  python -c "import time; print(f\"SINCE_C${C}_NS\", time.time_ns())"
  python $D/nvml_sampler.py --out $D/results/gpu_coloc_c${C}.csv --interval-ms 50 --gpus 0 &
  SP=$!
  python $D/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency $C --num-requests $N --label c${C} --out $D/results/c${C}.json 2>&1 | tail -3
  kill $SP 2>/dev/null; sleep 1
done
echo "SWEEP_DONE"
