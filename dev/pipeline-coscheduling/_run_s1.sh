#!/bin/bash
cd /workspace/sglang-omni
D=dev/pipeline-coscheduling
echo "=== warmup (3 req, discard) ==="
python $D/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 1 --num-requests 3 --label warmup --out $D/results/warmup.json 2>&1 | tail -5
echo "=== MARK measured-run start ==="
python -c "import time; print(\"SINCE_NS\", time.time_ns())"
# start NVML sampler for the measured window
python $D/nvml_sampler.py --out $D/results/gpu_coloc_s1.csv --interval-ms 50 --gpus 0 &
SP=$!
echo "=== measured S1: C=1 N=8 ==="
python $D/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 1 --num-requests 8 --label s1_coloc --out $D/results/s1_coloc.json 2>&1 | tail -20
kill $SP 2>/dev/null
echo "RUN_S1_DONE"
