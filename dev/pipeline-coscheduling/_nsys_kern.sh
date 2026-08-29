#!/bin/bash
cd /workspace/sglang-omni; R=dev/pipeline-coscheduling
# server under nsys: capture window at t=[200,215] (nsys-slowed load finishes well before)
CUDA_VISIBLE_DEVICES=7 TMPDIR=/tmp TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
nsys profile --trace=cuda --gpu-metrics-devices=7 --delay=200 --duration=15 --force-overwrite true \
  -o $R/results/nsk \
  python -m sglang_omni.cli serve --colocate --config examples/configs/qwen3_omni_colocated_h200.yaml --port 8000 --host 0.0.0.0 \
  > $R/logs/nsys_server.log 2>&1 &
NSYS=$!
echo "nsys+server launched t=$(date +%s)"
# wait ready
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)" = "200" ] && { echo "ready t=$(date +%s) (i=$i)"; break; }
  sleep 5
done
# warmup then sustained load that runs through the t=[200,215] capture window
python $R/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 4 --num-requests 8 --modalities text,audio --max-tokens 32 --ignore-eos --label nwarm >/dev/null 2>&1
echo "warm done t=$(date +%s); starting sustained load"
python $R/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 8 --num-requests 160 --modalities text,audio --max-tokens 32 --ignore-eos --label nload --out $R/results/nload.json >/dev/null 2>&1
echo "load done t=$(date +%s)"
wait $NSYS 2>/dev/null
echo "NSYS_DONE; report:"; ls -la $R/results/nsk.nsys-rep 2>/dev/null
