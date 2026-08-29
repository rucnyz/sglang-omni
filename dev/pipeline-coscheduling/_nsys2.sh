#!/bin/bash
cd /workspace/sglang-omni; R=dev/pipeline-coscheduling
CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
nsys profile --trace=cuda --sample=none --force-overwrite true -o $R/results/talkern \
  python -m sglang_omni.cli serve --colocate --config examples/configs/qwen3_omni_colocated_h200.yaml \
    --talker-cuda-graph off --port 8000 --host 0.0.0.0 > $R/logs/nsys2_server.log 2>&1 &
NSYS=$!
echo "nsys pid $NSYS"
for i in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)" = "200" ] && { echo "ready i=$i"; break; }
  sleep 5
done
python $R/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 4 --num-requests 8 --modalities text,audio --max-tokens 32 --ignore-eos --label n2warm >/dev/null 2>&1
echo "warm done; sustained load"
python $R/run_workload.py --base-url http://localhost:8000 --model qwen3-omni-colocated-h200 --concurrency 8 --num-requests 70 --modalities text,audio --max-tokens 32 --ignore-eos --label n2load >/dev/null 2>&1
echo "load done; stopping nsys (SIGINT)"
kill -INT $NSYS; wait $NSYS 2>/dev/null
echo "NSYS2_DONE"; ls -la $R/results/talkern.nsys-rep 2>/dev/null
