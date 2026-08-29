#!/bin/bash
# One command: run the full testbed matrix inside the omni-phase0 container.
# Pass-through args go to testbed.py, e.g.:
#   bash dev/testbed/run_all.sh                 # full matrix
#   bash dev/testbed/run_all.sh --filter omni   # subset
#   bash dev/testbed/run_all.sh --profile       # + nsys GR/SM
#   bash dev/testbed/run_all.sh --regression    # compare to baselines/
#   bash dev/testbed/run_all.sh --list          # show coverage + runnable/skipped
set -uo pipefail
CONTAINER="${OMNI_CONTAINER:-omni-phase0}"
GPU="${OMNI_GPU:-0}"
docker exec "$CONTAINER" bash -lc \
  "cd /workspace/sglang-omni && python dev/testbed/testbed.py --gpu $GPU $*"
