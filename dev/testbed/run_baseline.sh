#!/bin/bash
# Build the FIXED testbed baseline. Runs inside the omni-phase0 container.
# Order: fast models first (asr -> higgs -> qwen3tts) so their baselines lock early,
# then Qwen3-Omni last (slow cold start + heavy csweep). Each group is baselined as it
# finishes via --update-baseline, so an interruption during the slow group keeps the
# fast baselines already saved. Re-running is safe (idempotent; re-bakes that group).
#
#   docker exec omni-phase0 bash /workspace/sglang-omni/dev/testbed/run_baseline.sh
set -uo pipefail
cd /workspace/sglang-omni
GPU="${OMNI_GPU:-0}"
for FILT in asr higgs qwen3tts omni; do
  echo "######## BASELINE GROUP: ${FILT} ########"
  python dev/testbed/testbed.py --filter "${FILT}" --gpu "${GPU}" --update-baseline 2>&1
  echo "######## GROUP ${FILT} DONE ########"
done
echo "=== baselines now fixed in dev/testbed/baselines/ ==="
ls -1 dev/testbed/baselines/ 2>/dev/null
echo "BASELINE_ALL_DONE"
