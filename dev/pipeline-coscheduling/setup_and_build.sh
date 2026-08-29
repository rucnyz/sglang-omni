#!/bin/bash
# Idempotent B300 bring-up: install deps, patch sgl-kernel for sm_103 + fix
# DeepGEMM, build wheel (temp on /scratch), cache the wheel, install, verify.
# Safe to re-run; reuses cached wheel if present.
set -uo pipefail
DEV=/workspace/sglang-omni/dev/pipeline-coscheduling
WHEELDIR=$DEV/wheels
SK=/sgl-workspace/sglang/sgl-kernel
export TMPDIR=$DEV/buildtmp
export CCACHE_DIR=$DEV/.ccache
mkdir -p "$WHEELDIR" "$TMPDIR" "$CCACHE_DIR"

echo "=== [1/5] install missing pure-python deps ==="
uv pip install --system --break-system-packages -q \
  typer msgpack xxhash tabulate qwen-vl-utils librosa av numba openai 2>&1 | tail -3 || true

echo "=== [2/5] reuse cached sm_103 wheel if present ==="
CACHED=$(ls -t "$WHEELDIR"/sgl_kernel-*.whl 2>/dev/null | head -1)
if [ -n "$CACHED" ]; then
  echo "found cached wheel: $CACHED"
else
  echo "=== [3/5] patch sgl-kernel CMakeLists (idempotent) ==="
  cd "$SK"
  python - <<'PY'
p="CMakeLists.txt"; s=open(p).read(); ch=False
# (a) sm_103a in the 12.8 block
if "compute_103a,code=sm_103a" not in s.split("VERSION_GREATER_EQUAL \"13.0\"")[0]:
    a='        "-gencode=arch=compute_120a,code=sm_120a"\n    )'
    b='        "-gencode=arch=compute_120a,code=sm_120a"\n        "-gencode=arch=compute_103a,code=sm_103a"\n    )'
    if a in s: s=s.replace(a,b,1); ch=True
# (b) DeepGEMM era-matched commit (Jan-6-2026 tip) instead of dead pin / main
for dead in ["54f99a8af537b3c6eb4819b69907ccbe2b600792","main"]:
    tag="GIT_TAG        "+dead
    if tag in s: s=s.replace(tag,"GIT_TAG        3ccf40c53a979df8c2f5edf87a166beac0d7b42c",1); ch=True
# (c) drop USE_SABI for deep_gemm_cpp (pybind11 Py_buffer vs Py_LIMITED_API on py3.12)
old="Python_add_library(deep_gemm_cpp MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${DEEPGEMM_SOURCES})"
new="Python_add_library(deep_gemm_cpp MODULE WITH_SOABI ${DEEPGEMM_SOURCES})"
if old in s: s=s.replace(old,new,1); ch=True
open(p,"w").write(s)
print("patched" if ch else "already patched (idempotent)")
PY
  echo "=== [4/5] build wheel (TMPDIR + ccache on /scratch) ==="
  export MAX_JOBS=64 CMAKE_BUILD_PARALLEL_LEVEL=64
  python -m pip wheel --no-build-isolation --no-deps -w "$WHEELDIR" . 2>&1
  echo "WHEEL_BUILD_EXIT=$?"
  CACHED=$(ls -t "$WHEELDIR"/sgl_kernel-*.whl 2>/dev/null | head -1)
fi

echo "=== [5/5] install + verify ==="
if [ -z "$CACHED" ]; then echo "NO WHEEL PRODUCED — build failed"; exit 1; fi
pip install --break-system-packages --force-reinstall --no-deps "$CACHED" 2>&1 | tail -3
python - <<'PY'
import glob, os, subprocess, sgl_kernel
d=os.path.dirname(sgl_kernel.__file__); ok=False
for so in glob.glob(d+"/*.so"):
    if "sm_103" in subprocess.run(["cuobjdump","--list-elf",so],capture_output=True,text=True).stdout:
        print("HAS sm_103:", os.path.basename(so)); ok=True
import torch
from sglang.srt.layers.layernorm import RMSNorm
n=RMSNorm(2048,eps=1e-6).cuda().to(torch.bfloat16)
x=torch.randn(4,2048,device="cuda",dtype=torch.bfloat16)
y=n(x); y=y[0] if isinstance(y,tuple) else y
print("RMSNORM_TEST: EAGER OK", tuple(y.shape))
PY
echo "ALL_DONE"
