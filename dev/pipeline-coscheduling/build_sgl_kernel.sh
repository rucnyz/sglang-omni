#!/bin/bash
set -e
cd /sgl-workspace/sglang/sgl-kernel
echo "=== ensure build deps ==="
pip install --break-system-packages -q scikit-build-core ninja wheel setuptools 2>&1 | tail -2 || true
export CMAKE_BUILD_PARALLEL_LEVEL=64
export MAX_JOBS=64
export TORCH_CUDA_ARCH_LIST="10.0a 10.3a 12.0a"
echo "=== build+install sgl-kernel with sm_103a (this takes a while) ==="
pip install --no-build-isolation --break-system-packages --force-reinstall --no-deps . 2>&1
echo "BUILD_EXIT=$?"
echo "=== verify sm_103 now in installed common_ops ==="
python - <<'PY'
import sgl_kernel, glob, os, subprocess
d=os.path.dirname(sgl_kernel.__file__)
for so in glob.glob(d+"/*.so"):
    out=subprocess.run(["cuobjdump","--list-elf",so],capture_output=True,text=True).stdout
    if "sm_103" in out:
        print("HAS sm_103:", os.path.basename(so))
PY
PY_TEST=$(python -c "
import torch
from sglang.srt.layers.layernorm import RMSNorm
n=RMSNorm(2048,eps=1e-6).cuda().to(torch.bfloat16)
x=torch.randn(4,2048,device='cuda',dtype=torch.bfloat16)
print('RMSNORM_EAGER_OK', tuple(n(x)[0].shape) if isinstance(n(x),tuple) else 'ok')
" 2>&1 | tail -2)
echo "RMSNORM_TEST: $PY_TEST"
echo "ALL_DONE"
