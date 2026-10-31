#!/bin/bash
# Fresh venv: torch 2.10.0+cu130 + Transformer Engine 2.20.2 (core cu13 wheel + PyTorch extension built from sdist).
set -euo pipefail
cd /root/agents/b300_prep
t0=$(date +%s)
uv venv venv --python 3.12 -q
. venv/bin/activate
uv pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
t1=$(date +%s); echo "TIME torch $((t1-t0))s"
uv pip install -q numpy setuptools wheel ninja pybind11 packaging pip
uv pip install -q "transformer_engine_cu13==2.20.2" "nvidia-cudnn-frontend>=1.29.0" einops onnxscript onnx pydantic nvdlfw-inspect
t2=$(date +%s); echo "TIME te_core $((t2-t1))s"
MAX_JOBS=64 NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation "transformer_engine_torch==2.20.2" "transformer_engine==2.20.2"
t3=$(date +%s); echo "TIME te_torch_build $((t3-t2))s"
python -c "import torch, transformer_engine as te, transformer_engine.pytorch as tep; print(torch.__version__, torch.version.cuda, te.__version__); from transformer_engine.common.recipe import NVFP4BlockScaling; print(NVFP4BlockScaling())"
echo "TIME total $((t3-t0))s"
