#!/usr/bin/env bash
set -euo pipefail

export UV_SYSTEM_PYTHON=1
export UV_PYTHON_DOWNLOADS=0
export UV_TORCH_BACKEND=cu126
export PIP_REQUIRE_VIRTUALENV=true

[[ "$(command -v python)" == "/usr/local/bin/python" ]] || {
  echo "ABORT: expected /usr/local/bin/python, got $(command -v python)" >&2
  exit 1
}

command -v uv >/dev/null 2>&1 || {
  echo "ABORT: uv is not installed in the Docker image" >&2
  exit 1
}

python - <<'PY'
import sys
import torch

if sys.version_info[:3] != (3, 11, 16):
    raise SystemExit(f"expected Python 3.11.16, got {sys.version.split()[0]}")
if torch.__version__ != "2.9.1+cu126":
    raise SystemExit(f"expected torch 2.9.1+cu126, got {torch.__version__}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"expected torch CUDA build 12.6, got {torch.version.cuda}")
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
PY

if python - <<'PY' >/dev/null 2>&1
import torchvision
assert torchvision.__version__ == "0.24.1+cu126"
PY
then
  echo "torchvision 0.24.1+cu126 is already installed"
  exit 0
fi

echo '=== installing torchvision only ==='
uv pip install --system --no-deps 'torchvision==0.24.1'

python - <<'PY'
import torch
import torchvision

if torch.__version__ != "2.9.1+cu126":
    raise SystemExit(f"torch changed unexpectedly: {torch.__version__}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"torch CUDA build changed unexpectedly: {torch.version.cuda}")
if torchvision.__version__ != "0.24.1+cu126":
    raise SystemExit(f"expected torchvision 0.24.1+cu126, got {torchvision.__version__}")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA build:", torch.version.cuda)
PY

uv pip check --system
