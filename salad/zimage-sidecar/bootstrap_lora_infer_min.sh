#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_REQUIRE_VIRTUALENV=false

ZIMAGE_TURBO_REVISION="f332072aa78be7aecdf3ee76d5c247082da564a6"

CODE_DIR="/workspace/code/lora40"
VALIDATOR="$CODE_DIR/validate_lora40_turbo.py"
R2_VALIDATOR="r2:nana-storage/zimage-sidecar/code/lora40/validate_lora40_turbo.py"

mkdir -p \
  "$HF_HOME" \
  "$CODE_DIR" \
  /workspace/lora \
  /workspace/output \
  /root/.config/rclone

cat > /root/.config/rclone/rclone.conf <<EOF_RCLONE
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_KEY}
secret_access_key = ${R2_SECRET}
endpoint = ${R2_URL}
EOF_RCLONE
chmod 600 /root/.config/rclone/rclone.conf

echo '=== runtime ==='
python - <<'PY'
import sys
import torch
import diffusers
import transformers
import accelerate
import huggingface_hub

print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('huggingface_hub:', huggingface_hub.__version__)

if diffusers.__version__ != '0.40.0':
    raise SystemExit(f'expected diffusers 0.40.0, got {diffusers.__version__}')
PY

rclone version | head -n 1

echo
echo '=== LoRA inference dependency ==='
if ! python - <<'PY'
import peft
assert peft.__version__ == '0.21.2'
print('peft:', peft.__version__)
PY
then
    python -m pip install 'peft==0.21.2'
fi

python -m pip check

echo
echo '=== validator ==='
rclone copyto "$R2_VALIDATOR" "$VALIDATOR"
python -m py_compile "$VALIDATOR"
chmod +x "$VALIDATOR"
echo "validator: $VALIDATOR"

echo
echo '=== Z-Image-Turbo ==='
python - "$ZIMAGE_TURBO_REVISION" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

revision = sys.argv[1]

p = Path(snapshot_download(
    repo_id='Tongyi-MAI/Z-Image-Turbo',
    revision=revision,
    token=os.environ['HF_TOKEN'],
    allow_patterns=[
        'model_index.json',
        'scheduler/*',
        'text_encoder/*',
        'tokenizer/*',
        'transformer/*',
        'vae/*',
    ],
))

required = [
    'model_index.json',
    'scheduler',
    'text_encoder',
    'tokenizer',
    'transformer',
    'vae',
]
missing = [name for name in required if not (p / name).exists()]
broken = [x for x in p.rglob('*') if x.is_symlink() and not x.exists()]

print('snapshot:', p)
print('missing:', missing)
print('broken symlinks:', len(broken))

if missing or broken:
    raise SystemExit(1)
PY

echo
echo '=== ready ==='
echo "validator=$VALIDATOR"
echo "lora_dir=/workspace/lora"
echo "output_dir=/workspace/output"
echo "No dataset, Z-Image Base, train mix, trainer, or optimizer state was downloaded."
