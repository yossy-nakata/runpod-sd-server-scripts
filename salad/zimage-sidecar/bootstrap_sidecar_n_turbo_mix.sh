#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export UV_SYSTEM_PYTHON=1
export UV_PYTHON_DOWNLOADS=0
export UV_TORCH_BACKEND=cu126
export PIP_REQUIRE_VIRTUALENV=true

ZIMAGE_REVISION="f332072aa78be7aecdf3ee76d5c247082da564a6"
ADAPTER_SHA256="20a541d3e016ab8de0da076321b48b6cd9b3ffd072d9df830a068220ab2265f6"

CODE_DIR="/workspace/code"
IDENTITY="/workspace/identity/character-v3.safetensors"
ADAPTER="/workspace/adapters/zimage_turbo_training_adapter_v2.safetensors"
TRAIN_MIX="/workspace/hf/zimage_turbo_train_mix"

mkdir -p \
  /workspace \
  "$HF_HOME" \
  "$CODE_DIR" \
  /workspace/identity \
  /workspace/adapters \
  /workspace/runs \
  /workspace/validation \
  /workspace/final \
  /workspace/cache/sidecar-n \
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

check_dataset() {
  for d in core expression frame nsfw; do
    [[ -d "/workspace/dataset/$d" ]] || return 1
  done

  [[ "$(find /workspace/dataset/core       -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" -eq 20 ]] &&
  [[ "$(find /workspace/dataset/expression -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" -eq 8  ]] &&
  [[ "$(find /workspace/dataset/frame      -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" -eq 9  ]] &&
  [[ "$(find /workspace/dataset/nsfw       -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" -eq 3  ]]
}

echo '=== runtime ==='
[[ "$(command -v python)" == "/usr/local/bin/python" ]] || {
  echo "ERROR: expected /usr/local/bin/python, got $(command -v python)" >&2
  exit 1
}

python - <<'PY'
import sys
import torch
import torchvision
import diffusers
import transformers
import accelerate
import huggingface_hub

expected = {
    'python': '3.11.16',
    'torch': '2.9.1+cu126',
    'torchvision': '0.24.1+cu126',
    'diffusers': '0.40.0',
    'transformers': '5.17.0',
    'accelerate': '1.15.0',
}
actual = {
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'diffusers': diffusers.__version__,
    'transformers': transformers.__version__,
    'accelerate': accelerate.__version__,
}
for name, wanted in expected.items():
    got = actual[name]
    print(f'{name}: {got}')
    if got != wanted:
        raise SystemExit(f'expected {name} {wanted}, got {got}')

print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.version.cuda != '12.6':
    raise SystemExit(f'expected torch CUDA build 12.6, got {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print('huggingface_hub:', huggingface_hub.__version__)
PY
uv pip check --system
rclone version | head -n 1

echo
echo '=== dataset ==='
if check_dataset; then
  echo 'dataset: already present (20 / 8 / 9 / 3)'
else
  if [[ -e /workspace/dataset ]]; then
    echo 'ERROR: /workspace/dataset exists but is incomplete or unexpected; refusing to overwrite.' >&2
    exit 1
  fi

  rclone copyto \
    r2:nana-storage/zimage-sidecar/dataset/dataset.tar \
    /workspace/dataset.tar
  rclone copyto \
    r2:nana-storage/zimage-sidecar/dataset/dataset.tar.sha256 \
    /workspace/dataset.tar.sha256

  (
    cd /workspace
    sha256sum -c dataset.tar.sha256
    tar -xf dataset.tar
  )

  check_dataset || {
    echo 'ERROR: restored dataset failed validation.' >&2
    exit 1
  }

  rm -f /workspace/dataset.tar /workspace/dataset.tar.sha256
  echo 'dataset: restored and verified (20 / 8 / 9 / 3)'
fi

python - <<'PY'
from pathlib import Path
from PIL import Image

root = Path('/workspace/dataset')
images = sorted(
    p for p in root.rglob('*')
    if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
)
if len(images) != 40:
    raise SystemExit(f'expected 40 images, found {len(images)}')
for image in images:
    caption = image.with_suffix('.txt')
    if not caption.is_file():
        raise SystemExit(f'missing caption: {caption}')
    if not caption.read_text(encoding='utf-8').strip():
        raise SystemExit(f'empty caption: {caption}')
    with Image.open(image) as im:
        im.verify()
print('dataset preflight: 40 image/caption pairs decode OK')
PY

echo
echo '=== code ==='
rclone copy \
  r2:nana-storage/zimage-sidecar/code/ \
  "$CODE_DIR/" \
  --progress

for f in train_sidecar_n_turbo_mix.py run_sidecar_n_turbo_mix.sh generate_sidecar_batch.py; do
  [[ -f "$CODE_DIR/$f" ]] || {
    echo "ERROR: missing code after R2 restore: $CODE_DIR/$f" >&2
    exit 1
  }
done
python -m py_compile \
  "$CODE_DIR/train_sidecar_n_turbo_mix.py" \
  "$CODE_DIR/generate_sidecar_batch.py"
bash -n "$CODE_DIR/run_sidecar_n_turbo_mix.sh"
chmod +x "$CODE_DIR/run_sidecar_n_turbo_mix.sh"
echo 'code: restored and syntax OK'

echo
echo '=== identity ==='
if [[ ! -f "$IDENTITY" ]]; then
  rclone copyto \
    r2:nana-storage/zimage-sidecar/identity/character-v3.safetensors \
    "$IDENTITY" \
    --progress
fi

python - "$IDENTITY" <<'PY'
import math
import sys
import torch
from safetensors.torch import load_file

path = sys.argv[1]
data = load_file(path)
if 'embedding' not in data:
    raise SystemExit("identity key 'embedding' is missing")
x = data['embedding'].float()
if tuple(x.shape) != (1, 512):
    raise SystemExit(f'expected identity shape [1,512], got {tuple(x.shape)}')
if not torch.isfinite(x).all():
    raise SystemExit('identity contains NaN/Inf')
norm = float(x.norm(dim=-1).item())
if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
    raise SystemExit(f'expected identity norm 1 within 1e-4, got {norm}')
print('identity:', path)
print('identity norm:', f'{norm:.8f}')
PY

echo
echo '=== training adapter ==='
if [[ ! -f "$ADAPTER" ]]; then
  rclone copyto \
    r2:nana-storage/zimage-sidecar/adapters/zimage_turbo_training_adapter_v2.safetensors \
    "$ADAPTER" \
    --progress
fi

ACTUAL_ADAPTER_SHA256="$(sha256sum "$ADAPTER" | awk '{print $1}')"
if [[ "$ACTUAL_ADAPTER_SHA256" != "$ADAPTER_SHA256" ]]; then
  echo "ERROR: training adapter SHA256 mismatch: $ACTUAL_ADAPTER_SHA256" >&2
  exit 1
fi
echo "training adapter: $ADAPTER"
echo "training adapter sha256: $ACTUAL_ADAPTER_SHA256"

echo
echo '=== Z-Image-Turbo ==='
CLEAN_SNAPSHOT="$(python - "$ZIMAGE_REVISION" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

revision = sys.argv[1]
path = snapshot_download(
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
)

p = Path(path)
required = ['model_index.json', 'scheduler', 'text_encoder', 'tokenizer', 'transformer', 'vae']
missing = [name for name in required if not (p / name).exists()]
broken = [x for x in p.rglob('*') if x.is_symlink() and not x.exists()]
if missing or broken:
    print('missing:', missing, file=sys.stderr)
    print('broken symlinks:', len(broken), file=sys.stderr)
    raise SystemExit(1)
print(p)
PY
)"
echo "clean snapshot: $CLEAN_SNAPSHOT"

echo
echo '=== Turbo train-mix ==='
if [[ ! -f "$TRAIN_MIX/transformer/config.json" ]]; then
  mkdir -p "$TRAIN_MIX"
  rclone copy \
    r2:nana-storage/zimage-sidecar/hf/zimage_turbo_train_mix/ \
    "$TRAIN_MIX/" \
    --progress
fi

[[ -f "$TRAIN_MIX/transformer/config.json" ]] || {
  echo "ERROR: missing $TRAIN_MIX/transformer/config.json" >&2
  exit 1
}

python - "$CLEAN_SNAPSHOT" "$TRAIN_MIX" <<'PY'
import json
import sys
from pathlib import Path

clean = Path(sys.argv[1])
train = Path(sys.argv[2])
clean_cfg = json.loads((clean / 'transformer' / 'config.json').read_text(encoding='utf-8'))
train_cfg = json.loads((train / 'transformer' / 'config.json').read_text(encoding='utf-8'))
for key in ('dim', 'n_layers'):
    if clean_cfg.get(key) != train_cfg.get(key):
        raise SystemExit(
            f'clean/train-mix transformer mismatch at {key}: '
            f'{clean_cfg.get(key)} != {train_cfg.get(key)}'
        )
broken = [x for x in train.rglob('*') if x.is_symlink() and not x.exists()]
if broken:
    raise SystemExit(f'train-mix has {len(broken)} broken symlinks')
print('train-mix:', train)
print('transformer dim:', train_cfg.get('dim'))
print('transformer layers:', train_cfg.get('n_layers'))
PY

echo
echo '=== ready ==='
echo 'Training has NOT been started.'
echo "dataset=/workspace/dataset"
echo "clean_snapshot=$CLEAN_SNAPSHOT"
echo "train_mix=$TRAIN_MIX"
echo "training_adapter=$ADAPTER"
echo "identity=$IDENTITY"
echo "code_dir=$CODE_DIR"
echo
echo 'Start training manually when ready:'
echo
echo '  DATASET=/workspace/dataset EXPECTED_COUNT=40 EPOCHS=10 \'
echo '    bash /workspace/code/run_sidecar_n_turbo_mix.sh'
