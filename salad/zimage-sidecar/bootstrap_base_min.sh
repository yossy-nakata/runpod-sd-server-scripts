#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_REQUIRE_VIRTUALENV=false

TRIGGER="${TRIGGER:-nana_person}"
CLASS_WORD="${CLASS_WORD:-person}"
OVERFIT_IMAGE="${OVERFIT_IMAGE:-core/front_A_bg2.png}"

ZIMAGE_TURBO_REVISION="f332072aa78be7aecdf3ee76d5c247082da564a6"
ZIMAGE_BASE_REVISION="04cc4abb7c5069926f75c9bfde9ef43d49423021"

CODE_DIR="/workspace/code/lora40"
R2_CODE="r2:nana-storage/zimage-sidecar/code/lora40"
TRAINER="$CODE_DIR/train_dreambooth_lora_z_image.py"
TRAINER_URL="https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/examples/dreambooth/train_dreambooth_lora_z_image.py"

mkdir -p /workspace "$HF_HOME" "$CODE_DIR" /workspace/runs /root/.config/rclone

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
if torch.cuda.is_available():
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
echo '=== LoRA dependencies ==='
if ! python - <<'PY'
import peft
import datasets
import torchvision

assert peft.__version__ == '0.21.2'
assert datasets.__version__ == '5.0.1'
assert torchvision.__version__ == '0.24.1+cu126'
PY
then
    python -m pip install --no-deps \
      'torchvision==0.24.1+cu126' \
      --index-url https://download.pytorch.org/whl/cu126

    python -m pip install \
      'peft==0.21.2' \
      'datasets==5.0.1'
fi

python -m pip check

echo
echo '=== dataset ==='
if check_dataset; then
    echo 'dataset: already present (20 / 8 / 9 / 3)'
else
    if [[ -e /workspace/dataset ]]; then
        echo 'ERROR: /workspace/dataset exists but is incomplete; refusing to overwrite.' >&2
        exit 1
    fi

    rclone copyto r2:nana-storage/zimage-sidecar/dataset/dataset.tar /workspace/dataset.tar
    rclone copyto r2:nana-storage/zimage-sidecar/dataset/dataset.tar.sha256 /workspace/dataset.tar.sha256

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

echo
echo '=== code ==='
rclone copy "$R2_CODE/" "$CODE_DIR/" --progress

for f in \
    prepare_lora40_dataset.py \
    prepare_lora1_dataset.py \
    lora_checkpoint_sync.py \
    run_lora40.sh \
    run_lora1_overfit.sh \
    validate_lora40_turbo.py
do
    [[ -f "$CODE_DIR/$f" ]] || {
        echo "ERROR: missing code after R2 restore: $CODE_DIR/$f" >&2
        exit 1
    }
done

chmod +x "$CODE_DIR"/*.sh "$CODE_DIR"/*.py

echo
echo '=== official Diffusers v0.40.0 Z-Image trainer ==='
tmp="${TRAINER}.tmp.$$"
curl -fsSL "$TRAINER_URL" -o "$tmp"
python -m py_compile "$tmp"
mv "$tmp" "$TRAINER"
chmod +x "$TRAINER"

python "$TRAINER" --help >/dev/null

grep -Fq 'Saved state to' "$TRAINER" || {
    echo 'ERROR: trainer checkpoint completion marker not found.' >&2
    exit 1
}

sha256sum "$TRAINER"

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

required = ['model_index.json', 'scheduler', 'text_encoder', 'tokenizer', 'transformer', 'vae']
missing = [name for name in required if not (p / name).exists()]
broken = [x for x in p.rglob('*') if x.is_symlink() and not x.exists()]

print('snapshot:', p)
print('missing:', missing)
print('broken symlinks:', len(broken))

if missing or broken:
    raise SystemExit(1)
PY

echo
echo '=== Z-Image Base minimal ==='
python - "$ZIMAGE_BASE_REVISION" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

revision = sys.argv[1]

p = Path(snapshot_download(
    repo_id='Tongyi-MAI/Z-Image',
    revision=revision,
    token=os.environ['HF_TOKEN'],
    allow_patterns=[
        'transformer/*',
        'scheduler/scheduler_config.json',
    ],
))

required = ['transformer', 'scheduler/scheduler_config.json']
missing = [name for name in required if not (p / name).exists()]
broken = [x for x in p.rglob('*') if x.is_symlink() and not x.exists()]

print('snapshot:', p)
print('missing:', missing)
print('broken symlinks:', len(broken))

if missing or broken:
    raise SystemExit(1)
PY

echo
echo '=== Base training mix ==='
python - "$HF_HOME" "$ZIMAGE_TURBO_REVISION" "$ZIMAGE_BASE_REVISION" <<'PY'
import shutil
import sys
from pathlib import Path

hf_home = Path(sys.argv[1])
hub = hf_home / 'hub'
turbo = hub / 'models--Tongyi-MAI--Z-Image-Turbo' / 'snapshots' / sys.argv[2]
base = hub / 'models--Tongyi-MAI--Z-Image' / 'snapshots' / sys.argv[3]
mix = hf_home / 'zimage_base_train_mix'

if mix.exists() or mix.is_symlink():
    if mix.is_symlink() or mix.is_file():
        mix.unlink()
    else:
        shutil.rmtree(mix)

mix.mkdir(parents=True)

links = {
    'model_index.json': turbo / 'model_index.json',
    'tokenizer': turbo / 'tokenizer',
    'text_encoder': turbo / 'text_encoder',
    'vae': turbo / 'vae',
    'transformer': base / 'transformer',
}

for name, src in links.items():
    if not src.exists():
        raise FileNotFoundError(src)
    (mix / name).symlink_to(src, target_is_directory=src.is_dir())

(mix / 'scheduler').mkdir()
scheduler = base / 'scheduler' / 'scheduler_config.json'
if not scheduler.exists():
    raise FileNotFoundError(scheduler)
(mix / 'scheduler' / 'scheduler_config.json').symlink_to(scheduler)

broken = [p for p in mix.rglob('*') if p.is_symlink() and not p.exists()]
if broken:
    raise RuntimeError('broken mix symlinks: ' + ', '.join(map(str, broken)))

print('base_train_mix:', mix)
PY

echo
echo '=== prepared LoRA datasets ==='
python "$CODE_DIR/prepare_lora40_dataset.py" \
    --dataset /workspace/dataset \
    --out /workspace/lora40_dataset \
    --trigger "$TRIGGER" \
    --class-word "$CLASS_WORD"

python "$CODE_DIR/prepare_lora1_dataset.py" \
    --dataset /workspace/dataset \
    --image "$OVERFIT_IMAGE" \
    --out /workspace/lora1_dataset \
    --trigger "$TRIGGER" \
    --class-word "$CLASS_WORD"

python - <<'PY'
from datasets import load_dataset

for path, expected in [
    ('/workspace/lora40_dataset', 40),
    ('/workspace/lora1_dataset', 1),
]:
    ds = load_dataset(path, split='train')
    if len(ds) != expected:
        raise SystemExit(f'{path}: expected {expected}, got {len(ds)}')
    if not {'image', 'text'}.issubset(ds.column_names):
        raise SystemExit(f'{path}: unexpected columns {ds.column_names}')
    print(path, len(ds), ds.column_names)

print('DATASETS: OK')
PY

echo
echo '=== capacity report ==='
du -sh \
    /workspace/hf \
    /workspace/dataset \
    /workspace/code \
    /workspace/lora40_dataset \
    /workspace/lora1_dataset \
    /workspace/runs \
    2>/dev/null || true

echo
echo '=== ready ==='
echo "TRIGGER=$TRIGGER"
echo "overfit image=$OVERFIT_IMAGE"
echo "code=$CODE_DIR"
echo "Base train mix=/workspace/hf/zimage_base_train_mix"
echo "LoRA1 dataset=/workspace/lora1_dataset"
echo "LoRA40 dataset=/workspace/lora40_dataset"
echo "next: TRIGGER=$TRIGGER bash $CODE_DIR/run_lora1_overfit.sh"
