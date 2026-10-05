#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
ZIMAGE_TURBO_REVISION="f332072aa78be7aecdf3ee76d5c247082da564a6"
ZIMAGE_BASE_REVISION="04cc4abb7c5069926f75c9bfde9ef43d49423021"

mkdir -p /workspace "$HF_HOME" /root/.config/rclone

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
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('huggingface_hub:', huggingface_hub.__version__)
PY
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
echo '=== Z-Image-Turbo ==='
python - "$ZIMAGE_TURBO_REVISION" <<'PY'
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

print('snapshot:', p)
print('missing:', missing)
print('broken symlinks:', len(broken))

if missing or broken:
    raise SystemExit(1)
PY

echo
echo '=== Z-Image Base (minimal for training) ==='
python - "$ZIMAGE_BASE_REVISION" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

revision = sys.argv[1]
path = snapshot_download(
    repo_id='Tongyi-MAI/Z-Image',
    revision=revision,
    token=os.environ['HF_TOKEN'],
    allow_patterns=[
        'transformer/*',
        'scheduler/scheduler_config.json',
    ],
)

p = Path(path)
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
echo '=== train aliases ==='
python - "$HF_HOME" "$ZIMAGE_TURBO_REVISION" "$ZIMAGE_BASE_REVISION" <<'PY'
import os
import shutil
import sys
from pathlib import Path

hf_home = Path(sys.argv[1])
turbo_rev = sys.argv[2]
base_rev = sys.argv[3]

hub = hf_home / 'hub'
turbo = hub / 'models--Tongyi-MAI--Z-Image-Turbo' / 'snapshots' / turbo_rev
base  = hub / 'models--Tongyi-MAI--Z-Image' / 'snapshots' / base_rev
mix = Path('/workspace/hf/zimage_base_train_mix')

if mix.exists() or mix.is_symlink():
    if mix.is_symlink() or mix.is_file():
        mix.unlink()
    else:
        shutil.rmtree(mix)
mix.mkdir(parents=True, exist_ok=True)

links = {
    'model_index.json': turbo / 'model_index.json',
    'tokenizer': turbo / 'tokenizer',
    'text_encoder': turbo / 'text_encoder',
    'vae': turbo / 'vae',
    'transformer': base / 'transformer',
}

for name, src in links.items():
    dst = mix / name
    dst.symlink_to(src, target_is_directory=src.is_dir())

(mix / 'scheduler').mkdir(exist_ok=True)
base_sched = base / 'scheduler' / 'scheduler_config.json'
dst_sched = mix / 'scheduler' / 'scheduler_config.json'
if dst_sched.exists() or dst_sched.is_symlink():
    dst_sched.unlink()
dst_sched.symlink_to(base_sched)

print('turbo_snapshot:', turbo)
print('base_snapshot:', base)
print('base_train_mix:', mix)
for p in sorted(mix.rglob('*')):
    if p.is_symlink():
        print('  ', p, '->', os.readlink(p))
PY

echo
echo '=== ready ==='
echo "dataset: 40 images"
echo "Z-Image-Turbo revision: $ZIMAGE_TURBO_REVISION"
echo "Z-Image Base revision: $ZIMAGE_BASE_REVISION"
echo "Turbo snapshot alias: /workspace/hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/$ZIMAGE_TURBO_REVISION"
echo "Base minimal snapshot: /workspace/hf/hub/models--Tongyi-MAI--Z-Image/snapshots/$ZIMAGE_BASE_REVISION"
echo "Base train mix path: /workspace/hf/zimage_base_train_mix"
du -sh /workspace/dataset "$HF_HOME" /workspace/hf/zimage_base_train_mix 2>/dev/null || true
