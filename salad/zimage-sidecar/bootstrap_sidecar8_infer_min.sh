#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${R2_KEY:?R2_KEY is required}"
: "${R2_SECRET:?R2_SECRET is required}"
: "${R2_URL:?R2_URL is required}"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_REQUIRE_VIRTUALENV=false

ZIMAGE_TURBO_REVISION="f332072aa78be7aecdf3ee76d5c247082da564a6"
RUN_NAME="sidecar-n40-turbo-mix-v1"
R2_RUN="r2:nana-storage/zimage-sidecar/runs/${RUN_NAME}"
R2_IDENTITY="${R2_IDENTITY:-r2:nana-storage/zimage-sidecar/identity/character-v3.safetensors}"
R2_OUTPUT="${R2_OUTPUT:-r2:nana-storage/zimage-sidecar/validation/sidecar-int8-6000}"

CODE_DIR="/workspace/code/sidecar8"
VALIDATOR="$CODE_DIR/validate_sidecar8_turbo.py"
IDENTITY="/workspace/identity/character-v3.safetensors"
SIDECAR="/workspace/sidecar/sidecar-step-006000.safetensors"
META_DIR="/workspace/meta"
OUTPUT_DIR="/workspace/output/sidecar-int8-6000"

mkdir -p \
  "$HF_HOME" \
  "$CODE_DIR" \
  /workspace/identity \
  /workspace/sidecar \
  "$META_DIR" \
  "$OUTPUT_DIR" \
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
print('BF16 supported:', torch.cuda.is_bf16_supported())
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('huggingface_hub:', huggingface_hub.__version__)

if diffusers.__version__ != '0.40.0':
    raise SystemExit(f'expected diffusers 0.40.0, got {diffusers.__version__}')
PY

rclone version | head -n 1

echo
echo '=== bitsandbytes ==='
if ! python - <<'PY'
import bitsandbytes as bnb
print('bitsandbytes:', bnb.__version__)
PY
then
    python -m pip install bitsandbytes
fi
python - <<'PY'
import bitsandbytes as bnb
print('bitsandbytes:', bnb.__version__)
PY
python -m pip check

echo
echo '=== Sidecar assets ==='
rclone copyto "$R2_IDENTITY" "$IDENTITY" --retries 3 --low-level-retries 10
rclone copyto "$R2_RUN/sidecar-step-006000.safetensors" "$SIDECAR" --retries 3 --low-level-retries 10
rclone copyto "$R2_RUN/config.json" "$META_DIR/config.json" --retries 3 --low-level-retries 10
rclone copyto "$R2_RUN/dataset_manifest.json" "$META_DIR/dataset_manifest.json" --retries 3 --low-level-retries 10

python - "$IDENTITY" "$SIDECAR" <<'PY'
import sys
import torch
from safetensors.torch import load_file

identity_path, sidecar_path = sys.argv[1:]
identity = load_file(identity_path, device='cpu')
if 'embedding' not in identity or tuple(identity['embedding'].shape) != (1, 512):
    raise SystemExit(f'invalid identity: {identity_path}')
emb = identity['embedding'].float()
if not torch.isfinite(emb).all():
    raise SystemExit('identity contains NaN/Inf')
print('identity:', tuple(emb.shape), 'norm=', float(emb.norm().item()))

sidecar = load_file(sidecar_path, device='cpu')
if not sidecar:
    raise SystemExit(f'empty sidecar: {sidecar_path}')
print('sidecar tensors:', len(sidecar))
print('sidecar bytes:', sum(x.numel() * x.element_size() for x in sidecar.values()))
PY

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
echo '=== write INT8 validator ==='
cat > "$VALIDATOR" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.torch import load_file

KNOWN_VALIDATION_IMAGES = (
    'core/front_A_bg2.png',
    'core/left_3q_C_bg2.png',
    'core/right_profile_D_bg2.png',
    'expression/front_B_bg3_exp.png',
    'frame/front_A_bg1_half.png',
    'frame/front_B_bg3_full.png',
)

COFFEE_PROMPT = '''realistic photo, clear left three-quarter portrait, seated by a window, upper body visible, holding a coffee cup, calm gentle expression, subtle closed-mouth smile.

She has a neat bob haircut ending around the jawline, with soft side-swept bangs. She wears a charcoal blazer over a simple white blouse, with very minimal jewelry. The cafe background is softly blurred and secondary. Soft natural window light, realistic proportions, photorealistic, natural skin, restrained styling.'''


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_prompts(clean_snapshot: Path, captions: list[str], device, dtype) -> dict[str, torch.Tensor]:
    from transformers import AutoModel, AutoTokenizer

    unique = list(dict.fromkeys(captions))
    print(f'[text] unique prompts={len(unique)}')
    tokenizer = AutoTokenizer.from_pretrained(clean_snapshot / 'tokenizer', local_files_only=True)
    encoder = AutoModel.from_pretrained(
        clean_snapshot / 'text_encoder', dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    encoder.eval()
    out = {}
    with torch.no_grad():
        for cap in unique:
            text = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': cap}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            toks = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            ids = toks.input_ids.to(device)
            mask = toks.attention_mask.to(device).bool()
            hidden = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
            emb = hidden[0][mask[0]].detach().to('cpu', dtype=dtype).contiguous()
            if emb.ndim != 2 or emb.shape[1] != 2560 or emb.shape[0] < 1:
                raise ValueError(f'unexpected text embedding shape: {tuple(emb.shape)}')
            out[cap] = emb
            del hidden, ids, mask, toks
    del encoder, tokenizer
    cleanup_cuda()
    return out


class IdentityProjector(nn.Module):
    def __init__(self, identity_dim: int = 512, rank: int = 512, num_tokens: int = 8):
        super().__init__()
        self.rank = rank
        self.num_tokens = num_tokens
        self.proj = nn.Linear(identity_dim, num_tokens * rank, bias=True)
        self.norm = nn.LayerNorm(rank)

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        x = self.proj(identity.float()).view(identity.shape[0], self.num_tokens, self.rank)
        return self.norm(x)


class IdentityAttentionResidual(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 512):
        super().__init__()
        self.rank = rank
        self.hidden_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_q = nn.Linear(hidden_dim, rank, bias=False)
        self.to_k = nn.Linear(rank, rank, bias=False)
        self.to_v = nn.Linear(rank, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, hidden: torch.Tensor, identity_tokens: torch.Tensor) -> torch.Tensor:
        q = self.to_q(self.hidden_norm(hidden.float()))
        k = self.to_k(identity_tokens)
        v = self.to_v(identity_tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rank), dim=-1)
        return self.out_proj(torch.matmul(attn, v)).to(hidden.dtype)


class IdentitySidecar(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, num_tokens: int, layers: Iterable[int], scales: Iterable[float]):
        super().__init__()
        layers = tuple(int(x) for x in layers)
        scales = tuple(float(x) for x in scales)
        if len(layers) != len(scales) or len(set(layers)) != len(layers):
            raise ValueError('layers/scales must pair, with unique layers')
        self.layers = layers
        self.scales = dict(zip(layers, scales))
        self.projector = IdentityProjector(512, rank, num_tokens)
        self.slots = nn.ModuleDict({str(i): IdentityAttentionResidual(hidden_dim, rank) for i in layers})
        self.identity = None
        self.image_tokens = None
        self.global_scale = 1.0
        self._handles = []

    def set_context(self, identity: torch.Tensor, image_tokens: int, global_scale: float) -> None:
        self.identity = identity
        self.image_tokens = int(image_tokens)
        self.global_scale = float(global_scale)

    def _hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if self.identity is None or self.image_tokens is None or self.global_scale == 0.0:
                return output
            if self.image_tokens > output.shape[1]:
                raise ValueError('real image token count exceeds transformer block sequence length')
            n = self.image_tokens
            image_hidden = output[:, :n]
            identity_tokens = self.projector(self.identity)
            delta = self.slots[str(layer_idx)](image_hidden, identity_tokens)
            delta = delta * (self.global_scale * self.scales[layer_idx])
            return torch.cat([image_hidden + delta, output[:, n:]], dim=1)
        return hook

    def attach(self, transformer) -> None:
        if self._handles:
            return
        for layer in self.layers:
            if not 0 <= layer < len(transformer.layers):
                raise ValueError(f'invalid injection layer {layer}; model has {len(transformer.layers)} layers')
        self._handles = [transformer.layers[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_identity(path: Path) -> torch.Tensor:
    values = load_file(str(path), device='cpu')
    if 'embedding' not in values:
        raise KeyError(f'identity file has no embedding tensor: {path}')
    identity = values['embedding'].float().contiguous()
    if tuple(identity.shape) != (1, 512):
        raise ValueError(f'expected identity [1,512], got {tuple(identity.shape)}')
    if not torch.isfinite(identity).all():
        raise ValueError('identity contains NaN/Inf')
    norm = float(identity.norm(dim=-1).item())
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f'identity norm={norm:.8f}, expected 1')
    return identity


def safe_stem(rel: str) -> str:
    return rel.rsplit('.', 1)[0].replace('/', '-').replace('\\', '-')


def build_panel(step_dir: Path, rows: list[dict], seeds: list[int], scale: float) -> None:
    tile_w, tile_h, label_h = 320, 320, 28
    canvas = Image.new('RGB', (tile_w * len(seeds), (tile_h + label_h) * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for r, row in enumerate(rows):
        for c, seed in enumerate(seeds):
            p = step_dir / f"{safe_stem(row['image'])}--{seed}--s{scale:.2f}.png"
            with Image.open(p) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (tile_w, tile_h))
                x = c * tile_w + (tile_w - thumb.width) // 2
                y = r * (tile_h + label_h) + label_h + (tile_h - thumb.height) // 2
                canvas.paste(thumb, (x, y))
            draw.text((c * tile_w + 4, r * (tile_h + label_h) + 7), f"{Path(row['image']).stem} | {seed}", fill='black', font=font)
    canvas.save(step_dir / 'panel.jpg', quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean-snapshot', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--identity', type=Path, required=True)
    ap.add_argument('--sidecar', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--r2-out', default='')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')

    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    by_rel = {s['image']: s for s in manifest['samples']}
    missing = [x for x in KNOWN_VALIDATION_IMAGES if x not in by_rel]
    if missing:
        raise ValueError(f'validation entries missing: {missing}')

    samples = [by_rel[x] for x in KNOWN_VALIDATION_IMAGES]
    seeds = list(config['validation']['seeds'])
    scale = float(config['validation']['scale'])
    inference_steps = int(config['validation']['inference_steps'])
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    captions = [s['caption_text'] for s in samples] + [COFFEE_PROMPT]
    prompt_map = encode_prompts(args.clean_snapshot, captions, device, dtype)

    identity = load_identity(args.identity).to(device=device, dtype=torch.float32)

    cfg = json.loads((args.clean_snapshot / 'transformer' / 'config.json').read_text(encoding='utf-8'))
    dim = int(cfg['dim'])
    layers = list(config['injection_layers_0based'])
    sidecar = IdentitySidecar(
        dim,
        int(config['sidecar_rank']),
        int(config['identity_tokens']),
        layers,
        list(config['layer_scales']),
    ).to(device=device, dtype=torch.float32)
    sidecar.load_state_dict(load_file(str(args.sidecar), device='cpu'), strict=True)
    sidecar.eval()

    print('[model] loading Z-Image-Turbo transformer as bitsandbytes INT8')
    from diffusers import AutoModel as DiffusersAutoModel
    from diffusers import AutoencoderKL, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler, ZImagePipeline

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    transformer = DiffusersAutoModel.from_pretrained(
        args.clean_snapshot,
        subfolder='transformer',
        quantization_config=quant_config,
        dtype=dtype,
        device_map='cuda',
        local_files_only=True,
    )
    transformer.eval()
    print('[model] transformer footprint GiB:', round(transformer.get_memory_footprint() / 1024**3, 3))
    sidecar.attach(transformer)

    vae = AutoencoderKL.from_pretrained(
        args.clean_snapshot / 'vae', dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    vae.eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.clean_snapshot / 'scheduler', local_files_only=True)
    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)

    val_dir = args.out / 'validation-step-006000'
    val_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for sample in samples:
            prompt = prompt_map[sample['caption_text']].to(device=device, dtype=dtype)
            sidecar.set_context(identity, int(sample['image_tokens']), scale)
            rows.append({'image': sample['image']})
            for seed in seeds:
                g = torch.Generator(device='cuda').manual_seed(seed)
                image = pipe(
                    prompt=None,
                    prompt_embeds=[prompt],
                    height=int(sample['train_height']),
                    width=int(sample['train_width']),
                    num_inference_steps=inference_steps,
                    guidance_scale=0.0,
                    generator=g,
                ).images[0]
                p = val_dir / f"{safe_stem(sample['image'])}--{seed}--s{scale:.2f}.png"
                image.save(p)
                print('[gen]', p.name)

        build_panel(val_dir, rows, seeds, scale)

        coffee_prompt = prompt_map[COFFEE_PROMPT].to(device=device, dtype=dtype)
        sidecar.set_context(identity, (1024 // 16) * (1024 // 16), scale)
        coffee_seed = 12664542
        coffee = pipe(
            prompt=None,
            prompt_embeds=[coffee_prompt],
            height=1024,
            width=1024,
            num_inference_steps=inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device='cuda').manual_seed(coffee_seed),
        ).images[0]
        coffee_path = args.out / f'coffee-step-006000--seed-{coffee_seed}--s{scale:.2f}.png'
        coffee.save(coffee_path)
        print('[gen]', coffee_path.name)

    elapsed = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    report = {
        'gpu': torch.cuda.get_device_name(0),
        'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        'dtype': str(dtype),
        'quantization': 'bitsandbytes_8bit_transformer_only',
        'sidecar_step': 6000,
        'sidecar_scale': scale,
        'inference_steps': inference_steps,
        'validation_seeds': seeds,
        'coffee_seed': coffee_seed,
        'transformer_memory_footprint_gib': transformer.get_memory_footprint() / 1024**3,
        'peak_allocated_gib': peak_alloc,
        'peak_reserved_gib': peak_reserved,
        'elapsed_seconds_total': elapsed,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

    sidecar.detach()
    if args.r2_out:
        subprocess.run(['rclone', 'copy', str(args.out), args.r2_out, '--retries', '3', '--low-level-retries', '10'], check=True)
        print('[r2]', args.r2_out)


if __name__ == '__main__':
    main()
PY

python -m py_compile "$VALIDATOR"
chmod +x "$VALIDATOR"

echo
echo '=== ready ==='
echo "validator=$VALIDATOR"
echo "identity=$IDENTITY"
echo "sidecar=$SIDECAR"
echo "output=$OUTPUT_DIR"
echo "r2_output=$R2_OUTPUT"
echo
echo 'Run:'
printf 'python -u %q --clean-snapshot %q --config %q --manifest %q --identity %q --sidecar %q --out %q --r2-out %q\n' \
  "$VALIDATOR" \
  "$HF_HOME/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/$ZIMAGE_TURBO_REVISION" \
  "$META_DIR/config.json" \
  "$META_DIR/dataset_manifest.json" \
  "$IDENTITY" \
  "$SIDECAR" \
  "$OUTPUT_DIR" \
  "$R2_OUTPUT"
echo
echo 'No dataset, train mix, trainer, optimizer state, or LoRA/PEFT dependency was downloaded.'
