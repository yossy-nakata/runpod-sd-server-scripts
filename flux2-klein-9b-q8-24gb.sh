#!/usr/bin/env bash
set -euo pipefail

# FLUX.2 Klein 9B (Q8) + Qwen3 8B (Q4_K_M) + FLUX.2 VAE
# Intended for stable-diffusion.cpp sd-server on a 24GB GPU.
#
# Usage examples:
#   bash /workspace/flux2-klein-9b-q8-24gb.sh
#   MODEL_DIR=/workspace/models PORT=1234 bash /workspace/flux2-klein-9b-q8-24gb.sh
#
# Optional environment variables:
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN : Hugging Face access token
#   MODEL_DIR   : model directory (default: /workspace/models)
#   PORT        : server port (default: 1234)
#   HOST        : listen IP (default: 0.0.0.0)
#   SD_SERVER   : sd-server path (default: /sd/bin/sd-server)

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
PORT="${PORT:-1234}"
HOST="${HOST:-0.0.0.0}"
SD_SERVER="${SD_SERVER:-/sd/bin/sd-server}"

mkdir -p "$MODEL_DIR"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
CURL_AUTH_ARGS=()
if [ -n "$HF_TOKEN_VALUE" ]; then
    CURL_AUTH_ARGS=(-H "Authorization: Bearer $HF_TOKEN_VALUE")
fi

download() {
    local url="$1"
    local file="$2"

    local dst="$MODEL_DIR/$file"
    local tmp="$dst.part"

    if [ -s "$dst" ]; then
        echo "[skip] $file"
        return
    fi

    echo
    echo "[download] $file"

    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        "${CURL_AUTH_ARGS[@]}" \
        --output "$tmp" \
        "$url"

    mv "$tmp" "$dst"
    echo "[done] $file"
}

echo "=== disk ==="
df -h /workspace || true
echo

echo "=== gpu ==="
nvidia-smi || true
echo

# DiT (official sd.cpp GGUF by leejet)
download \
"https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf?download=true" \
"flux-2-klein-9b-Q8_0.gguf"

# Text encoder (Qwen3 8B GGUF)
download \
"https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true" \
"Qwen3-8B-Q4_K_M.gguf"

# Official FLUX.2 VAE (gated; requires HF token accepted for black-forest-labs/FLUX.2-dev)
download \
"https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors?download=true" \
"flux2_ae.safetensors"

echo
echo "=== models ==="
ls -lh "$MODEL_DIR"
echo

echo "=== starting sd-server ==="
exec "$SD_SERVER" \
  --diffusion-model "$MODEL_DIR/flux-2-klein-9b-Q8_0.gguf" \
  --vae "$MODEL_DIR/flux2_ae.safetensors" \
  --vae-format flux \
  --llm "$MODEL_DIR/Qwen3-8B-Q4_K_M.gguf" \
  --backend diffusion=cuda0,te=cuda0,vae=cuda0 \
  --auto-fit off \
  --cfg-scale 1 \
  --steps 4 \
  --sampling-method euler \
  --diffusion-fa \
  --listen-ip "$HOST" \
  --listen-port "$PORT" \
  -v
