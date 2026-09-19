#!/usr/bin/env bash
set -euo pipefail

# Krea-2 Turbo Q6_K + Qwen3-VL 4B Q4_K_M + Wan 2.1 VAE
# LoRAs: Fedor Filter Bypass + Krea2 Realism V2
# Intended for stable-diffusion.cpp sd-server on a 24GB NVIDIA GPU.
#
# This RunPod version intentionally follows the simple FLUX.2 startup script:
#   1. create directories
#   2. download files sequentially with resume support
#   3. show downloaded files
#   4. exec sd-server
#
# Optional environment variables:
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN : Hugging Face token
#   MODEL_DIR : model directory (default: /workspace/models)
#   LORA_DIR  : LoRA directory  (default: $MODEL_DIR/loras)
#   PORT      : server port     (default: 1234)
#   HOST      : listen IP       (default: 0.0.0.0)
#   SD_SERVER : sd-server path  (default: /sd/bin/sd-server)

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
LORA_DIR="${LORA_DIR:-$MODEL_DIR/loras}"
PORT="${PORT:-1234}"
HOST="${HOST:-0.0.0.0}"
SD_SERVER="${SD_SERVER:-/sd/bin/sd-server}"

mkdir -p "$MODEL_DIR" "$LORA_DIR"

command -v curl >/dev/null 2>&1 || {
    echo "[fatal] curl is required" >&2
    exit 1
}

if [ ! -x "$SD_SERVER" ]; then
    echo "[fatal] sd-server not found or not executable: $SD_SERVER" >&2
    exit 1
fi

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
CURL_AUTH_ARGS=()
if [ -n "$HF_TOKEN_VALUE" ]; then
    CURL_AUTH_ARGS=(-H "Authorization: Bearer $HF_TOKEN_VALUE")
fi

download() {
    local url="$1"
    local dir="$2"
    local file="$3"

    local dst="$dir/$file"
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

# Krea-2 Turbo diffusion model (GGUF Q6_K)
download \
"https://huggingface.co/realrebelai/KREA-2_GGUFs/resolve/main/TURBO/Krea-2-Turbo-Q6_K.gguf?download=true" \
"$MODEL_DIR" \
"Krea-2-Turbo-Q6_K.gguf"

# Qwen3-VL 4B text encoder (GGUF Q4_K_M)
download \
"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf?download=true" \
"$MODEL_DIR" \
"Qwen3VL-4B-Instruct-Q4_K_M.gguf"

# Wan 2.1 VAE
download \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true" \
"$MODEL_DIR" \
"wan_2.1_vae.safetensors"

# Fedor Krea2 Filter Bypass LoRA
download \
"https://huggingface.co/diobrando0/krea2_loras_public/resolve/main/fedor_bypass.safetensors?download=true" \
"$LORA_DIR" \
"fedor_bypass.safetensors"

# Krea2 Realism V2 LoRA
download \
"https://huggingface.co/RudySen/Krea2-realism-V2/resolve/main/Krea2-realism-V2.safetensors?download=true" \
"$LORA_DIR" \
"Krea2-realism-V2.safetensors"

echo
echo "=== models ==="
ls -lh "$MODEL_DIR"
echo

echo "=== loras ==="
ls -lh "$LORA_DIR"
echo

echo "=== starting sd-server ==="
echo "LoRAs are available in $LORA_DIR but are not force-applied."

exec "$SD_SERVER" \
  --diffusion-model "$MODEL_DIR/Krea-2-Turbo-Q6_K.gguf" \
  --vae "$MODEL_DIR/wan_2.1_vae.safetensors" \
  --llm "$MODEL_DIR/Qwen3VL-4B-Instruct-Q4_K_M.gguf" \
  --lora-model-dir "$LORA_DIR" \
  --backend diffusion=cuda0,te=cuda0,vae=cuda0 \
  --auto-fit off \
  --cfg-scale 1 \
  --steps 8 \
  --sampling-method euler \
  --diffusion-fa \
  --listen-ip "$HOST" \
  --listen-port "$PORT" \
  -v
