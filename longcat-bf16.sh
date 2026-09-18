#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/workspace/models
mkdir -p "$MODEL_DIR"

download() {
    local url="$1"
    local dst="$2"

    if [ -s "$dst" ]; then
        echo "[skip] $(basename "$dst")"
        return
    fi

    echo "[download] $(basename "$dst")"

    curl \
        -fL \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        -o "${dst}.part" \
        "$url"

    mv "${dst}.part" "$dst"
}

download \
  "https://huggingface.co/Comfy-Org/LongCat-Image/resolve/main/split_files/diffusion_models/longcat_image_edit_turbo_bf16.safetensors?download=true" \
  "$MODEL_DIR/longcat_image_edit_turbo_bf16.safetensors"

download \
  "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct.Q6_K.gguf?download=true" \
  "$MODEL_DIR/Qwen2.5-VL-7B-Instruct.Q6_K.gguf"

download \
  "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf?download=true" \
  "$MODEL_DIR/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf"

download \
  "https://huggingface.co/flux-safetensors/flux-safetensors/resolve/main/ae.safetensors?download=true" \
  "$MODEL_DIR/ae.safetensors"

exec /sd/bin/sd-server \
  --diffusion-model "$MODEL_DIR/longcat_image_edit_turbo_bf16.safetensors" \
  --vae "$MODEL_DIR/ae.safetensors" \
  --vae-format flux \
  --llm "$MODEL_DIR/Qwen2.5-VL-7B-Instruct.Q6_K.gguf" \
  --llm_vision "$MODEL_DIR/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf" \
  --backend diffusion=cuda0,te=cpu,vae=cpu \
  --auto-fit off \
  --cfg-scale 1 \
  --steps 8 \
  --sampling-method euler \
  --flow-shift 3 \
  --diffusion-fa \
  --listen-ip 0.0.0.0 \
  --listen-port 1234 \
  -v
