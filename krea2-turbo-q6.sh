#!/usr/bin/env bash
set -euo pipefail

# Krea2 Turbo Q6 - Salad / stable-diffusion.cpp
#
# Core:
#   - Krea-2-Turbo Q6_K
#   - Qwen3-VL 4B Q4_K_M
#   - Qwen3-VL 4B mmproj Q8_0
#   - Wan 2.1 VAE
#
# LoRA candidates are downloaded but NOT enabled globally.
# Select LoRAs per API request through sd-server's structured LoRA field.
#
# Hugging Face files used here are public, so normal aria2c is used.
# /usr/local/bin/aria2c_hf remains available for gated/private HF files.

SD_SERVER="${SD_SERVER:-/sd/bin/sd-server}"
MODEL_ROOT="${MODEL_ROOT:-/sd/models}"

DIFFUSION_DIR="${MODEL_ROOT}/diffusion_models"
TEXT_ENCODER_DIR="${MODEL_ROOT}/text_encoders"
VAE_DIR="${MODEL_ROOT}/vae"
LORA_DIR="${MODEL_ROOT}/loras"

mkdir -p \
  "$DIFFUSION_DIR" \
  "$TEXT_ENCODER_DIR" \
  "$VAE_DIR" \
  "$LORA_DIR"

ARIA_CONNECTIONS="${ARIA_CONNECTIONS:-8}"
ARIA_SPLIT="${ARIA_SPLIT:-8}"
ARIA_PIECE_SIZE="${ARIA_PIECE_SIZE:-4M}"

download() {
  local url="$1"
  local dir="$2"
  local out="$3"

  echo "[download] ${out}"

  aria2c \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --max-connection-per-server="$ARIA_CONNECTIONS" \
    --split="$ARIA_SPLIT" \
    --min-split-size="$ARIA_PIECE_SIZE" \
    --file-allocation=none \
    --connect-timeout=15 \
    --timeout=30 \
    --retry-wait=2 \
    --max-tries=0 \
    --summary-interval=10 \
    --dir="$dir" \
    --out="$out" \
    "$url"
}

# ---------------------------------------------------------------------------
# Core model files
# ---------------------------------------------------------------------------

KREA2_FILE="Krea-2-Turbo-Q6_K.gguf"
QWEN_FILE="Qwen3VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
VAE_FILE="wan_2.1_vae.safetensors"

download \
  "https://huggingface.co/realrebelai/KREA-2_GGUFs/resolve/main/TURBO/Krea-2-Turbo-Q6_K.gguf?download=true" \
  "$DIFFUSION_DIR" \
  "$KREA2_FILE"

download \
  "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf?download=true" \
  "$TEXT_ENCODER_DIR" \
  "$QWEN_FILE"

download \
  "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf?download=true" \
  "$TEXT_ENCODER_DIR" \
  "$MMPROJ_FILE"

download \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true" \
  "$VAE_DIR" \
  "$VAE_FILE"

# ---------------------------------------------------------------------------
# LoRA candidates
#
# These are only placed in --lora-model-dir.
# Nothing is permanently applied here.
# ---------------------------------------------------------------------------

download \
  "https://huggingface.co/Quiho/Krea2_TextFusion_Refusal-Reduction_LoRA_v1.0_lora/resolve/main/Krea2_TextFusion_Refusal_Reduction.safetensors?download=true" \
  "$LORA_DIR" \
  "Krea2_TextFusion_Refusal_Reduction.safetensors"

download \
  "https://huggingface.co/diobrando0/krea2_loras_public/resolve/main/Krea2_HMNSFW_AIO.safetensors?download=true" \
  "$LORA_DIR" \
  "Krea2_HMNSFW_AIO.safetensors"

download \
  "https://huggingface.co/RudySen/Krea2-realism-V2/resolve/main/Krea2-realism-V2.safetensors?download=true" \
  "$LORA_DIR" \
  "Krea2-realism-V2.safetensors"

download \
  "https://huggingface.co/diobrando0/krea2_loras_public/resolve/main/fedor_bypass.safetensors?download=true" \
  "$LORA_DIR" \
  "fedor_bypass.safetensors"

# Ostris/Comfy Krea2 reference-conditioning LoRA.
# Downloaded as a candidate; not globally enabled.
download \
  "https://huggingface.co/ostris/krea2_turbo_style_reference/resolve/main/krea2_style_reference.safetensors?download=true" \
  "$LORA_DIR" \
  "krea2_style_reference.safetensors"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

for f in \
  "$DIFFUSION_DIR/$KREA2_FILE" \
  "$TEXT_ENCODER_DIR/$QWEN_FILE" \
  "$TEXT_ENCODER_DIR/$MMPROJ_FILE" \
  "$VAE_DIR/$VAE_FILE"
do
  if [[ ! -s "$f" ]]; then
    echo "[error] missing or empty file: $f" >&2
    exit 1
  fi
done

if [[ ! -x "$SD_SERVER" ]]; then
  echo "[error] sd-server not executable: $SD_SERVER" >&2
  exit 1
fi

echo "[models] download complete"
du -sh "$MODEL_ROOT" || true
df -h "$MODEL_ROOT" || true

# ---------------------------------------------------------------------------
# Start sd-server
#
# - Native IPv6 bind for Salad Container Gateway.
# - Q6 Krea2 + Q4 text encoder + Q8 vision projector.
# - LoRAs remain selectable per request.
# - krea2_ostris_edit prepares the reference-image path used by compatible
#   Krea2 community reference/edit LoRAs.
# ---------------------------------------------------------------------------

echo "[server] starting sd-server on [::]:1234"

exec "$SD_SERVER" \
  --listen-ip "::" \
  --listen-port 1234 \
  --diffusion-model "$DIFFUSION_DIR/$KREA2_FILE" \
  --llm "$TEXT_ENCODER_DIR/$QWEN_FILE" \
  --llm_vision "$TEXT_ENCODER_DIR/$MMPROJ_FILE" \
  --vae "$VAE_DIR/$VAE_FILE" \
  --lora-model-dir "$LORA_DIR" \
  --lora-apply-mode at_runtime \
  --ref-image-args "preset=krea2_ostris_edit" \
  --cfg-scale 1.0 \
  --steps 8 \
  --sampling-method euler \
  --diffusion-fa \
  --offload-to-cpu \
  --verbose
