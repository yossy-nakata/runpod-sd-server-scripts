#!/usr/bin/env bash
set -euo pipefail
# FLUX.1-Krea-dev - Salad / stable-diffusion.cpp
#
# What this script is:
#   - start.sh meant to be fetched by bootstrap.sh via GITURL
#   - model + LoRA downloader
#   - fixed-LoRA list writer for the Python proxy (/tmp/fixed-loras.tsv)
#
# What is verified here:
#   - FLUX in stable-diffusion.cpp uses:
#       diffusion gguf + ae.safetensors + clip_l.safetensors + t5xxl_fp16.safetensors
#   - FLUX.1 Krea [dev] is intended as a drop-in replacement for FLUX.1 [dev]
#   - the LoRA URLs below are verified for 7 files
#
# What is NOT fully resolved here:
#   - the exact community GGUF URL for FLUX.1-Krea-dev itself
#   - 2 sensitive-content LoRA file names (NSFWMaster / Dynamic_Pose_Uncensored)
#
# So:
#   1) fill KREA_GGUF_URL
#   2) optionally fill the 2 gated LoRA file names / URLs
#   3) keep lora_loads empty at first, then decide after GUI testing

SD_SERVER="${SD_SERVER:-/sd/bin/sd-server}"
MODEL_ROOT="${MODEL_ROOT:-/sd/models}"

# IMPORTANT:
# bootstrap.sh launches this script expecting sd-server on 127.0.0.1:1235 by default.
SD_LISTEN_IP="${SD_LISTEN_IP:-127.0.0.1}"
SD_LISTEN_PORT="${SD_LISTEN_PORT:-1235}"

FIXED_LORAS_FILE="${FIXED_LORAS_FILE:-/tmp/fixed-loras.tsv}"

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
  local downloader="${4:-aria2c}"

  echo "[download] ${out}"

  "$downloader" \
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

download_hf() {
  local url="$1"
  local dir="$2"
  local out="$3"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    download "$url" "$dir" "$out" /usr/local/bin/aria2c_hf
  else
    download "$url" "$dir" "$out"
  fi
}

lora_download() {
  local filename="$1"
  local url="$2"
  local mode="${3:-public}"   # public | gated

  if [[ "$mode" == "gated" ]]; then
    download_hf "$url" "$LORA_DIR" "$filename"
  else
    download "$url" "$LORA_DIR" "$filename"
  fi
}

lora_loads() {
  if (( $# % 3 != 0 )); then
    echo "[error] lora_loads expects groups of: path multiplier is_high_noise" >&2
    return 2
  fi

  : > "$FIXED_LORAS_FILE"

  while (( $# > 0 )); do
    local path="$1"
    local multiplier="$2"
    local is_high_noise="$3"
    shift 3

    if [[ ! "$multiplier" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
      echo "[error] invalid LoRA multiplier for ${path}: ${multiplier}" >&2
      return 2
    fi

    if [[ "$is_high_noise" != "true" && "$is_high_noise" != "false" ]]; then
      echo "[error] is_high_noise for ${path} must be true or false" >&2
      return 2
    fi

    printf '%s\t%s\t%s\n' \
      "$path" \
      "$multiplier" \
      "$is_high_noise" \
      >> "$FIXED_LORAS_FILE"
  done
}

# -----------------------------------------------------------------------------
# Core model files
# -----------------------------------------------------------------------------

# REQUIRED: fill this with your chosen community-converted FLUX.1-Krea-dev GGUF.
# Example shape only:
# KREA_GGUF_URL="https://huggingface.co/<repo>/resolve/main/flux1-krea-dev-Q8_0.gguf?download=true"
KREA_GGUF_URL="${KREA_GGUF_URL:-}"
KREA_GGUF_FILE="${KREA_GGUF_FILE:-flux1-krea-dev-Q8_0.gguf}"

AE_URL="${AE_URL:-https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors?download=true}"
AE_FILE="${AE_FILE:-ae.safetensors}"

CLIP_L_URL="${CLIP_L_URL:-https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true}"
CLIP_L_FILE="${CLIP_L_FILE:-clip_l.safetensors}"

T5_URL="${T5_URL:-https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true}"
T5_FILE="${T5_FILE:-t5xxl_fp16.safetensors}"

if [[ -z "$KREA_GGUF_URL" ]]; then
  echo "[error] KREA_GGUF_URL is empty. Set the exact FLUX.1-Krea-dev GGUF URL first." >&2
  exit 1
fi

download_hf "$KREA_GGUF_URL" "$DIFFUSION_DIR" "$KREA_GGUF_FILE"
download_hf "$AE_URL" "$VAE_DIR" "$AE_FILE"
download "$CLIP_L_URL" "$TEXT_ENCODER_DIR" "$CLIP_L_FILE"
download "$T5_URL" "$TEXT_ENCODER_DIR" "$T5_FILE"

# -----------------------------------------------------------------------------
# LoRA downloads
#
# Download availability and fixed loading are deliberately separate.
# -----------------------------------------------------------------------------

# 1) Verified realism / photo / anatomy / speed LoRAs

lora_download \
  "lora.safetensors" \
  "https://huggingface.co/XLabs-AI/flux-RealismLora/resolve/main/lora.safetensors?download=true"

lora_download \
  "amateurphoto-v6-forcu.safetensors" \
  "https://huggingface.co/ujouy/Amateur_Photography_FluxDev/resolve/main/amateurphoto-v6-forcu.safetensors?download=true"

lora_download \
  "aidmaNSFWunlock-FLUX-V0.2.safetensors" \
  "https://huggingface.co/shahtab/FLUXNSFWunlock/resolve/main/aidmaNSFWunlock-FLUX-V0.2.safetensors?download=true"

lora_download \
  "flux-female-anatomy.safetensors" \
  "https://huggingface.co/uriel353/flux-female-anatomy/resolve/main/flux-female-anatomy.safetensors?download=true"

lora_download \
  "diffusion_pytorch_model.safetensors" \
  "https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors?download=true"

lora_download \
  "Hyper-FLUX.1-dev-8steps-lora.safetensors" \
  "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true"

lora_download \
  "FLUX.1-dev_tdd_adv_lora_weights.safetensors" \
  "https://huggingface.co/RED-AIGC/TDD/resolve/main/FLUX.1-dev_tdd_adv_lora_weights.safetensors?download=true"

# 2) Sensitive-content repo candidates
#
# The repo identities are confirmed, but the web-side file listing is hidden behind
# the sensitive-content gate, so the exact filename must be checked once in browser.
# After you confirm the filename, uncomment and fill these.

# lora_download \
#   "NSFWMaster.safetensors" \
#   "https://huggingface.co/Jonjew/NSFWMaster/resolve/main/NSFWMaster.safetensors?download=true" \
#   gated
#
# lora_download \
#   "Dynamic_Pose_Uncensored.safetensors" \
#   "https://huggingface.co/Keltezaa/Dynamic_Pose_Uncensored/resolve/main/Dynamic_Pose_Uncensored.safetensors?download=true" \
#   gated

# -----------------------------------------------------------------------------
# Fixed LoRA load list
#
# Each entry is:
#   path  multiplier  is_high_noise
#
# Leave this empty initially. Test from GUI first.
# -----------------------------------------------------------------------------

# Default: no fixed LoRAs
lora_loads

# Example normal-quality stack (not enabled):
#
# lora_loads \
#   "aidmaNSFWunlock-FLUX-V0.2.safetensors" "0.55" "false" \
#   "flux-female-anatomy.safetensors"      "0.60" "false" \
#   "amateurphoto-v6-forcu.safetensors"    "0.45" "false" \
#   "lora.safetensors"                     "0.25" "false"
#
# Example speed stack with exactly one 8-step LoRA (not enabled):
#
# lora_loads \
#   "diffusion_pytorch_model.safetensors"  "1.00" "false"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

for f in \
  "$DIFFUSION_DIR/$KREA_GGUF_FILE" \
  "$VAE_DIR/$AE_FILE" \
  "$TEXT_ENCODER_DIR/$CLIP_L_FILE" \
  "$TEXT_ENCODER_DIR/$T5_FILE"
do
  if [[ ! -s "$f" ]]; then
    echo "[error] missing or empty file: $f" >&2
    exit 1
  fi
done

while IFS=$'\t' read -r lora_path _multiplier _is_high_noise; do
  [[ -z "$lora_path" ]] && continue
  if [[ ! -s "$LORA_DIR/$lora_path" ]]; then
    echo "[error] fixed LoRA was not downloaded: $LORA_DIR/$lora_path" >&2
    exit 1
  fi
done < "$FIXED_LORAS_FILE"

if [[ ! -x "$SD_SERVER" ]]; then
  echo "[error] sd-server not executable: $SD_SERVER" >&2
  exit 1
fi

echo "[models] download complete"
du -sh "$MODEL_ROOT" || true
df -h "$MODEL_ROOT" || true

echo "[lora] fixed load list:"
if [[ -s "$FIXED_LORAS_FILE" ]]; then
  cat "$FIXED_LORAS_FILE"
else
  echo "[lora]   (empty)"
fi

# -----------------------------------------------------------------------------
# Start sd-server
# -----------------------------------------------------------------------------

echo "[server] starting sd-server on ${SD_LISTEN_IP}:${SD_LISTEN_PORT}"

exec "$SD_SERVER" \
  --listen-ip "$SD_LISTEN_IP" \
  --listen-port "$SD_LISTEN_PORT" \
  --diffusion-model "$DIFFUSION_DIR/$KREA_GGUF_FILE" \
  --vae "$VAE_DIR/$AE_FILE" \
  --clip_l "$TEXT_ENCODER_DIR/$CLIP_L_FILE" \
  --t5xxl "$TEXT_ENCODER_DIR/$T5_FILE" \
  --lora-model-dir "$LORA_DIR" \
  --lora-apply-mode at_runtime \
  --cfg-scale 1.0 \
  --steps 28 \
  --sampling-method euler \
  --diffusion-fa \
  --offload-to-cpu \
  --verbose
