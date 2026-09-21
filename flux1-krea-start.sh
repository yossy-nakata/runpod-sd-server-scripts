#!/usr/bin/env bash
set -euo pipefail

# FLUX.1-Krea-dev / Salad / stable-diffusion.cpp
#
# Salad-side user inputs remain exactly:
#   GITURL
#   HF_TOKEN
#
# Everything else is fixed here so a new Container Group does not require
# additional manual configuration.

SD_SERVER="/sd/bin/sd-server"
MODEL_ROOT="/sd/models"

DIFFUSION_DIR="${MODEL_ROOT}/diffusion_models"
TEXT_ENCODER_DIR="${MODEL_ROOT}/text_encoders"
VAE_DIR="${MODEL_ROOT}/vae"
LORA_DIR="${MODEL_ROOT}/loras"

FIXED_LORAS_FILE="/tmp/fixed-loras.tsv"

SD_LISTEN_IP="127.0.0.1"
SD_LISTEN_PORT="1235"

ARIA_CONNECTIONS="8"
ARIA_SPLIT="8"
ARIA_PIECE_SIZE="4M"

mkdir -p \
  "$DIFFUSION_DIR" \
  "$TEXT_ENCODER_DIR" \
  "$VAE_DIR" \
  "$LORA_DIR"

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

download_hf() {
  local url="$1"
  local dir="$2"
  local out="$3"

  # aria2c_hf itself requires HF_TOKEN and adds the Authorization header.
  /usr/local/bin/aria2c_hf \
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

lora_download() {
  local filename="$1"
  local url="$2"

  download "$url" "$LORA_DIR" "$filename"
}

lora_loads() {
  if (( $# % 3 != 0 )); then
    echo "[error] lora_loads expects: path multiplier is_high_noise ..." >&2
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

# ---------------------------------------------------------------------------
# Core FLUX.1-Krea-dev files
# ---------------------------------------------------------------------------

KREA_FILE="flux1-krea-dev-Q6_K.gguf"
AE_FILE="ae.safetensors"
CLIP_L_FILE="clip_l.safetensors"
T5_FILE="t5xxl_fp16.safetensors"

# Public community GGUF conversion of FLUX.1-Krea-dev.
download \
  "https://huggingface.co/QuantStack/FLUX.1-Krea-dev-GGUF/resolve/main/flux1-krea-dev-Q6_K.gguf?download=true" \
  "$DIFFUSION_DIR" \
  "$KREA_FILE"

# Official FLUX.1-dev VAE is gated; this is the only core download here that
# deliberately uses HF_TOKEN through aria2c_hf.
download_hf \
  "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors?download=true" \
  "$VAE_DIR" \
  "$AE_FILE"

# Public text encoders.
download \
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" \
  "$TEXT_ENCODER_DIR" \
  "$CLIP_L_FILE"

download \
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true" \
  "$TEXT_ENCODER_DIR" \
  "$T5_FILE"

# ---------------------------------------------------------------------------
# LoRA candidates
#
# All are downloaded so they are visible in the sd-server GUI.
# The GUI uses /sdcpp/v1/* and the Python proxy does not alter those requests.
# ---------------------------------------------------------------------------

# NSFW / anatomy / pose
lora_download \
  "aidmaNSFWunlock-FLUX-V0.2.safetensors" \
  "https://huggingface.co/shahtab/FLUXNSFWunlock/resolve/main/aidmaNSFWunlock-FLUX-V0.2.safetensors?download=true"

lora_download \
  "NSFW_master.safetensors" \
  "https://huggingface.co/Jonjew/NSFWMaster/resolve/main/NSFW_master.safetensors?download=true"

lora_download \
  "Dynamic_Poses-nsfw09.safetensors" \
  "https://huggingface.co/Keltezaa/Dynamic_Pose_Uncensored/resolve/main/Dynamic_Poses-nsfw09.safetensors?download=true"

lora_download \
  "flux-female-anatomy.safetensors" \
  "https://huggingface.co/uriel353/flux-female-anatomy/resolve/main/flux-female-anatomy.safetensors?download=true"

# Photography / realism
lora_download \
  "amateurphoto-v6-forcu.safetensors" \
  "https://huggingface.co/ujouy/Amateur_Photography_FluxDev/resolve/main/amateurphoto-v6-forcu.safetensors?download=true"

# Use the Comfy-converted XLabs file recommended by stable-diffusion.cpp docs
# for FLUX LoRA compatibility.
lora_download \
  "realism_lora_comfy_converted.safetensors" \
  "https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/realism_lora_comfy_converted.safetensors?download=true"

# Speed experiments: downloaded for GUI A/B testing, not fixed-loaded.
lora_download \
  "FLUX.1-Turbo-Alpha.safetensors" \
  "https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors?download=true"

lora_download \
  "Hyper-FLUX.1-dev-8steps-lora.safetensors" \
  "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true"

lora_download \
  "FLUX.1-dev_tdd_adv_lora_weights.safetensors" \
  "https://huggingface.co/RED-AIGC/TDD/resolve/main/FLUX.1-dev_tdd_adv_lora_weights.safetensors?download=true"

# ---------------------------------------------------------------------------
# Fixed LoRAs for OpenAI-compatible /v1/images/generations only.
#
# The proxy injects these into OpenAI-compatible requests.
# sd-server GUI requests (/sdcpp/v1/*) remain untouched and can freely test
# any downloaded LoRA.
#
# Speed LoRAs are intentionally excluded because they need their own step /
# guidance settings and must not be stacked blindly.
# ---------------------------------------------------------------------------

lora_loads \
  "aidmaNSFWunlock-FLUX-V0.2.safetensors" "0.55" "false" \
  "NSFW_master.safetensors"               "0.40" "false" \
  "Dynamic_Poses-nsfw09.safetensors"      "0.60" "false" \
  "flux-female-anatomy.safetensors"       "0.60" "false" \
  "amateurphoto-v6-forcu.safetensors"     "0.45" "false" \
  "realism_lora_comfy_converted.safetensors" "0.25" "false"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

for f in \
  "$DIFFUSION_DIR/$KREA_FILE" \
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

echo "[lora] fixed OpenAI load list:"
cat "$FIXED_LORAS_FILE"

# ---------------------------------------------------------------------------
# Start sd-server
#
# Text encoders stay on CPU. The Q6_K diffusion model remains on GPU; unlike
# --offload-to-cpu this avoids requiring the 24 GB system RAM to hold all
# diffusion + text-encoder weights at once.
# ---------------------------------------------------------------------------

echo "[server] starting sd-server on ${SD_LISTEN_IP}:${SD_LISTEN_PORT}"

exec "$SD_SERVER" \
  --listen-ip "$SD_LISTEN_IP" \
  --listen-port "$SD_LISTEN_PORT" \
  --diffusion-model "$DIFFUSION_DIR/$KREA_FILE" \
  --vae "$VAE_DIR/$AE_FILE" \
  --clip_l "$TEXT_ENCODER_DIR/$CLIP_L_FILE" \
  --t5xxl "$TEXT_ENCODER_DIR/$T5_FILE" \
  --lora-model-dir "$LORA_DIR" \
  --lora-apply-mode at_runtime \
  --cfg-scale 1.0 \
  --steps 28 \
  --sampling-method euler \
  --clip-on-cpu \
  --diffusion-fa \
  --verbose
