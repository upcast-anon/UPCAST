#!/usr/bin/env bash
set -euo pipefail

download_dinov2="${DOWNLOAD_DINOV2:-true}"
download_vjepa="${DOWNLOAD_VJEPA:-true}"
vjepa_variant="${VJEPA_VARIANT:-vitl16}"
default_vjepa_dir="./metric_assets/jepa_checkpoints"
checkpoint_dir="${VJEPA_CHECKPOINT_DIR:-${default_vjepa_dir}}"
torch_cache_dir="${TORCH_CACHE_DIR:-${HOME}/.cache/torch/hub/checkpoints}"

python_bin="${PYTHON:-python3}"
download_timeout="${DOWNLOAD_TIMEOUT:-600}"
download_connect_timeout="${DOWNLOAD_CONNECT_TIMEOUT:-30}"
download_retries="${DOWNLOAD_RETRIES:-5}"
aria2_timeout="${download_timeout}"
if (( aria2_timeout > 600 )); then
  aria2_timeout=600
fi

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --file-allocation=none \
      --connect-timeout="${download_connect_timeout}" \
      --timeout="${aria2_timeout}" \
      --max-tries="${download_retries}" \
      -x 8 -s 8 \
      -o "$(basename "${out}")" \
      -d "$(dirname "${out}")" \
      "${url}"
  else
    curl -L --fail --connect-timeout "${download_connect_timeout}" --max-time "${download_timeout}" --retry "${download_retries}" "${url}" -o "${out}"
  fi
}

mkdir -p "${torch_cache_dir}"
if [[ "${download_dinov2}" == true ]]; then
  "${python_bin}" - <<'PY'
import torch

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
model.eval()
print("DINOv2 ViT-S/14 is cached.")
PY
fi

if [[ "${download_vjepa}" == "true" ]]; then
  mkdir -p "${checkpoint_dir}"
  case "${vjepa_variant}" in
    vitl16)
      url="https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar"
      out="${checkpoint_dir}/vitl16.pth.tar"
      ;;
    vith16)
      url="https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar"
      out="${checkpoint_dir}/vith16.pth.tar"
      ;;
    vith16-384|vith16_384)
      url="https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar"
      out="${checkpoint_dir}/vith16-384.pth.tar"
      ;;
    *)
      echo "Unknown VJEPA_VARIANT=${vjepa_variant}. Use vitl16, vith16, or vith16-384." >&2
      exit 1
      ;;
  esac
  if [[ ! -s "${out}" || -f "${out}.aria2" ]]; then
    echo "Downloading ${url} -> ${out}"
    download_file "${url}" "${out}"
  else
    echo "V-JEPA checkpoint already exists: ${out}"
  fi
fi
