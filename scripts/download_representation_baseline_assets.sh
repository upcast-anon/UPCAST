#!/usr/bin/env bash
set -euo pipefail

asset_root="${REPRESENTATION_ASSET_ROOT:-.}"
cache_root="${asset_root}/model_cache"
dino_repo="${DINO_SOURCE_DIR:-${asset_root}/metric_cache/torch/hub/facebookresearch_dinov2_main}"
videorepa_root="${VIDEOREPA_SOURCE_DIR:-${asset_root}/references/VideoREPA}"
videomae_source="${videorepa_root}/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py"
dino_weight="${cache_root}/dinov2/dinov2_vitb14_pretrain.pth"
videomae_weight="${cache_root}/VideoMAEv2/vit_b_k710_dl_from_giant.pth"

download() {
  local url="$1" output="$2"
  mkdir -p "$(dirname "${output}")"
  aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
    --retry-wait=3 --max-tries=10 \
    -d "$(dirname "${output}")" -o "$(basename "${output}")" "${url}"
}

if [ ! -s "${dino_repo}/hubconf.py" ]; then
  mkdir -p "$(dirname "${dino_repo}")"
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git "${dino_repo}"
fi

if [ ! -s "${videomae_source}" ]; then
  download \
    "https://raw.githubusercontent.com/oooolga/VideoREPA/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py" \
    "${videomae_source}"
fi

download \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" \
  "${dino_weight}"
download \
  "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth?download=true" \
  "${videomae_weight}"

expected_source_sha="2a162b1a8f99ebbd7ff4a7bc116ff30c992cc57015a02a4387c0843b1de51029"
actual_source_sha=$(sha256sum "${videomae_source}" | cut -d' ' -f1)
if [ "${actual_source_sha}" != "${expected_source_sha}" ]; then
  printf 'Unexpected VideoMAEv2 source checksum: %s\n' "${actual_source_sha}" >&2
  exit 1
fi
expected_dino_sha="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
expected_videomae_sha="8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec"
actual_dino_sha=$(sha256sum "${dino_weight}" | cut -d' ' -f1)
actual_videomae_sha=$(sha256sum "${videomae_weight}" | cut -d' ' -f1)
if [ "${actual_dino_sha}" != "${expected_dino_sha}" ]; then
  printf 'Unexpected DINOv2 checkpoint checksum: %s\n' "${actual_dino_sha}" >&2
  exit 1
fi
if [ "${actual_videomae_sha}" != "${expected_videomae_sha}" ]; then
  printf 'Unexpected VideoMAEv2 checkpoint checksum: %s\n' \
    "${actual_videomae_sha}" >&2
  exit 1
fi

"${PYTHON:-python}" - \
  "${dino_weight}" "${videomae_weight}" <<'PY'
from pathlib import Path
import sys
import torch

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file() or path.stat().st_size < 100 * 1024 * 1024:
        raise SystemExit(f"Incomplete representation checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected checkpoint payload: {path}")
    print(f"verified {path} ({path.stat().st_size / 2**20:.1f} MiB)")
PY

sha256sum "${dino_weight}" "${videomae_weight}" "${videomae_source}"
