#!/usr/bin/env bash
set -euo pipefail

video_id="${1:?usage: download_arkitscenes_video.sh VIDEO_ID}"
download_root="${ARKITSCENES_DOWNLOAD_ROOT:-./data/arkitscenes}"
split="${ARKITSCENES_SPLIT:-Validation}"
base_url="${ARKITSCENES_BASE_URL:-https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1}"
retries="${DOWNLOAD_RETRIES:-8}"
connections="${DOWNLOAD_CONNECTIONS_PER_FILE:-16}"
video_root="${download_root}/raw/${split}/${video_id}"
log_root="${download_root}/logs"

mkdir -p "${video_root}" "${log_root}"
exec >>"${log_root}/${video_id}.log" 2>&1
printf '[%s] starting/resuming %s\n' "$(date --iso-8601=seconds)" "${video_id}"

download_file() {
  local url="$1" destination="$2" temporary="${2}.tmp"
  if [ -s "${destination}" ]; then
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server="${connections}" \
      --split="${connections}" \
      --min-split-size=1M \
      --file-allocation=none \
      --max-tries="${retries}" \
      --retry-wait=2 \
      --timeout=30 \
      --summary-interval=15 \
      --dir="$(dirname "${temporary}")" \
      --out="$(basename "${temporary}")" \
      "${url}"
  else
    curl_args=(
      --fail
      --location
      --retry "${retries}"
      --retry-delay 2
      --retry-all-errors
      --connect-timeout 30
      --speed-limit 1024
      --speed-time 120
    )
    if [ -s "${temporary}" ]; then
      if ! curl "${curl_args[@]}" --continue-at - --output "${temporary}" "${url}"; then
        rm -f "${temporary}"
        curl "${curl_args[@]}" --output "${temporary}" "${url}"
      fi
    else
      curl "${curl_args[@]}" --output "${temporary}" "${url}"
    fi
  fi
  mv "${temporary}" "${destination}"
}

for asset in lowres_wide lowres_depth confidence lowres_wide_intrinsics; do
  asset_dir="${video_root}/${asset}"
  archive="${video_root}/${asset}.zip"
  if [ -d "${asset_dir}" ] && find "${asset_dir}" -type f -print -quit | grep -q .; then
    rm -f "${archive}" "${archive}.tmp"
    continue
  fi
  download_file \
    "${base_url}/raw/${split}/${video_id}/${asset}.zip" \
    "${archive}"
  unzip -oq "${archive}" -d "${video_root}"
  if [ ! -d "${asset_dir}" ] || ! find "${asset_dir}" -type f -print -quit | grep -q .; then
    printf 'Archive did not produce %s\n' "${asset_dir}" >&2
    exit 1
  fi
  rm -f "${archive}"
done

trajectory="${video_root}/lowres_wide.traj"
download_file \
  "${base_url}/raw/${split}/${video_id}/lowres_wide.traj" \
  "${trajectory}"
printf '[%s] complete %s\n' "$(date --iso-8601=seconds)" "${video_id}"
