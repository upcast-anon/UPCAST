#!/usr/bin/env bash
set -euo pipefail

download_root="${ARKITSCENES_DOWNLOAD_ROOT:-./data/arkitscenes}"
video_list="${ARKITSCENES_VIDEO_LIST:-evaluation/arkitscenes_video_ids.txt}"
workers="${DOWNLOAD_WORKERS:-3}"
minimum_free_gb="${ARKITSCENES_MIN_FREE_GB:-15}"
worker_script="scripts/download_arkitscenes_video.sh"

if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'DOWNLOAD_WORKERS must be a positive integer\n' >&2
  exit 1
fi
if [ ! -x "${worker_script}" ]; then
  printf 'ARKitScenes resumable downloader is missing: %s\n' "${worker_script}" >&2
  exit 1
fi
if [ ! -f "${video_list}" ]; then
  printf 'ARKitScenes list is missing: %s\n' "${video_list}" >&2
  exit 1
fi

available_kb="$(df -Pk "$(dirname "${download_root}")" | awk 'NR==2 {print $4}')"
required_kb="$((minimum_free_gb * 1024 * 1024))"
if (( available_kb < required_kb )); then
  printf 'ARKitScenes download needs at least %s GB free; only %s GB is available.\n' \
    "${minimum_free_gb}" "$((available_kb / 1024 / 1024))" >&2
  exit 1
fi

mapfile -t video_ids < <(sed 's/#.*//' "${video_list}" | awk 'NF {print $1}')
if [ "${#video_ids[@]}" -eq 0 ]; then
  printf 'No video IDs found in %s\n' "${video_list}" >&2
  exit 1
fi

mkdir -p "${download_root}/logs"
printf 'Downloading/resuming %s ARKitScenes videos with %s workers\n' \
  "${#video_ids[@]}" "${workers}"
printf '%s\n' "${video_ids[@]}" | \
  ARKITSCENES_DOWNLOAD_ROOT="${download_root}" \
  xargs -n 1 -P "${workers}" bash "${worker_script}"

for video_id in "${video_ids[@]}"; do
  video_root="${download_root}/raw/Validation/${video_id}"
  for asset in lowres_wide lowres_depth confidence lowres_wide_intrinsics; do
    if [ ! -d "${video_root}/${asset}" ]; then
      printf 'Download is incomplete for %s: missing %s\n' "${video_id}" "${asset}" >&2
      exit 1
    fi
  done
  if [ ! -s "${video_root}/lowres_wide.traj" ]; then
    printf 'Download is incomplete for %s: missing trajectory\n' "${video_id}" >&2
    exit 1
  fi
done

df -h "${download_root}"
printf 'Downloaded %s ARKitScenes RGB-D videos under %s\n' "${#video_ids[@]}" "${download_root}"
