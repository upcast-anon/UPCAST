#!/usr/bin/env bash
set -euo pipefail

# Matched zero-shot ARKitScenes generation and sensor-referenced evaluation.

python_bin="${PYTHON:-python}"
download_root="${ARKITSCENES_DOWNLOAD_ROOT:-./data/arkitscenes}"
prepared_root="${ARKITSCENES_PREPARED_ROOT:-./data/arkitscenes-prepared}"
result_root="${ARKITSCENES_RESULT_ROOT:-./output/evaluations/arkitscenes}"
cache_root="${ARKITSCENES_CACHE_ROOT:-./metric_cache/arkitscenes}"
video_list="${ARKITSCENES_VIDEO_LIST:-evaluation/arkitscenes_video_ids.txt}"
dataset_name="${ARKITSCENES_DATASET_NAME:-ARKitScenes v1 raw RGB-D}"
summary_prefix="${ARKITSCENES_SUMMARY_PREFIX:-arkitscenes_results}"
methods_csv="${METHODS:-GF,Ours}"
count="${NUM_EVAL_VIDEOS:-12}"
frames="${N_FRAMES:-64}"
wandb_mode="${WANDB_MODE:-offline}"
eval_seed="${EVAL_SEED:-20260814}"
generation_chunk="${ARKITSCENES_GENERATION_CHUNK:-6}"
run_download="${RUN_DOWNLOAD:-true}"
run_prepare="${RUN_PREPARE:-true}"
run_generation="${RUN_GENERATION:-true}"
run_rgb="${RUN_RGB_METRICS:-true}"
run_depth="${RUN_DEPTH_METRICS:-true}"
run_sensor_pose="${RUN_SENSOR_POSE:-true}"

dfot_ckpt="${DFOT_CKPT:-checkpoints/DFoT_16f_state_dict.ckpt}"
gf_ckpt="${GF_CKPT:-checkpoints/geometry_forcing_state_dict.ckpt}"
ours_ckpt="${OURS_CKPT:-checkpoints/upcast.ckpt}"

for value_name in run_download run_prepare run_generation run_rgb run_depth run_sensor_pose; do
  case "${!value_name}" in
    true|false) ;;
    *) printf '%s must be true or false\n' "${value_name}" >&2; exit 1 ;;
  esac
done
if ! [[ "${count}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${frames}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${generation_chunk}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_EVAL_VIDEOS, N_FRAMES, and ARKITSCENES_GENERATION_CHUNK must be positive integers\n' >&2
  exit 1
fi

mkdir -p "${prepared_root}" "${result_root}" "${cache_root}"
export HF_HOME="${HF_HOME:-./cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TORCH_HOME="${TORCH_HOME:-./metric_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-./metric_cache/xdg}"
export TMPDIR="${TMPDIR:-./tmp/arkitscenes}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

if [ "${run_download}" = true ]; then
  PYTHON="${python_bin}" \
  ARKITSCENES_DOWNLOAD_ROOT="${download_root}" \
  ARKITSCENES_VIDEO_LIST="${video_list}" \
    bash scripts/download_arkitscenes.sh
fi

if [ "${run_prepare}" = true ] && [ ! -s "${prepared_root}/manifest.json" ]; then
  "${python_bin}" evaluation/prepare_arkitscenes.py \
    --raw-root "${download_root}/raw" \
    --output-root "${prepared_root}" \
    --video-list "${video_list}" \
    --max-videos "${count}" \
    --clip-length "${frames}" \
    --frame-stride "${ARKITSCENES_FRAME_STRIDE:-1}" \
    --resolution 256 \
    --confidence-threshold "${ARKITSCENES_CONFIDENCE_THRESHOLD:-1}"
fi

if [ ! -s "${prepared_root}/manifest.json" ]; then
  printf 'Prepared ARKitScenes manifest is missing: %s\n' "${prepared_root}/manifest.json" >&2
  exit 1
fi

prepared_count="$("${python_bin}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["clips"]))' "${prepared_root}/manifest.json")"
if (( prepared_count < count )); then
  printf 'Prepared manifest has only %s/%s clips\n' "${prepared_count}" "${count}" >&2
  exit 1
fi

declare -A checkpoints=(
  [DFoT]="${dfot_ckpt}"
  [GF]="${gf_ckpt}"
  [Ours]="${ours_ckpt}"
)
declare -A strict=(
  [DFoT]=true
  [GF]=true
  [Ours]=true
)
IFS=',' read -r -a methods <<<"${methods_csv}"
for method in "${methods[@]}"; do
  if [ -z "${checkpoints[$method]+set}" ]; then
    printf 'Unsupported method: %s\n' "${method}" >&2
    exit 1
  fi
  if [ ! -s "${checkpoints[$method]}" ]; then
    printf 'Missing checkpoint for %s: %s\n' "${method}" "${checkpoints[$method]}" >&2
    exit 1
  fi
  checkpoints[$method]="$(realpath "${checkpoints[$method]}")"
done

mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
if [ "${#gpu_ids[@]}" -eq 0 ]; then
  printf 'At least one CUDA GPU is required\n' >&2
  exit 1
fi
max_parallel="${MAX_PARALLEL_EVALS:-${#gpu_ids[@]}}"
if (( max_parallel < 1 || max_parallel > ${#gpu_ids[@]} )); then
  printf 'MAX_PARALLEL_EVALS must be in [1,%s]\n' "${#gpu_ids[@]}" >&2
  exit 1
fi

find_gif_dir() {
  find "$1" -type d -path '*/files/media/videos/prediction_vis' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

chunk_complete() {
  local root="$1" expected="$2" gif_dir raw_count gif_count
  raw_count="$(find "${root}/raw" -mindepth 2 -maxdepth 2 -name data.npz -type f 2>/dev/null | wc -l)"
  gif_dir="$(find_gif_dir "${root}")"
  gif_count=0
  if [ -n "${gif_dir}" ]; then
    gif_count="$(find "${gif_dir}" -maxdepth 1 -name '*.gif' -type f | wc -l)"
  fi
  [ "${raw_count}" -ge "${expected}" ] && [ "${gif_count}" -ge "${expected}" ]
}

generate_chunk() {
  local gpu="$1" method="$2" offset="$3" chunk_count="$4"
  local root="${result_root}/generation/${method}/chunk_${offset}_${chunk_count}"
  mkdir -p "${root}"
  if chunk_complete "${root}" "${chunk_count}"; then
    printf '[reuse] ARKitScenes %s offset=%s count=%s\n' \
      "${method}" "${offset}" "${chunk_count}"
  else
    printf '[gpu %s] ARKitScenes %s offset=%s count=%s\n' \
      "${gpu}" "${method}" "${offset}" "${chunk_count}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON="${python_bin}" \
    EVAL_CKPT="${checkpoints[$method]}" \
    RESULT_DIR="${root}" \
    RAW_DIR="${root}/raw" \
    SAVE_RAW=true \
    EVAL_RUN_NAME="ARKitScenes_${method}_64f_${offset}" \
    EVAL_DATASET=arkitscenes_eval \
    DATASET_SAVE_DIR="${prepared_root}" \
    NUM_EVAL_VIDEOS="${chunk_count}" \
    EVAL_VIDEO_OFFSET="${offset}" \
    N_FRAMES="${frames}" \
    CONTEXT_LENGTH=1 \
    FRAME_SKIP=1 \
    EVAL_BATCH_SIZE=1 \
    EVAL_NUM_WORKERS=0 \
    EVAL_METRICS='[]' \
    EVAL_GENERATIVE_UNIT_ENABLE=false \
    CHECKPOINT_STRICT="${strict[$method]}" \
    EVAL_SEED="$((eval_seed + offset))" \
    WANDB_MODE="${wandb_mode}" \
      bash scripts/eval_upcast.sh >"${root}/eval.log" 2>&1; then
      tail -80 "${root}/eval.log" >&2
      return 1
    fi
  fi
  if ! chunk_complete "${root}" "${chunk_count}"; then
    printf 'Incomplete generation output: %s\n' "${root}" >&2
    tail -80 "${root}/eval.log" >&2
    return 1
  fi
}

if [ "${run_generation}" = true ]; then
  jobs=()
  for ((offset=0; offset<count; offset+=generation_chunk)); do
    chunk_count=$((count - offset))
    (( chunk_count > generation_chunk )) && chunk_count="${generation_chunk}"
    for method in "${methods[@]}"; do
      jobs+=("${method}|${offset}|${chunk_count}")
    done
  done
  cursor=0
  while (( cursor < ${#jobs[@]} )); do
    pids=()
    labels=()
    for ((slot=0; slot<max_parallel && cursor<${#jobs[@]}; slot++,cursor++)); do
      IFS='|' read -r method offset chunk_count <<<"${jobs[$cursor]}"
      generate_chunk "${gpu_ids[$slot]}" "${method}" "${offset}" "${chunk_count}" &
      pids+=("$!")
      labels+=("${method}/${offset}")
    done
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        printf 'ARKitScenes generation failed for %s\n' "${labels[$index]}" >&2
        exit 1
      fi
    done
  done
fi

for method in "${methods[@]}"; do
  collect_args=()
  for ((offset=0; offset<count; offset+=generation_chunk)); do
    chunk_count=$((count - offset))
    (( chunk_count > generation_chunk )) && chunk_count="${generation_chunk}"
    collect_args+=(
      --chunk
      "${offset}=${result_root}/generation/${method}/chunk_${offset}_${chunk_count}"
    )
  done
  "${python_bin}" evaluation/collect_eval_outputs.py \
    "${collect_args[@]}" \
    --output_dir "${result_root}/combined/${method}" \
    --expected_count "${count}"
done

run_args=()
for method in "${methods[@]}"; do
  raw_dir="${result_root}/combined/${method}/raw"
  if [ ! -d "${raw_dir}" ]; then
    printf 'Missing collected outputs for %s: %s\n' "${method}" "${raw_dir}" >&2
    exit 1
  fi
  run_args+=(--run "${method}=${raw_dir}")
done
indices="0-$((count - 1))"

if [ "${run_rgb}" = true ]; then
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${python_bin}" \
    evaluation/metrics/compute_gif_video_metrics.py \
    "${run_args[@]}" \
    --output_dir "${result_root}/metrics/rgb" \
    --indices "${indices}" \
    --input_format npz \
    --max_frames "${frames}" \
    --context_frames 1 \
    --no-compute_distribution_metrics
fi

if [ "${run_depth}" = true ]; then
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${python_bin}" \
    evaluation/eval_sensor_depth.py \
    --prepared-root "${prepared_root}" \
    --dataset-name "${dataset_name}" \
    "${run_args[@]}" \
    --output-dir "${result_root}/geometry/metric_depth" \
    --indices "${indices}" \
    --context-frames 1 \
    --cache-dir "${cache_root}/depth" \
    --device cuda \
    --amp
fi

if [ "${run_sensor_pose}" = true ]; then
  "${python_bin}" evaluation/eval_arkitscenes_sensor_pose.py \
    --prepared-root "${prepared_root}" \
    --dataset-name "${dataset_name}" \
    "${run_args[@]}" \
    --output-dir "${result_root}/geometry/sensor_pose" \
    --indices "${indices}" \
    --max-frames "${frames}" \
    --context-frames 1
fi

if [ "${run_rgb}" = true ] && [ "${run_depth}" = true ] \
  && [ "${run_sensor_pose}" = true ]; then
  "${python_bin}" evaluation/summarize_sensor_metrics.py \
    --root "${result_root}" \
    --methods "${methods_csv}" \
    --dataset-name "ARKitScenes" \
    --output-prefix "${summary_prefix}"
fi

printf 'ARKitScenes complete: %s\n' "${result_root}"
