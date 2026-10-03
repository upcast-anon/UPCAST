#!/usr/bin/env bash
set -euo pipefail

# Matched training-stage analysis for the final model:
# DFoT -> shared physical/private appearance space -> conditional distillation
# -> rollout refinement.  DFoT and Full reuse the lossless main-suite samples;
# only the two intermediate checkpoints are generated here.

python_bin="${PYTHON:-python}"
jedi_python="${JEDI_PYTHON:-${PYTHON:-python}}"
output_root="${OUTPUT_DIR:-./output}"
main_root="${MAIN_RESULT_DIR:-${output_root}/evaluations/upcast_main}"
result_root="${ABLATION_RESULT_DIR:-${output_root}/evaluations/training_stages}"
metric_cache="${METRIC_CACHE_DIR:-./metric_cache}"
tmp_root="${TMPDIR_ROOT:-./tmp/training_stages}"
seed="${EVAL_SEED:-20260814}"
wandb_mode="${WANDB_MODE:-offline}"
long_count="${ABLATION_LONG_COUNT:-25}"
max_parallel="${MAX_PARALLEL_EVALS:-2}"

shared_checkpoint="${SHARED_SPACE_CKPT:-checkpoints/upcast_acquired.ckpt}"
distill_checkpoint="${DISTILL_CKPT:-checkpoints/upcast_consolidated.ckpt}"

for checkpoint in "${shared_checkpoint}" "${distill_checkpoint}"; do
  if [ ! -s "${checkpoint}" ] || [ -f "${checkpoint}.aria2" ]; then
    printf 'Missing or incomplete ablation checkpoint: %s\n' "${checkpoint}" >&2
    exit 1
  fi
done

for path in \
  "${main_root}/combined/long100/DFoT/raw" \
  "${main_root}/combined/long100/Ours/raw"; do
  if [ ! -d "${path}" ]; then
    printf 'Main-suite lossless samples are required: %s\n' "${path}" >&2
    exit 1
  fi
done

mkdir -p "${result_root}" "${tmp_root}"
export WANDB_CACHE_DIR="${metric_cache}/wandb"
export WANDB_DATA_DIR="${metric_cache}/wandb-data"
export WANDB_ARTIFACT_DIR="${metric_cache}/wandb-artifacts"
export TORCH_HOME="${metric_cache}/torch"
export XDG_CACHE_HOME="${metric_cache}/xdg"
export TMPDIR="${tmp_root}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${GPU_IDS:-}" ]; then
  IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
else
  mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
if (( ${#gpu_ids[@]} < 1 )); then
  printf 'At least one CUDA GPU is required.\n' >&2
  exit 1
fi
for gpu_id in "${gpu_ids[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]] \
    || ! nvidia-smi --id="${gpu_id}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    printf 'Invalid GPU id in GPU_IDS: %s\n' "${gpu_id}" >&2
    exit 1
  fi
done
(( max_parallel > ${#gpu_ids[@]} )) && max_parallel=${#gpu_ids[@]}

declare -A checkpoints=(
  [SharedSpace]="$(realpath "${shared_checkpoint}")"
  [SharedSpaceDistill]="$(realpath "${distill_checkpoint}")"
)

find_gif_dir() {
  find "$1" -type d -path '*/files/media/videos/prediction_vis' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

chunk_complete() {
  local root="$1" expected="$2" gif_dir raw_count gif_count
  raw_count=$(find "${root}/raw" -mindepth 2 -maxdepth 2 -name data.npz -type f 2>/dev/null | wc -l)
  gif_dir=$(find_gif_dir "${root}")
  gif_count=0
  [ -n "${gif_dir}" ] && gif_count=$(find "${gif_dir}" -maxdepth 1 -name '*.gif' -type f | wc -l)
  [ "${raw_count}" -ge "${expected}" ] && [ "${gif_count}" -ge "${expected}" ]
}

run_chunk() {
  local gpu="$1" protocol="$2" method="$3" offset="$4" count="$5"
  local frames="$6" context="$7" skip="$8"
  local root="${result_root}/generation/${protocol}/${method}/chunk_${offset}_${count}"
  mkdir -p "${root}"
  if chunk_complete "${root}" "${count}"; then
    printf '[reuse] %s/%s offset=%s\n' "${protocol}" "${method}" "${offset}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHON="${python_bin}" \
  EVAL_CKPT="${checkpoints[$method]}" \
  RESULT_DIR="${root}" \
  RAW_DIR="${root}/raw" \
  SAVE_RAW=true \
  EVAL_RUN_NAME="ablation_${protocol}_${method}_${offset}" \
  NUM_EVAL_VIDEOS="${count}" \
  EVAL_VIDEO_OFFSET="${offset}" \
  N_FRAMES="${frames}" \
  CONTEXT_LENGTH="${context}" \
  FRAME_SKIP="${skip}" \
  EVAL_BATCH_SIZE=2 \
  EVAL_NUM_WORKERS=0 \
  EVAL_METRICS='[]' \
  EVAL_GENERATIVE_UNIT_ENABLE=false \
  CHECKPOINT_STRICT=true \
  EVAL_SEED="$((seed + offset))" \
  WANDB_MODE="${wandb_mode}" \
    bash scripts/eval_upcast.sh >"${root}/eval.log" 2>&1
  chunk_complete "${root}" "${count}" || {
    tail -80 "${root}/eval.log" >&2
    return 1
  }
}

run_protocol() {
  local protocol="$1" total="$2" chunk="$3" frames="$4" context="$5" skip="$6"
  local offset count method slot cursor status
  local -a jobs=()
  for ((offset=0; offset<total; offset+=chunk)); do
    count=$((total - offset)); (( count > chunk )) && count=${chunk}
    for method in SharedSpace SharedSpaceDistill; do
      jobs+=("${protocol}|${method}|${offset}|${count}|${frames}|${context}|${skip}")
    done
  done
  cursor=0
  while (( cursor < ${#jobs[@]} )); do
    local -a pids=()
    for ((slot=0; slot<max_parallel && cursor<${#jobs[@]}; slot++,cursor++)); do
      IFS='|' read -r protocol method offset count frames context skip <<<"${jobs[$cursor]}"
      run_chunk "${gpu_ids[$slot]}" "${protocol}" "${method}" "${offset}" "${count}" "${frames}" "${context}" "${skip}" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    (( status == 0 )) || return 1
  done
}

collect_method() {
  local protocol="$1" total="$2" chunk="$3" method="$4" offset count
  local -a args=()
  for ((offset=0; offset<total; offset+=chunk)); do
    count=$((total - offset)); (( count > chunk )) && count=${chunk}
    args+=(--chunk "${offset}=${result_root}/generation/${protocol}/${method}/chunk_${offset}_${count}")
  done
  "${python_bin}" evaluation/collect_eval_outputs.py \
    "${args[@]}" --expected_count "${total}" \
    --output_dir "${result_root}/combined/${protocol}/${method}"
}

run_protocol long25 "${long_count}" 10 64 1 1
for method in SharedSpace SharedSpaceDistill; do
  collect_method long25 "${long_count}" 10 "${method}"
done

verify_protocol_pairing() {
  local protocol="$1" count="$2" main_protocol="$3"
  "${python_bin}" evaluation/verify_paired_eval_outputs.py \
    --run "DFoT=${main_root}/combined/${main_protocol}/DFoT/raw" \
    --run "SharedSpace=${result_root}/combined/${protocol}/SharedSpace/raw" \
    --run "SharedSpace+Distill=${result_root}/combined/${protocol}/SharedSpaceDistill/raw" \
    --run "Full=${main_root}/combined/${main_protocol}/Ours/raw" \
    --expected_count "${count}" \
    --allow-extra \
    --output_json "${result_root}/combined/${protocol}/paired_audit.json"
}

verify_protocol_pairing long25 "${long_count}" long100

run_metrics() {
  local protocol="$1" count="$2" context="$3" horizon="$4"
  local metric_dir="${result_root}/metrics/${protocol}/pixel"
  mkdir -p "${metric_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${python_bin}" \
    evaluation/metrics/compute_gif_video_metrics.py \
    --run "DFoT=${main_root}/combined/long100/DFoT/raw" \
    --run "SharedSpace=${result_root}/combined/${protocol}/SharedSpace/raw" \
    --run "SharedSpace+Distill=${result_root}/combined/${protocol}/SharedSpaceDistill/raw" \
    --run "Full=${main_root}/combined/long100/Ours/raw" \
    --output_dir "${metric_dir}" \
    --input_format npz \
    --indices "0-$((count - 1))" \
    --context_frames "${context}" \
    --max_frames "${horizon}" \
    --frame_chunk_size 32 \
    --device cuda \
    --include_fid

  local jedi_dir="${result_root}/metrics/${protocol}/jedi"
  local base_protocol
  base_protocol=long100
  mkdir -p "${jedi_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${jedi_python}" \
    evaluation/metrics/compute_official_jedi.py \
    --run "DFoT=${main_root}/combined/${base_protocol}/DFoT/gifs" \
    --run "SharedSpace=${result_root}/combined/${protocol}/SharedSpace/gifs" \
    --run "SharedSpace+Distill=${result_root}/combined/${protocol}/SharedSpaceDistill/gifs" \
    --run "Full=${main_root}/combined/${base_protocol}/Ours/gifs" \
    --output_dir "${jedi_dir}" \
    --indices "0-$((count - 1))" \
    --context_frames "${context}" \
    --max_frames "${horizon}" \
    --num_frames 16 \
    --batch_size 2 \
    --num_workers 0
}

run_metrics long25 "${long_count}" 1 64
"${python_bin}" evaluation/summarize_training_stages.py \
  --root "${result_root}"

printf '\nTraining-stage analysis complete: %s\n' "${result_root}"
