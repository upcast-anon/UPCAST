#!/usr/bin/env bash
set -euo pipefail

# Paper-facing matched evaluation for same-backbone baselines and the final
# shared-physical-space model. Generation is sharded by dataset offset,
# lossless outputs are retained, and every stage is restartable.

python_bin="${PYTHON:-python}"
output_root="${OUTPUT_DIR:-./output}"
result_root="${MAIN_RESULT_DIR:-${output_root}/evaluations/upcast_main}"
metric_cache="${METRIC_CACHE_DIR:-./metric_cache}"
tmp_root="${TMPDIR_ROOT:-./tmp/upcast_main}"

dfot_ckpt="${DFOT_CKPT:-checkpoints/DFoT_16f_state_dict.ckpt}"
gf_ckpt="${GF_CKPT:-checkpoints/geometry_forcing_state_dict.ckpt}"
repa_ckpt="${REPA_CKPT:-${output_root}/train/dfot-repa-re10k-16f/checkpoints/inference.ckpt}"
videorepa_ckpt="${VIDEOREPA_CKPT:-${output_root}/train/dfot-videorepa-re10k-16f/checkpoints/inference.ckpt}"
ours_ckpt="${OURS_CKPT:-checkpoints/upcast.ckpt}"
methods_csv="${METHODS:-DFoT,REPA,VideoREPA,GF,Ours}"

run_generation="${RUN_GENERATION:-true}"
run_metrics="${RUN_METRICS:-true}"
wandb_mode="${WANDB_MODE:-offline}"
eval_seed="${EVAL_SEED:-20260814}"
min_free_gib="${MIN_FREE_GIB:-30}"

long_count="${LONG_COUNT:-100}"
stress_count="${STRESS_COUNT:-12}"
long_chunk="${LONG_CHUNK_SIZE:-10}"
stress_chunk="${STRESS_CHUNK_SIZE:-2}"
stress_protocol="${STRESS_PROTOCOL:-stress12}"

for value_name in run_generation run_metrics; do
  case "${!value_name}" in
    true|false) ;;
    *) printf '%s must be true or false\n' "${value_name}" >&2; exit 1 ;;
  esac
done
if [[ ! "${stress_protocol}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'STRESS_PROTOCOL contains unsupported characters: %s\n' \
    "${stress_protocol}" >&2
  exit 1
fi

for value_name in long_count stress_count long_chunk stress_chunk; do
  if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer\n' "${value_name}" >&2
    exit 1
  fi
done

mkdir -p "${result_root}" "${metric_cache}" "${tmp_root}"
export WANDB_CACHE_DIR="${metric_cache}/wandb"
export WANDB_DATA_DIR="${metric_cache}/wandb-data"
export WANDB_ARTIFACT_DIR="${metric_cache}/wandb-artifacts"
export TORCH_HOME="${metric_cache}/torch"
export XDG_CACHE_HOME="${metric_cache}/xdg"
export TMPDIR="${tmp_root}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

free_gib=$(df --output=avail -BG "${result_root}" | tail -1 | tr -dc '0-9')
if (( free_gib < min_free_gib )); then
  printf 'Need at least %s GiB free under %s; found %s GiB\n' \
    "${min_free_gib}" "${result_root}" "${free_gib}" >&2
  exit 1
fi

declare -A checkpoints=(
  [DFoT]="${dfot_ckpt}"
  [REPA]="${repa_ckpt}"
  [VideoREPA]="${videorepa_ckpt}"
  [GF]="${gf_ckpt}"
  [Ours]="${ours_ckpt}"
)
declare -A checkpoint_strict=(
  [DFoT]=true
  [REPA]=false
  [VideoREPA]=false
  [GF]=true
  [Ours]=true
)
IFS=',' read -r -a methods <<<"${methods_csv}"
if [ "${#methods[@]}" -eq 0 ]; then
  printf 'METHODS must contain at least one method.\n' >&2
  exit 1
fi
for method in "${methods[@]}"; do
  if [ -z "${checkpoints[$method]+set}" ]; then
    printf 'Unknown method in METHODS: %s\n' "${method}" >&2
    exit 1
  fi
  checkpoint="${checkpoints[$method]}"
  if [ ! -s "${checkpoint}" ] || [ -f "${checkpoint}.aria2" ]; then
    printf 'Missing or incomplete %s checkpoint: %s\n' "${method}" "${checkpoint}" >&2
    exit 1
  fi
  checkpoints[$method]="$(realpath "${checkpoint}")"
done
mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
if [ "${#gpu_ids[@]}" -eq 0 ]; then
  printf 'At least one CUDA GPU is required.\n' >&2
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
  raw_count=$(find "${root}/raw" -mindepth 2 -maxdepth 2 -name data.npz -type f 2>/dev/null | wc -l)
  gif_dir=$(find_gif_dir "${root}")
  gif_count=0
  if [ -n "${gif_dir}" ]; then
    gif_count=$(find "${gif_dir}" -maxdepth 1 -name '*.gif' -type f | wc -l)
  fi
  [ "${raw_count}" -ge "${expected}" ] && [ "${gif_count}" -ge "${expected}" ]
}

run_generation_job() {
  local gpu="$1" protocol="$2" method="$3" offset="$4" count="$5"
  local frames="$6" context="$7" skip="$8" batch="$9"
  local root="${result_root}/generation/${protocol}/${method}/chunk_${offset}_${count}"
  local chunk_seed=$((eval_seed + offset))
  mkdir -p "${root}"
  if chunk_complete "${root}" "${count}"; then
    printf '[reuse] %s/%s offset=%s count=%s\n' "${protocol}" "${method}" "${offset}" "${count}"
    return 0
  fi
  printf '[gpu %s] %s/%s offset=%s count=%s frames=%s\n' \
    "${gpu}" "${protocol}" "${method}" "${offset}" "${count}" "${frames}"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON="${python_bin}" \
    EVAL_CKPT="${checkpoints[$method]}" \
    RESULT_DIR="${root}" \
    RAW_DIR="${root}/raw" \
    SAVE_RAW=true \
    EVAL_RUN_NAME="main_${protocol}_${method}_${offset}" \
    NUM_EVAL_VIDEOS="${count}" \
    EVAL_VIDEO_OFFSET="${offset}" \
    N_FRAMES="${frames}" \
    CONTEXT_LENGTH="${context}" \
    FRAME_SKIP="${skip}" \
    EVAL_BATCH_SIZE="${batch}" \
    EVAL_NUM_WORKERS=0 \
    EVAL_METRICS='[]' \
    EVAL_GENERATIVE_UNIT_ENABLE=false \
    CHECKPOINT_STRICT="${checkpoint_strict[$method]}" \
    EVAL_SEED="${chunk_seed}" \
    WANDB_MODE="${wandb_mode}" \
      bash scripts/eval_upcast.sh >"${root}/eval.log" 2>&1; then
    tail -80 "${root}/eval.log" >&2
    return 1
  fi
  if ! chunk_complete "${root}" "${count}"; then
    printf 'Incomplete generation output: %s\n' "${root}" >&2
    tail -80 "${root}/eval.log" >&2
    return 1
  fi
}

build_jobs() {
  local protocol="$1" total="$2" chunk="$3" frames="$4" context="$5" skip="$6" batch="$7"
  local method offset count job_root
  for ((offset=0; offset<total; offset+=chunk)); do
    for method in "${methods[@]}"; do
      count=$((total - offset))
      (( count > chunk )) && count="${chunk}"
      job_root="${result_root}/generation/${protocol}/${method}/chunk_${offset}_${count}"
      if chunk_complete "${job_root}" "${count}"; then
        printf '[reuse] %s/%s offset=%s count=%s\n' \
          "${protocol}" "${method}" "${offset}" "${count}" >&2
        continue
      fi
      printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "${protocol}" "${method}" "${offset}" "${count}" \
        "${frames}" "${context}" "${skip}" "${batch}"
    done
  done
}

execute_jobs() {
  local -a jobs=("$@")
  local cursor=0 slot pid status
  while (( cursor < ${#jobs[@]} )); do
    local -a pids=()
    local -a labels=()
    for ((slot=0; slot<max_parallel && cursor<${#jobs[@]}; slot++,cursor++)); do
      IFS='|' read -r protocol method offset count frames context skip batch <<<"${jobs[$cursor]}"
      run_generation_job "${gpu_ids[$slot]}" "${protocol}" "${method}" \
        "${offset}" "${count}" "${frames}" "${context}" "${skip}" "${batch}" &
      pids+=("$!")
      labels+=("${protocol}/${method}/${offset}")
    done
    status=0
    for slot in "${!pids[@]}"; do
      if ! wait "${pids[$slot]}"; then
        printf 'Generation job failed: %s\n' "${labels[$slot]}" >&2
        status=1
      fi
    done
    (( status == 0 )) || return 1
  done
}

collect_protocol() {
  local protocol="$1" total="$2" chunk="$3" method offset count
  for method in "${methods[@]}"; do
    local -a args=()
    for ((offset=0; offset<total; offset+=chunk)); do
      count=$((total - offset))
      (( count > chunk )) && count="${chunk}"
      args+=(--chunk "${offset}=${result_root}/generation/${protocol}/${method}/chunk_${offset}_${count}")
    done
    "${python_bin}" evaluation/collect_eval_outputs.py \
      "${args[@]}" \
      --expected_count "${total}" \
      --output_dir "${result_root}/combined/${protocol}/${method}"
  done

  local -a paired_args=()
  for method in "${methods[@]}"; do
    paired_args+=(
      --run "${method}=${result_root}/combined/${protocol}/${method}/raw"
    )
  done
  "${python_bin}" evaluation/verify_paired_eval_outputs.py \
    "${paired_args[@]}" \
    --expected_count "${total}" \
    --output_json "${result_root}/combined/${protocol}/paired_audit.json"
}

if [ "${run_generation}" = true ]; then
  mapfile -t jobs < <(build_jobs long100 "${long_count}" "${long_chunk}" 64 1 1 2)
  execute_jobs "${jobs[@]}"
  mapfile -t jobs < <(build_jobs "${stress_protocol}" "${stress_count}" "${stress_chunk}" 256 1 1 2)
  execute_jobs "${jobs[@]}"
fi

collect_protocol long100 "${long_count}" "${long_chunk}"
collect_protocol "${stress_protocol}" "${stress_count}" "${stress_chunk}"

if [ "${run_metrics}" = true ]; then
  PYTHON="${python_bin}" MAIN_RESULT_DIR="${result_root}" \
  METHODS="${methods_csv}" LONG_COUNT="${long_count}" \
  STRESS_COUNT="${stress_count}" STRESS_PROTOCOL="${stress_protocol}" \
    bash scripts/run_main_metrics.sh
fi
printf 'Matched evaluation complete: %s\n' "${result_root}"
