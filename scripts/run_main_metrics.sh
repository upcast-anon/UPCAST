#!/usr/bin/env bash
set -euo pipefail

# Compute only the metrics required by the current RealEstate10K main table.
# Generated videos are reused from run_main_experiments.sh.

python_bin="${PYTHON:-python}"
rpe_python="${RPE_PYTHON:-python}"
root="${MAIN_RESULT_DIR:-./output/evaluations/upcast_main}"
cache_root="${METRIC_CACHE_DIR:-./metric_cache}"
tmp_root="${TMPDIR_ROOT:-./tmp/upcast_main_metrics}"
pixel_gpu="${PIXEL_GPU:-0}"
jedi_gpu="${JEDI_GPU:-${pixel_gpu}}"
jedi_python="${JEDI_PYTHON:-${PYTHON:-python}}"
long_count="${LONG_COUNT:-100}"
stress_count="${STRESS_COUNT:-12}"
stress_protocol="${STRESS_PROTOCOL:-stress12}"
IFS=',' read -r -a methods <<<"${METHODS:-DFoT,REPA,VideoREPA,GF,Ours}"

mkdir -p "${cache_root}" "${tmp_root}" "${root}/logs"
export TORCH_HOME="${cache_root}/torch"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TMPDIR="${tmp_root}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  printf 'A working CUDA GPU is required.\n' >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null \
  || ! command -v "${jedi_python}" >/dev/null \
  || ! command -v "${rpe_python}" >/dev/null; then
  printf 'Missing evaluation Python environment.\n' >&2
  exit 1
fi

for protocol in long100 "${stress_protocol}"; do
  audit="${root}/combined/${protocol}/paired_audit.json"
  if [ ! -s "${audit}" ]; then
    printf 'Missing paired audit: %s\n' "${audit}" >&2
    exit 1
  fi
  for method in "${methods[@]}"; do
    if [ ! -d "${root}/combined/${protocol}/${method}/raw" ] \
      || [ ! -d "${root}/combined/${protocol}/${method}/gifs" ]; then
      printf 'Missing combined outputs for %s/%s.\n' "${protocol}" "${method}" >&2
      exit 1
    fi
  done
done

report_has_all_methods() {
  local report="$1"
  [ -s "${report}" ] || return 1
  "${python_bin}" - "${report}" "${methods[@]}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
if isinstance(payload.get("runs"), list):
    available = {str(run.get("name")) for run in payload["runs"]}
elif isinstance(payload.get("results"), dict):
    available = set(payload["results"])
else:
    available = set()
required = set(sys.argv[2:])
raise SystemExit(0 if required <= available else 1)
PY
}

metric_args() {
  local protocol="$1" kind="$2" method
  for method in "${methods[@]}"; do
    printf '%s\n' --run "${method}=${root}/combined/${protocol}/${method}/${kind}"
  done
}

run_pixel() {
  local protocol="$1" count="$2" context="$3" horizon="$4"
  local out="${root}/metrics/${protocol}/horizon_${horizon}/pixel"
  local report="${out}/gif_video_metrics_report.json"
  local log="${out}/eval_main_table.log"
  local -a args=()
  mkdir -p "${out}"
  if report_has_all_methods "${report}"; then
    printf '[reuse] pixel %s horizon=%s\n' "${protocol}" "${horizon}"
    return 0
  fi
  mapfile -t args < <(metric_args "${protocol}" raw)
  printf '[gpu %s] pixel %s horizon=%s\n' "${pixel_gpu}" "${protocol}" "${horizon}"
  if ! CUDA_VISIBLE_DEVICES="${pixel_gpu}" "${python_bin}" \
    evaluation/metrics/compute_gif_video_metrics.py \
    "${args[@]}" \
    --output_dir "${out}" \
    --input_format npz \
    --indices "0-$((count - 1))" \
    --context_frames "${context}" \
    --max_frames "${horizon}" \
    --frame_chunk_size 32 \
    --device cuda \
    --include_fid >"${log}" 2>&1; then
    tail -80 "${log}" >&2
    return 1
  fi
}

run_jedi() {
  local protocol="$1" count="$2" context="$3" horizon="$4"
  local out="${root}/metrics/${protocol}/jedi"
  local report="${out}/official_jedi_report.json"
  local log="${out}/eval_main_table.log"
  local -a args=()
  mkdir -p "${out}"
  if report_has_all_methods "${report}"; then
    printf '[reuse] JEDi %s horizon=%s\n' "${protocol}" "${horizon}"
    return 0
  fi
  mapfile -t args < <(metric_args "${protocol}" gifs)
  printf '[gpu %s] JEDi %s horizon=%s\n' "${jedi_gpu}" "${protocol}" "${horizon}"
  if ! CUDA_VISIBLE_DEVICES="${jedi_gpu}" "${jedi_python}" \
    evaluation/metrics/compute_official_jedi.py \
    "${args[@]}" \
    --output_dir "${out}" \
    --indices "0-$((count - 1))" \
    --context_frames "${context}" \
    --max_frames "${horizon}" \
    --num_frames 16 \
    --batch_size 2 \
    --num_workers 0 >"${log}" 2>&1; then
    tail -80 "${log}" >&2
    return 1
  fi
}

run_droid() {
  local gpu="$1" method="$2"
  local out="${root}/metrics/long100/geometry/droid/${method}"
  local gif_dir="${root}/combined/long100/${method}/gifs"
  local log="${out}/eval.log"
  mkdir -p "${out}/video_vis" "${out}/eval"
  if [ -s "${out}/eval/video_evaluation_results.json" ]; then
    printf '[reuse] DROID RPE %s\n' "${method}"
    return 0
  fi
  printf '[gpu %s] DROID RPE %s\n' "${gpu}" "${method}"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${rpe_python}" evaluation/eval_rpe.py \
    --temp_dir "${out}/video_vis" \
    --output_dir "${out}/eval" \
    --gif_dir "${gif_dir}" \
    --indices "0-$((long_count < 12 ? long_count - 1 : 11))" \
    --max_files 12 >"${log}" 2>&1; then
    tail -80 "${log}" >&2
    return 1
  fi
  touch "${out}/.complete"
}

# Run metrics sequentially so the default also works on a single GPU.
run_pixel long100 "${long_count}" 1 64
run_pixel "${stress_protocol}" "${stress_count}" 1 128
run_pixel "${stress_protocol}" "${stress_count}" 1 256
run_jedi long100 "${long_count}" 1 64
run_jedi "${stress_protocol}" "${stress_count}" 1 256

if [[ ",${METHODS:-DFoT,REPA,VideoREPA,GF,Ours}," == *,Ours,* ]]; then
  "${python_bin}" evaluation/eval_vjepa_feature_fscore.py \
    --run "Ours=${root}/metrics/long100/jedi/Ours" \
    --output-dir "${root}/metrics/long100/jedi/vjepa_feature_fscore" \
    --neighbors 3 --primary-neighbor 3 --normalization l2 \
    --source-report "${root}/metrics/long100/jedi/official_jedi_report.json"
fi

for method in "${methods[@]}"; do
  run_droid "${pixel_gpu}" "${method}"
done
"${python_bin}" evaluation/summarize_main_experiments.py \
  --root "${root}" \
  --methods "$(IFS=','; printf '%s' "${methods[*]}")" \
  --protocols "long100:64,${stress_protocol}:128,256"

printf 'Main-table metrics complete: %s/summary/main_results.md\n' "${root}"
