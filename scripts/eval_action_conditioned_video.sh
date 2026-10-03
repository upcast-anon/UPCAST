#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON:-python}"
asset_root="${ACTION_ASSET_ROOT:-./action_eval_assets}"
eval_root="${OUTPUT_ROOT:-${asset_root}/evaluations/action_conditioned_minecraft_16f}"
report_dir="${REPORT_DIR:-${eval_root}/report}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TORCH_HOME="${TORCH_HOME:-${asset_root}/torch_cache}"
export HF_HOME="${HF_HOME:-${asset_root}/hf_cache}"

"${python_bin}" evaluation/eval_action_conditioned_video.py \
  --dataset-root "${asset_root}/data/minecraft" \
  --output-dir "${report_dir}" \
  --run "dfot_correct=${eval_root}/dfot_correct/raw" \
  --run "dfot_donor=${eval_root}/dfot_donor/raw" \
  --run "upcast_correct=${eval_root}/upcast_correct/raw" \
  --run "upcast_donor=${eval_root}/upcast_donor/raw" \
  --run-mode dfot_correct=correct \
  --run-mode dfot_donor=donor \
  --run-mode upcast_correct=correct \
  --run-mode upcast_donor=donor \
  --causal-pair dfot=dfot_correct,dfot_donor \
  --causal-pair upcast=upcast_correct,upcast_donor \
  --probe-epochs "${PROBE_EPOCHS:-20}" \
  --probe-batch-size "${PROBE_BATCH_SIZE:-64}" \
  --metric-batch-size "${METRIC_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --device cuda
