#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON:-python}"
asset_root="${ACTION_ASSET_ROOT:-./action_eval_assets}"
train_dir="${TRAIN_DIR:-./output/train/upcast-action-minecraft}"
eval_root="${OUTPUT_ROOT:-${asset_root}/evaluations/action_conditioned_minecraft_16f}"
base_checkpoint="${BASE_CHECKPOINT:-${asset_root}/huggingface/pretrained_models/DFoT_MCRAFT.ckpt}"
adapted_checkpoint="${ADAPTED_CKPT:-${train_dir}/checkpoints/last.ckpt}"
compact_checkpoint="${eval_root}/upcast_inference.ckpt"
gpu="${GPU:-0}"

if [ ! -s "${adapted_checkpoint}" ]; then
  printf 'Missing adapted checkpoint: %s\n' "${adapted_checkpoint}" >&2
  exit 1
fi

mkdir -p "${eval_root}/logs"
"${python_bin}" evaluation/export_inference_checkpoint.py \
  --input "${adapted_checkpoint}" --output "${compact_checkpoint}"

for action_mode in correct donor; do
  GPU="${gpu}" CHECKPOINT="${base_checkpoint}" MODEL_NAME=dfot \
    ACTION_MODE="${action_mode}" OUTPUT_ROOT="${eval_root}" \
    bash scripts/generate_action_conditioned_videos.sh \
    >"${eval_root}/logs/dfot_${action_mode}.log" 2>&1
  GPU="${gpu}" CHECKPOINT="${compact_checkpoint}" MODEL_NAME=upcast \
    ACTION_MODE="${action_mode}" OUTPUT_ROOT="${eval_root}" \
    bash scripts/generate_action_conditioned_videos.sh \
    >"${eval_root}/logs/upcast_${action_mode}.log" 2>&1
done

GPU="${gpu}" OUTPUT_ROOT="${eval_root}" REPORT_DIR="${eval_root}/report" \
  bash scripts/eval_action_conditioned_video.sh \
  >"${eval_root}/logs/evaluate.log" 2>&1

printf 'Action-conditioned evaluation complete: %s/report/summary.md\n' "${eval_root}"
