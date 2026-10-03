#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
# Probe the complete acquired representation. Consolidated checkpoints intentionally
# omit semantic fusion and decoder modules because final inference is DFoT-only.
checkpoint="${PROBE_CKPT:-checkpoints/upcast_acquired.ckpt}"
output_dir="${PROBE_OUTPUT_DIR:-./output/evaluations/upcast_main/factorized_space_probe}"
num_clips="${PROBE_NUM_CLIPS:-64}"
batch_size="${PROBE_BATCH_SIZE:-1}"
num_workers="${PROBE_NUM_WORKERS:-0}"
noise_level="${PROBE_NOISE_LEVEL:-0.1}"
spans="${PROBE_SPANS:-0-3,0-7,0-15}"
device="${PROBE_DEVICE:-cuda}"
skip_decoder_ablation="${PROBE_SKIP_DECODER_ABLATION:-false}"

if [ ! -s "${checkpoint}" ] || [ -f "${checkpoint}.aria2" ]; then
  printf 'Missing or incomplete probe checkpoint: %s\n' "${checkpoint}" >&2
  exit 1
fi
if ! [[ "${num_clips}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PROBE_NUM_CLIPS must be a positive integer, got %s\n' "${num_clips}" >&2
  exit 1
fi
case "${skip_decoder_ablation}" in
  true|false) ;;
  *)
    printf 'PROBE_SKIP_DECODER_ABLATION must be true or false, got %s\n' \
      "${skip_decoder_ablation}" >&2
    exit 1
    ;;
esac

mkdir -p "${output_dir}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
args=(
  evaluation/eval_transition_probes.py
  --ckpt "${checkpoint}" \
  --output-dir "${output_dir}" \
  --num-clips "${num_clips}" \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --noise-level "${noise_level}" \
  --spans "${spans}" \
  --device "${device}"
)
if [ -n "${PROBE_CONFIG:-}" ]; then
  args+=(--config "${PROBE_CONFIG}")
fi
if [ "${skip_decoder_ablation}" = true ]; then
  args+=(--skip-decoder-ablation)
fi
exec "${python_bin}" "${args[@]}"
