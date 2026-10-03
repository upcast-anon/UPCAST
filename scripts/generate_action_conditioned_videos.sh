#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON:-python}"
asset_root="${ACTION_ASSET_ROOT:-./action_eval_assets}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to an official DFoT or exported adapted checkpoint.}"
model_name="${MODEL_NAME:?Set MODEL_NAME, for example dfot or upcast.}"
action_mode="${ACTION_MODE:-correct}"
num_samples="${NUM_SAMPLES:-100}"
sampling_steps="${SAMPLING_STEPS:-50}"
gpu="${GPU:-0}"
output_root="${OUTPUT_ROOT:-${asset_root}/evaluations/action_conditioned_minecraft_16f}"
run_dir="${output_root}/${model_name}_${action_mode}"
raw_dir="${run_dir}/raw"
vae_checkpoint="${VAE_CHECKPOINT:-${asset_root}/huggingface/pretrained_models/ImageVAE_MCRAFT.ckpt}"

case "${action_mode}" in
  correct|donor) ;;
  *)
    echo "ACTION_MODE must be correct or donor." >&2
    exit 1
    ;;
esac
for path in "${checkpoint}" "${vae_checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required checkpoint is missing: ${path}" >&2
    exit 1
  fi
done
existing_count=0
if [[ -d "${raw_dir}" ]]; then
  existing_count="$(find "${raw_dir}" -name data.npz | wc -l)"
fi
if [[ "${existing_count}" -eq "${num_samples}" && "${FORCE:-false}" != true ]]; then
  echo "Reusing ${existing_count} completed clips in ${raw_dir}"
  exit 0
fi
if [[ "${existing_count}" -gt 0 && "${FORCE:-false}" != true ]]; then
  echo "Found ${existing_count}/${num_samples} clips in ${raw_dir}; set FORCE=true to replace the partial run." >&2
  exit 1
fi
if [[ "${FORCE:-false}" == true ]]; then
  rm -rf "${run_dir}"
fi

mkdir -p "${run_dir}" "${asset_root}/torch_cache" "${asset_root}/hf_cache" "${asset_root}/wandb"
export CUDA_VISIBLE_DEVICES="${gpu}"
export TORCH_HOME="${TORCH_HOME:-${asset_root}/torch_cache}"
export HF_HOME="${HF_HOME:-${asset_root}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${asset_root}/wandb}"

echo "Generating ${num_samples} paired Minecraft clips: model=${model_name}, action=${action_mode}, NFE=${sampling_steps}"
"${python_bin}" -m main \
  +name="Minecraft_${model_name}_${action_mode}" \
  +seed=0 \
  dataset=minecraft \
  dataset.save_dir="${asset_root}/data/minecraft" \
  algorithm=dfot_video \
  experiment=video_generation \
  '++algorithm={diffusion: {is_continuous: True, precond_scale: 0.125}, backbone: {use_fourier_noise_embedding: True}}' \
  '++algorithm={backbone: {depth: 12, hidden_size: 768, num_heads: 12}}' \
  "load=${checkpoint}" \
  wandb.mode=offline \
  algorithm.compile=false \
  algorithm.checkpoint.strict=true \
  algorithm.checkpoint.weights_source=raw \
  algorithm.vae.pretrained_path="${vae_checkpoint}" \
  algorithm.diffusion.sampling_timesteps="${sampling_steps}" \
  dataset.max_frames=16 \
  dataset.n_frames=16 \
  dataset.context_length=1 \
  dataset.num_eval_videos=null \
  dataset.eval_action_mode="${action_mode}" \
  algorithm.max_frames=16 \
  algorithm.n_frames=16 \
  algorithm.context_frames=1 \
  'experiment.tasks=[validation]' \
  experiment.validation.batch_size=1 \
  experiment.validation.limit_batch="${num_samples}" \
  experiment.validation.num_sanity_val_steps=0 \
  experiment.validation.data.num_workers="${NUM_WORKERS:-2}" \
  experiment.ema.enable=false \
  algorithm.logging.deterministic=0 \
  algorithm.logging.max_num_videos="${num_samples}" \
  'algorithm.logging.metrics=[]' \
  algorithm.logging.raw_dir="${raw_dir}" \
  hydra.run.dir="${run_dir}"

count="$(find "${raw_dir}" -name data.npz | wc -l)"
if [[ "${count}" -ne "${num_samples}" ]]; then
  echo "Expected ${num_samples} raw clips but found ${count} in ${raw_dir}." >&2
  exit 1
fi
echo "Saved ${count} clips to ${raw_dir}"
