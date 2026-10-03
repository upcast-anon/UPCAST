#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON:-python}"
asset_root="${ACTION_ASSET_ROOT:-./action_eval_assets}"
result_dir="${RESULT_DIR:-./output/train/upcast-action-minecraft}"
base_checkpoint="${BASE_CHECKPOINT:-${asset_root}/huggingface/pretrained_models/DFoT_MCRAFT.ckpt}"
vae_checkpoint="${VAE_CHECKPOINT:-${asset_root}/huggingface/pretrained_models/ImageVAE_MCRAFT.ckpt}"
page_checkpoint="${PAGE_CHECKPOINT:-${project_root}/huggingface/zhouk777--PAGE4D/checkpoint_nomask.pt}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
max_steps="${MAX_STEPS:-500}"
bootstrap_steps="${BOOTSTRAP_STEPS:-100}"
checkpoint_every="${CHECKPOINT_EVERY:-250}"
global_batch="${GLOBAL_BATCH:-40}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-2}"
wandb_mode="${WANDB_MODE:-offline}"

IFS=',' read -ra gpu_ids <<< "${cuda_devices}"
num_gpus="${#gpu_ids[@]}"
denominator=$((batch_size * num_gpus))
if (( global_batch % denominator != 0 )); then
  echo "GLOBAL_BATCH=${global_batch} must be divisible by BATCH_SIZE*GPUS=${denominator}." >&2
  exit 1
fi
accumulate=$((global_batch / denominator))

for path in "${base_checkpoint}" "${vae_checkpoint}" "${page_checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required checkpoint is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${result_dir}" "${asset_root}/torch_cache" "${asset_root}/hf_cache" "${asset_root}/wandb"
export CUDA_VISIBLE_DEVICES="${cuda_devices}"
export TORCH_HOME="${TORCH_HOME:-${asset_root}/torch_cache}"
export HF_HOME="${HF_HOME:-${asset_root}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${asset_root}/wandb}"

echo "Action-conditioned UPCAST adaptation"
echo "  base=${base_checkpoint}"
echo "  batch=${batch_size} x ${num_gpus} GPUs x accum${accumulate} = ${global_batch}"
echo "  steps=${max_steps} (representation bootstrap=${bootstrap_steps}); output=${result_dir}"

"${python_bin}" -m main \
  +name=UPCAST_action_minecraft \
  +seed=0 \
  dataset=minecraft \
  dataset.save_dir="${asset_root}/data/minecraft" \
  algorithm=dfot_action_upcast \
  experiment=video_generation \
  '++algorithm={diffusion: {is_continuous: True, precond_scale: 0.125}, backbone: {use_fourier_noise_embedding: True}}' \
  '++algorithm={backbone: {depth: 12, hidden_size: 768, num_heads: 12}}' \
  "load=${base_checkpoint}" \
  wandb.mode="${wandb_mode}" \
  algorithm.compile=false \
  algorithm.checkpoint.strict=true \
  +algorithm.checkpoint.reset_training_state=true \
  algorithm.vae.pretrained_path="${vae_checkpoint}" \
  algorithm.alignment.geometry_teacher_checkpoint="${page_checkpoint}" \
  algorithm.alignment.unit_bootstrap_steps="${bootstrap_steps}" \
  algorithm.lr_scheduler.num_warmup_steps=40 \
  algorithm.lr_scheduler.num_training_steps="${max_steps}" \
  dataset.max_frames=16 \
  dataset.n_frames=16 \
  dataset.context_length=1 \
  dataset.subdataset_size=2000 \
  dataset.latent.load_video_during_training=true \
  algorithm.max_frames=16 \
  algorithm.n_frames=16 \
  algorithm.context_frames=1 \
  'algorithm.logging.metrics=[]' \
  algorithm.logging.loss_freq=20 \
  algorithm.logging.grad_norm_freq=0 \
  experiment.training.max_steps="${max_steps}" \
  experiment.training.max_epochs=20 \
  experiment.training.batch_size="${batch_size}" \
  experiment.training.optim.accumulate_grad_batches="${accumulate}" \
  experiment.training.data.num_workers="${num_workers}" \
  experiment.training.log_every_n_steps=20 \
  experiment.training.checkpointing.every_n_train_steps="${checkpoint_every}" \
  experiment.training.checkpointing.every_n_epochs=null \
  +experiment.training.checkpointing.save_last=link \
  experiment.validation.limit_batch=0 \
  experiment.validation.num_sanity_val_steps=0 \
  experiment.ema.enable=false \
  hydra.run.dir="${result_dir}"
