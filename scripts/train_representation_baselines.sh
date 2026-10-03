#!/usr/bin/env bash
set -euo pipefail

# Accumulation keeps parameter gradients resident between micro-batches.  The
# expandable allocator avoids fragmentation when frozen-teacher workspaces are
# subsequently reused by DFoT.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON:-python}"
output_root="${OUTPUT_DIR:-./output}"
source_checkpoint="${DFOT_CKPT:-checkpoints/DFoT_16f_state_dict.ckpt}"
baselines="${BASELINES:-repa,video_repa}"
target_global_batch="${TARGET_GLOBAL_BATCH:-40}"
batch_size="${TRAIN_BATCH_SIZE:-1}"
max_steps="${MAX_STEPS:-500}"
warmup_steps="${WARMUP_STEPS:-0}"
num_workers="${NUM_WORKERS:-0}"
wandb_mode="${WANDB_MODE:-offline}"
alignment_coeff="${ALIGNMENT_COEFF:-0.5}"
loss_freq="${LOSS_FREQ:-10}"
checkpoint_every_n_steps="${CHECKPOINT_EVERY_N_STEPS:-50}"
checkpoint_save_top_k="${CHECKPOINT_SAVE_TOP_K:-1}"
seed="${SEED:-20260823}"

if [ ! -s "${source_checkpoint}" ]; then
  printf 'Missing DFoT source checkpoint: %s\n' "${source_checkpoint}" >&2
  exit 1
fi
source_checkpoint=$(realpath "${source_checkpoint}")

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  world_size=${#visible_gpus[@]}
else
  world_size=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
if (( world_size < 1 )); then
  printf 'At least one CUDA GPU is required.\n' >&2
  exit 1
fi
micro_global=$((batch_size * world_size))
if (( target_global_batch % micro_global != 0 )); then
  printf 'TARGET_GLOBAL_BATCH=%s is not divisible by batch=%s x GPUs=%s\n' \
    "${target_global_batch}" "${batch_size}" "${world_size}" >&2
  exit 1
fi
accumulate=$((target_global_batch / micro_global))

IFS=',' read -r -a requested <<<"${baselines}"
for method in "${requested[@]}"; do
  method="${method//-/_}"
  case "${method}" in
    repa)
      experiment_name="dfot-repa-re10k-16f"
      mode="${REPA_MODE:-fresh}"
      ;;
    video_repa|videorepa)
      method="video_repa"
      experiment_name="dfot-videorepa-re10k-16f"
      mode="${VIDEOREPA_MODE:-fresh}"
      ;;
    *)
      printf 'Unknown baseline %s; expected repa or video_repa\n' "${method}" >&2
      exit 1
      ;;
  esac

  result_dir="${output_root}/train/${experiment_name}"
  checkpoint_dir="${result_dir}/checkpoints"
  mkdir -p "${result_dir}"
  load_checkpoint="${source_checkpoint}"
  reset_state=true
  checkpoint_strict=false
  if [ "${mode}" = resume ]; then
    latest=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "${latest}" ]; then
      printf 'No checkpoint available to resume under %s\n' "${checkpoint_dir}" >&2
      exit 1
    fi
    load_checkpoint=$(realpath "${latest}")
    reset_state=false
    checkpoint_strict=true
  elif [ "${mode}" = skip ]; then
    printf '[skip] %s\n' "${experiment_name}"
    continue
  elif [ "${mode}" != fresh ]; then
    printf '%s mode must be fresh, resume, or skip\n' "${method}" >&2
    exit 1
  fi

  printf 'Training %s\n' "${experiment_name}"
  printf '  source=%s\n' "${load_checkpoint}"
  printf '  batch=%s x %s GPUs x accum=%s = %s\n' \
    "${batch_size}" "${world_size}" "${accumulate}" "${target_global_batch}"
  printf '  method=%s steps=%s lr=8e-6 coeff=%s warmup=%s\n' \
    "${method}" "${max_steps}" "${alignment_coeff}" "${warmup_steps}"
  printf '  checkpoint_every=%s keep=%s + last\n' \
    "${checkpoint_every_n_steps}" "${checkpoint_save_top_k}"

  "${python_bin}" -m main \
    "+name=RE10k_${method}" \
    "+seed=${seed}" \
    dataset=realestate10k \
    algorithm=dfot_representation_alignment \
    experiment=video_generation \
    @diffusion/continuous \
    "wandb.mode=${wandb_mode}" \
    algorithm.compile=false \
    "algorithm.checkpoint.strict=${checkpoint_strict}" \
    "+algorithm.checkpoint.reset_training_state=${reset_state}" \
    algorithm.checkpoint.weights_source=raw \
    "algorithm.representation_alignment.method=${method}" \
    "algorithm.representation_alignment.alignment_coeff=${alignment_coeff}" \
    algorithm.backbone.latent_return_mode=same_level_blocks \
    algorithm.backbone.latent_return_stage=down \
    algorithm.backbone.latent_return_level=2 \
    'algorithm.backbone.latent_return_blocks=[-1]' \
    algorithm.backbone.generative_unit.enable=false \
    algorithm.backbone.world_state.enable=false \
    algorithm.tasks.prediction.enabled=false \
    algorithm.tasks.interpolation.enabled=false \
    "algorithm.lr_scheduler.num_warmup_steps=${warmup_steps}" \
    "algorithm.lr_scheduler.num_training_steps=${max_steps}" \
    algorithm.lr_scheduler_frequency=1 \
    "algorithm.logging.loss_freq=${loss_freq}" \
    algorithm.logging.grad_norm_freq=0 \
    "load=\"${load_checkpoint}\"" \
    dataset.subdataset_size=10000 \
    experiment.ema.enable=false \
    experiment.training.lr=8e-6 \
    experiment.training.max_epochs=2 \
    "experiment.training.max_steps=${max_steps}" \
    "experiment.training.batch_size=${batch_size}" \
    experiment.training.data.shuffle=true \
    "experiment.training.data.num_workers=${num_workers}" \
    "experiment.training.optim.accumulate_grad_batches=${accumulate}" \
    experiment.training.optim.gradient_clip_val=1.0 \
    "experiment.training.log_every_n_steps=${loss_freq}" \
    "experiment.training.checkpointing.every_n_train_steps=${checkpoint_every_n_steps}" \
    experiment.training.checkpointing.every_n_epochs=null \
    "experiment.training.checkpointing.save_top_k=${checkpoint_save_top_k}" \
    +experiment.training.checkpointing.save_last=true \
    experiment.validation.limit_batch=0 \
    experiment.validation.num_sanity_val_steps=0 \
    "experiment.validation.data.num_workers=${num_workers}" \
    hydra.run.dir="${result_dir}"

  if [ ! -s "${checkpoint_dir}/last.ckpt" ]; then
    printf 'Training finished without a last checkpoint: %s\n' "${checkpoint_dir}" >&2
    exit 1
  fi
  "${python_bin}" evaluation/export_inference_checkpoint.py \
    --input "${checkpoint_dir}/last.ckpt" \
    --output "${checkpoint_dir}/inference.ckpt"
done
