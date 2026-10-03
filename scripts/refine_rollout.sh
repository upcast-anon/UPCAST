#!/usr/bin/env bash
set -euo pipefail

# UPCAST rollout refinement.
#
# Continue the consolidated student with three bounded objectives:
# motion-aware UPCAST distillation, coarse multi-span latent motion consistency,
# and sparse generated-context exposure. The frozen consolidated model is also the
# prior, so the continuation cannot buy motion by discarding its appearance
# and exposure improvements.

python_bin="${PYTHON:-python}"
output_root="${OUTPUT_DIR:-./output}"
base_name="${BASE_EXP_NAME:-upcast-rollout}"
run_name="${RUN_NAME:-UPCAST_rollout_refinement}"
result_root="${RESULT_DIR:-${output_root}/train/${base_name}}"
start_ckpt="${START_CKPT:-checkpoints/upcast_consolidated.ckpt}"
teacher_ckpt="${TEACHER_CKPT:-checkpoints/upcast_teacher.ckpt}"
generative_mode="${GENERATIVE_UNIT_MODE:-student_distill}"

train_mode="${TRAIN_MODE:-fresh}"
wandb_mode="${WANDB_MODE:-offline}"
run_eval="${RUN_EVAL:-false}"
checkpoint_strict="${CHECKPOINT_STRICT:-true}"
generative_unit_initialize="${GENERATIVE_UNIT_INITIALIZE:-false}"
experiment_seed="${EXPERIMENT_SEED:-}"
training_shuffle="${TRAINING_SHUFFLE:-true}"
max_steps="${MAX_STEPS:-150}"
warmup_steps="${WARMUP_STEPS:-10}"
batch_size="${BATCH_SIZE:-2}"
target_global_batch="${TARGET_GLOBAL_BATCH:-40}"
num_workers="${NUM_WORKERS:-0}"
subdataset_size="${SUBDATASET_SIZE:-10000}"
loss_freq="${LOSS_FREQ:-10}"
checkpoint_freq="${CHECKPOINT_EVERY_N_STEPS:-50}"
save_last="${SAVE_LAST:-link}"
min_free_gb="${MIN_CHECKPOINT_FREE_GB:-20}"

diffusion_lr="${DIFFUSION_LR:-3e-7}"
unit_lr="${UNIT_LR:-5e-5}"
advantage_weight="${ADVANTAGE_DISTILL_WEIGHT:-0.25}"
distill_motion_boost="${DISTILL_MOTION_BOOST:-1.0}"
motion_weight="${MOTION_LOSS_WEIGHT:-0.01}"
motion_freq="${MOTION_LOSS_FREQ:-5}"
motion_phase="${MOTION_LOSS_PHASE:-0}"
motion_grad_ratio="${MOTION_MAX_GRAD_RATIO:-0.05}"
prior_weight="${VIDEO_PRIOR_WEIGHT:-0.02}"
prior_freq="${VIDEO_PRIOR_FREQ:-20}"
context_weight="${GENERATED_CONTEXT_WEIGHT:-0.05}"
context_freq="${GENERATED_CONTEXT_FREQ:-20}"
context_phase="${GENERATED_CONTEXT_PHASE:-2}"

runtime_cache_dir="${RUNTIME_CACHE_DIR:-${output_root}/runtime_cache}"
mkdir -p \
  "${runtime_cache_dir}/tmp" \
  "${runtime_cache_dir}/torchinductor" \
  "${runtime_cache_dir}/triton"
export TMPDIR="${runtime_cache_dir}/tmp"
export TORCHINDUCTOR_CACHE_DIR="${runtime_cache_dir}/torchinductor"
export TRITON_CACHE_DIR="${runtime_cache_dir}/triton"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

count_devices() {
  if [ -n "${NUM_DEVICES:-}" ]; then
    printf '%s\n' "${NUM_DEVICES}"
  elif [ -n "${LOCAL_WORLD_SIZE:-}" ]; then
    printf '%s\n' "${LOCAL_WORLD_SIZE}"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
  else
    printf '1\n'
  fi
}

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name '*.ckpt' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

ensure_checkpoint() {
  if [ ! -s "$1" ] || [ -f "$1.aria2" ]; then
    printf 'Checkpoint is missing or incomplete: %s\n' "$1" >&2
    exit 1
  fi
}

positive_integer() {
  if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$1" "$2" >&2
    exit 1
  fi
}

nonnegative_integer() {
  if ! [[ "$2" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "$1" "$2" >&2
    exit 1
  fi
}

case "${train_mode}" in
  fresh|resume) ;;
  *) printf 'TRAIN_MODE must be fresh or resume\n' >&2; exit 1 ;;
esac
case "${wandb_mode}" in
  online|offline|disabled) ;;
  *) printf 'WANDB_MODE must be online, offline, or disabled\n' >&2; exit 1 ;;
esac
case "${run_eval}" in
  true|false) ;;
  *) printf 'RUN_EVAL must be true or false\n' >&2; exit 1 ;;
esac
case "${checkpoint_strict}" in
  true|false) ;;
  *) printf 'CHECKPOINT_STRICT must be true or false\n' >&2; exit 1 ;;
esac
case "${generative_unit_initialize}" in
  true|false) ;;
  *) printf 'GENERATIVE_UNIT_INITIALIZE must be true or false\n' >&2; exit 1 ;;
esac
case "${training_shuffle}" in
  true|false) ;;
  *) printf 'TRAINING_SHUFFLE must be true or false\n' >&2; exit 1 ;;
esac
seed_args=()
if [ -n "${experiment_seed}" ]; then
  nonnegative_integer EXPERIMENT_SEED "${experiment_seed}"
  seed_args=("+seed=${experiment_seed}")
fi
case "${generative_mode}" in
  student_distill|motion_calibration) ;;
  *)
    printf 'GENERATIVE_UNIT_MODE must be student_distill or motion_calibration\n' >&2
    exit 1
    ;;
esac
for pair in \
  "MAX_STEPS:${max_steps}" \
  "WARMUP_STEPS:${warmup_steps}" \
  "BATCH_SIZE:${batch_size}" \
  "TARGET_GLOBAL_BATCH:${target_global_batch}" \
  "LOSS_FREQ:${loss_freq}" \
  "CHECKPOINT_EVERY_N_STEPS:${checkpoint_freq}" \
  "MOTION_LOSS_FREQ:${motion_freq}" \
  "VIDEO_PRIOR_FREQ:${prior_freq}" \
  "GENERATED_CONTEXT_FREQ:${context_freq}" \
  "MIN_CHECKPOINT_FREE_GB:${min_free_gb}"; do
  positive_integer "${pair%%:*}" "${pair#*:}"
done
nonnegative_integer MOTION_LOSS_PHASE "${motion_phase}"
nonnegative_integer GENERATED_CONTEXT_PHASE "${context_phase}"
if [ $((max_steps % checkpoint_freq)) -ne 0 ]; then
  printf 'MAX_STEPS must be divisible by CHECKPOINT_EVERY_N_STEPS\n' >&2
  exit 1
fi
if [ $((motion_phase % motion_freq)) -ne "${motion_phase}" ]; then
  printf 'MOTION_LOSS_PHASE must be in [0, MOTION_LOSS_FREQ)\n' >&2
  exit 1
fi
if [ $((context_phase % context_freq)) -ne "${context_phase}" ]; then
  printf 'GENERATED_CONTEXT_PHASE must be in [0, GENERATED_CONTEXT_FREQ)\n' >&2
  exit 1
fi
if [ "${prior_freq}" -le 1 ] || [ "${context_freq}" -le 1 ]; then
  printf 'VIDEO_PRIOR_FREQ and GENERATED_CONTEXT_FREQ must exceed one\n' >&2
  exit 1
fi
for ((step = 0; step < motion_freq * context_freq; step++)); do
  if [ $((step % motion_freq)) -eq "${motion_phase}" ] \
    && [ $((step % context_freq)) -eq "${context_phase}" ]; then
    printf 'Motion and generated-context sparse phases overlap at step %s\n' \
      "${step}" >&2
    exit 1
  fi
done
for ((step = 0; step < motion_freq * prior_freq; step++)); do
  if [ $((step % motion_freq)) -eq "${motion_phase}" ] \
    && [ $((step % prior_freq)) -eq 1 ]; then
    printf 'Motion and video-prior sparse phases overlap at step %s\n' \
      "${step}" >&2
    exit 1
  fi
done
for ((step = 0; step < context_freq * prior_freq; step++)); do
  if [ $((step % context_freq)) -eq "${context_phase}" ] \
    && [ $((step % prior_freq)) -eq 1 ]; then
    printf 'Generated-context and video-prior phases overlap at step %s\n' \
      "${step}" >&2
    exit 1
  fi
done

ensure_checkpoint "${start_ckpt}"
start_ckpt=$(realpath "${start_ckpt}")
if [ "${generative_mode}" = student_distill ]; then
  ensure_checkpoint "${teacher_ckpt}"
  teacher_ckpt=$(realpath "${teacher_ckpt}")
  teacher_override="algorithm.alignment.generative_teacher_checkpoint=\"${teacher_ckpt}\""
  require_qualified_teacher=true
else
  if [[ ! "${advantage_weight}" =~ ^0+([.]0+)?([eE][+-]?0+)?$ ]]; then
    printf 'ADVANTAGE_DISTILL_WEIGHT must be 0 in motion_calibration mode\n' >&2
    exit 1
  fi
  teacher_ckpt=""
  teacher_override="algorithm.alignment.generative_teacher_checkpoint=null"
  require_qualified_teacher=false
fi

num_devices=$(count_devices)
positive_integer NUM_DEVICES "${num_devices}"
if [ -n "${ACCUM:-}" ]; then
  accumulate="${ACCUM}"
else
  accumulate=$(
    "${python_bin}" - "${target_global_batch}" "${batch_size}" "${num_devices}" <<'PY'
import math
import sys

target, batch, devices = map(int, sys.argv[1:])
print(max(1, math.ceil(target / (batch * devices))))
PY
  )
fi
positive_integer ACCUM "${accumulate}"
effective_batch=$((batch_size * num_devices * accumulate))

checkpoint_dir="${result_root}/motion_calibration/checkpoints"
phase_dir="${result_root}/motion_calibration"
mkdir -p "${checkpoint_dir}"
if [ "${train_mode}" = resume ]; then
  load_checkpoint=$(latest_checkpoint "${checkpoint_dir}")
  if [ -z "${load_checkpoint}" ]; then
    printf 'No rollout-refined checkpoint is available to resume in %s\n' "${checkpoint_dir}" >&2
    exit 1
  fi
  reset_state=false
else
  if [ -n "$(latest_checkpoint "${checkpoint_dir}")" ]; then
    printf 'Fresh rollout-refinement run refuses existing checkpoints in %s\n' "${checkpoint_dir}" >&2
    exit 1
  fi
  load_checkpoint="${start_ckpt}"
  reset_state=true
fi
ensure_checkpoint "${load_checkpoint}"
load_checkpoint=$(realpath "${load_checkpoint}")

available_kib=$(df -Pk "${result_root}" | awk 'NR == 2 {print $4}')
required_kib=$((min_free_gb * 1024 * 1024))
if [ -z "${available_kib}" ] || [ "${available_kib}" -lt "${required_kib}" ]; then
  printf 'At least %s GiB free space is required under %s\n' \
    "${min_free_gb}" "${result_root}" >&2
  exit 1
fi

printf 'UPCAST motion-aware rollout refinement\n'
printf '  source=%s\n' "${load_checkpoint}"
printf '  mode=%s, teacher=%s\n' \
  "${generative_mode}" "${teacher_ckpt:-disabled}"
printf '  batch=%s x %s GPUs x accum%s = %s\n' \
  "${batch_size}" "${num_devices}" "${accumulate}" "${effective_batch}"
printf '  steps=%s, lr=%s, motion=%s, prior=%s, context=%s\n' \
  "${max_steps}" "${diffusion_lr}" "${motion_weight}" \
  "${prior_weight}" "${context_weight}"

"${python_bin}" -m main \
  "${seed_args[@]}" \
  "+name=${run_name}" \
  "dataset=realestate10k" \
  "algorithm=dfot_upcast" \
  "experiment=video_generation" \
  "@diffusion/continuous" \
  "wandb.mode=${wandb_mode}" \
  "algorithm.compile=false" \
  "algorithm.checkpoint.strict=${checkpoint_strict}" \
  "algorithm.checkpoint.weights_source=raw" \
  "algorithm.alignment.alignment_mode=unit_world_latent" \
  "algorithm.alignment.unit_arch_version=7" \
  "algorithm.alignment.unit_physical_only_continuation=true" \
  "algorithm.alignment.unit_inference_enable=false" \
  "algorithm.alignment.unit_video_only_inference=false" \
  "algorithm.alignment.unit_migrate_from_version=null" \
  "algorithm.alignment.unit_full_teacher_checkpoint=null" \
  "algorithm.alignment.unit_allow_curriculum_override=true" \
  "algorithm.alignment.alignment_coeff=0.0" \
  "algorithm.alignment.unit_geometry_tokenizer_steps=0" \
  "algorithm.alignment.unit_bootstrap_steps=0" \
  "algorithm.alignment.unit_joint_ramp_steps=0" \
  "algorithm.alignment.unit_cotrain_start_step=-1" \
  "algorithm.alignment.alignment_context_length=16" \
  "algorithm.alignment.keyframe_stride=1" \
  "algorithm.alignment.latents_info=[[576,32,32]]" \
  "algorithm.alignment.unit_video_tail_checkpoint=null" \
  "algorithm.alignment.unit_tail_fm_loss_weight=0.0" \
  "algorithm.alignment.compute_generation_loss=false" \
  "algorithm.alignment.generation_dino_patch_loss_weight=0.0" \
  "algorithm.alignment.generation_vjepa_spatial_loss_weight=0.0" \
  "algorithm.alignment.generation_vjepa_temporal_loss_weight=0.0" \
  "algorithm.alignment.generation_vjepa_global_loss_weight=0.0" \
  "algorithm.alignment.rollout_training_enable=false" \
  "algorithm.alignment.world_state_training_mode=joint" \
  "algorithm.alignment.world_state_lr=1e-5" \
  "algorithm.alignment.generative_unit_mode=${generative_mode}" \
  "algorithm.alignment.generative_unit_initialize=${generative_unit_initialize}" \
  "algorithm.alignment.generative_unit_lr=${unit_lr}" \
  "algorithm.alignment.generative_diffusion_lr=${diffusion_lr}" \
  "${teacher_override}" \
  "algorithm.alignment.generative_physical_velocity_weight=0.0" \
  "algorithm.alignment.generative_advantage_distill_weight=${advantage_weight}" \
  "algorithm.alignment.generative_advantage_max=1.0" \
  "algorithm.alignment.generative_distill_signal_min=0.1" \
  "algorithm.alignment.generative_distill_signal_max=0.9" \
  "algorithm.alignment.generative_distill_motion_boost=${distill_motion_boost}" \
  "algorithm.alignment.generative_motion_loss_weight=${motion_weight}" \
  "algorithm.alignment.generative_motion_loss_freq=${motion_freq}" \
  "algorithm.alignment.generative_motion_loss_phase=${motion_phase}" \
  "algorithm.alignment.generative_motion_frequency_compensation=true" \
  "algorithm.alignment.generative_motion_spatial_grid=8" \
  "algorithm.alignment.generative_motion_strides=[1,2,4,8]" \
  "algorithm.alignment.generative_motion_signal_threshold=0.1" \
  "algorithm.alignment.generative_motion_magnitude_weight=0.25" \
  "algorithm.alignment.generative_motion_under_weight=1.25" \
  "algorithm.alignment.generative_motion_camera_boost=0.5" \
  "algorithm.alignment.generative_motion_max_grad_ratio=${motion_grad_ratio}" \
  "algorithm.alignment.generative_motion_gradient_probe_scale=65536.0" \
  "algorithm.alignment.strict_generative_motion=true" \
  "algorithm.alignment.generative_require_qualified_teacher=${require_qualified_teacher}" \
  "algorithm.alignment.generative_shuffle_diagnostic_freq=20" \
  "algorithm.alignment.video_prior_preservation_weight=${prior_weight}" \
  "algorithm.alignment.video_prior_checkpoint=\"${start_ckpt}\"" \
  "algorithm.alignment.video_prior_cpu_offload=true" \
  "algorithm.alignment.video_prior_mode=full_model" \
  "algorithm.alignment.video_prior_preservation_freq=${prior_freq}" \
  "algorithm.alignment.video_prior_preservation_phase=1" \
  "algorithm.alignment.video_prior_frequency_compensation=true" \
  "algorithm.alignment.strict_video_prior=true" \
  "algorithm.alignment.generated_context_exposure_weight=${context_weight}" \
  "algorithm.alignment.generated_context_exposure_start_step=0" \
  "algorithm.alignment.generated_context_exposure_freq=${context_freq}" \
  "algorithm.alignment.generated_context_exposure_phase=${context_phase}" \
  "algorithm.alignment.generated_context_frequency_compensation=true" \
  "algorithm.alignment.generated_context_num_frames=4" \
  "algorithm.alignment.generated_context_blend=0.25" \
  "algorithm.alignment.generated_context_max_residual=0.5" \
  "algorithm.alignment.generated_context_min_signal=0.1" \
  "algorithm.alignment.generated_context_activation_checkpointing=true" \
  "algorithm.alignment.generated_context_skip_alignment=true" \
  "algorithm.alignment.generated_context_skip_base_diffusion=true" \
  "algorithm.alignment.strict_generated_context_exposure=true" \
  "algorithm.backbone.latent_return_mode=same_level_blocks" \
  "algorithm.backbone.latent_return_stage=down" \
  "algorithm.backbone.latent_return_level=2" \
  "algorithm.backbone.latent_return_blocks=[-1]" \
  "algorithm.backbone.generative_unit.enable=true" \
  "algorithm.backbone.generative_unit.physical_dim=64" \
  "algorithm.backbone.generative_unit.spatial_grid=8" \
  "algorithm.backbone.generative_unit.num_heads=8" \
  "algorithm.backbone.generative_unit.max_read_scale=0.08" \
  "algorithm.backbone.generative_unit.local_read_fraction=0.5" \
  "algorithm.backbone.generative_unit.local_read_radius=1" \
  "algorithm.backbone.generative_unit.pair_encoder_layers=2" \
  "algorithm.backbone.generative_unit.pair_gate_prior=0.8" \
  "algorithm.backbone.generative_unit.pair_update_scale=0.25" \
  "algorithm.backbone.use_checkpointing=[false,false,true,true]" \
  "algorithm.backbone.world_state.enable=false" \
  "algorithm.lr_scheduler.num_warmup_steps=${warmup_steps}" \
  "algorithm.lr_scheduler.num_training_steps=${max_steps}" \
  "algorithm.lr_scheduler_frequency=1" \
  "+algorithm.checkpoint.reset_training_state=${reset_state}" \
  "load=\"${load_checkpoint}\"" \
  "dataset.subdataset_size=${subdataset_size}" \
  "experiment.ema.enable=false" \
  "experiment.training.lr=${diffusion_lr}" \
  "experiment.training.max_epochs=999" \
  "experiment.training.max_steps=${max_steps}" \
  "experiment.training.batch_size=${batch_size}" \
  "experiment.training.data.shuffle=${training_shuffle}" \
  "experiment.training.data.num_workers=${num_workers}" \
  "experiment.training.optim.accumulate_grad_batches=${accumulate}" \
  "experiment.training.optim.gradient_clip_val=1.0" \
  "experiment.training.log_every_n_steps=${loss_freq}" \
  "experiment.training.checkpointing.every_n_train_steps=${checkpoint_freq}" \
  "experiment.training.checkpointing.every_n_epochs=null" \
  "experiment.training.checkpointing.save_top_k=-1" \
  "+experiment.training.checkpointing.save_last=${save_last}" \
  "experiment.find_unused_parameters=false" \
  "experiment.validation.limit_batch=0.0" \
  "experiment.validation.num_sanity_val_steps=0" \
  "algorithm.logging.grad_norm_freq=0" \
  "algorithm.logging.loss_freq=${loss_freq}" \
  "hydra.run.dir=${phase_dir}"

result=$(
  find "${checkpoint_dir}" -maxdepth 1 -type f \
    -name "*-step=${max_steps}.ckpt" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
)
if [ -z "${result}" ]; then
  printf 'Rollout refinement completed without an exact step-%s checkpoint\n' "${max_steps}" >&2
  exit 1
fi
result=$(realpath "${result}")
ln -sfn "${result}" "${result_root}/latest.ckpt"
printf 'UPCAST refinement complete: %s\n' "${result}"

if [ "${run_eval}" = true ]; then
  EVAL_CKPT="${result}" \
  RESULT_DIR="${output_root}/evaluations/${base_name}" \
  WANDB_MODE="${wandb_mode}" \
  bash scripts/eval_upcast.sh
fi
