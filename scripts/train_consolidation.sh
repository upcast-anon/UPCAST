#!/usr/bin/env bash
set -euo pipefail

# Fit a conditional teacher and selectively consolidate its corrections.

python_bin="${PYTHON:-python}"
output_root="${OUTPUT_DIR:-./output}"
base_name="${BASE_EXP_NAME:-upcast-consolidation}"
wandb_mode="${WANDB_MODE:-offline}"
run_eval="${RUN_EVAL:-false}"
stop_after_phase="${STOP_AFTER_PHASE:-student_distill}"
start_ckpt="${START_CKPT:-checkpoints/upcast_acquired.ckpt}"

phase1_mode="${PHASE1_MODE:-fresh}"
phase2_mode="${PHASE2_MODE:-fresh}"
phase3_mode="${PHASE3_MODE:-fresh}"
phase1_steps="${PHASE1_STEPS:-150}"
phase2_steps="${PHASE2_STEPS:-800}"
phase3_steps="${PHASE3_STEPS:-250}"
phase1_warmup="${PHASE1_WARMUP:-20}"
phase2_warmup="${PHASE2_WARMUP:-20}"
phase3_warmup="${PHASE3_WARMUP:-20}"

batch_size="${BATCH_SIZE:-2}"
target_global_batch="${TARGET_GLOBAL_BATCH:-40}"
num_workers="${NUM_WORKERS:-0}"
subdataset_size="${SUBDATASET_SIZE:-10000}"
loss_freq="${LOSS_FREQ:-10}"
checkpoint_freq="${CHECKPOINT_EVERY_N_STEPS:-50}"
save_last="${SAVE_LAST:-link}"
prune_completed_phases="${PRUNE_COMPLETED_PHASE_CHECKPOINTS:-true}"
# The default cadence retains several recovery points per phase. Disk space is
# checked below from the phases and steps that still need to run; callers can
# retain a stricter fixed floor through MIN_CHECKPOINT_FREE_GB.
min_checkpoint_free_gb_override="${MIN_CHECKPOINT_FREE_GB:-}"

phase1_unit_lr="${PHASE1_UNIT_LR:-1e-4}"
phase2_unit_lr="${PHASE2_UNIT_LR:-5e-5}"
diffusion_lr="${DIFFUSION_LR:-5e-7}"
prior_weight="${VIDEO_PRIOR_WEIGHT:-0.05}"
prior_freq="${VIDEO_PRIOR_FREQ:-20}"
phase1_physical_weight="${PHASE1_PHYSICAL_WEIGHT:-0.25}"
phase2_physical_weight="${PHASE2_PHYSICAL_WEIGHT:-0.0}"
teacher_min_gain="${TEACHER_MIN_RELATIVE_GAIN:-0.00025}"
teacher_min_win="${TEACHER_MIN_WIN_FRACTION:-0.52}"
teacher_min_shuffle_gain="${TEACHER_MIN_SHUFFLE_GAIN:-0.000075}"
teacher_min_condition_win="${TEACHER_MIN_CONDITION_WIN_FRACTION:-0.53}"
teacher_min_samples="${TEACHER_MIN_SAMPLES:-256}"
distill_signal_min="${DISTILL_SIGNAL_MIN:-0.1}"
distill_signal_max="${DISTILL_SIGNAL_MAX:-0.9}"
allow_weak_teacher="${ALLOW_WEAK_TEACHER:-false}"
require_qualified_teacher=true
if [ "${allow_weak_teacher}" = true ]; then
  require_qualified_teacher=false
fi

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
  local directory="$1"
  find "${directory}" -maxdepth 1 -type f -name '*.ckpt' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

ensure_checkpoint() {
  local checkpoint="$1"
  if [ ! -s "${checkpoint}" ] || [ -f "${checkpoint}.aria2" ]; then
    printf 'Checkpoint is missing or incomplete: %s\n' "${checkpoint}" >&2
    exit 1
  fi
}

validate_integer() {
  local name="$1" value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "${name}" "${value}" >&2
    exit 1
  fi
}

for mode in "${phase1_mode}" "${phase2_mode}" "${phase3_mode}"; do
  case "${mode}" in
    fresh|resume|skip) ;;
    *) printf 'Phase mode must be fresh, resume, or skip; got %s\n' "${mode}" >&2; exit 1 ;;
  esac
done
case "${wandb_mode}" in
  online|offline|disabled) ;;
  *) printf 'WANDB_MODE must be online, offline, or disabled\n' >&2; exit 1 ;;
esac
case "${run_eval}" in
  true|false) ;;
  *) printf 'RUN_EVAL must be true or false\n' >&2; exit 1 ;;
esac
case "${stop_after_phase}" in
  adapter_warmup|conditional_teacher|student_distill) ;;
  *) printf 'STOP_AFTER_PHASE has an invalid value: %s\n' "${stop_after_phase}" >&2; exit 1 ;;
esac
case "${save_last}" in
  true|false|link) ;;
  *) printf 'SAVE_LAST must be true, false, or link\n' >&2; exit 1 ;;
esac
case "${prune_completed_phases}" in
  true|false) ;;
  *) printf 'PRUNE_COMPLETED_PHASE_CHECKPOINTS must be true or false\n' >&2; exit 1 ;;
esac
case "${allow_weak_teacher}" in
  true|false) ;;
  *) printf 'ALLOW_WEAK_TEACHER must be true or false\n' >&2; exit 1 ;;
esac

validate_integer BATCH_SIZE "${batch_size}"
validate_integer TARGET_GLOBAL_BATCH "${target_global_batch}"
validate_integer LOSS_FREQ "${loss_freq}"
validate_integer PHASE1_STEPS "${phase1_steps}"
validate_integer PHASE2_STEPS "${phase2_steps}"
validate_integer PHASE3_STEPS "${phase3_steps}"
validate_integer TEACHER_MIN_SAMPLES "${teacher_min_samples}"
validate_integer CHECKPOINT_EVERY_N_STEPS "${checkpoint_freq}"
for phase_steps in "${phase1_steps}" "${phase2_steps}" "${phase3_steps}"; do
  if [ $((phase_steps % checkpoint_freq)) -ne 0 ]; then
    printf 'Each phase step count must be divisible by checkpoint cadence; got steps=%s, cadence=%s\n' \
      "${phase_steps}" "${checkpoint_freq}" >&2
    exit 1
  fi
done
ensure_checkpoint "${start_ckpt}"
start_ckpt=$(realpath "${start_ckpt}")

num_devices=$(count_devices)
validate_integer NUM_DEVICES "${num_devices}"
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
validate_integer ACCUM "${accumulate}"
effective_batch=$((batch_size * num_devices * accumulate))

result_root="${RESULT_DIR:-${output_root}/train/${base_name}}"
mkdir -p "${result_root}"

latest_saved_step() {
  local directory="$1"
  local step
  step=$(
    find "${directory}" -maxdepth 1 -type f -name '*-step=*.ckpt' \
      -printf '%f\n' 2>/dev/null \
      | sed -nE 's/.*-step=([0-9]+)\.ckpt$/\1/p' \
      | sort -n \
      | tail -1
  )
  printf '%s\n' "${step:-0}"
}

remaining_checkpoint_count() {
  local mode="$1" steps="$2" directory="$3"
  if [ "${mode}" = skip ]; then
    printf '0\n'
    return
  fi
  local completed=0
  if [ "${mode}" = resume ]; then
    completed=$(latest_saved_step "${directory}")
  fi
  if [ "${completed}" -ge "${steps}" ]; then
    printf '0\n'
    return
  fi
  printf '%s\n' $((
    (steps - completed + checkpoint_freq - 1) / checkpoint_freq
  ))
}

if [ -n "${min_checkpoint_free_gb_override}" ]; then
  validate_integer MIN_CHECKPOINT_FREE_GB "${min_checkpoint_free_gb_override}"
  min_checkpoint_free_gb="${min_checkpoint_free_gb_override}"
else
  phase1_checkpoint_count=$(remaining_checkpoint_count \
    "${phase1_mode}" "${phase1_steps}" \
    "${result_root}/adapter_warmup/checkpoints")
  phase2_checkpoint_count=0
  phase3_checkpoint_count=0
  if [ "${stop_after_phase}" != adapter_warmup ]; then
    phase2_checkpoint_count=$(remaining_checkpoint_count \
      "${phase2_mode}" "${phase2_steps}" \
      "${result_root}/conditional_teacher/checkpoints")
  fi
  if [ "${stop_after_phase}" = student_distill ]; then
    phase3_checkpoint_count=$(remaining_checkpoint_count \
      "${phase3_mode}" "${phase3_steps}" \
      "${result_root}/student_distill/checkpoints")
  fi
  # Warmup/teacher checkpoints are currently 2.28 GiB and student checkpoints
  # about 3.15 GiB. During a phase all recovery points remain available. With
  # pruning enabled, only its exact endpoint remains before the next phase.
  # Estimate peak additional storage rather than summing all three phases.
  if [ "${prune_completed_phases}" = true ]; then
    retained_gb=0
    peak_checkpoint_gb=0
    if [ "${phase1_checkpoint_count}" -gt 0 ]; then
      phase_peak=$((retained_gb + 3 * phase1_checkpoint_count))
      [ "${phase_peak}" -gt "${peak_checkpoint_gb}" ] \
        && peak_checkpoint_gb="${phase_peak}"
      retained_gb=$((retained_gb + 3))
    fi
    if [ "${phase2_checkpoint_count}" -gt 0 ]; then
      phase_peak=$((retained_gb + 3 * phase2_checkpoint_count))
      [ "${phase_peak}" -gt "${peak_checkpoint_gb}" ] \
        && peak_checkpoint_gb="${phase_peak}"
      retained_gb=$((retained_gb + 3))
    fi
    if [ "${phase3_checkpoint_count}" -gt 0 ]; then
      phase_peak=$((retained_gb + 4 * phase3_checkpoint_count))
      [ "${phase_peak}" -gt "${peak_checkpoint_gb}" ] \
        && peak_checkpoint_gb="${phase_peak}"
    fi
    min_checkpoint_free_gb=$((8 + peak_checkpoint_gb))
  else
    min_checkpoint_free_gb=$((
      8
      + 3 * (phase1_checkpoint_count + phase2_checkpoint_count)
      + 4 * phase3_checkpoint_count
    ))
  fi
fi
available_kib=$(df -Pk "${result_root}" | awk 'NR == 2 {print $4}')
required_kib=$((min_checkpoint_free_gb * 1024 * 1024))
if [ -z "${available_kib}" ] || [ "${available_kib}" -lt "${required_kib}" ]; then
  printf 'At least %s GiB free space is required under %s\n' \
    "${min_checkpoint_free_gb}" "${result_root}" >&2
  exit 1
fi
available_gib=$(awk -v kib="${available_kib}" \
  'BEGIN {printf "%.1f", kib / 1048576}')
printf '  disk_guard=%s GiB required, %s GiB currently free\n' \
  "${min_checkpoint_free_gb}" "${available_gib}"

printf 'Generative UPCAST V14.1\n'
printf '  source=%s\n' "${start_ckpt}"
printf '  batch=%s x %s GPUs x accum%s = %s\n' \
  "${batch_size}" "${num_devices}" "${accumulate}" "${effective_batch}"
printf '  phases=%s/%s/%s steps; W&B=%s; post-eval=%s\n' \
  "${phase1_steps}" "${phase2_steps}" "${phase3_steps}" \
  "${wandb_mode}" "${run_eval}"
printf '  prune_completed_phase_checkpoints=%s\n' \
  "${prune_completed_phases}"

common_args=(
  "dataset=realestate10k"
  "algorithm=dfot_upcast"
  "experiment=video_generation"
  "@diffusion/continuous"
  "wandb.mode=${wandb_mode}"
  "algorithm.compile=false"
  "algorithm.checkpoint.strict=true"
  "algorithm.checkpoint.weights_source=raw"
  "algorithm.alignment.alignment_mode=unit_world_latent"
  "algorithm.alignment.unit_arch_version=7"
  "algorithm.alignment.unit_physical_only_continuation=true"
  "algorithm.alignment.unit_inference_enable=false"
  "algorithm.alignment.unit_video_only_inference=false"
  "algorithm.alignment.unit_migrate_from_version=null"
  "algorithm.alignment.unit_full_teacher_checkpoint=null"
  "algorithm.alignment.unit_allow_curriculum_override=true"
  "algorithm.alignment.alignment_coeff=0.0"
  "algorithm.alignment.unit_geometry_tokenizer_steps=0"
  "algorithm.alignment.unit_bootstrap_steps=0"
  "algorithm.alignment.unit_joint_ramp_steps=0"
  "algorithm.alignment.unit_cotrain_start_step=-1"
  "algorithm.alignment.alignment_context_length=16"
  "algorithm.alignment.keyframe_stride=1"
  "algorithm.alignment.latents_info=[[576,32,32]]"
  "algorithm.alignment.unit_video_tail_checkpoint=null"
  "algorithm.alignment.unit_tail_fm_loss_weight=0.0"
  "algorithm.alignment.compute_generation_loss=false"
  "algorithm.alignment.generation_dino_patch_loss_weight=0.0"
  "algorithm.alignment.generation_vjepa_spatial_loss_weight=0.0"
  "algorithm.alignment.generation_vjepa_temporal_loss_weight=0.0"
  "algorithm.alignment.generation_vjepa_global_loss_weight=0.0"
  "algorithm.alignment.generated_context_exposure_weight=0.0"
  "algorithm.alignment.rollout_training_enable=false"
  "algorithm.alignment.world_state_training_mode=joint"
  "algorithm.alignment.world_state_lr=1e-5"
  "algorithm.alignment.generative_video_fm_weight=1.0"
  "algorithm.alignment.generative_conditional_fm_weight=1.0"
  "algorithm.alignment.generative_baseline_rank_weight=1.0"
  "algorithm.alignment.generative_shuffle_rank_weight=1.0"
  "algorithm.alignment.generative_shuffle_identity_weight=0.25"
  "algorithm.alignment.generative_pair_gate_rank_weight=0.25"
  "algorithm.alignment.generative_pair_gate_calibration_weight=0.01"
  "algorithm.alignment.generative_pair_gate_margin=0.10"
  "algorithm.alignment.generative_rank_margin=0.002"
  "algorithm.alignment.generative_fm_error_floor=1e-4"
  "algorithm.alignment.generative_teacher_initial_gate=0.30"
  "algorithm.alignment.generative_teacher_quality_ema_decay=0.995"
  "algorithm.alignment.generative_teacher_min_relative_gain=${teacher_min_gain}"
  "algorithm.alignment.generative_teacher_min_win_fraction=${teacher_min_win}"
  "algorithm.alignment.generative_teacher_min_shuffle_gain=${teacher_min_shuffle_gain}"
  "algorithm.alignment.generative_teacher_min_condition_win_fraction=${teacher_min_condition_win}"
  "algorithm.alignment.generative_teacher_min_samples=${teacher_min_samples}"
  "algorithm.alignment.generative_max_consecutive_amp_skips=3"
  "algorithm.alignment.generative_require_qualified_teacher=${require_qualified_teacher}"
  "algorithm.alignment.generative_advantage_distill_weight=0.5"
  "algorithm.alignment.generative_advantage_max=1.0"
  "algorithm.alignment.generative_distill_signal_min=${distill_signal_min}"
  "algorithm.alignment.generative_distill_signal_max=${distill_signal_max}"
  "algorithm.alignment.generative_shuffle_diagnostic_freq=20"
  "algorithm.backbone.latent_return_mode=same_level_blocks"
  "algorithm.backbone.latent_return_stage=down"
  "algorithm.backbone.latent_return_level=2"
  "algorithm.backbone.latent_return_blocks=[-1]"
  "algorithm.backbone.generative_unit.enable=true"
  "algorithm.backbone.generative_unit.physical_dim=64"
  "algorithm.backbone.generative_unit.spatial_grid=8"
  "algorithm.backbone.generative_unit.num_heads=8"
  "algorithm.backbone.generative_unit.max_read_scale=0.08"
  "algorithm.backbone.generative_unit.local_read_fraction=0.5"
  "algorithm.backbone.generative_unit.local_read_radius=1"
  "algorithm.backbone.generative_unit.pair_encoder_layers=2"
  "algorithm.backbone.generative_unit.pair_gate_prior=0.8"
  "algorithm.backbone.generative_unit.pair_update_scale=0.25"
  "algorithm.backbone.use_checkpointing=[false,false,true,true]"
  "algorithm.backbone.world_state.enable=false"
  "dataset.subdataset_size=${subdataset_size}"
  "experiment.ema.enable=false"
  "experiment.training.max_epochs=999"
  "experiment.training.batch_size=${batch_size}"
  "experiment.training.data.num_workers=${num_workers}"
  "experiment.training.data.shuffle=true"
  "experiment.training.optim.accumulate_grad_batches=${accumulate}"
  "experiment.training.optim.gradient_clip_val=1.0"
  "experiment.training.log_every_n_steps=${loss_freq}"
  "experiment.find_unused_parameters=false"
  "experiment.training.checkpointing.every_n_epochs=null"
  "experiment.training.checkpointing.save_top_k=-1"
  "+experiment.training.checkpointing.save_last=${save_last}"
  "experiment.validation.limit_batch=0.0"
  "experiment.validation.num_sanity_val_steps=0"
  "algorithm.logging.grad_norm_freq=0"
  "algorithm.logging.loss_freq=${loss_freq}"
)

prune_phase_checkpoints() {
  local checkpoint_dir="$1" keep_checkpoint="$2"
  if [ "${prune_completed_phases}" != true ]; then
    return
  fi
  local keep_real removed=0 candidate candidate_real
  keep_real=$(realpath "${keep_checkpoint}")
  while IFS= read -r candidate; do
    candidate_real=$(realpath "${candidate}")
    if [ "${candidate_real}" != "${keep_real}" ]; then
      rm -f "${candidate}"
      removed=$((removed + 1))
    fi
  done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' -print)
  rm -f "${checkpoint_dir}/last.ckpt"
  if [ "${save_last}" != false ]; then
    ln -s "$(basename "${keep_real}")" "${checkpoint_dir}/last.ckpt"
  fi
  printf 'Pruned %s completed recovery checkpoint(s) from %s\n' \
    "${removed}" "${checkpoint_dir}"
}

run_phase() {
  local phase="$1" mode="$2" source="$3" steps="$4" warmup="$5"
  local unit_lr="$6" prior="$7" teacher_checkpoint="$8" initialize="$9"
  local physical_weight="${10}"
  local phase_dir="${result_root}/${phase}"
  local checkpoint_dir="${phase_dir}/checkpoints"
  mkdir -p "${checkpoint_dir}"

  if [ "${mode}" = skip ]; then
    local skipped
    skipped=$(latest_checkpoint "${checkpoint_dir}")
    if [ -z "${skipped}" ]; then
      printf 'Cannot skip %s: no checkpoint in %s\n' "${phase}" "${checkpoint_dir}" >&2
      exit 1
    fi
    PHASE_RESULT=$(realpath "${skipped}")
    return
  fi

  local load_checkpoint reset_state
  if [ "${mode}" = resume ]; then
    load_checkpoint=$(latest_checkpoint "${checkpoint_dir}")
    if [ -z "${load_checkpoint}" ]; then
      printf 'Cannot resume %s: no checkpoint in %s\n' "${phase}" "${checkpoint_dir}" >&2
      exit 1
    fi
    reset_state=false
  else
    if [ -n "$(latest_checkpoint "${checkpoint_dir}")" ]; then
      printf 'Fresh %s refuses to mix with existing checkpoints in %s\n' \
        "${phase}" "${checkpoint_dir}" >&2
      exit 1
    fi
    load_checkpoint="${source}"
    reset_state=true
  fi
  ensure_checkpoint "${load_checkpoint}"
  load_checkpoint=$(realpath "${load_checkpoint}")

  local teacher_arg="null"
  if [ -n "${teacher_checkpoint}" ]; then
    ensure_checkpoint "${teacher_checkpoint}"
    teacher_arg=$(realpath "${teacher_checkpoint}")
  fi
  local prior_checkpoint="null"
  if [ "${prior}" != 0 ] && [ "${prior}" != 0.0 ]; then
    prior_checkpoint="${start_ckpt}"
  fi

  printf 'Running %s: mode=%s, source=%s, steps=%s\n' \
    "${phase}" "${mode}" "${load_checkpoint}" "${steps}"
  "${python_bin}" -m main \
    "+name=UPCAST_${phase}" \
    "${common_args[@]}" \
    "algorithm.alignment.generative_unit_mode=${phase}" \
    "algorithm.alignment.generative_unit_initialize=${initialize}" \
    "algorithm.alignment.generative_unit_lr=${unit_lr}" \
    "algorithm.alignment.generative_physical_velocity_weight=${physical_weight}" \
    "algorithm.alignment.generative_diffusion_lr=${diffusion_lr}" \
    "algorithm.alignment.generative_teacher_checkpoint=\"${teacher_arg}\"" \
    "algorithm.alignment.video_prior_preservation_weight=${prior}" \
    "algorithm.alignment.video_prior_checkpoint=\"${prior_checkpoint}\"" \
    "algorithm.alignment.video_prior_cpu_offload=true" \
    "algorithm.alignment.video_prior_mode=full_model" \
    "algorithm.alignment.video_prior_preservation_freq=${prior_freq}" \
    "algorithm.alignment.video_prior_preservation_phase=1" \
    "algorithm.alignment.video_prior_frequency_compensation=true" \
    "algorithm.alignment.strict_video_prior=true" \
    "algorithm.lr_scheduler.num_warmup_steps=${warmup}" \
    "algorithm.lr_scheduler.num_training_steps=${steps}" \
    "algorithm.lr_scheduler_frequency=1" \
    "+algorithm.checkpoint.reset_training_state=${reset_state}" \
    "load=\"${load_checkpoint}\"" \
    "experiment.training.lr=${diffusion_lr}" \
    "experiment.training.max_steps=${steps}" \
    "experiment.training.checkpointing.every_n_train_steps=${checkpoint_freq}" \
    "hydra.run.dir=${phase_dir}"

  local result
  result=$(
    find "${checkpoint_dir}" -maxdepth 1 -type f \
      -name "*-step=${steps}.ckpt" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 {print $2}'
  )
  if [ -z "${result}" ]; then
    printf '%s completed without its exact step-%s checkpoint in %s\n' \
      "${phase}" "${steps}" "${checkpoint_dir}" >&2
    exit 1
  fi
  PHASE_RESULT=$(realpath "${result}")
}

PHASE_RESULT=""
run_phase \
  adapter_warmup "${phase1_mode}" "${start_ckpt}" \
  "${phase1_steps}" "${phase1_warmup}" "${phase1_unit_lr}" \
  0 "" true "${phase1_physical_weight}"
phase1_checkpoint="${PHASE_RESULT}"
prune_phase_checkpoints \
  "${result_root}/adapter_warmup/checkpoints" \
  "${phase1_checkpoint}"
if [ "${stop_after_phase}" = adapter_warmup ]; then
  ln -sfn "${phase1_checkpoint}" "${result_root}/latest.ckpt"
  printf 'Stopped after adapter_warmup: %s\n' "${phase1_checkpoint}"
  exit 0
fi
run_phase \
  conditional_teacher "${phase2_mode}" "${phase1_checkpoint}" \
  "${phase2_steps}" "${phase2_warmup}" "${phase2_unit_lr}" \
  0 "" false "${phase2_physical_weight}"
phase2_checkpoint="${PHASE_RESULT}"
teacher_audit_dir="${result_root}/conditional_teacher/teacher_quality"
audit_args=(
  --checkpoint "${phase2_checkpoint}"
  --output-dir "${teacher_audit_dir}"
  --min-relative-gain "${teacher_min_gain}"
  --min-win-fraction "${teacher_min_win}"
  --min-shuffle-gain "${teacher_min_shuffle_gain}"
  --min-condition-win-fraction "${teacher_min_condition_win}"
  --min-samples "${teacher_min_samples}"
)
if [ "${allow_weak_teacher}" = true ]; then
  audit_args+=(--allow-failure)
fi
"${python_bin}" evaluation/validate_conditional_teacher.py "${audit_args[@]}"
prune_phase_checkpoints \
  "${result_root}/conditional_teacher/checkpoints" \
  "${phase2_checkpoint}"
if [ "${stop_after_phase}" = conditional_teacher ]; then
  ln -sfn "${phase2_checkpoint}" "${result_root}/latest.ckpt"
  printf 'Stopped after conditional_teacher: %s\n' "${phase2_checkpoint}"
  exit 0
fi
run_phase \
  student_distill "${phase3_mode}" "${phase2_checkpoint}" \
  "${phase3_steps}" "${phase3_warmup}" "${phase2_unit_lr}" \
  "${prior_weight}" "${phase2_checkpoint}" false 0
phase3_checkpoint="${PHASE_RESULT}"
prune_phase_checkpoints \
  "${result_root}/student_distill/checkpoints" \
  "${phase3_checkpoint}"

ln -sfn "${phase3_checkpoint}" "${result_root}/latest.ckpt"
printf 'UPCAST consolidation complete: %s\n' "${phase3_checkpoint}"

if [ "${run_eval}" = true ]; then
  printf 'RUN_EVAL=true: evaluating the distilled video-only checkpoint.\n'
  EVAL_CKPT="${phase3_checkpoint}" \
  RESULT_DIR="${output_root}/evaluations/${base_name}" \
  WANDB_MODE="${wandb_mode}" \
  bash scripts/eval_upcast.sh
fi
