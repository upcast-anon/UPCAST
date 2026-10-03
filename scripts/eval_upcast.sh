#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
output_root="${OUTPUT_DIR:-./output}"
checkpoint="${EVAL_CKPT:-${CHECKPOINT_PATH:-}}"
result_dir="${RESULT_DIR:-${output_root}/evaluations/upcast/re10k64}"
save_raw="${SAVE_RAW:-true}"
raw_dir="${RAW_DIR:-${result_dir}/raw}"
wandb_mode="${WANDB_MODE:-offline}"
run_name="${EVAL_RUN_NAME:-UPCAST_RE10K_eval}"
num_videos="${NUM_EVAL_VIDEOS:-100}"
eval_video_offset="${EVAL_VIDEO_OFFSET:-0}"
n_frames="${N_FRAMES:-64}"
context_length="${CONTEXT_LENGTH:-1}"
frame_skip="${FRAME_SKIP:-1}"
batch_size="${EVAL_BATCH_SIZE:-2}"
num_workers="${EVAL_NUM_WORKERS:-0}"
metrics="${EVAL_METRICS:-[fvd,is,fid,lpips,mse,ssim,psnr]}"
keyframe_density="${KEYFRAME_DENSITY:-1.0}"
sliding_context_len="${SLIDING_CONTEXT_LEN:-4}"
history_guidance_scale="${HISTORY_GUIDANCE_SCALE:-4.0}"
history_stabilization="${HISTORY_STABILIZATION_LEVEL:-0.02}"
generative_unit_enable="${EVAL_GENERATIVE_UNIT_ENABLE:-false}"
checkpoint_strict="${CHECKPOINT_STRICT:-true}"
eval_seed="${EVAL_SEED:-20260814}"
eval_dataset="${EVAL_DATASET:-realestate10k}"
dataset_save_dir="${DATASET_SAVE_DIR:-}"

case "${generative_unit_enable}" in
  true|false) ;;
  *) printf 'EVAL_GENERATIVE_UNIT_ENABLE must be true or false\n' >&2; exit 1 ;;
esac
case "${checkpoint_strict}" in
  true|false) ;;
  *) printf 'CHECKPOINT_STRICT must be true or false\n' >&2; exit 1 ;;
esac
case "${save_raw}" in
  true|false) ;;
  *) printf 'SAVE_RAW must be true or false\n' >&2; exit 1 ;;
esac
if ! [[ "${eval_seed}" =~ ^[0-9]+$ ]]; then
  printf 'EVAL_SEED must be a non-negative integer\n' >&2
  exit 1
fi

if [ -z "${checkpoint}" ] || [ ! -s "${checkpoint}" ]; then
  printf 'Set EVAL_CKPT to a complete UPCAST checkpoint\n' >&2
  exit 1
fi
checkpoint=$(realpath "${checkpoint}")
mkdir -p "${result_dir}"
raw_args=()
if [ "${save_raw}" = true ]; then
  mkdir -p "${raw_dir}"
  raw_args+=("algorithm.logging.raw_dir=${raw_dir}")
fi

dataset_args=()
if [ -n "${dataset_save_dir}" ]; then
  dataset_args+=("dataset.save_dir=${dataset_save_dir}")
fi

rollout_args=()
if (( n_frames > 16 )); then
  rollout_step=$((16 - sliding_context_len))
  if (( rollout_step <= 0 )); then
    printf 'SLIDING_CONTEXT_LEN must be smaller than 16, got %s\n' \
      "${sliding_context_len}" >&2
    exit 1
  fi
  if [[ "${keyframe_density}" == "1" || "${keyframe_density}" == "1.0" ]] \
    && (( (n_frames - 16) % rollout_step != 0 )); then
    printf 'N_FRAMES=%s is incompatible with SLIDING_CONTEXT_LEN=%s\n' \
      "${n_frames}" "${sliding_context_len}" >&2
    exit 1
  fi
  rollout_args=(
    "algorithm.tasks.prediction.keyframe_density=${keyframe_density}"
    "algorithm.tasks.prediction.sliding_context_len=${sliding_context_len}"
    "algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla"
    "+algorithm.tasks.prediction.history_guidance.guidance_scale=${history_guidance_scale}"
    "+algorithm.tasks.prediction.history_guidance.stabilization_level=${history_stabilization}"
    "algorithm.tasks.interpolation.history_guidance.name=vanilla"
    "+algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5"
    "algorithm.tasks.interpolation.max_batch_size=4"
  )
fi

printf 'Evaluating UPCAST with backbone-only inference\n'
printf '  checkpoint=%s\n  frames=%s videos=%s offset=%s context=%s skip=%s result=%s\n' \
  "${checkpoint}" "${n_frames}" "${num_videos}" "${eval_video_offset}" \
  "${context_length}" "${frame_skip}" "${result_dir}"
printf '  generative_unit_module=%s (runtime conditioning remains off)\n' \
  "${generative_unit_enable}"
printf '  evaluation_seed=%s\n' "${eval_seed}"
printf '  lossless_raw=%s%s\n' "${save_raw}" "$([ "${save_raw}" = true ] && printf ' (%s)' "${raw_dir}")"
if (( n_frames > 16 )); then
  printf '  history_guidance=%s stabilization=%s\n' \
    "${history_guidance_scale}" "${history_stabilization}"
fi

"${python_bin}" -m main \
  "+name=${run_name}" \
  "+seed=${eval_seed}" \
  "dataset=${eval_dataset}" \
  algorithm=dfot_upcast \
  experiment=video_generation \
  @diffusion/continuous \
  "wandb.mode=${wandb_mode}" \
  algorithm.compile=false \
  "algorithm.checkpoint.strict=${checkpoint_strict}" \
  algorithm.checkpoint.weights_source=raw \
  algorithm.alignment.alignment_coeff=0.0 \
  algorithm.alignment.generative_unit_mode=off \
  algorithm.alignment.unit_inference_enable=false \
  algorithm.alignment.unit_allow_curriculum_override=true \
  algorithm.alignment.unit_geometry_tokenizer_steps=0 \
  algorithm.alignment.unit_bootstrap_steps=0 \
  algorithm.alignment.unit_joint_ramp_steps=0 \
  algorithm.alignment.compute_generation_loss=false \
  algorithm.alignment.video_prior_preservation_weight=0.0 \
  algorithm.backbone.latent_return_mode=same_level_blocks \
  algorithm.backbone.latent_return_stage=down \
  algorithm.backbone.latent_return_level=2 \
  'algorithm.backbone.latent_return_blocks=[-1]' \
  "algorithm.backbone.generative_unit.enable=${generative_unit_enable}" \
  algorithm.backbone.world_state.enable=false \
  "load=\"${checkpoint}\"" \
  experiment.ema.enable=false \
  'experiment.tasks=[validation]' \
  experiment.validation.data.shuffle=false \
  "dataset.context_length=${context_length}" \
  "dataset.frame_skip=${frame_skip}" \
  "dataset.n_frames=${n_frames}" \
  "dataset.num_eval_videos=${num_videos}" \
  "dataset.eval_video_offset=${eval_video_offset}" \
  "${dataset_args[@]}" \
  "experiment.validation.batch_size=${batch_size}" \
  "experiment.validation.data.num_workers=${num_workers}" \
  "algorithm.logging.max_num_videos=${num_videos}" \
  "algorithm.logging.metrics=${metrics}" \
  "${raw_args[@]}" \
  "${rollout_args[@]}" \
  "hydra.run.dir=${result_dir}"
