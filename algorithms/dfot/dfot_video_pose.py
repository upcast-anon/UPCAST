import torch
import torch.distributed as dist
import time
import copy
import contextlib
import hashlib
import json
import math
from pathlib import Path
import torch.nn.functional as F
from torch.func import functional_call
from omegaconf import OmegaConf
from typing import Any, Optional
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange
from utils.geometry_utils import CameraPose
from utils.distributed_utils import rank_zero_print
from .dfot_video import DFoTVideo
from transformers import get_scheduler
from upcast.transition_learning import FactorizedTransitionLearning
from upcast.state_compat import (
    CounterfactualUtilityCache,
    continuous_preference_loss,
    continuous_targets_from_repeats,
    detached_policy_inputs,
    deterministic_noise,
    policy_spec_dict,
    preference_sample_keys,
    split_for_key,
)


def _validate_world_state_learning_rate(
    learning_rate: float,
    *,
    collection_only: bool,
) -> None:
    """Keep normal training trainable while making cache collection immutable."""
    if collection_only:
        if learning_rate != 0.0:
            raise ValueError(
                "world_state_lr must be zero for preference cache collection-only mode"
            )
        return
    if learning_rate <= 0.0:
        raise ValueError(
            "world_state_lr must be positive outside preference cache "
            "collection-only mode"
        )


class DFoTVideoPose(DFoTVideo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self.camera_pose_conditioning = cfg.camera_pose_conditioning
        self.conditioning_type = cfg.camera_pose_conditioning.type
        self._warned_unbounded_pose = False
        self._check_cfg(cfg)
        self._update_backbone_cfg(cfg)
        super().__init__(cfg)
        # import pdb; pdb.set_trace()

    def _check_cfg(self, cfg: DictConfig):
        """
        Check if the config is valid
        """
        if cfg.backbone.name not in {"dit3d_pose", "u_vit3d_pose","u_vit3d_pose_controlnet"}:
            raise ValueError(
                f"DiffusionForcingVideo3D only supports backbone 'dit3d_pose' or 'u_vit3d_pose_controlnet' 'dit3d_pose_controlnet', got {cfg.backbone.name}"
            )

        if (
            cfg.backbone.name == "dit3d_pose"
            and self.conditioning_type == "global"
            and cfg.backbone.conditioning.modeling != "film"
        ):
            raise ValueError(
                f"When using global camera pose conditioning, `algorithm.backbone.conditioning.modeling` should be 'film', got {cfg.backbone.conditioning.modeling}"
            )
        if (cfg.backbone.name in ["u_vit3d_pose","u_vit3d_pose_controlnet"]) and self.conditioning_type == "global":
            raise ValueError(
                "Global camera pose conditioning is not supported for U-ViT3DPose"
            )
        ## adding additional control to diffusion backbone

    def _update_backbone_cfg(self, cfg: DictConfig):
        """
        Update backbone config with camera pose conditioning
        """
        conditioning_dim = None
        match self.conditioning_type:
            case "global":
                conditioning_dim = 12
            case "ray" | "plucker":
                conditioning_dim = 6
            case "ray_encoding":
                conditioning_dim = 180
            case _:
                raise ValueError(
                    f"Unknown camera pose conditioning type: {self.conditioning_type}"
                )
        cfg.backbone.conditioning.dim = conditioning_dim

    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def _process_conditions(
        self, conditions: Tensor, noise_levels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Process conditions (raw camera poses) to desired format for the model
        Args:
            conditions (Tensor): raw camera poses (B, T, 12)
        """
        # import pdb; pdb.set_trace()
        camera_poses = CameraPose.from_vectors(conditions)
        if self.cfg.tasks.prediction.history_guidance.name == "temporal":
            # NOTE: when using temporal history guidance,
            # some frames are fully masked out and thus their camera poses are not needed
            # so we replace them with interpolated camera poses from the nearest non-masked frames
            # this is important b/c we normalize camera poses by the first frame
            camera_poses.replace_with_interpolation(
                mask=noise_levels == self.timesteps - 1
            )

        match self.camera_pose_conditioning.normalize_by:
            case "first":
                camera_poses.normalize_by_first()
            case "mean":
                camera_poses.normalize_by_mean()
            case _:
                raise ValueError(
                    f"Unknown camera pose normalization method: {self.camera_pose_conditioning.normalize_by}"
                )

        if self.camera_pose_conditioning.bound is not None:
            camera_poses.scale_within_bounds(self.camera_pose_conditioning.bound)
        elif (
            not self._warned_unbounded_pose
            and self.conditioning_type in {"ray", "ray_encoding", "plucker"}
        ):
            # NeRF-style positional encoding is periodic; if translation/origin is unbounded,
            # the encoding aliases and the model often becomes rotation-only sensitive.
            rank_zero_print(
                "[DFoTVideoPose] WARNING: camera_pose_conditioning.bound is null while using "
                f"{self.conditioning_type}. Translation/origin may be unbounded and positional "
                "encoding can alias. Consider setting algorithm.camera_pose_conditioning.bound=1.0 "
                "(or similar) to improve translation sensitivity."
            )
            self._warned_unbounded_pose = True

        match self.conditioning_type:
            case "global":
                return camera_poses.extrinsics(flatten=True)
            case "ray" | "ray_encoding" | "plucker":
                rays = camera_poses.rays(resolution=self.x_shape[1])
                if self.conditioning_type == "ray_encoding":
                    rays = rays.to_pos_encoding()[0]
                else:
                    rays = rays.to_tensor(
                        use_plucker=self.conditioning_type == "plucker"
                    )
                return rearrange(rays, "b t h w c -> b t c h w")


class DFoTUPCAST(DFoTVideoPose):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._diffusion_frozen = False
        self._fit_start_walltime = None
        self._fit_start_step = 0
        self.build_alignment_module(cfg)

    def on_fit_start(self) -> None:
        self._fit_start_walltime = time.time()
        self._fit_start_step = int(self.global_step)
        if hasattr(self, "vggt_alignment_loss"):
            self.vggt_alignment_loss.discard_unit_codebook_update()
        if (
            self.video_prior_preservation_weight > 0
            and (
                self.video_prior_cpu_offload
                or self.video_prior_mode == "selected_functional"
            )
        ):
            # Build and offload before the first training graph exists. Creating
            # a second 458M-parameter DFoT while UPCAST activations are live can
            # otherwise exceed a 40 GB device even at batch size one.
            self._ensure_video_prior_model()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if not self.generative_unit_enabled:
            return
        conditioner = (
            self._generative_unit_backbone().generative_unit_conditioner
        )
        checkpoint["generative_unit"] = {
            "version": int(conditioner.architecture_version.item()),
            "training_contract_version": 6,
            "mode": self.generative_unit_mode,
            "latent_return_stage": str(
                self.cfg.backbone.latent_return_stage
            ),
            "latent_return_level": int(
                self.cfg.backbone.latent_return_level
            ),
            "latent_return_blocks": [
                int(value) for value in self.cfg.backbone.latent_return_blocks
            ],
            "physical_dim": int(conditioner.physical_dim),
            "spatial_grid": int(conditioner.spatial_grid),
            "num_heads": int(conditioner.cross_attention.num_heads),
            "max_read_scale": float(conditioner.max_read_scale),
            "local_read_fraction": float(conditioner.local_read_fraction),
            "local_read_radius": int(conditioner.local_read_radius),
            "pair_encoder_layers": int(conditioner.pair_encoder_layers),
            "pair_gate_prior": float(conditioner.pair_gate_prior),
            "pair_update_scale": float(conditioner.pair_update_scale),
            "condition_interface": "paired_video_fused",
            "objective_contract": "calibrated_gate_rms_identity",
            "teacher_quality": self._generative_teacher_quality_snapshot(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if not self.generative_unit_enabled:
            return
        metadata = checkpoint.get("generative_unit") or {}
        if str(metadata.get("mode")) == "conditional_teacher":
            # A resumed teacher has already chosen its gate. Do not reapply the
            # phase-entry floor and overwrite a learned conservative value.
            self._generative_teacher_gate_initialized = True
        if (
            self.generative_unit_mode == "conditional_teacher"
            and int(metadata.get("training_contract_version", -1)) < 6
        ):
            # Older contracts accumulated rank-local squared-error statistics.
            # Retain the model and optimizer, but reset statistics for the
            # globally reduced calibrated/RMS objective.
            rank_zero_print(
                "[DFoTUPCAST] Reset conditional-teacher quality "
                "statistics for training contract 6."
            )
            return
        quality = metadata.get("teacher_quality")
        if not isinstance(quality, dict):
            return
        for name, default in self._generative_teacher_quality.items():
            if name not in quality:
                continue
            value = quality[name]
            self._generative_teacher_quality[name] = (
                bool(value) if isinstance(default, bool) else float(value)
            )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.log(
            "eval/weights_source_ema",
            float(getattr(self, "loaded_weights_source", "raw") == "ema"),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        output = super().validation_step(batch, batch_idx, namespace=namespace)
        if self.unit_inference_enable and self._latest_world_state_metrics:
            # DDP collectives must be issued in exactly the same order even if
            # an upstream diagnostic dictionary was assembled dynamically.
            for name in sorted(self._latest_world_state_metrics):
                value = self._latest_world_state_metrics[name]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=self.device)
                if value.numel() != 1:
                    continue
                self.log(
                    f"{namespace}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch[0].shape[0],
                )
        return output

    @staticmethod
    def _is_frequency_step(batch_idx: int, frequency: int, phase: int = 0) -> bool:
        frequency = max(int(frequency), 1)
        return int(batch_idx) % frequency == int(phase) % frequency

    def _should_compute_video_prior(self, namespace: str, batch_idx: int) -> bool:
        """Schedule the frozen-base prior independently of alignment."""
        return bool(
            not self._diffusion_frozen
            and self.video_prior_preservation_weight > 0
            and (
                namespace != "training"
                or self._is_frequency_step(
                    batch_idx,
                    self.video_prior_preservation_freq,
                    self.video_prior_preservation_phase,
                )
            )
        )

    def _is_unit_cotrain_step(self, step: int) -> bool:
        return bool(
            self.unit_cotrain_start_step >= 0
            and int(step) >= self.unit_cotrain_start_step
            and (
                int(step) - self.unit_cotrain_start_step
            )
            % self.unit_cotrain_interval
            == 0
        )

    def _apply_unit_cotrain_lr(self) -> None:
        if self._unit_cotrain_lr_applied:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None or not getattr(trainer, "optimizers", None):
            return
        optimizer = trainer.optimizers[0]
        base_optimizer = getattr(optimizer, "optimizer", optimizer)
        unit_parameter_ids = {
            id(parameter)
            for parameter in self.vggt_alignment_loss.parameters()
        }
        unit_indices = []
        for index, group in enumerate(base_optimizer.param_groups):
            named_unit_group = group.get("group_name") == "unit"
            contains_unit_parameter = any(
                id(parameter) in unit_parameter_ids
                for parameter in group["params"]
            )
            if not (named_unit_group or contains_unit_parameter):
                continue
            group["group_name"] = "unit"
            group["lr"] = self.unit_cotrain_lr
            group["initial_lr"] = self.unit_cotrain_lr
            unit_indices.append(index)
        if not unit_indices:
            raise RuntimeError("UPCAST optimizer group is missing during cotrain")

        for scheduler_config in getattr(trainer, "lr_scheduler_configs", ()):
            scheduler = scheduler_config.scheduler
            if hasattr(scheduler, "base_lrs"):
                for index in unit_indices:
                    if index < len(scheduler.base_lrs):
                        scheduler.base_lrs[index] = self.unit_cotrain_lr
        self._unit_cotrain_lr_applied = True
        rank_zero_print(
            "[DFoTUPCAST] Set selective UPCAST cotrain LR to "
            f"{self.unit_cotrain_lr:.3g}."
        )

    def _set_diffusion_trainable(
        self,
        trainable: bool,
        *,
        keep_ddp_registered: bool = False,
    ) -> None:
        self.diffusion_model.train(mode=trainable)
        for param in self.diffusion_model.parameters():
            param.requires_grad = bool(trainable or keep_ddp_registered)
        self._diffusion_frozen = not trainable

    def _generative_unit_backbone(self):
        backbone = self._world_state_backbone()
        conditioner = getattr(
            backbone,
            "generative_unit_conditioner",
            None,
        )
        if conditioner is None:
            raise RuntimeError(
                "Physical conditioner mode requires backbone.generative_unit.enable=true"
            )
        return backbone

    def _matches_generative_diffusion_prefix(self, name: str) -> bool:
        canonical = self._canonical_diffusion_parameter_name(name)
        return any(
            canonical == prefix or canonical.startswith(prefix + ".")
            for prefix in self.generative_diffusion_trainable_prefixes
        )

    def _set_generative_unit_trainable(self) -> None:
        """Apply the exact trainable set for one generative continuation phase."""
        backbone = self._generative_unit_backbone()
        conditioner = backbone.generative_unit_conditioner
        conditioner_ids = {
            id(parameter) for parameter in conditioner.parameters()
        }
        # The conditional teacher must not update the unconditioned suffix.
        # Otherwise the shared suffix can reduce both teacher and student FM
        # losses without ever using the UPCAST tokens. Only the final student
        # phase is allowed to move the video model.
        train_suffix = self.generative_unit_mode in {
            "student_distill",
            "motion_calibration",
        }
        train_conditioner = self.generative_unit_mode in {
            "adapter_warmup",
            "conditional_teacher",
        }
        matched = {prefix: 0 for prefix in self.generative_diffusion_trainable_prefixes}
        for name, parameter in self.diffusion_model.named_parameters():
            canonical = self._canonical_diffusion_parameter_name(name)
            is_conditioner = id(parameter) in conditioner_ids
            suffix_trainable = bool(
                train_suffix
                and self._matches_generative_diffusion_prefix(canonical)
            )
            parameter.requires_grad = bool(
                (train_conditioner and is_conditioner)
                or (suffix_trainable and not is_conditioner)
            )
            if suffix_trainable and not is_conditioner:
                for prefix in matched:
                    if canonical == prefix or canonical.startswith(prefix + "."):
                        matched[prefix] += parameter.numel()
        if train_suffix:
            missing = [prefix for prefix, count in matched.items() if count == 0]
            if missing:
                raise RuntimeError(
                    "Physical conditioner suffix prefixes matched no parameters: "
                    + ", ".join(missing)
                )
        for parameter in self.vggt_alignment_loss.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.vggt_alignment_loss.set_unit_phase("joint")
        self.vggt_alignment_loss.eval()
        # Disable stochastic block dropout so the base and replayed suffix are
        # directly comparable under exactly the same denoising task.
        self.diffusion_model.eval()
        conditioner.train(train_conditioner)
        self._diffusion_frozen = not train_suffix

    def _generative_parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        backbone = self._generative_unit_backbone()
        conditioner_ids = {
            id(parameter)
            for parameter in backbone.generative_unit_conditioner.parameters()
        }
        conditioner_parameters = []
        diffusion_parameters = []
        for parameter in self.diffusion_model.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in conditioner_ids:
                conditioner_parameters.append(parameter)
            else:
                diffusion_parameters.append(parameter)
        return conditioner_parameters, diffusion_parameters

    def _set_world_state_adapter_only_trainable(self) -> None:
        """Freeze the acquired model and train only the zero-initialized adapter."""
        for parameter in self.diffusion_model.parameters():
            parameter.requires_grad = False
        adapter = getattr(self._world_state_backbone(), "world_state_adapter", None)
        if adapter is None:
            raise RuntimeError("adapter_only training requires an enabled world state")
        for parameter in adapter.parameters():
            parameter.requires_grad = True
        self.diffusion_model.eval()
        adapter.train()
        if hasattr(self, "vggt_alignment_loss"):
            self.vggt_alignment_loss.eval()
            for parameter in self.vggt_alignment_loss.parameters():
                parameter.requires_grad = False
        # The frozen base still executes with gradients enabled so the loss can
        # reach the injected adapter. This is not a joint DFoT update.
        self._diffusion_frozen = False

    def _set_world_state_gate_only_trainable(self) -> None:
        """Freeze the generator and expose only the preference policy."""
        for parameter in self.diffusion_model.parameters():
            parameter.requires_grad = False
        adapter = getattr(self._world_state_backbone(), "world_state_adapter", None)
        if adapter is None:
            raise RuntimeError("gate_only training requires an enabled world state")
        preference_parameters = adapter.preference_parameters()
        preference_ids = {id(parameter) for parameter in preference_parameters}
        for parameter in adapter.parameters():
            parameter.requires_grad = id(parameter) in preference_ids
        self.diffusion_model.eval()
        adapter.train()
        if hasattr(self, "vggt_alignment_loss"):
            self.vggt_alignment_loss.eval()
            for parameter in self.vggt_alignment_loss.parameters():
                parameter.requires_grad = False
        # The frozen read path remains differentiable with respect to its two
        # branch gates; no Video-DiT or UPCAST parameter is updated.
        self._diffusion_frozen = False

    @staticmethod
    def _canonical_diffusion_parameter_name(name: str) -> str:
        name = str(name).removeprefix("_orig_mod.")
        return name.replace("model._orig_mod.", "model.", 1)

    def _matches_selective_diffusion_prefix(self, name: str) -> bool:
        canonical = self._canonical_diffusion_parameter_name(name)
        return any(
            canonical == prefix or canonical.startswith(prefix + ".")
            for prefix in self.selective_diffusion_trainable_prefixes
        )

    def _set_selective_joint_trainable(self) -> None:
        """Expose only the DFoT regions used by controlled continuation."""
        matched_prefixes = {
            prefix: 0 for prefix in self.selective_diffusion_trainable_prefixes
        }
        trainable_names = set()
        for name, parameter in self.diffusion_model.named_parameters():
            canonical = self._canonical_diffusion_parameter_name(name)
            trainable = self._matches_selective_diffusion_prefix(canonical)
            parameter.requires_grad = trainable
            if not trainable:
                parameter.grad = None
                continue
            trainable_names.add(canonical)
            for prefix in matched_prefixes:
                if canonical == prefix or canonical.startswith(prefix + "."):
                    matched_prefixes[prefix] += parameter.numel()
        missing = [name for name, count in matched_prefixes.items() if count == 0]
        if missing:
            raise RuntimeError(
                "Selective DFoT prefixes matched no parameters: "
                + ", ".join(missing)
            )
        self._selective_trainable_parameter_names = trainable_names
        trainable_count = sum(matched_prefixes.values())
        if not hasattr(self, "_selective_trainable_parameter_count"):
            self._selective_trainable_parameter_count = trainable_count
            total_count = sum(
                parameter.numel() for parameter in self.diffusion_model.parameters()
            )
            rank_zero_print(
                "[DFoTUPCAST] Selective joint update exposes "
                f"{trainable_count / 1e6:.1f}M / {total_count / 1e6:.1f}M "
                "DFoT parameters."
            )
        elif self._selective_trainable_parameter_count != trainable_count:
            raise RuntimeError("Selective trainable parameter set changed at runtime")
        self.diffusion_model.train()
        if hasattr(self, "vggt_alignment_loss"):
            if getattr(self.vggt_alignment_loss, "unit_phase", None) != "joint":
                self.vggt_alignment_loss.set_unit_phase("joint")
            self.vggt_alignment_loss.eval()
        self._diffusion_frozen = False
        self._ensure_selective_rollout_gradient_hooks()

    def _selective_parameter_groups(self) -> tuple[list[Tensor], list[Tensor]]:
        adapter = getattr(self._world_state_backbone(), "world_state_adapter", None)
        adapter_ids = (
            {id(parameter) for parameter in adapter.parameters()}
            if adapter is not None
            else set()
        )
        diffusion_parameters = []
        world_state_parameters = []
        for parameter in self.diffusion_model.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in adapter_ids:
                world_state_parameters.append(parameter)
            else:
                diffusion_parameters.append(parameter)
        return diffusion_parameters, world_state_parameters

    def _ensure_selective_rollout_gradient_hooks(self) -> None:
        """Keep rollout-only updates in the recurrent adapter.

        A rollout micro-batch still needs gradients through the frozen DFoT
        suffix so that its diffusion and trajectory objectives can reach the
        injected state. Hooks suppress only that micro-batch's parameter
        gradients for the selected video backbone; gradients accumulated from
        ordinary/alignment micro-batches remain untouched.
        """
        if not getattr(self, "rollout_adapter_only_gradients", False):
            return
        if getattr(self, "_selective_rollout_gradient_hooks", None):
            return
        adapter = getattr(self._world_state_backbone(), "world_state_adapter", None)
        if adapter is None:
            raise RuntimeError(
                "rollout_adapter_only_gradients requires a world-state adapter"
            )
        adapter_ids = {id(parameter) for parameter in adapter.parameters()}
        handles = []

        def suppress_rollout_gradient(gradient: Tensor) -> Tensor:
            if getattr(self, "_suppress_selective_diffusion_gradients", False):
                return torch.zeros_like(gradient)
            return gradient

        for parameter in self.diffusion_model.parameters():
            if parameter.requires_grad and id(parameter) not in adapter_ids:
                handles.append(parameter.register_hook(suppress_rollout_gradient))
        if not handles:
            raise RuntimeError(
                "rollout_adapter_only_gradients found no selected DFoT parameters"
            )
        self._selective_rollout_gradient_hooks = handles

    def _release_frozen_unit_optimizer_state(self) -> None:
        trainer = getattr(self, "_trainer", None)
        if trainer is None or not getattr(trainer, "optimizers", None):
            return
        optimizer = trainer.optimizers[0]
        base_optimizer = getattr(optimizer, "optimizer", optimizer)
        released = 0
        for parameter in self.vggt_alignment_loss.parameters():
            parameter.grad = None
            state = base_optimizer.state.pop(parameter, None)
            if state:
                released += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rank_zero_print(
            "[DFoTUPCAST] Released optimizer state for "
            f"{released} frozen UPCAST parameters."
        )

    def build_alignment_module(self, cfg):
        alignment_config = cfg.alignment
        alignment_dict = OmegaConf.to_container(alignment_config, resolve=True)
        encoder_type = alignment_dict.pop("encoder_type", "vggt")
        self.encoder_type = encoder_type
        self.alignment_coeff = alignment_dict.pop("alignment_coeff", 0.5)
        self.generative_unit_mode = str(
            alignment_dict.pop("generative_unit_mode", "off") or "off"
        ).lower()
        if self.generative_unit_mode not in {
            "off",
            "adapter_warmup",
            "conditional_teacher",
            "student_distill",
            "motion_calibration",
        }:
            raise ValueError(
                "generative_unit_mode must be off, adapter_warmup, "
                "conditional_teacher, student_distill, or motion_calibration"
            )
        self.generative_unit_initialize = bool(
            alignment_dict.pop("generative_unit_initialize", False)
        )
        self.generative_unit_lr = float(
            alignment_dict.pop("generative_unit_lr", 1.0e-4)
        )
        self.generative_diffusion_lr = float(
            alignment_dict.pop("generative_diffusion_lr", 5.0e-7)
        )
        self.generative_video_fm_weight = float(
            alignment_dict.pop("generative_video_fm_weight", 1.0)
        )
        self.generative_conditional_fm_weight = float(
            alignment_dict.pop("generative_conditional_fm_weight", 1.0)
        )
        self.generative_baseline_rank_weight = float(
            alignment_dict.pop("generative_baseline_rank_weight", 1.0)
        )
        self.generative_shuffle_rank_weight = float(
            alignment_dict.pop("generative_shuffle_rank_weight", 0.5)
        )
        self.generative_shuffle_identity_weight = float(
            alignment_dict.pop("generative_shuffle_identity_weight", 0.25)
        )
        self.generative_pair_gate_rank_weight = float(
            alignment_dict.pop("generative_pair_gate_rank_weight", 0.25)
        )
        self.generative_pair_gate_calibration_weight = float(
            alignment_dict.pop(
                "generative_pair_gate_calibration_weight",
                0.01,
            )
        )
        self.generative_pair_gate_margin = float(
            alignment_dict.pop("generative_pair_gate_margin", 0.10)
        )
        self.generative_rank_margin = float(
            alignment_dict.pop("generative_rank_margin", 0.002)
        )
        self.generative_fm_error_floor = float(
            alignment_dict.pop("generative_fm_error_floor", 1.0e-4)
        )
        self.generative_teacher_initial_gate = float(
            alignment_dict.pop("generative_teacher_initial_gate", 0.10)
        )
        self.generative_teacher_quality_ema_decay = float(
            alignment_dict.pop("generative_teacher_quality_ema_decay", 0.995)
        )
        self.generative_teacher_min_relative_gain = float(
            alignment_dict.pop("generative_teacher_min_relative_gain", 0.00025)
        )
        self.generative_teacher_min_win_fraction = float(
            alignment_dict.pop("generative_teacher_min_win_fraction", 0.52)
        )
        self.generative_teacher_min_shuffle_gain = float(
            alignment_dict.pop("generative_teacher_min_shuffle_gain", 0.000075)
        )
        self.generative_teacher_min_condition_win_fraction = float(
            alignment_dict.pop(
                "generative_teacher_min_condition_win_fraction",
                0.53,
            )
        )
        self.generative_teacher_min_samples = int(
            alignment_dict.pop("generative_teacher_min_samples", 256)
        )
        self.generative_max_consecutive_amp_skips = int(
            alignment_dict.pop("generative_max_consecutive_amp_skips", 3)
        )
        self.generative_require_qualified_teacher = bool(
            alignment_dict.pop("generative_require_qualified_teacher", True)
        )
        self.generative_physical_velocity_weight = float(
            alignment_dict.pop(
                "generative_physical_velocity_weight",
                0.25,
            )
        )
        self.generative_advantage_distill_weight = float(
            alignment_dict.pop(
                "generative_advantage_distill_weight",
                0.1,
            )
        )
        self.generative_advantage_max = float(
            alignment_dict.pop("generative_advantage_max", 1.0)
        )
        self.generative_distill_signal_min = float(
            alignment_dict.pop("generative_distill_signal_min", 0.2)
        )
        self.generative_distill_signal_max = float(
            alignment_dict.pop("generative_distill_signal_max", 0.8)
        )
        self.generative_distill_motion_boost = float(
            alignment_dict.pop("generative_distill_motion_boost", 1.0)
        )
        self.generative_motion_loss_weight = float(
            alignment_dict.pop("generative_motion_loss_weight", 0.0)
        )
        self.generative_motion_loss_freq = max(
            1,
            int(alignment_dict.pop("generative_motion_loss_freq", 5) or 5),
        )
        self.generative_motion_loss_phase = int(
            alignment_dict.pop("generative_motion_loss_phase", 0) or 0
        ) % self.generative_motion_loss_freq
        self.generative_motion_frequency_compensation = bool(
            alignment_dict.pop(
                "generative_motion_frequency_compensation",
                True,
            )
        )
        self.generative_motion_spatial_grid = int(
            alignment_dict.pop("generative_motion_spatial_grid", 8)
        )
        raw_motion_strides = alignment_dict.pop(
            "generative_motion_strides",
            (1, 2, 4, 8),
        )
        if isinstance(raw_motion_strides, str):
            raw_motion_strides = tuple(
                int(value)
                for value in raw_motion_strides.split(",")
                if value.strip()
            )
        self.generative_motion_strides = tuple(
            sorted({int(value) for value in raw_motion_strides})
        )
        self.generative_motion_signal_threshold = float(
            alignment_dict.pop("generative_motion_signal_threshold", 0.1)
        )
        self.generative_motion_magnitude_weight = float(
            alignment_dict.pop("generative_motion_magnitude_weight", 0.25)
        )
        self.generative_motion_under_weight = float(
            alignment_dict.pop("generative_motion_under_weight", 1.25)
        )
        self.generative_motion_camera_boost = float(
            alignment_dict.pop("generative_motion_camera_boost", 0.5)
        )
        self.generative_motion_max_grad_ratio = float(
            alignment_dict.pop("generative_motion_max_grad_ratio", 0.05)
        )
        self.generative_motion_gradient_probe_scale = float(
            alignment_dict.pop(
                "generative_motion_gradient_probe_scale",
                65536.0,
            )
        )
        self.strict_generative_motion = bool(
            alignment_dict.pop("strict_generative_motion", True)
        )
        raw_generative_teacher = alignment_dict.pop(
            "generative_teacher_checkpoint",
            None,
        )
        self.generative_teacher_checkpoint = (
            None
            if raw_generative_teacher in (None, "", "null")
            else str(raw_generative_teacher)
        )
        default_generative_prefixes = (
            "model.mid_blocks.8",
            "model.mid_blocks.9",
            "model.mid_blocks.10",
            "model.mid_blocks.11",
            "model.up_blocks.0",
            "model.up_blocks.1",
            "model.project_output",
        )
        raw_generative_prefixes = alignment_dict.pop(
            "generative_diffusion_trainable_prefixes",
            default_generative_prefixes,
        )
        if isinstance(raw_generative_prefixes, str):
            raw_generative_prefixes = tuple(
                value
                for value in raw_generative_prefixes.split(",")
                if value
            )
        self.generative_diffusion_trainable_prefixes = tuple(
            str(value).strip().rstrip(".")
            for value in raw_generative_prefixes
        )
        self.generative_shuffle_diagnostic_freq = max(
            0,
            int(
                alignment_dict.pop(
                    "generative_shuffle_diagnostic_freq",
                    20,
                )
                or 0
            ),
        )
        self.generative_unit_enabled = self.generative_unit_mode != "off"
        # A motion-calibration continuation keeps the learned UPCAST state in the
        # checkpoint but skips PAGE/RVQ and a teacher that failed its utility
        # audit. Only the bounded motion/context/prior objectives update DFoT.
        self.generative_conditioning_enabled = self.generative_unit_mode in {
            "adapter_warmup",
            "conditional_teacher",
            "student_distill",
        }
        if self.generative_unit_enabled:
            if self.generative_unit_lr <= 0:
                raise ValueError("generative_unit_lr must be positive")
            if self.generative_diffusion_lr <= 0:
                raise ValueError("generative_diffusion_lr must be positive")
            if min(
                self.generative_video_fm_weight,
                self.generative_conditional_fm_weight,
                self.generative_baseline_rank_weight,
                self.generative_shuffle_rank_weight,
                self.generative_shuffle_identity_weight,
                self.generative_pair_gate_rank_weight,
                self.generative_pair_gate_calibration_weight,
                self.generative_physical_velocity_weight,
                self.generative_advantage_distill_weight,
            ) < 0:
                raise ValueError("Physical conditioner loss weights must be non-negative")
            if self.generative_rank_margin < 0:
                raise ValueError("generative_rank_margin must be non-negative")
            if self.generative_pair_gate_margin < 0:
                raise ValueError(
                    "generative_pair_gate_margin must be non-negative"
                )
            if self.generative_fm_error_floor <= 0:
                raise ValueError("generative_fm_error_floor must be positive")
            if not 0.0 <= self.generative_teacher_initial_gate < 1.0:
                raise ValueError(
                    "generative_teacher_initial_gate must be in [0, 1)"
                )
            if not 0.0 <= self.generative_teacher_quality_ema_decay < 1.0:
                raise ValueError(
                    "generative_teacher_quality_ema_decay must be in [0, 1)"
                )
            if not 0.0 <= self.generative_teacher_min_win_fraction <= 1.0:
                raise ValueError(
                    "generative_teacher_min_win_fraction must be in [0, 1]"
                )
            if not (
                0.0
                <= self.generative_teacher_min_condition_win_fraction
                <= 1.0
            ):
                raise ValueError(
                    "generative_teacher_min_condition_win_fraction must be "
                    "in [0, 1]"
                )
            if self.generative_teacher_min_samples < 1:
                raise ValueError("generative_teacher_min_samples must be positive")
            if self.generative_max_consecutive_amp_skips < 1:
                raise ValueError(
                    "generative_max_consecutive_amp_skips must be positive"
                )
            if self.generative_advantage_max <= 0:
                raise ValueError("generative_advantage_max must be positive")
            if not (
                0.0
                <= self.generative_distill_signal_min
                < self.generative_distill_signal_max
                <= 1.0
            ):
                raise ValueError(
                    "Generative distillation signal range must satisfy "
                    "0 <= min < max <= 1"
                )
            if self.generative_distill_motion_boost < 0:
                raise ValueError(
                    "generative_distill_motion_boost must be non-negative"
                )
            if self.generative_motion_loss_weight < 0:
                raise ValueError(
                    "generative_motion_loss_weight must be non-negative"
                )
            if self.generative_motion_spatial_grid < 1:
                raise ValueError(
                    "generative_motion_spatial_grid must be positive"
                )
            if not self.generative_motion_strides or any(
                stride < 1 for stride in self.generative_motion_strides
            ):
                raise ValueError(
                    "generative_motion_strides must contain positive integers"
                )
            if not 0.0 <= self.generative_motion_signal_threshold < 1.0:
                raise ValueError(
                    "generative_motion_signal_threshold must be in [0,1)"
                )
            if min(
                self.generative_motion_magnitude_weight,
                self.generative_motion_under_weight,
                self.generative_motion_camera_boost,
                self.generative_motion_max_grad_ratio,
            ) < 0:
                raise ValueError(
                    "Generative motion weights must be non-negative"
                )
            if self.generative_motion_gradient_probe_scale <= 0:
                raise ValueError(
                    "generative_motion_gradient_probe_scale must be positive"
                )
            if (
                self.generative_unit_mode == "student_distill"
                and self.generative_teacher_checkpoint is None
            ):
                raise ValueError(
                    "student_distill requires generative_teacher_checkpoint"
                )
        object.__setattr__(self, "_generative_teacher_tail", None)
        object.__setattr__(self, "_generative_teacher_conditioner", None)
        self._generative_teacher_failed = False
        self._generative_tail_replay_validated = False
        self._generative_tail_replay_mean_error = 0.0
        self._generative_tail_replay_max_error = 0.0
        self._generative_teacher_gate_initialized = False
        self._generative_teacher_quality = {
            "samples": 0.0,
            "relative_gain_sum": 0.0,
            "teacher_wins": 0.0,
            "shuffle_gain_sum": 0.0,
            "condition_wins": 0.0,
            "ema_relative_gain": 0.0,
            "ema_win_fraction": 0.0,
            "ema_shuffle_gain": 0.0,
            "ema_condition_win_fraction": 0.0,
            "ema_initialized": False,
        }
        self._latest_generative_unit_metrics: dict[str, Tensor] = {}
        self._generative_nonfinite_gradient_pending = False
        self._generative_nonfinite_gradient_detail = ""
        self._generative_amp_skip_streak = 0
        self.unit_inference_enable = bool(
            alignment_dict.pop("unit_inference_enable", False)
        )
        self.unit_video_only_inference = bool(
            alignment_dict.get("unit_video_only_inference", False)
        )
        self.unit_migrate_from_version = alignment_dict.pop(
            "unit_migrate_from_version",
            None,
        )
        if self.unit_migrate_from_version in ("", "null"):
            self.unit_migrate_from_version = None
        elif self.unit_migrate_from_version is not None:
            self.unit_migrate_from_version = int(self.unit_migrate_from_version)
        full_teacher_checkpoint = alignment_dict.pop(
            "unit_full_teacher_checkpoint",
            None,
        )
        self.unit_full_teacher_checkpoint = (
            None
            if full_teacher_checkpoint in (None, "", "null")
            else str(full_teacher_checkpoint)
        )
        self.world_state_runtime_scale = float(
            alignment_dict.pop("world_state_runtime_scale", 1.0)
        )
        self.world_state_runtime_branch_mode = str(
            alignment_dict.pop("world_state_runtime_branch_mode", "adaptive")
            or "adaptive"
        ).lower()
        if self.world_state_runtime_branch_mode not in {
            "adaptive",
            "full",
            "physical",
            "appearance",
            "none",
        }:
            raise ValueError(
                "world_state_runtime_branch_mode must be adaptive, full, "
                "physical, appearance, or none"
            )
        self.world_state_lr = float(
            alignment_dict.pop("world_state_lr", 1.0e-5)
        )
        self.world_state_training_mode = str(
            alignment_dict.pop("world_state_training_mode", "joint") or "joint"
        ).lower()
        if self.world_state_training_mode not in {
            "joint",
            "adapter_only",
            "gate_only",
            "selective_joint",
        }:
            raise ValueError(
                "world_state_training_mode must be joint, adapter_only, "
                "gate_only, or selective_joint"
            )
        default_selective_prefixes = (
            "model.external_cond_embedding",
            "model.down_blocks.1",
            "model.down_blocks.2",
            "model.mid_blocks.8",
            "model.mid_blocks.9",
            "model.mid_blocks.10",
            "model.mid_blocks.11",
            "model.up_blocks.0",
            "model.up_blocks.1",
            "model.project_output",
            "model.world_state_adapter",
        )
        raw_selective_prefixes = alignment_dict.pop(
            "selective_diffusion_trainable_prefixes",
            default_selective_prefixes,
        )
        if isinstance(raw_selective_prefixes, str):
            raw_selective_prefixes = tuple(
                value for value in raw_selective_prefixes.split(",") if value
            )
        self.selective_diffusion_trainable_prefixes = tuple(
            str(value).strip().rstrip(".") for value in raw_selective_prefixes
        )
        if (
            self.world_state_training_mode == "selective_joint"
            and not self.selective_diffusion_trainable_prefixes
        ):
            raise ValueError(
                "selective_joint requires selective_diffusion_trainable_prefixes"
            )
        self.unit_allow_curriculum_override = bool(
            alignment_dict.pop("unit_allow_curriculum_override", False)
        )
        self.world_state_require_rollout_reads = bool(
            alignment_dict.pop("world_state_require_rollout_reads", False)
        )
        self.world_state_initialize_from_context = bool(
            alignment_dict.pop("world_state_initialize_from_context", False)
        )
        self.world_state_context_min_frames = max(
            2,
            int(alignment_dict.pop("world_state_context_min_frames", 2) or 2),
        )
        self.world_state_context_max_frames = max(
            self.world_state_context_min_frames,
            int(alignment_dict.pop("world_state_context_max_frames", 4) or 4),
        )
        if self.world_state_runtime_scale < 0:
            raise ValueError("world_state_runtime_scale must be non-negative")
        object.__setattr__(self, "_rollout_world_state", None)
        self._latest_world_state_metrics: dict[str, Tensor] = {}
        self._world_state_read_count = 0
        self._world_state_write_count = 0
        self.bootstrap_alignment_coeff = float(self.alignment_coeff)
        self.unit_bootstrap_steps = int(
            alignment_dict.pop("unit_bootstrap_steps", 0)
        )
        self.unit_geometry_tokenizer_steps = max(
            0,
            int(alignment_dict.pop("unit_geometry_tokenizer_steps", 0)),
        )
        if self.unit_geometry_tokenizer_steps > self.unit_bootstrap_steps:
            raise ValueError(
                "unit_geometry_tokenizer_steps cannot exceed "
                f"unit_bootstrap_steps ({self.unit_geometry_tokenizer_steps} > "
                f"{self.unit_bootstrap_steps})"
            )
        self.unit_health_check_start = max(
            self.unit_geometry_tokenizer_steps,
            int(alignment_dict.pop("unit_health_check_start", 150)),
        )
        self.unit_health_check_patience = max(
            1,
            int(alignment_dict.pop("unit_health_check_patience", 20)),
        )
        self.unit_health_min_perplexity = float(
            alignment_dict.pop("unit_health_min_perplexity", 1.5)
        )
        self.unit_health_min_projected_std = float(
            alignment_dict.pop("unit_health_min_projected_std", 0.01)
        )
        self._unit_health_last_step = -1
        self._unit_health_unhealthy_steps = 0
        self.joint_alignment_coeff = float(
            alignment_dict.pop("unit_joint_alignment_coeff", self.alignment_coeff)
        )
        self.unit_joint_ramp_steps = max(
            0,
            int(alignment_dict.pop("unit_joint_ramp_steps", 0)),
        )
        self.unit_freeze_after_bootstrap = bool(
            alignment_dict.pop("unit_freeze_after_bootstrap", True)
        )
        self.unit_cotrain_start_step = int(
            alignment_dict.pop("unit_cotrain_start_step", -1)
        )
        self.unit_cotrain_interval = max(
            1,
            int(alignment_dict.pop("unit_cotrain_interval", 4) or 4),
        )
        self.unit_cotrain_lr = float(
            alignment_dict.pop("unit_cotrain_lr", 1e-6)
        )
        if (
            self.unit_cotrain_start_step >= 0
            and self.unit_cotrain_start_step < self.unit_bootstrap_steps
        ):
            raise ValueError(
                "unit_cotrain_start_step must not precede unit_bootstrap_steps "
                f"({self.unit_cotrain_start_step} < {self.unit_bootstrap_steps})"
            )
        if self.unit_cotrain_lr <= 0:
            raise ValueError("unit_cotrain_lr must be positive")
        self._unit_cotrain_lr_applied = False
        self.compute_generation_loss = bool(alignment_dict.pop("compute_generation_loss", True))
        self.generation_loss_freq = int(alignment_dict.pop("generation_loss_freq", 1) or 1)
        self.generation_loss_phase = int(
            alignment_dict.pop("generation_loss_phase", 0) or 0
        ) % max(self.generation_loss_freq, 1)
        self.generation_loss_frequency_compensation = bool(
            alignment_dict.pop("generation_loss_frequency_compensation", True)
        )
        self.strict_generation_protection = bool(
            alignment_dict.pop("strict_generation_protection", True)
        )
        self._warned_alignment_generation_loss_failed = False
        self._latest_generation_expected = None
        self.video_prior_preservation_weight = float(
            alignment_dict.pop(
                "video_prior_preservation_weight",
                alignment_dict.pop("preservation_loss_weight", 0.0),
            )
        )
        self.video_prior_cpu_offload = bool(
            alignment_dict.pop("video_prior_cpu_offload", False)
        )
        self.video_prior_mode = str(
            alignment_dict.pop("video_prior_mode", "full_model") or "full_model"
        ).lower()
        if self.video_prior_mode not in {"full_model", "selected_functional"}:
            raise ValueError(
                "video_prior_mode must be full_model or selected_functional"
            )
        if (
            self.video_prior_mode == "selected_functional"
            and self.world_state_training_mode != "selective_joint"
        ):
            raise ValueError(
                "selected_functional video prior requires selective_joint training"
            )
        prior_checkpoint = alignment_dict.pop("video_prior_checkpoint", None)
        self.video_prior_checkpoint = (
            None if prior_checkpoint in (None, "", "null") else str(prior_checkpoint)
        )
        self.video_prior_preservation_freq = max(
            1,
            int(alignment_dict.pop("video_prior_preservation_freq", 1) or 1),
        )
        self.video_prior_preservation_phase = int(
            alignment_dict.pop("video_prior_preservation_phase", 0) or 0
        ) % self.video_prior_preservation_freq
        self.video_prior_frequency_compensation = bool(
            alignment_dict.pop("video_prior_frequency_compensation", True)
        )
        self.strict_video_prior = bool(alignment_dict.pop("strict_video_prior", True))
        self.generated_context_exposure_weight = float(
            alignment_dict.pop("generated_context_exposure_weight", 0.0)
        )
        self.rollout_training_enable = bool(
            alignment_dict.pop("rollout_training_enable", False)
        )
        self.rollout_training_start_step = int(
            alignment_dict.pop("rollout_training_start_step", -1)
        )
        self.rollout_training_freq = max(
            1,
            int(alignment_dict.pop("rollout_training_freq", 1) or 1),
        )
        self.rollout_training_phase = int(
            alignment_dict.pop("rollout_training_phase", 0) or 0
        ) % self.rollout_training_freq
        self.rollout_sampling_steps = max(
            1,
            int(alignment_dict.pop("rollout_sampling_steps", 8) or 8),
        )
        self.rollout_overlap_frames = max(
            1,
            int(alignment_dict.pop("rollout_overlap_frames", 4) or 4),
        )
        self.rollout_initial_context_frames = max(
            1,
            int(alignment_dict.pop("rollout_initial_context_frames", 1) or 1),
        )
        self.rollout_training_frames = max(
            self.max_tokens,
            int(alignment_dict.pop("rollout_training_frames", 28) or 28),
        )
        self.rollout_num_windows = int(
            alignment_dict.pop("rollout_num_windows", 2) or 2
        )
        raw_horizon_probabilities = alignment_dict.pop(
            "rollout_horizon_probabilities",
            None,
        )
        # Retain the three-window fields as a config fallback so old scripts
        # remain loadable. New runs should provide the complete horizon
        # distribution explicitly.
        self.rollout_three_window_start_step = int(
            alignment_dict.pop("rollout_three_window_start_step", -1)
        )
        self.rollout_three_window_probability = float(
            alignment_dict.pop("rollout_three_window_probability", 0.0)
        )
        self.rollout_generated_mix_ramp_steps = max(
            1,
            int(
                alignment_dict.pop(
                    "rollout_generated_mix_ramp_steps",
                    100,
                )
                or 100
            ),
        )
        self.rollout_generated_mix_max_prob = float(
            alignment_dict.pop("rollout_generated_mix_max_prob", 0.75)
        )
        self.rollout_generated_mix_min_prob = float(
            alignment_dict.pop("rollout_generated_mix_min_prob", 0.0)
        )
        self.rollout_boundary_temporal_weight = float(
            alignment_dict.pop("rollout_boundary_temporal_weight", 0.02)
        )
        self.rollout_boundary_color_weight = float(
            alignment_dict.pop("rollout_boundary_color_weight", 0.005)
        )
        self.rollout_trajectory_weight = float(
            alignment_dict.pop("rollout_trajectory_weight", 0.0)
        )
        self.rollout_state_regret_weight = float(
            alignment_dict.pop("rollout_state_regret_weight", 0.0)
        )
        self.rollout_adapter_diffusion_weight = float(
            alignment_dict.pop("rollout_adapter_diffusion_weight", 1.0)
        )
        self.rollout_sampled_preference_enable = bool(
            alignment_dict.pop("rollout_sampled_preference_enable", False)
        )
        self.rollout_sampled_preference_weight = float(
            alignment_dict.pop("rollout_sampled_preference_weight", 0.0)
        )
        self.rollout_sampled_preference_freq = max(
            1,
            int(
                alignment_dict.pop("rollout_sampled_preference_freq", 2)
                or 2
            ),
        )
        self.rollout_sampled_preference_phase = int(
            alignment_dict.pop("rollout_sampled_preference_phase", 0) or 0
        ) % self.rollout_sampled_preference_freq
        self.rollout_sampled_preference_frequency_compensation = bool(
            alignment_dict.pop(
                "rollout_sampled_preference_frequency_compensation",
                True,
            )
        )
        self.rollout_sampled_preference_temperature = float(
            alignment_dict.pop("rollout_sampled_preference_temperature", 0.25)
        )
        self.rollout_sampled_preference_num_windows = max(
            1,
            int(
                alignment_dict.pop(
                    "rollout_sampled_preference_num_windows",
                    1,
                )
                or 1
            ),
        )
        self.rollout_preference_context_init_probability = float(
            alignment_dict.pop(
                "rollout_preference_context_init_probability",
                0.0,
            )
        )
        self.rollout_preference_relative_floor = float(
            alignment_dict.pop("rollout_preference_relative_floor", 1.0e-3)
        )
        self.rollout_preference_absolute_floor = float(
            alignment_dict.pop("rollout_preference_absolute_floor", 1.0e-6)
        )
        self.rollout_preference_confidence_temperature = float(
            alignment_dict.pop(
                "rollout_preference_confidence_temperature",
                0.25,
            )
        )
        self.rollout_preference_identity_weight = float(
            alignment_dict.pop("rollout_preference_identity_weight", 0.05)
        )
        self.rollout_preference_regression_beta = float(
            alignment_dict.pop("rollout_preference_regression_beta", 0.05)
        )
        self.rollout_preference_noise_repeats = max(
            1,
            int(alignment_dict.pop("rollout_preference_noise_repeats", 1) or 1),
        )
        self.rollout_preference_holdout_fraction = float(
            alignment_dict.pop("rollout_preference_holdout_fraction", 0.2)
        )
        self.rollout_preference_cache_mode = str(
            alignment_dict.pop("rollout_preference_cache_mode", "off") or "off"
        ).lower()
        self.rollout_preference_cache_collection_only = bool(
            alignment_dict.pop(
                "rollout_preference_cache_collection_only",
                False,
            )
        )
        _validate_world_state_learning_rate(
            self.world_state_lr,
            collection_only=self.rollout_preference_cache_collection_only,
        )
        raw_preference_cache_dir = alignment_dict.pop(
            "rollout_preference_cache_dir",
            None,
        )
        self.rollout_preference_cache_dir = (
            None
            if raw_preference_cache_dir in (None, "", "null")
            else str(raw_preference_cache_dir)
        )
        self.rollout_preference_cache_namespace = str(
            alignment_dict.pop(
                "rollout_preference_cache_namespace",
                "upcast-state",
            )
            or "upcast-state"
        )
        self.rollout_preference_source_id = str(
            alignment_dict.pop("rollout_preference_source_id", "unknown")
            or "unknown"
        )
        self.rollout_preference_predecessor_branch_mode = str(
            alignment_dict.pop(
                "rollout_preference_predecessor_branch_mode",
                "full",
            )
            or "full"
        ).lower()
        self.rollout_preference_min_validation_correlation = float(
            alignment_dict.pop(
                "rollout_preference_min_validation_correlation",
                0.2,
            )
        )
        self.allow_unvalidated_preference_policy = bool(
            alignment_dict.pop(
                "allow_unvalidated_preference_policy",
                False,
            )
        )
        self.world_state_preference_migrate_from_version = alignment_dict.pop(
            "world_state_preference_migrate_from_version",
            None,
        )
        if self.world_state_preference_migrate_from_version in ("", "null"):
            self.world_state_preference_migrate_from_version = None
        elif self.world_state_preference_migrate_from_version is not None:
            self.world_state_preference_migrate_from_version = int(
                self.world_state_preference_migrate_from_version
            )
        self.rollout_preference_state_weight = float(
            alignment_dict.pop("rollout_preference_state_weight", 0.25)
        )
        self.rollout_preference_dino_weight = float(
            alignment_dict.pop("rollout_preference_dino_weight", 0.30)
        )
        self.rollout_preference_vjepa_temporal_weight = float(
            alignment_dict.pop(
                "rollout_preference_vjepa_temporal_weight",
                0.35,
            )
        )
        self.rollout_preference_vjepa_spatial_weight = float(
            alignment_dict.pop(
                "rollout_preference_vjepa_spatial_weight",
                0.15,
            )
        )
        self.rollout_preference_physical_weight = float(
            alignment_dict.pop("rollout_preference_physical_weight", 0.20)
        )
        self.strict_rollout_sampled_preference = bool(
            alignment_dict.pop("strict_rollout_sampled_preference", True)
        )
        self.rollout_horizon_power = float(
            alignment_dict.pop("rollout_horizon_power", 1.0)
        )
        self.rollout_boundary_signal_threshold = float(
            alignment_dict.pop("rollout_boundary_signal_threshold", 0.1)
        )
        self.rollout_boundary_max_grad_ratio = float(
            alignment_dict.pop("rollout_boundary_max_grad_ratio", 0.1)
        )
        self.rollout_gradient_probe_scale = float(
            alignment_dict.pop("rollout_gradient_probe_scale", 65536.0)
        )
        self.rollout_frequency_compensation = bool(
            alignment_dict.pop("rollout_frequency_compensation", True)
        )
        self.rollout_skip_alignment = bool(
            alignment_dict.pop("rollout_skip_alignment", True)
        )
        self.rollout_adapter_only_gradients = bool(
            alignment_dict.pop("rollout_adapter_only_gradients", False)
        )
        self.strict_rollout_training = bool(
            alignment_dict.pop("strict_rollout_training", True)
        )
        if not 2 <= self.rollout_num_windows <= 5:
            raise ValueError("rollout_num_windows must be in [2,5]")
        num_horizons = self.rollout_num_windows - 1
        if raw_horizon_probabilities in (None, "", "null"):
            if self.rollout_num_windows == 3:
                long_probability = self.rollout_three_window_probability
                raw_horizon_probabilities = [
                    1.0 - long_probability,
                    long_probability,
                ]
                self.rollout_horizon_start_step = (
                    self.rollout_three_window_start_step
                )
            else:
                raw_horizon_probabilities = [1.0 / num_horizons] * num_horizons
                self.rollout_horizon_start_step = self.rollout_training_start_step
        else:
            self.rollout_horizon_start_step = self.rollout_training_start_step
        if not isinstance(raw_horizon_probabilities, (list, tuple)):
            raise ValueError("rollout_horizon_probabilities must be a list")
        if len(raw_horizon_probabilities) != num_horizons:
            raise ValueError(
                "rollout_horizon_probabilities must contain one probability "
                f"for each target horizon 1..{num_horizons}, got "
                f"{len(raw_horizon_probabilities)} values"
            )
        horizon_probabilities = tuple(
            float(value) for value in raw_horizon_probabilities
        )
        if any(value < 0 for value in horizon_probabilities):
            raise ValueError("rollout horizon probabilities must be non-negative")
        probability_sum = sum(horizon_probabilities)
        if probability_sum <= 0:
            raise ValueError("rollout horizon probabilities must have positive mass")
        self.rollout_horizon_probabilities = tuple(
            value / probability_sum for value in horizon_probabilities
        )
        if self.rollout_initial_context_frames > self.rollout_overlap_frames:
            raise ValueError(
                "rollout_initial_context_frames cannot exceed "
                "rollout_overlap_frames"
            )
        rollout_stride = self.max_tokens - self.rollout_overlap_frames
        required_rollout_frames = self.max_tokens + (
            self.rollout_num_windows - 1
        ) * rollout_stride
        if self.rollout_training_enable:
            if not self.unit_inference_enable:
                raise ValueError(
                    "rollout_training_enable requires unit_inference_enable=true"
                )
            if self.rollout_training_frames != required_rollout_frames:
                raise ValueError(
                    f"{self.rollout_num_windows}-window rollout requires "
                    "training_frames=max_tokens+(num_windows-1)*(max_tokens-overlap), "
                    f"got {self.rollout_training_frames} instead of {required_rollout_frames}"
                )
            if not 0.0 <= self.rollout_generated_mix_min_prob <= 1.0:
                raise ValueError("rollout_generated_mix_min_prob must be in [0,1]")
            if not 0.0 <= self.rollout_generated_mix_max_prob <= 1.0:
                raise ValueError("rollout_generated_mix_max_prob must be in [0,1]")
            if (
                self.rollout_generated_mix_min_prob
                > self.rollout_generated_mix_max_prob
            ):
                raise ValueError(
                    "rollout_generated_mix_min_prob cannot exceed max_prob"
                )
            if self.rollout_sampling_steps > self.sampling_timesteps:
                raise ValueError(
                    "rollout_sampling_steps cannot exceed normal sampling_timesteps"
                )
            if not 0.0 <= self.rollout_three_window_probability <= 1.0:
                raise ValueError("rollout_three_window_probability must be in [0,1]")
            if not 0.0 <= self.rollout_boundary_signal_threshold < 1.0:
                raise ValueError("rollout_boundary_signal_threshold must be in [0,1)")
            if self.rollout_boundary_max_grad_ratio < 0:
                raise ValueError("rollout_boundary_max_grad_ratio must be non-negative")
            if self.rollout_gradient_probe_scale < 1.0:
                raise ValueError("rollout_gradient_probe_scale must be at least 1")
            if self.rollout_trajectory_weight < 0:
                raise ValueError("rollout_trajectory_weight must be non-negative")
            if self.rollout_state_regret_weight < 0:
                raise ValueError("rollout_state_regret_weight must be non-negative")
            if self.rollout_adapter_diffusion_weight < 0:
                raise ValueError(
                    "rollout_adapter_diffusion_weight must be non-negative"
                )
            if (
                self.rollout_state_regret_weight > 0
                and self.world_state_training_mode
                not in {"adapter_only", "gate_only"}
            ):
                raise ValueError(
                    "rollout_state_regret_weight requires adapter_only or gate_only "
                    "training so the no-state branch remains an immutable baseline"
                )
            if self.rollout_sampled_preference_weight < 0:
                raise ValueError(
                    "rollout_sampled_preference_weight must be non-negative"
                )
            if self.rollout_sampled_preference_temperature <= 0:
                raise ValueError(
                    "rollout_sampled_preference_temperature must be positive"
                )
            if not 1 <= self.rollout_sampled_preference_num_windows <= self.rollout_num_windows:
                raise ValueError(
                    "rollout_sampled_preference_num_windows must be in "
                    f"[1,{self.rollout_num_windows}]"
                )
            if not 0.0 <= self.rollout_preference_context_init_probability < 1.0:
                raise ValueError(
                    "rollout_preference_context_init_probability must be in [0,1)"
                )
            if self.rollout_preference_relative_floor <= 0:
                raise ValueError("rollout_preference_relative_floor must be positive")
            if self.rollout_preference_absolute_floor <= 0:
                raise ValueError("rollout_preference_absolute_floor must be positive")
            if self.rollout_preference_confidence_temperature <= 0:
                raise ValueError(
                    "rollout_preference_confidence_temperature must be positive"
                )
            if self.rollout_preference_identity_weight < 0:
                raise ValueError(
                    "rollout_preference_identity_weight must be non-negative"
                )
            if self.rollout_preference_regression_beta <= 0:
                raise ValueError(
                    "rollout_preference_regression_beta must be positive"
                )
            if not 0.0 <= self.rollout_preference_holdout_fraction < 1.0:
                raise ValueError(
                    "rollout_preference_holdout_fraction must be in [0,1)"
                )
            if self.rollout_preference_cache_mode not in {
                "off",
                "read",
                "write",
                "read_write",
            }:
                raise ValueError(
                    "rollout_preference_cache_mode must be off, read, write, "
                    "or read_write"
                )
            if (
                self.rollout_preference_cache_mode != "off"
                and self.rollout_preference_cache_dir is None
            ):
                raise ValueError(
                    "A preference cache mode requires rollout_preference_cache_dir"
                )
            if self.rollout_preference_predecessor_branch_mode not in {
                "none",
                "physical",
                "appearance",
                "full",
            }:
                raise ValueError(
                    "rollout_preference_predecessor_branch_mode must be a fixed "
                    "branch (none, physical, appearance, or full)"
                )
            if (
                self.rollout_preference_cache_mode != "off"
                and self.rollout_preference_source_id == "unknown"
            ):
                raise ValueError(
                    "A preference cache requires rollout_preference_source_id so "
                    "utilities cannot be reused across generator checkpoints"
                )
            if self.rollout_preference_cache_collection_only and (
                self.rollout_preference_cache_mode not in {"write", "read_write"}
                or not self.rollout_sampled_preference_enable
            ):
                raise ValueError(
                    "rollout_preference_cache_collection_only requires an enabled "
                    "sampled preference writer"
                )
            if not -1.0 <= self.rollout_preference_min_validation_correlation <= 1.0:
                raise ValueError(
                    "rollout_preference_min_validation_correlation must be in [-1,1]"
                )
            if self.rollout_preference_state_weight < 0:
                raise ValueError("rollout_preference_state_weight must be non-negative")
            preference_weights = (
                self.rollout_preference_dino_weight,
                self.rollout_preference_vjepa_temporal_weight,
                self.rollout_preference_vjepa_spatial_weight,
                self.rollout_preference_physical_weight,
            )
            if any(weight < 0 for weight in preference_weights) or sum(
                preference_weights
            ) <= 0:
                raise ValueError(
                    "Sampled-preference component weights must be non-negative "
                    "with positive total mass"
                )
            if (
                self.rollout_sampled_preference_enable
                and self.world_state_training_mode != "gate_only"
            ):
                raise ValueError(
                    "rollout_sampled_preference_enable requires gate_only training"
                )
            if (
                self.rollout_sampled_preference_enable
                and self.rollout_sampled_preference_weight <= 0
            ):
                raise ValueError(
                    "Enabled sampled preference requires a positive loss weight"
                )
            if (
                self.rollout_preference_context_init_probability > 0
                and self.rollout_initial_context_frames
                < self.world_state_context_min_frames
            ):
                raise ValueError(
                    "Context-initialized preference requires at least "
                    "world_state_context_min_frames observed rollout frames"
                )
            max_preference_start = (
                self.rollout_num_windows
                - self.rollout_sampled_preference_num_windows
            )
            unavailable_mass = sum(
                probability
                for horizon, probability in enumerate(
                    self.rollout_horizon_probabilities,
                    start=1,
                )
                if horizon > max_preference_start
            )
            if (
                self.rollout_sampled_preference_enable
                and unavailable_mass > 1.0e-8
            ):
                raise ValueError(
                    "rollout_horizon_probabilities assigns mass to a target "
                    "without enough future windows for cumulative preference; "
                    f"maximum target is {max_preference_start}"
                )
            if self.rollout_horizon_power < 0:
                raise ValueError("rollout_horizon_power must be non-negative")
            if (
                self.rollout_adapter_only_gradients
                and self.world_state_training_mode != "selective_joint"
            ):
                raise ValueError(
                    "rollout_adapter_only_gradients requires selective_joint training"
                )
        self._latest_rollout_metrics: dict[str, Tensor] = {}
        self._preference_cache = (
            None
            if self.rollout_preference_cache_mode == "off"
            else CounterfactualUtilityCache(
                self.rollout_preference_cache_dir,
                self.rollout_preference_cache_namespace,
            )
        )
        self._rollout_horizon_step_metrics: dict[int, dict[str, Tensor | int]] = {}
        self._rollout_horizon_log_metrics: dict[int, dict[str, Tensor | int]] = {}
        self._rollout_horizon_total_counts: dict[int, int] = {}
        self._suppress_selective_diffusion_gradients = False
        self._selective_rollout_gradient_hooks = []
        self.generated_context_exposure_start_step = int(
            alignment_dict.pop("generated_context_exposure_start_step", -1)
        )
        self.generated_context_exposure_freq = max(
            1,
            int(alignment_dict.pop("generated_context_exposure_freq", 1) or 1),
        )
        self.generated_context_exposure_phase = int(
            alignment_dict.pop("generated_context_exposure_phase", 0) or 0
        ) % self.generated_context_exposure_freq
        self.generated_context_frequency_compensation = bool(
            alignment_dict.pop(
                "generated_context_frequency_compensation",
                True,
            )
        )
        self.generated_context_num_frames = max(
            1,
            int(alignment_dict.pop("generated_context_num_frames", 4) or 4),
        )
        self.generated_context_blend = float(
            alignment_dict.pop("generated_context_blend", 0.5)
        )
        self.generated_context_max_residual = float(
            alignment_dict.pop("generated_context_max_residual", 1.0)
        )
        self.generated_context_min_signal = float(
            alignment_dict.pop("generated_context_min_signal", 0.1)
        )
        self.generated_context_activation_checkpointing = bool(
            alignment_dict.pop(
                "generated_context_activation_checkpointing",
                True,
            )
        )
        self.generated_context_skip_alignment = bool(
            alignment_dict.pop("generated_context_skip_alignment", True)
        )
        self.generated_context_skip_base_diffusion = bool(
            alignment_dict.pop(
                "generated_context_skip_base_diffusion",
                True,
            )
        )
        self.strict_generated_context_exposure = bool(
            alignment_dict.pop("strict_generated_context_exposure", True)
        )
        if self.generated_context_exposure_weight < 0:
            raise ValueError("generated_context_exposure_weight must be non-negative")
        if not 0.0 <= self.generated_context_blend <= 1.0:
            raise ValueError("generated_context_blend must be in [0,1]")
        if self.generated_context_max_residual <= 0:
            raise ValueError("generated_context_max_residual must be positive")
        if not 0.0 <= self.generated_context_min_signal < 1.0:
            raise ValueError("generated_context_min_signal must be in [0,1)")
        if (
            self.generated_context_exposure_weight > 0
            and (
                self.generated_context_skip_alignment
                or self.generated_context_skip_base_diffusion
            )
            and self.generated_context_exposure_freq <= 1
        ):
            raise ValueError(
                "generated_context_exposure_freq must exceed 1 when exposure "
                "microbatches replace a dense training objective"
            )
        self._warned_generated_context_exposure_failed = False
        self._latest_generated_context_expected = None
        self._latest_generated_context_active_fraction = None
        self._latest_generated_context_residual_rms = None
        if (
            self.video_prior_preservation_weight > 0
            and self.strict_video_prior
            and self.video_prior_checkpoint is None
        ):
            raise ValueError(
                "A strict frozen-base prior requires alignment.video_prior_checkpoint"
            )
        object.__setattr__(self, "_video_prior_model", None)
        object.__setattr__(self, "_video_prior_selected_state", None)
        self._video_prior_snapshot_failed = False
        self._latest_video_prior_expected = None
        self._latest_video_prior_expected_weighted = None
        self._latest_video_prior_valid = None
        self.current_alignment_coeff = float(self.alignment_coeff)
        if self.unit_geometry_tokenizer_steps > 0:
            self._curriculum_phase = "geometry"
        elif self.unit_bootstrap_steps > 0:
            self._curriculum_phase = "alignment"
        else:
            self._curriculum_phase = "joint"
        # Optimizer-only keys are consumed by configure_optimizers and must not be
        # forwarded into the alignment module constructor.
        alignment_dict.pop("diffusion_lr", None)
        alignment_dict.pop("unit_lr", None)
        if (
            self.alignment_coeff > 0
            or self.unit_inference_enable
            or self.generative_unit_enabled
        ):
            if self.is_latent_diffusion:
                raise ValueError(
                    "UPCAST currently requires pixel-space DFoT so DFoT, "
                    "PAGE-4D, DINO, and V-JEPA receive the same RGB clip. A latent "
                    "backbone needs an explicit differentiable RGB decoder first."
                )
            if encoder_type == "vggt":
                self.vggt_alignment_loss = FactorizedTransitionLearning(**alignment_dict)
                if hasattr(self.vggt_alignment_loss, "set_video_tail_model"):
                    self.vggt_alignment_loss.set_video_tail_model(self.diffusion_model)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.requires_find_unused_parameters = bool(
            self.alignment_coeff > 0
            and (self.unit_bootstrap_steps > 0 or self.unit_inference_enable)
        )
        if self.generative_unit_enabled:
            capture_contract = (
                str(self.cfg.backbone.latent_return_stage).lower(),
                int(self.cfg.backbone.latent_return_level),
                tuple(int(value) for value in self.cfg.backbone.latent_return_blocks),
            )
            if capture_contract != ("down", 2, (-1,)):
                raise ValueError(
                    "Physical conditioner requires the transition-source interface "
                    "latent_return_stage=down, level=2, blocks=[-1]; got "
                    f"{capture_contract}"
                )
            if not bool(
                getattr(
                    getattr(self.cfg.backbone, "generative_unit", None),
                    "enable",
                    False,
                )
            ):
                raise ValueError(
                    "generative_unit_mode requires backbone.generative_unit.enable=true"
                )
            if bool(getattr(self.cfg.backbone.world_state, "enable", False)):
                raise ValueError(
                    "Physical conditioner continuation must disable the legacy "
                    "up-path world_state adapter"
                )
            if self.alignment_coeff != 0:
                raise ValueError(
                    "Physical conditioner continuation replaces the old decoder/alignment "
                    "objective and requires alignment_coeff=0"
                )
            if self.unit_video_only_inference:
                raise ValueError(
                    "Physical conditioner requires the complete PAGE/fusion UPCAST teacher"
                )
            if self.unit_bootstrap_steps != 0 or self.unit_geometry_tokenizer_steps != 0:
                raise ValueError(
                    "Physical conditioner is a continuation from a trained UPCAST and "
                    "requires bootstrap/tokenizer steps to be zero"
                )
            if self.rollout_training_enable:
                raise ValueError(
                    "Validate short-video generative distillation before enabling "
                    "persistent rollout training"
                )
            self.requires_find_unused_parameters = False
            self._set_generative_unit_trainable()
        elif self.world_state_training_mode == "selective_joint":
            if self.alignment_coeff <= 0:
                raise ValueError("selective_joint requires alignment_coeff > 0")
            if self.unit_bootstrap_steps != 0:
                raise ValueError(
                    "selective_joint is a continuation mode and requires "
                    "unit_bootstrap_steps=0"
                )
            if self.unit_video_only_inference:
                raise ValueError(
                    "selective_joint requires the complete UPCAST teacher"
                )
            self.requires_find_unused_parameters = True
            self._set_selective_joint_trainable()
        elif self.requires_find_unused_parameters:
            # DDP fixes its reducer parameter set when wrapping the module.
            # Keep DFoT registered, then suppress its bootstrap graph with
            # no_grad in training_step.
            self._set_diffusion_trainable(
                False,
                keep_ddp_registered=True,
            )
        if self.world_state_training_mode in {"adapter_only", "gate_only"}:
            mode_name = self.world_state_training_mode
            if not self.rollout_training_enable:
                raise ValueError(
                    f"{mode_name} world-state training requires rollout training"
                )
            if self.rollout_training_freq != 1:
                raise ValueError(
                    f"{mode_name} training requires rollout_training_freq=1 so every "
                    "microbatch has an adapter gradient"
                )
            if self.alignment_coeff != 0:
                raise ValueError(
                    f"{mode_name} training requires alignment_coeff=0"
                )
            if not self.unit_video_only_inference:
                raise ValueError(
                    f"{mode_name} training requires unit_video_only_inference=true"
                )
            self.requires_find_unused_parameters = False
            if mode_name == "gate_only":
                self._set_world_state_gate_only_trainable()
            else:
                self._set_world_state_adapter_only_trainable()

    def _world_state_backbone(self):
        diffusion_model = getattr(self.diffusion_model, "_orig_mod", self.diffusion_model)
        backbone = getattr(diffusion_model, "model", None)
        if backbone is None:
            raise RuntimeError("DFoT diffusion wrapper has no backbone model")
        return getattr(backbone, "_orig_mod", backbone)

    @torch.no_grad()
    def _camera_ray_anchors(
        self,
        conditions: Tensor,
        *,
        reference_origin: Tensor | None = None,
        reference_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Represent camera rays in one rollout-stable coordinate frame."""
        if conditions.ndim != 3 or conditions.shape[-1] != 16:
            raise ValueError(
                "Raw camera conditions must be [B,T,16], got "
                f"{tuple(conditions.shape)}"
            )
        camera = CameraPose.from_vectors(conditions.float())
        rays = camera.rays(resolution=8).to_tensor(use_plucker=False)
        origin = rays[..., :3]
        direction = F.normalize(rays[..., 3:], dim=-1)
        camera_centers = origin[:, :, 0, 0]
        if reference_origin is None:
            reference_origin = camera_centers[:, 0]
        else:
            reference_origin = reference_origin.to(origin).reshape(-1, 3)
        if reference_scale is None:
            displacement = camera_centers - reference_origin[:, None]
            reference_scale = displacement.norm(dim=-1).amax(dim=1, keepdim=True)
            reference_scale = reference_scale.clamp_min(1e-2)
        else:
            reference_scale = reference_scale.to(origin).reshape(-1, 1).clamp_min(1e-2)
        normalized_origin = (
            origin - reference_origin[:, None, None, None]
        ) / reference_scale[:, None, None, None]
        moment = torch.cross(normalized_origin, direction, dim=-1)
        anchors = torch.cat((direction, moment, normalized_origin), dim=-1)
        return (
            rearrange(anchors, "b t h w c -> b t (h w) c"),
            reference_origin,
            reference_scale,
        )

    def _anchor_world_observation(
        self,
        observation: dict[str, Tensor],
        conditions: Tensor,
        previous_state: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        reference_origin = (
            None if previous_state is None else previous_state.get("reference_origin")
        )
        reference_scale = (
            None if previous_state is None else previous_state.get("reference_scale")
        )
        anchors, reference_origin, reference_scale = self._camera_ray_anchors(
            conditions,
            reference_origin=reference_origin,
            reference_scale=reference_scale,
        )
        future_indices = observation["span_future_indices"].long()
        anchor_spans = anchors.index_select(1, future_indices)
        if anchor_spans.shape[2] != observation["physical_spans"].shape[2]:
            raise RuntimeError(
                "Camera anchor and UPCAST spatial grids differ: "
                f"{anchor_spans.shape[2]} != {observation['physical_spans'].shape[2]}"
            )

        indices = observation["indices"].long()
        modal_indices = torch.mode(indices, dim=1).values
        code_agreement = (
            indices == modal_indices[:, None]
        ).float().mean(dim=-1)
        commitment = observation["commitment_per_token"].float()
        commitment_scale = commitment.flatten(2).median(dim=-1).values
        commitment_scale = commitment_scale[:, :, None].clamp_min(1e-6)
        commitment_quality = torch.exp(-commitment / commitment_scale)
        confidence = (
            (0.25 + 0.75 * code_agreement)
            * (0.5 + 0.5 * commitment_quality)
        ).clamp(0.0, 1.0)

        anchored = dict(observation)
        anchored.update(
            {
                "anchor_spans": anchor_spans.to(observation["physical_spans"]),
                "confidence_spans": confidence[..., None].to(
                    observation["physical_spans"]
                ),
                "reference_origin": reference_origin,
                "reference_scale": reference_scale,
            }
        )
        return anchored

    def _world_state_query_anchors(
        self,
        conditions: Tensor,
        state: dict[str, Tensor] | None,
    ) -> Tensor | None:
        if state is None:
            return None
        anchors, _, _ = self._camera_ray_anchors(
            conditions,
            reference_origin=state["reference_origin"],
            reference_scale=state["reference_scale"],
        )
        return anchors

    @torch.no_grad()
    def _encode_observed_context_world_state(
        self,
        context: Tensor,
        conditions: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode only genuinely observed frames for the first-window state.

        The fixed-length U-ViT is filled only by repeating the last observed
        frame and pose; its padded outputs are discarded before UPCAST.  Thus no
        generated or future ground-truth frame enters the transition spans. A
        minimum of two observed frames is required because UPCAST is a
        transition tokenizer rather than a single-image appearance encoder.
        """
        num_observed = min(
            int(context.shape[1]),
            int(conditions.shape[1]),
            self.world_state_context_max_frames,
        )
        if num_observed < self.world_state_context_min_frames:
            raise ValueError(
                "Observed-context world state requires at least "
                f"{self.world_state_context_min_frames} frames, got {num_observed}"
            )
        observed_context = context[:, :num_observed]
        observed_conditions = conditions[:, :num_observed]
        padding = self.max_tokens - num_observed
        if padding < 0:
            raise ValueError(
                f"Observed context exceeds DFoT capacity: {num_observed} > {self.max_tokens}"
            )
        if padding > 0:
            backbone_context = torch.cat(
                (
                    observed_context,
                    observed_context[:, -1:].expand(-1, padding, -1, -1, -1),
                ),
                dim=1,
            )
            backbone_conditions = torch.cat(
                (
                    observed_conditions,
                    observed_conditions[:, -1:].expand(-1, padding, -1),
                ),
                dim=1,
            )
        else:
            backbone_context = observed_context
            backbone_conditions = observed_conditions
        self._world_state_backbone().clear_world_state_condition()
        latents, logsnr = self.diffusion_model.extract_clean_alignment_latents(
            backbone_context,
            self._process_conditions(backbone_conditions),
        )
        latents = [latent[:, :num_observed] for latent in latents]
        logsnr = logsnr[:, :num_observed]
        observation = self.vggt_alignment_loss.encode_video_world_observation(
            latents,
            logsnr,
        )
        observation = self._anchor_world_observation(
            observation,
            observed_conditions,
            previous_state=None,
        )
        state = self._update_world_state_from_observation(
            observation,
            previous_state=None,
            detach_previous=True,
        )
        return (
            {key: value.detach() for key, value in state.items()},
            {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in observation.items()
            },
        )

    def _world_state_begin_rollout(
        self,
        context: Tensor,
        conditions: Optional[Tensor],
    ) -> None:
        if not self.unit_inference_enable:
            return
        backbone = self._world_state_backbone()
        backbone.clear_world_state_condition()
        object.__setattr__(self, "_rollout_world_state", None)
        self._latest_world_state_metrics = {}
        self._world_state_read_count = 0
        self._world_state_write_count = 0
        self._world_state_context_init_count = 0
        self._world_state_read_diagnostic_sums: dict[str, Tensor] = {}
        self._world_state_read_diagnostic_count = 0
        if (
            self.world_state_initialize_from_context
            and conditions is not None
            and min(context.shape[1], conditions.shape[1])
            >= self.world_state_context_min_frames
        ):
            state, observation = self._encode_observed_context_world_state(
                context,
                conditions,
            )
            object.__setattr__(self, "_rollout_world_state", state)
            self._world_state_context_init_count = 1
            self._latest_world_state_metrics.update(
                {
                    "world_state_context_initialized": context.new_ones(()),
                    "world_state_context_frames": context.new_tensor(
                        float(
                            min(
                                context.shape[1],
                                conditions.shape[1],
                                self.world_state_context_max_frames,
                            )
                        )
                    ),
                    "world_state_num_spans": observation["num_spans"],
                    "world_state_span_code_agreement": observation[
                        "span_code_agreement"
                    ],
                }
            )
        # The evaluation script enables this only for direct long-video runs.
        # Do not infer the requested rollout length from the initial context:
        # single-image long generation necessarily starts with one frame.
        self._world_state_expect_reads = self.world_state_require_rollout_reads

    @torch.no_grad()
    def _accumulate_world_state_read_diagnostics(self) -> None:
        backbone = self._world_state_backbone()
        diagnostics = getattr(
            backbone,
            "world_state_window_diagnostics",
            backbone.world_state_diagnostics,
        )
        if not diagnostics:
            return
        for name, value in diagnostics.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            value = value.detach().float()
            previous = self._world_state_read_diagnostic_sums.get(name)
            if previous is None:
                aggregate = value
            elif name.endswith("_max"):
                aggregate = torch.maximum(previous, value)
            elif name.endswith("_min"):
                aggregate = torch.minimum(previous, value)
            else:
                aggregate = previous + value
            self._world_state_read_diagnostic_sums[name] = aggregate
        self._world_state_read_diagnostic_count += 1

    def _world_state_before_chunk(
        self,
        context: Tensor,
        context_mask: Tensor,
        conditions: Optional[Tensor],
    ) -> None:
        if not self.unit_inference_enable:
            return
        query_anchors = None
        if self._rollout_world_state is not None:
            if conditions is None:
                raise RuntimeError("Pose-aware world-state read requires camera conditions")
            query_anchors = self._world_state_query_anchors(
                conditions,
                self._rollout_world_state,
            )
            self._world_state_read_count += 1
        self._world_state_backbone().set_world_state_condition(
            self._rollout_world_state,
            query_anchors=query_anchors,
            scale=self.world_state_runtime_scale,
            detach=True,
            branch_mode=self.world_state_runtime_branch_mode,
        )

    @torch.no_grad()
    def _encode_finalized_world_state(
        self,
        finalized_chunk: Tensor,
        conditions: Tensor,
        previous_state: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        if finalized_chunk.shape[1] != self.max_tokens:
            raise ValueError(
                "World-state updates require one complete DFoT window, got "
                f"{finalized_chunk.shape[1]} tokens instead of {self.max_tokens}"
            )
        # Encode what is visibly present in the finalized chunk. Re-reading the
        # previous state during this clean feature pass would feed memory back
        # into itself even when the generated pixels did not preserve it.
        self._world_state_backbone().clear_world_state_condition()
        processed_conditions = self._process_conditions(conditions)
        latents, logsnr = self.diffusion_model.extract_clean_alignment_latents(
            finalized_chunk,
            processed_conditions,
        )
        observation = self.vggt_alignment_loss.encode_video_world_observation(
            latents,
            logsnr,
        )
        observation = self._anchor_world_observation(
            observation,
            conditions,
            previous_state,
        )
        state = self._update_world_state_from_observation(
            observation,
            previous_state,
            detach_previous=True,
        )
        self._latest_world_state_metrics.update(
            {
                "world_state_num_spans": observation["num_spans"].detach(),
                "world_state_span_code_agreement": observation[
                    "span_code_agreement"
                ].detach(),
                "world_state_physical_rms": state[
                    "physical"
                ].float().square().mean().sqrt(),
                "world_state_appearance_rms": state[
                    "appearance"
                ].float().square().mean().sqrt(),
                "world_state_confidence_mean": state["confidence"].float().mean(),
                "world_state_age_mean": state["age"].float().mean(),
                "world_state_age_max": state["age"].float().amax(),
                "world_state_window_count": state["window_count"].float().mean(),
                "world_state_memory_tokens": state["physical"].new_tensor(
                    float(state["physical"].shape[1])
                ),
            }
        )
        return {key: value.detach() for key, value in state.items()}

    def _world_state_after_chunk(
        self,
        finalized_chunk: Tensor,
        context_mask: Tensor,
        conditions: Optional[Tensor],
    ) -> None:
        if (
            not self.unit_inference_enable
            or conditions is None
            or finalized_chunk.shape[1] != self.max_tokens
        ):
            return
        self._accumulate_world_state_read_diagnostics()
        state = self._encode_finalized_world_state(
            finalized_chunk,
            conditions,
            self._rollout_world_state,
        )
        object.__setattr__(self, "_rollout_world_state", state)
        self._world_state_write_count += 1

    def _update_world_state_from_observation(
        self,
        observation: dict[str, Tensor],
        previous_state: dict[str, Tensor] | None,
        *,
        detach_previous: bool,
    ) -> dict[str, Tensor]:
        """Apply the same within-window state updates in training and sampling."""
        physical_spans = observation.get("physical_spans")
        appearance_spans = observation.get("appearance_spans")
        anchor_spans = observation.get("anchor_spans")
        confidence_spans = observation.get("confidence_spans")
        if physical_spans is None or appearance_spans is None:
            physical_spans = observation["physical"][:, None]
            appearance_spans = observation["appearance"][:, None]
        if anchor_spans is None:
            anchor_spans = physical_spans.new_zeros(
                *physical_spans.shape[:3], 9
            )
        if confidence_spans is None:
            confidence_spans = physical_spans.new_ones(
                *physical_spans.shape[:3], 1
            )
        if physical_spans.shape != appearance_spans.shape:
            raise ValueError(
                "Factorized world-state spans must have matching shapes, got "
                f"{tuple(physical_spans.shape)} and {tuple(appearance_spans.shape)}"
            )
        backbone = self._world_state_backbone()
        # A finalized clip is one temporal update regardless of how many local
        # transition spans UPCAST emits. Advance old tokens once, then append all
        # spans from this clip with age zero.
        state = backbone.advance_world_state_age(previous_state)
        for span_index in range(physical_spans.shape[1]):
            state = backbone.update_world_state(
                physical_spans[:, span_index],
                appearance_spans[:, span_index],
                state,
                anchors=anchor_spans[:, span_index],
                confidence=confidence_spans[:, span_index],
                detach_previous=detach_previous,
            )
        if state is None:
            raise RuntimeError("World-state observation did not contain any spans")
        for key in ("reference_origin", "reference_scale"):
            if key in observation:
                state[key] = observation[key]
            elif previous_state is not None and key in previous_state:
                state[key] = previous_state[key]
        return state

    def _world_state_end_rollout(self) -> None:
        if not self.unit_inference_enable:
            return
        # _world_state_after_chunk is intentionally skipped for the final
        # generated chunk, so retain its read diagnostics here.
        self._accumulate_world_state_read_diagnostics()
        if (
            self.world_state_require_rollout_reads
            and getattr(self, "_world_state_expect_reads", False)
            and self._world_state_read_count == 0
        ):
            raise RuntimeError(
                "Long-video rollout completed without reading persistent world state"
            )
        self._latest_world_state_metrics.update(
            {
                "world_state_rollout_reads": torch.tensor(
                    float(self._world_state_read_count), device=self.device
                ),
                "world_state_rollout_writes": torch.tensor(
                    float(self._world_state_write_count), device=self.device
                ),
                "world_state_context_initializations": torch.tensor(
                    float(getattr(self, "_world_state_context_init_count", 0)),
                    device=self.device,
                ),
            }
        )
        diagnostic_count = int(self._world_state_read_diagnostic_count)
        if diagnostic_count > 0:
            self._latest_world_state_metrics.update(
                {
                    name: (
                        total
                        if name.endswith(("_max", "_min"))
                        else total / float(diagnostic_count)
                    )
                    for name, total in self._world_state_read_diagnostic_sums.items()
                }
            )
            self._latest_world_state_metrics[
                "world_state_read_diagnostic_windows"
            ] = torch.tensor(float(diagnostic_count), device=self.device)
        self._world_state_backbone().clear_world_state_condition()
        object.__setattr__(self, "_rollout_world_state", None)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        processed = super().on_after_batch_transfer(batch, dataloader_idx)
        if (
            not self.rollout_training_enable
            or not getattr(self.trainer, "training", False)
        ):
            return processed
        xs, conditions, masks, gt_videos = processed
        expected = self.rollout_training_frames
        if xs.shape[1] != expected:
            raise RuntimeError(
                "Rollout dataset returned the wrong temporal length: "
                f"got {xs.shape[1]}, expected {expected}. Set "
                f"dataset.rollout_training_frames={expected}."
            )
        stride = self.max_tokens - self.rollout_overlap_frames
        windows = []
        for window_index in range(self.rollout_num_windows):
            start = window_index * stride
            window_slice = slice(start, start + self.max_tokens)
            windows.append(
                {
                    "xs": xs[:, window_slice],
                    "conditions": (
                        None if conditions is None else conditions[:, window_slice]
                    ),
                    "masks": masks[:, window_slice],
                    "gt_videos": (
                        None if gt_videos is None else gt_videos[:, window_slice]
                    ),
                }
            )
        rollout_batch = {"windows": windows}
        target_window = windows[1]
        return (
            target_window["xs"],
            target_window["conditions"],
            target_window["masks"],
            target_window["gt_videos"],
            rollout_batch,
        )

    def alignment_loss(self, latents_list, context_images, **kwargs):
        if self.encoder_type == "vggt":
            return self.vggt_alignment_loss(
                latents_list,
                context_images,
                frozen_diffusion=self._diffusion_frozen,
                **kwargs,
            )
        else:
            raise ValueError(f"[alignment_loss] Unknown encoder_type: {self.encoder_type}")

    def _alignment_coeff_tensor(self, reference):
        return torch.as_tensor(
            float(self.current_alignment_coeff),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _ensure_video_prior_model(self):
        if self.video_prior_preservation_weight <= 0:
            return None
        if getattr(self, "video_prior_mode", "full_model") == "selected_functional":
            return self._ensure_selected_video_prior_state()
        if self._video_prior_model is not None:
            return self._video_prior_model
        if self._video_prior_snapshot_failed:
            if self.strict_video_prior:
                raise RuntimeError("Frozen-base prior snapshot previously failed")
            return None
        try:
            prior_model = copy.deepcopy(self.diffusion_model)
            if self.video_prior_checkpoint is not None:
                checkpoint_path = Path(self.video_prior_checkpoint).expanduser()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Frozen-base prior checkpoint does not exist: {checkpoint_path}"
                    )
                try:
                    payload = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        mmap=True,
                        weights_only=False,
                    )
                except TypeError:
                    payload = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_state = payload.get("state_dict", payload)
                prefix = "diffusion_model."
                prior_state = {
                    key[len(prefix) :]: value
                    for key, value in checkpoint_state.items()
                    if key.startswith(prefix)
                }
                if self.cfg.compile:
                    prior_state = {
                        (
                            key
                            if key.startswith("_orig_mod.")
                            else "_orig_mod." + key
                        ): value
                        for key, value in prior_state.items()
                    }
                else:
                    prior_state = {
                        key.removeprefix("_orig_mod."): value
                        for key, value in prior_state.items()
                    }
                incompatible = prior_model.load_state_dict(prior_state, strict=False)
                allowed_missing = [
                    key
                    for key in incompatible.missing_keys
                    if ".world_state_adapter." in key
                    or ".generative_unit_conditioner." in key
                ]
                disallowed_missing = sorted(
                    set(incompatible.missing_keys) - set(allowed_missing)
                )
                allowed_unexpected = [
                    key
                    for key in incompatible.unexpected_keys
                    if ".world_state_adapter." in key
                ]
                disallowed_unexpected = sorted(
                    set(incompatible.unexpected_keys)
                    - set(allowed_unexpected)
                )
                if disallowed_missing or disallowed_unexpected:
                    raise RuntimeError(
                        "Frozen-base prior checkpoint is incompatible: "
                        f"missing={disallowed_missing[:5]}, "
                        f"unexpected={disallowed_unexpected[:5]}"
                    )
                del payload, checkpoint_state, prior_state
            prior_model.eval()
            for param in prior_model.parameters():
                param.requires_grad = False
            if self.video_prior_cpu_offload:
                prior_model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            object.__setattr__(self, "_video_prior_model", prior_model)
            rank_zero_print(
                "[DFoTUPCAST] Built frozen diffusion prior from "
                f"{self.video_prior_checkpoint or 'the bootstrap-start weights'}."
            )
        except Exception as exc:
            self._video_prior_snapshot_failed = True
            if self.strict_video_prior:
                raise RuntimeError("Failed to snapshot frozen-base DFoT prior") from exc
            rank_zero_print(
                "[DFoTUPCAST] WARNING: failed to create frozen diffusion prior; "
                f"video-prior preservation is disabled. Error: {exc}"
            )
            return None
        return self._video_prior_model

    def _ensure_selected_video_prior_state(self) -> dict[str, Tensor] | None:
        """Load only the selectively trainable tensors for a frozen teacher."""
        state = self._video_prior_selected_state
        if state is not None:
            return state
        if self._video_prior_snapshot_failed:
            if self.strict_video_prior:
                raise RuntimeError("Functional video-prior snapshot previously failed")
            return None
        try:
            if self.video_prior_checkpoint is None:
                raise ValueError(
                    "selected_functional prior requires video_prior_checkpoint"
                )
            checkpoint_path = Path(self.video_prior_checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Frozen-base prior checkpoint does not exist: {checkpoint_path}"
                )
            try:
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_state = payload.get("state_dict", payload)
            backbone = self._world_state_backbone()
            current_parameters = dict(backbone.named_parameters())
            selected_state: dict[str, Tensor] = {}
            missing = []
            for backbone_name, parameter in current_parameters.items():
                canonical_backbone_name = str(backbone_name).removeprefix(
                    "_orig_mod."
                )
                diffusion_name = "model." + canonical_backbone_name
                if not self._matches_selective_diffusion_prefix(diffusion_name):
                    continue
                candidates = (
                    "diffusion_model.model." + canonical_backbone_name,
                    "diffusion_model._orig_mod.model." + canonical_backbone_name,
                    "diffusion_model.model._orig_mod." + canonical_backbone_name,
                )
                source = next(
                    (
                        checkpoint_state[key]
                        for key in candidates
                        if key in checkpoint_state
                    ),
                    None,
                )
                if source is None:
                    missing.append(diffusion_name)
                    continue
                if source.shape != parameter.shape:
                    raise ValueError(
                        "Functional prior shape mismatch for "
                        f"{diffusion_name}: checkpoint={tuple(source.shape)}, "
                        f"model={tuple(parameter.shape)}"
                    )
                target_device = (
                    torch.device("cpu")
                    if self.video_prior_cpu_offload
                    else parameter.device
                )
                selected_state[backbone_name] = source.detach().to(
                    device=target_device,
                    dtype=parameter.dtype,
                    copy=True,
                )
            if missing:
                raise RuntimeError(
                    "Functional prior is missing selected parameters: "
                    + ", ".join(missing[:5])
                )
            if not selected_state:
                raise RuntimeError("Functional prior selected no parameters")
            object.__setattr__(self, "_video_prior_selected_state", selected_state)
            del payload, checkpoint_state
            parameter_count = sum(value.numel() for value in selected_state.values())
            rank_zero_print(
                "[DFoTUPCAST] Built selected functional video prior "
                f"with {parameter_count / 1e6:.1f}M reference parameters from "
                f"{checkpoint_path}."
            )
            return selected_state
        except Exception as exc:
            self._video_prior_snapshot_failed = True
            if self.strict_video_prior:
                raise RuntimeError(
                    "Failed to build selected functional video prior"
                ) from exc
            rank_zero_print(
                "[DFoTUPCAST] WARNING: failed to create selected "
                f"functional prior; preservation is disabled. Error: {exc}"
            )
            return None

    def _compute_video_prior_target(
        self,
        denoising_terms,
        processed_conditions,
    ):
        device = processed_conditions.device
        if (
            self.video_prior_preservation_weight <= 0
            or self._diffusion_frozen
            or denoising_terms is None
        ):
            if (
                self.strict_video_prior
                and self.video_prior_preservation_weight > 0
                and not self._diffusion_frozen
                and denoising_terms is None
            ):
                raise RuntimeError("Frozen-base prior requires denoising terms")
            return None, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )

        prior = self._ensure_video_prior_model()
        if prior is None:
            return None, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )

        x_t = denoising_terms.get("x_t")
        logsnr = denoising_terms.get("logsnr")
        if x_t is None or logsnr is None:
            if self.strict_video_prior:
                raise RuntimeError("Frozen-base prior is missing x_t or logSNR")
            return None, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )

        try:
            if self.video_prior_mode == "selected_functional":
                backbone = self._world_state_backbone()
                functional_state = {
                    name: value.to(device=device, non_blocking=True)
                    for name, value in prior.items()
                }
                was_training = backbone.training
                backbone.eval()
                try:
                    with torch.no_grad():
                        if hasattr(backbone, "clear_world_state_condition"):
                            backbone.clear_world_state_condition()
                        prior_v_pred = functional_call(
                            backbone,
                            functional_state,
                            (
                                x_t,
                                self.diffusion_model.precond_scale * logsnr,
                                processed_conditions,
                            ),
                            {
                                "external_cond_mask": denoising_terms.get(
                                    "external_cond_mask"
                                )
                            },
                            strict=False,
                        )
                finally:
                    backbone.train(was_training)
                return prior_v_pred.detach(), torch.tensor(
                    1.0,
                    device=device,
                    dtype=processed_conditions.dtype,
                )

            prior_model = prior
            prior_parameter = next(prior_model.parameters(), None)
            if prior_parameter is not None and prior_parameter.device != device:
                prior_model.to(device)
            prior_model.eval()
            with torch.no_grad():
                prior_backbone = getattr(prior_model, "model", None)
                if hasattr(prior_backbone, "clear_world_state_condition"):
                    prior_backbone.clear_world_state_condition()
                prior_v_pred = prior_model.model(
                    x_t,
                    prior_model.precond_scale * logsnr,
                    processed_conditions,
                    external_cond_mask=denoising_terms.get("external_cond_mask"),
                )
            return prior_v_pred.detach(), torch.tensor(
                1.0, device=device, dtype=processed_conditions.dtype
            )
        except Exception as exc:
            if self.strict_video_prior:
                raise RuntimeError("Frozen-base prior forward failed") from exc
            rank_zero_print(
                "[DFoTUPCAST] WARNING: video-prior preservation forward failed; "
                f"skipping this batch. Error: {exc}"
            )
            return None, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )
        finally:
            if (
                self.video_prior_mode == "full_model"
                and self.video_prior_cpu_offload
                and prior is not None
            ):
                prior_model = prior
                prior_model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _compute_video_prior_preservation_loss(
        self,
        denoising_terms,
        processed_conditions,
        prepared_target=None,
        prepared_valid=None,
    ):
        device = processed_conditions.device
        zero = torch.zeros((), device=device, dtype=processed_conditions.dtype)
        if denoising_terms is None:
            if (
                self.strict_video_prior
                and self.video_prior_preservation_weight > 0
                and not self._diffusion_frozen
            ):
                raise RuntimeError("Frozen-base prior requires denoising terms")
            return zero, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )
        v_pred = denoising_terms.get("v_pred")
        if v_pred is None:
            if self.strict_video_prior:
                raise RuntimeError("Frozen-base prior is missing current v_pred")
            return zero, torch.tensor(
                0.0,
                device=device,
                dtype=processed_conditions.dtype,
            )
        if prepared_valid is None:
            prepared_target, valid = self._compute_video_prior_target(
                denoising_terms,
                processed_conditions,
            )
        else:
            valid = prepared_valid
        if prepared_target is None:
            return zero, valid
        return F.mse_loss(v_pred, prepared_target.to(v_pred)), valid

    def _compute_generated_context_exposure_loss(
        self,
        xs: Tensor,
        xs_pred: Tensor,
        masks: Tensor,
        denoising_terms: dict | None,
        processed_conditions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = xs.new_zeros(())
        metrics = {
            "valid": zero,
            "active_fraction": zero,
            "context_residual_rms": zero,
        }
        if denoising_terms is None:
            raise RuntimeError(
                "Generated-context exposure requires denoising terms"
            )
        logsnr = denoising_terms.get("logsnr")
        if logsnr is None:
            raise RuntimeError(
                "Generated-context exposure requires frame-wise logSNR"
            )
        context_frames = min(
            self.generated_context_num_frames,
            int(xs.shape[1]) - 1,
        )
        if context_frames < 1:
            return zero, metrics

        signal = torch.sigmoid(logsnr[:, :context_frames]).to(xs)
        active = signal >= self.generated_context_min_signal
        quality = (
            (signal - self.generated_context_min_signal)
            / max(1.0 - self.generated_context_min_signal, 1e-6)
        ).clamp(0.0, 1.0)
        strength = self.generated_context_blend * (0.25 + 0.75 * quality)
        strength = strength * active.to(strength)
        while strength.ndim < xs.ndim:
            strength = strength.unsqueeze(-1)

        raw_residual = xs_pred.detach()[:, :context_frames] - xs[:, :context_frames]
        residual = raw_residual.clamp(
            -self.generated_context_max_residual,
            self.generated_context_max_residual,
        )
        context_estimate = xs.detach().clone()
        context_estimate[:, :context_frames] = (
            xs[:, :context_frames] + strength * residual
        ).detach()
        context_mask = torch.zeros(
            xs.shape[:2],
            device=xs.device,
            dtype=torch.bool,
        )
        context_mask[:, :context_frames] = True
        loss_map = self.diffusion_model.context_exposure_loss(
            processed_conditions,
            denoising_terms,
            context_estimate,
            context_mask,
            activation_checkpointing=(
                self.generated_context_activation_checkpointing
            ),
        )
        future_masks = masks.clone()
        future_masks[:, :context_frames] = False
        loss = self._reweight_loss(loss_map, future_masks)
        if torch.is_grad_enabled() and not loss.requires_grad:
            raise RuntimeError(
                "Generated-context future loss is detached from Video DiT"
            )
        applied_residual = strength * residual
        metrics = {
            "valid": xs.new_ones(()),
            "active_fraction": active.float().mean().to(xs),
            "context_residual_rms": applied_residual.float().square().mean().sqrt().to(xs),
        }
        return loss, metrics

    def on_train_batch_start(self, batch, batch_idx):
        # Backward for the preceding micro-batch has completed before this
        # hook. Reset the per-micro-batch selective-gradient policy here.
        self._suppress_selective_diffusion_gradients = False
        step = int(self.global_step)
        if self.generative_unit_enabled:
            self.current_alignment_coeff = 0.0
            self._curriculum_phase = self.generative_unit_mode
            self._set_generative_unit_trainable()
            return
        if self.world_state_training_mode in {"adapter_only", "gate_only"}:
            self.current_alignment_coeff = 0.0
            self._curriculum_phase = self.world_state_training_mode
            if self.world_state_training_mode == "gate_only":
                self._set_world_state_gate_only_trainable()
            else:
                self._set_world_state_adapter_only_trainable()
            return
        if self.world_state_training_mode == "selective_joint":
            if self.unit_joint_ramp_steps > 0:
                ramp_progress = min(
                    max((step + 1) / self.unit_joint_ramp_steps, 0.0),
                    1.0,
                )
            else:
                ramp_progress = 1.0
            self.current_alignment_coeff = (
                self.joint_alignment_coeff * ramp_progress
            )
            self._curriculum_phase = (
                "selective_ramp" if ramp_progress < 1.0 else "selective_joint"
            )
            self._set_selective_joint_trainable()
            return
        if step < self.unit_geometry_tokenizer_steps:
            self.current_alignment_coeff = self.bootstrap_alignment_coeff
            entering_geometry = self._curriculum_phase != "geometry"
            self._curriculum_phase = "geometry"
            self.diffusion_model.eval()
            if (
                hasattr(self, "vggt_alignment_loss")
                and self.vggt_alignment_loss.unit_phase != "geometry"
            ):
                self.vggt_alignment_loss.set_unit_phase("geometry")
            if entering_geometry:
                rank_zero_print(
                    "[DFoTUPCAST] Entered geometry-tokenizer warmup."
                )
            return

        if step < self.unit_bootstrap_steps:
            entering_alignment = self._curriculum_phase != "alignment"
            if entering_alignment:
                self._validate_geometry_teacher_health()
            self.current_alignment_coeff = self.bootstrap_alignment_coeff
            self._curriculum_phase = "alignment"
            self.diffusion_model.eval()
            if (
                hasattr(self, "vggt_alignment_loss")
                and self.vggt_alignment_loss.unit_phase != "alignment"
            ):
                self.vggt_alignment_loss.set_unit_phase("alignment")
            self._check_unit_student_health(step)
            if entering_alignment:
                rank_zero_print(
                    "[DFoTUPCAST] Geometry tokenizer fixed; entered "
                    "video/fusion-to-geometry alignment."
                )
            return

        entering_joint = self._curriculum_phase not in {
            "ramp",
            "joint",
            "cotrain",
        }
        if self._diffusion_frozen:
            self._set_diffusion_trainable(True)
        cotrain_active = self._is_unit_cotrain_step(step)
        if self.unit_freeze_after_bootstrap and hasattr(
            self,
            "vggt_alignment_loss",
        ):
            current_unit_phase = self.vggt_alignment_loss.unit_phase
            desired_unit_phase = "cotrain" if cotrain_active else "joint"
            if current_unit_phase != desired_unit_phase:
                leaving_bootstrap = current_unit_phase not in {
                    "joint",
                    "cotrain",
                }
                self.vggt_alignment_loss.set_unit_phase(desired_unit_phase)
                if leaving_bootstrap:
                    self._release_frozen_unit_optimizer_state()
            if cotrain_active:
                self._apply_unit_cotrain_lr()
        if self.unit_joint_ramp_steps > 0:
            ramp_progress = min(
                max(step - self.unit_bootstrap_steps + 1, 0)
                / self.unit_joint_ramp_steps,
                1.0,
            )
        else:
            ramp_progress = 1.0
        self.current_alignment_coeff = self.joint_alignment_coeff * ramp_progress
        if cotrain_active:
            self._curriculum_phase = "cotrain"
        else:
            self._curriculum_phase = "ramp" if ramp_progress < 1.0 else "joint"
        if entering_joint:
            rank_zero_print(
                "[DFoTUPCAST] Alignment complete: froze the complete UPCAST "
                "teacher and entered Video DiT distillation."
            )
        if cotrain_active and step == self.unit_cotrain_start_step:
            rank_zero_print(
                "[DFoTUPCAST] Entered selective video-side UPCAST cotrain; "
                f"one shared update every {self.unit_cotrain_interval} optimizer steps."
            )

    def _distributed_health_flag(self, unhealthy: bool) -> bool:
        device = next(self.diffusion_model.parameters()).device
        flag = torch.tensor(int(bool(unhealthy)), device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            # A fail-fast exception must be raised by every rank. Requiring all
            # ranks to observe collapse avoids terminating one reducer early.
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @staticmethod
    def _metric_float(metrics: dict, name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        if isinstance(value, Tensor):
            return float(value.detach().float().cpu())
        return float(value)

    def _validate_geometry_teacher_health(self) -> None:
        if not hasattr(self, "vggt_alignment_loss"):
            return
        metrics = self.vggt_alignment_loss.last_metrics
        perplexities = [
            self._metric_float(metrics, f"alignment_unit_g_perplexity_q{level}")
            for level in range(
                self.vggt_alignment_loss.unit_quantizer.num_quantizers
            )
        ]
        perplexities = [value for value in perplexities if value is not None]
        projected_std = self._metric_float(
            metrics,
            "alignment_unit_geometry_projected_std",
        )
        if not perplexities:
            counts = (
                self.vggt_alignment_loss.unit_quantizer.ema_counts.detach().float()
            )
            probabilities = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
            perplexities = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    dim=-1
                )
            ).cpu().tolist()
        if projected_std is None:
            # last_metrics are intentionally not checkpointed. On an exact
            # phase-boundary resume, persistent EMA usage is still sufficient
            # to reject a one-code teacher.
            unhealthy = min(perplexities) < self.unit_health_min_perplexity
        else:
            unhealthy = (
                min(perplexities) < self.unit_health_min_perplexity
                and projected_std < self.unit_health_min_projected_std
            )
        if self._distributed_health_flag(unhealthy):
            std_description = (
                "unavailable-after-resume"
                if projected_std is None
                else f"{projected_std:.6f}"
            )
            raise RuntimeError(
                "Geometry tokenizer collapsed before alignment: "
                f"min_perplexity={min(perplexities):.4f}, "
                f"projected_std={std_description}. Restart instead of freezing "
                "this teacher."
            )

    def _check_unit_student_health(self, step: int) -> None:
        if step < self.unit_health_check_start or step == self._unit_health_last_step:
            return
        self._unit_health_last_step = step
        metrics = self.vggt_alignment_loss.last_metrics
        perplexity = self._metric_float(
            metrics,
            "alignment_unit_student_min_perplexity",
        )
        projected_std = self._metric_float(
            metrics,
            "alignment_unit_v_projected_std",
        )
        if perplexity is None or projected_std is None:
            return
        unhealthy = (
            perplexity < self.unit_health_min_perplexity
            and projected_std < self.unit_health_min_projected_std
        )
        unhealthy = self._distributed_health_flag(unhealthy)
        self._unit_health_unhealthy_steps = (
            self._unit_health_unhealthy_steps + 1 if unhealthy else 0
        )
        if unhealthy and (
            self._unit_health_unhealthy_steps == 1
            or self._unit_health_unhealthy_steps % 5 == 0
        ):
            rank_zero_print(
                "[DFoTUPCAST] UPCAST health warning: "
                f"step={step}, min_perplexity={perplexity:.4f}, "
                f"projected_std={projected_std:.6f}, consecutive="
                f"{self._unit_health_unhealthy_steps}/"
                f"{self.unit_health_check_patience}."
            )
        if self._unit_health_unhealthy_steps >= self.unit_health_check_patience:
            raise RuntimeError(
                "Video/fusion UPCAST student collapsed for "
                f"{self._unit_health_unhealthy_steps} consecutive optimizer steps "
                f"(min_perplexity={perplexity:.4f}, "
                f"projected_std={projected_std:.6f})."
            )

    def configure_optimizers(self):
        diffusion_lr = self.cfg.lr
        unit_lr = self.cfg.lr
        if self.alignment_coeff > 0 and hasattr(self.cfg.alignment, "unit_lr"):
            unit_lr = self.cfg.alignment.unit_lr
        if self.alignment_coeff > 0 and hasattr(self.cfg.alignment, "diffusion_lr"):
            diffusion_lr = self.cfg.alignment.diffusion_lr

        if self.generative_unit_enabled:
            conditioner_params, diffusion_params = (
                self._generative_parameter_groups()
            )
            param_groups = []
            if conditioner_params:
                param_groups.append(
                    {
                        "params": conditioner_params,
                        "lr": self.generative_unit_lr,
                        "ema": True,
                        "group_name": "generative_unit",
                    }
                )
            if diffusion_params:
                param_groups.append(
                    {
                        "params": diffusion_params,
                        "lr": self.generative_diffusion_lr,
                        "ema": True,
                        "group_name": "generative_diffusion",
                    }
                )
            if not param_groups:
                raise RuntimeError(
                    f"{self.generative_unit_mode} has no trainable parameters"
                )
            optimizer_dynamics = torch.optim.AdamW(
                param_groups,
                lr=self.generative_diffusion_lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
        elif self.world_state_training_mode == "selective_joint":
            diffusion_params, world_state_params = self._selective_parameter_groups()
            if not diffusion_params:
                raise RuntimeError(
                    "selective_joint has no trainable DFoT backbone parameters"
                )
            param_groups = [
                {
                    "params": diffusion_params,
                    "lr": diffusion_lr,
                    "ema": True,
                    "group_name": "selective_diffusion",
                }
            ]
            if world_state_params:
                param_groups.append(
                    {
                        "params": world_state_params,
                        "lr": self.world_state_lr,
                        "ema": True,
                        "group_name": "world_state",
                    }
                )
            optimizer_dynamics = torch.optim.AdamW(
                param_groups,
                lr=diffusion_lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
        elif self.world_state_training_mode in {"adapter_only", "gate_only"}:
            world_state_adapter = getattr(
                self._world_state_backbone(),
                "world_state_adapter",
                None,
            )
            if world_state_adapter is None:
                raise RuntimeError("Missing world-state adapter")
            if self.world_state_training_mode == "gate_only":
                world_state_params = [
                    parameter
                    for parameter in world_state_adapter.preference_parameters()
                    if parameter.requires_grad
                ]
                group_name = "world_state_preference_gates"
            else:
                world_state_params = [
                    parameter
                    for parameter in world_state_adapter.parameters()
                    if parameter.requires_grad
                ]
                group_name = "world_state_adapter_only"
            if not world_state_params:
                raise RuntimeError("World-state adapter has no trainable parameters")
            optimizer_dynamics = torch.optim.AdamW(
                [
                    {
                        "params": world_state_params,
                        "lr": self.world_state_lr,
                        "ema": False,
                        "group_name": group_name,
                    }
                ],
                lr=self.world_state_lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
        elif self.alignment_coeff > 0:
            # The diffusion model starts frozen but must already belong to the
            # optimizer so that a single run can unfreeze it after bootstrap.
            world_state_adapter = getattr(
                self._world_state_backbone(),
                "world_state_adapter",
                None,
            )
            world_state_params = (
                list(world_state_adapter.parameters())
                if world_state_adapter is not None
                else []
            )
            world_state_param_ids = {id(parameter) for parameter in world_state_params}
            diffusion_params = [
                parameter
                for parameter in self.diffusion_model.parameters()
                if id(parameter) not in world_state_param_ids
            ]
            # Phase-specific requires_grad flags change after DDP setup. Register
            # every UPCAST parameter once so later curriculum phases are complete.
            unit_params = list(self.vggt_alignment_loss.parameters())
            param_groups = []
            if len(diffusion_params) > 0:
                param_groups.append(
                    {
                        "params": diffusion_params,
                        "lr": diffusion_lr,
                        "ema": True,
                        "group_name": "diffusion",
                    }
                )
            if len(world_state_params) > 0:
                param_groups.append(
                    {
                        "params": world_state_params,
                        "lr": self.world_state_lr,
                        "ema": True,
                        "group_name": "world_state",
                    }
                )
            if len(unit_params) > 0:
                param_groups.append(
                    {
                        "params": unit_params,
                        "lr": unit_lr,
                        "ema": False,
                        "group_name": "unit",
                    }
                )
            optimizer_dynamics = torch.optim.AdamW(
                param_groups,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
        else:
            optimizer_dynamics = torch.optim.AdamW(
                list(self.diffusion_model.parameters()),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
        lr_scheduler_frequency = int(getattr(self.cfg, "lr_scheduler_frequency", 1) or 1)
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer_dynamics,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": max(1, lr_scheduler_frequency),
        }
        return {
            "optimizer": optimizer_dynamics,
            "lr_scheduler": lr_scheduler_config,
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        if self.rollout_training_enable:
            self._commit_rollout_horizon_metrics()
        if self.generative_unit_enabled:
            conditioner_params, diffusion_params = (
                self._generative_parameter_groups()
            )

            conditioner_names = {
                id(parameter): name
                for name, parameter in self._generative_unit_backbone()
                .generative_unit_conditioner.named_parameters()
            }
            diffusion_names = {
                id(parameter): self._canonical_diffusion_parameter_name(name)
                for name, parameter in self.diffusion_model.named_parameters()
            }

            def gradient_report(
                parameters: list[Tensor],
                parameter_names: dict[int, str],
            ) -> tuple[Tensor, list[str]]:
                terms = []
                nonfinite = []
                for parameter in parameters:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad.detach().float()
                    if not torch.isfinite(gradient).all():
                        invalid = ~torch.isfinite(gradient)
                        nonfinite.append(
                            f"{parameter_names.get(id(parameter), '<unnamed>')}"
                            f"[{int(invalid.sum().item())}/{gradient.numel()}]"
                        )
                    else:
                        terms.append(gradient.square().sum())
                device = next(self.diffusion_model.parameters()).device
                norm = (
                    torch.stack(terms).sum().sqrt()
                    if terms
                    else torch.zeros((), device=device)
                )
                if nonfinite:
                    norm = norm.new_tensor(float("nan"))
                return norm, nonfinite

            conditioner_norm, conditioner_nonfinite = gradient_report(
                conditioner_params,
                conditioner_names,
            )
            diffusion_norm, diffusion_nonfinite = gradient_report(
                diffusion_params,
                diffusion_names,
            )
            expected = (
                conditioner_norm
                if self.generative_unit_mode
                in {"adapter_warmup", "conditional_teacher"}
                else diffusion_norm
            )
            if not torch.isfinite(expected):
                detail = ", ".join(
                    (conditioner_nonfinite + diffusion_nonfinite)[:8]
                )
                scaler = getattr(
                    getattr(self.trainer, "precision_plugin", None),
                    "scaler",
                    None,
                )
                if scaler is None:
                    raise RuntimeError(
                        "Physical conditioner optimizer gradient is non-finite "
                        "without an AMP GradScaler: "
                        f"mode={self.generative_unit_mode}, parameters={detail}"
                    )
                # A single FP16 overflow is recoverable: GradScaler has already
                # inspected the unscaled gradients and will skip this update,
                # then reduce its scale. Raising here would bypass precisely
                # that recovery path. Persistent overflows are rejected in
                # optimizer_step below.
                self._generative_nonfinite_gradient_pending = True
                self._generative_nonfinite_gradient_detail = detail
                return
            if expected.item() == 0.0:
                raise RuntimeError(
                    "Physical conditioner optimizer gradient is invalid: "
                    f"mode={self.generative_unit_mode}, "
                    f"conditioner={conditioner_norm.item()}, "
                    f"diffusion={diffusion_norm.item()}"
                )
            self._generative_nonfinite_gradient_pending = False
            self._generative_nonfinite_gradient_detail = ""
            unit_leaks = [
                name
                for name, parameter in self.vggt_alignment_loss.named_parameters()
                if parameter.grad is not None
            ]
            if unit_leaks:
                raise RuntimeError(
                    "Generative continuation leaked gradients into frozen UPCAST: "
                    + ", ".join(unit_leaks[:5])
                )
            self.log(
                "training/generative_unit_grad_norm",
                conditioner_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "training/generative_diffusion_grad_norm",
                diffusion_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            return
        if self.world_state_training_mode == "selective_joint":
            diffusion_params, world_state_params = self._selective_parameter_groups()

            def gradient_norm(parameters: list[Tensor]) -> Tensor:
                terms = [
                    parameter.grad.detach().float().square().sum()
                    for parameter in parameters
                    if parameter.grad is not None
                ]
                device = next(self.diffusion_model.parameters()).device
                if not terms:
                    return torch.zeros((), device=device)
                return torch.stack(terms).sum().sqrt()

            diffusion_norm = gradient_norm(diffusion_params)
            state_norm = gradient_norm(world_state_params)
            leaked = [
                name
                for name, parameter in self.diffusion_model.named_parameters()
                if not self._matches_selective_diffusion_prefix(name)
                and parameter.grad is not None
            ]
            if leaked:
                raise RuntimeError(
                    "Selective joint training leaked gradients into frozen DFoT "
                    "parameters: " + ", ".join(leaked[:5])
                )
            unit_leaks = [
                name
                for name, parameter in self.vggt_alignment_loss.named_parameters()
                if parameter.grad is not None
            ]
            if unit_leaks:
                raise RuntimeError(
                    "Selective joint training leaked gradients into frozen UPCAST "
                    "parameters: " + ", ".join(unit_leaks[:5])
                )
            if not torch.isfinite(diffusion_norm) or diffusion_norm.item() == 0.0:
                raise RuntimeError(
                    "Selective DFoT gradient is invalid: "
                    f"norm={diffusion_norm.item()}"
                )
            self.log(
                "training/selective_diffusion_grad_norm",
                diffusion_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "training/selective_world_state_grad_norm",
                state_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                "training/selective_world_state_to_diffusion_grad_ratio",
                state_norm / diffusion_norm.clamp_min(1e-12),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            return
        if self.world_state_training_mode not in {"adapter_only", "gate_only"}:
            return
        adapter = self._world_state_backbone().world_state_adapter
        adapter_parameters = (
            adapter.preference_parameters()
            if self.world_state_training_mode == "gate_only"
            else list(adapter.parameters())
        )
        parameter_names = {
            id(parameter): name
            for name, parameter in adapter.named_parameters()
        }
        gradient_terms = []
        nonfinite_gradients = []
        for parameter in adapter_parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            gradient_terms.append(gradient.square().sum())
            if not torch.isfinite(gradient).all():
                nonfinite_gradients.append(
                    parameter_names.get(id(parameter), "<unnamed>")
                )
        collection_only = bool(
            getattr(self, "rollout_preference_cache_collection_only", False)
        )
        if not gradient_terms and not collection_only:
            raise RuntimeError(
                f"{self.world_state_training_mode} optimizer step has no gradients"
            )
        gradient_norm = (
            torch.stack(gradient_terms).sum().sqrt()
            if gradient_terms
            else next(iter(adapter_parameters)).new_zeros((), dtype=torch.float32)
        )
        if not torch.isfinite(gradient_norm) or (
            gradient_norm.item() == 0.0 and not collection_only
        ):
            detail = (
                "; nonfinite_parameters=" + ", ".join(nonfinite_gradients)
                if nonfinite_gradients
                else ""
            )
            raise RuntimeError(
                f"World-state gradient is invalid: norm={gradient_norm.item()}"
                + detail
            )
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        leaked = [
            name
            for name, parameter in self.named_parameters()
            if id(parameter) not in adapter_ids and parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(
                f"{self.world_state_training_mode} training leaked gradients "
                "into frozen parameters: "
                + ", ".join(leaked[:5])
            )
        self.log(
            (
                "training/world_state_preference_gate_grad_norm"
                if self.world_state_training_mode == "gate_only"
                else "training/world_state_adapter_grad_norm"
            ),
            gradient_norm,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        if collection_only:
            self.log(
                "training/preference_collection_zero_gradient",
                float(gradient_norm.item() == 0.0),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            # The online optimizer only advances Lightning's collection loop.
            # The deployable policy is fitted from the complete train split in
            # the dedicated offline pass, so keep the collector itself frozen.
            for parameter in adapter_parameters:
                if parameter.grad is not None:
                    parameter.grad = None

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ):
        scaler = getattr(getattr(self.trainer, "precision_plugin", None), "scaler", None)
        scale_before = None if scaler is None else float(scaler.get_scale())
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )

        scale_after = None if scaler is None else float(scaler.get_scale())
        optimizer_was_skipped = bool(
            scale_before is not None
            and scale_after is not None
            and scale_after < scale_before
        )
        if getattr(self, "generative_unit_enabled", False):
            nonfinite_pending = bool(
                self._generative_nonfinite_gradient_pending
            )
            if nonfinite_pending:
                if not optimizer_was_skipped:
                    raise RuntimeError(
                        "Physical conditioner produced non-finite gradients, but "
                        "GradScaler did not skip the optimizer update: "
                        + self._generative_nonfinite_gradient_detail
                    )
                self._generative_amp_skip_streak += 1
                rank_zero_print(
                    "[DFoTUPCAST] GradScaler safely skipped one "
                    "Physical conditioner update and reduced the FP16 scale "
                    f"({scale_before:g} -> {scale_after:g}); invalid="
                    f"{self._generative_nonfinite_gradient_detail}."
                )
            else:
                self._generative_amp_skip_streak = 0
            self.log(
                "training/generative_amp_update_skipped",
                float(nonfinite_pending and optimizer_was_skipped),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            if scale_after is not None:
                self.log(
                    "training/generative_amp_scale",
                    float(scale_after),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
            self._generative_nonfinite_gradient_pending = False
            self._generative_nonfinite_gradient_detail = ""
            if (
                self._generative_amp_skip_streak
                >= self.generative_max_consecutive_amp_skips
            ):
                raise RuntimeError(
                    "Physical conditioner exceeded the allowed consecutive AMP "
                    "overflow count: "
                    f"{self._generative_amp_skip_streak}"
                )

        if not (
            self.alignment_coeff > 0
            and hasattr(self, "vggt_alignment_loss")
            and self.vggt_alignment_loss.unit_phase == "geometry"
        ):
            return

        if optimizer_was_skipped:
            self.vggt_alignment_loss.discard_unit_codebook_update()
            rank_zero_print(
                "[DFoTUPCAST] Discarded RVQ EMA statistics because "
                "GradScaler skipped the optimizer step."
            )
            return

        accumulation = max(1, int(self.trainer.accumulate_grad_batches))
        expected_microbatches = int(batch_idx) % accumulation + 1
        self.vggt_alignment_loss.commit_unit_codebook_update(
            expected_microbatches=expected_microbatches,
        )

    def _is_rollout_training_step(
        self,
        batch_idx: int,
        namespace: str,
        rollout_batch: dict | None,
    ) -> bool:
        return bool(
            namespace == "training"
            and rollout_batch is not None
            and self.rollout_training_enable
            and not self._diffusion_frozen
            and self.rollout_training_start_step >= 0
            and int(self.global_step) >= self.rollout_training_start_step
            and self._is_frequency_step(
                batch_idx,
                self.rollout_training_freq,
                self.rollout_training_phase,
            )
        )

    @torch.no_grad()
    def _sample_rollout_window(
        self,
        xs: Tensor,
        conditions: Tensor,
        *,
        context_source: Tensor | None = None,
        state: dict[str, Tensor] | None = None,
        initial_noise: Tensor | None = None,
        branch_mode: str | None = None,
    ) -> Tensor:
        context_frames = self.rollout_overlap_frames
        if context_source is None:
            context_source = xs[:, : self.rollout_initial_context_frames]
        known_context_frames = int(context_source.shape[1])
        if not 1 <= known_context_frames <= context_frames:
            raise ValueError(
                "Rollout context must contain between 1 and "
                f"{context_frames} frames, got {known_context_frames}"
            )
        if context_source.shape[0] != xs.shape[0] or context_source.shape[2:] != xs.shape[2:]:
            raise ValueError(
                "Rollout context source has the wrong non-temporal shape: "
                f"{tuple(context_source.shape)} for {tuple(xs.shape)}"
            )
        context = torch.zeros_like(xs)
        context[:, :known_context_frames] = context_source
        context_mask = torch.zeros(
            xs.shape[:2],
            device=xs.device,
            dtype=torch.long,
        )
        context_mask[:, :known_context_frames] = 1
        backbone = self._world_state_backbone()
        backbone.clear_world_state_condition()
        if state is not None:
            backbone.set_world_state_condition(
                state,
                query_anchors=self._world_state_query_anchors(conditions, state),
                scale=self.world_state_runtime_scale,
                detach=True,
                branch_mode=(
                    self.world_state_runtime_branch_mode
                    if branch_mode is None
                    else branch_mode
                ),
            )
        was_training = self.diffusion_model.training
        world_state_hooks_enabled = self.unit_inference_enable
        # This helper explicitly installs the intended state. Disable the
        # normal long-rollout hooks so _sample_sequence cannot clear it.
        self.unit_inference_enable = False
        self.diffusion_model.eval()
        try:
            generated, _ = self._sample_sequence(
                xs.shape[0],
                length=self.max_tokens,
                context=context,
                context_mask=context_mask,
                conditions=conditions,
                reconstruction_guidance=0.0,
                history_guidance=None,
                return_all=False,
                sampling_timesteps_override=self.rollout_sampling_steps,
                disable_progress=True,
                initial_noise=initial_noise,
            )
        finally:
            self.unit_inference_enable = world_state_hooks_enabled
            self.diffusion_model.train(was_training)
            backbone.clear_world_state_condition()
        return generated.detach()

    def _generated_context_probability(self) -> float:
        elapsed_steps = max(
            int(self.global_step) - self.rollout_training_start_step,
            0,
        )
        progress = min(
            elapsed_steps / self.rollout_generated_mix_ramp_steps,
            1.0,
        )
        return self.rollout_generated_mix_min_prob + progress * (
            self.rollout_generated_mix_max_prob
            - self.rollout_generated_mix_min_prob
        )

    def _mix_rollout_context(
        self,
        target_xs: Tensor,
        generated_previous: Tensor,
    ) -> tuple[Tensor, Tensor, float]:
        context_frames = self.rollout_overlap_frames
        generated_probability = self._generated_context_probability()
        use_generated = (
            torch.rand(target_xs.shape[0], device=target_xs.device)
            < generated_probability
        )
        generated_context = generated_previous[:, -context_frames:]
        mixed_context = torch.where(
            use_generated[:, None, None, None, None],
            generated_context,
            target_xs[:, :context_frames],
        )
        return mixed_context, use_generated, generated_probability

    def _sample_rollout_target_index(self) -> int:
        """Select one target horizon deterministically across all DDP ranks."""
        if int(self.global_step) < self.rollout_horizon_start_step:
            return 1
        # Knuth's multiplicative hash gives a reproducible low-cost schedule.
        # A million bins preserve the configured probabilities accurately while
        # keeping all ranks on identical sampling workloads.
        sample = (
            ((int(self.global_step) + 1) * 2654435761) % 1_000_000
        ) / 1_000_000
        context_probability = getattr(
            self,
            "rollout_preference_context_init_probability",
            0.0,
        )
        if sample < context_probability:
            return 0
        sample = (sample - context_probability) / max(
            1.0 - context_probability,
            1.0e-8,
        )
        cumulative = 0.0
        for horizon, probability in enumerate(
            self.rollout_horizon_probabilities,
            start=1,
        ):
            cumulative += probability
            if sample < cumulative:
                return horizon
        return self.rollout_num_windows - 1

    @staticmethod
    def _add_rollout_metric_sample(
        accumulator: dict[int, dict[str, Tensor | int]],
        horizon: int,
        metrics: dict[str, Tensor],
    ) -> None:
        metric_names = (
            "state_improvement",
            "state_win_fraction",
            "state_regret",
            "generated_fraction",
            "context_rmse",
        )
        bucket = accumulator.setdefault(horizon, {"count": 0})
        bucket["count"] = int(bucket["count"]) + 1
        for name in metric_names:
            if name not in metrics:
                continue
            value = metrics[name].detach().float()
            bucket[name] = bucket.get(name, torch.zeros_like(value)) + value

    def _accumulate_rollout_horizon_metrics(
        self,
        metrics: dict[str, Tensor],
    ) -> None:
        horizon = int(round(float(metrics["target_window_index"].item())))
        self._add_rollout_metric_sample(
            self._rollout_horizon_step_metrics,
            horizon,
            metrics,
        )

    def _commit_rollout_horizon_metrics(self) -> None:
        """Collapse microbatches into optimizer windows, then log intervals."""
        for horizon, bucket in self._rollout_horizon_step_metrics.items():
            microbatch_count = int(bucket["count"])
            if microbatch_count <= 0:
                continue
            optimizer_window_metrics = {
                name: value / microbatch_count
                for name, value in bucket.items()
                if name != "count"
            }
            self._add_rollout_metric_sample(
                self._rollout_horizon_log_metrics,
                horizon,
                optimizer_window_metrics,
            )
            self._rollout_horizon_total_counts[horizon] = (
                self._rollout_horizon_total_counts.get(horizon, 0) + 1
            )
        self._rollout_horizon_step_metrics = {}

        loss_frequency = max(1, int(self.cfg.logging.loss_freq))
        completed_step = int(self.global_step) + 1
        if completed_step % loss_frequency != 0:
            return
        total_horizon_samples = max(
            sum(self._rollout_horizon_total_counts.values()),
            1,
        )
        first_horizon = (
            0
            if getattr(
                self,
                "rollout_preference_context_init_probability",
                0.0,
            )
            > 0
            else 1
        )
        for horizon in range(first_horizon, self.rollout_num_windows):
            bucket = self._rollout_horizon_log_metrics.get(horizon)
            count = 0 if bucket is None else int(bucket["count"])
            device = self.device
            self.log(
                f"training/rollout_horizon_{horizon}_count",
                torch.tensor(float(count), device=device),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            cumulative_count = self._rollout_horizon_total_counts.get(horizon, 0)
            self.log(
                f"training/rollout_horizon_{horizon}_count_total",
                torch.tensor(float(cumulative_count), device=device),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log(
                f"training/rollout_horizon_{horizon}_fraction_total",
                torch.tensor(
                    cumulative_count / total_horizon_samples,
                    device=device,
                ),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            if not bucket or count <= 0:
                continue
            for name, value in bucket.items():
                if name == "count":
                    continue
                self.log(
                    f"training/rollout_horizon_{horizon}_{name}",
                    value / count,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
        self._rollout_horizon_log_metrics = {}

    def _encode_rollout_state(
        self,
        generated: Tensor,
        conditions: Tensor,
        previous_state: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        with torch.no_grad():
            latents, clean_logsnr = self.diffusion_model.extract_clean_alignment_latents(
                generated,
                self._process_conditions(conditions),
            )
            observation = self.vggt_alignment_loss.encode_video_world_observation(
                latents,
                clean_logsnr,
            )
            observation = self._anchor_world_observation(
                observation,
                conditions,
                previous_state,
            )
            detached_observation = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in observation.items()
            }
        state = self._update_world_state_from_observation(
            detached_observation,
            previous_state=previous_state,
            detach_previous=False,
        )
        return state, detached_observation

    def _prepare_real_rollout_training_batch(
        self,
        rollout_batch: dict[str, Tensor | None],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor],
        Tensor,
        dict[str, Tensor],
        Tensor,
        dict[str, Any],
    ]:
        windows = rollout_batch["windows"]
        if len(windows) != self.rollout_num_windows:
            raise RuntimeError(
                f"Expected {self.rollout_num_windows} rollout windows, got {len(windows)}"
            )
        if any(window["conditions"] is None for window in windows):
            raise RuntimeError("Pose-conditioned rollout requires every condition window")

        target_index = self._sample_rollout_target_index()
        if not 0 <= target_index < self.rollout_num_windows:
            raise RuntimeError(
                f"Invalid rollout target index {target_index} for "
                f"{self.rollout_num_windows} windows"
            )
        generated_fractions = []
        generated_probabilities = []
        state = None
        observation = None
        generated_previous = None
        target = windows[target_index]
        preference_keys = preference_sample_keys(
            target["xs"],
            target["conditions"],
            target_index=target_index,
            namespace=getattr(
                self,
                "rollout_preference_cache_namespace",
                "upcast-state",
            ),
        )
        context_frames = self.rollout_overlap_frames
        if target_index == 0:
            observed_frames = self.rollout_initial_context_frames
            state, observation = self._encode_observed_context_world_state(
                target["xs"][:, :observed_frames],
                target["conditions"][:, :observed_frames],
            )
            mixed_xs = target["xs"].clone()
            generated_fractions.append(target["xs"].new_zeros(()))
            generated_probabilities.append(0.0)
            context_rmse = target["xs"].new_zeros(())
        else:
            for predecessor_index in range(target_index):
                predecessor = windows[predecessor_index]
                if predecessor_index == 0:
                    generated_previous = self._sample_rollout_window(
                        predecessor["xs"],
                        predecessor["conditions"],
                        initial_noise=deterministic_noise(
                            predecessor["xs"],
                            preference_keys,
                            repeat_index=-1,
                            window_index=predecessor_index,
                        ),
                        branch_mode=(
                            getattr(
                                self,
                                "rollout_preference_predecessor_branch_mode",
                                "full",
                            )
                        ),
                    )
                else:
                    (
                        predecessor_context,
                        use_generated,
                        generated_probability,
                    ) = self._mix_rollout_context(
                        predecessor["xs"],
                        generated_previous,
                    )
                    generated_previous = self._sample_rollout_window(
                        predecessor["xs"],
                        predecessor["conditions"],
                        context_source=predecessor_context,
                        state=state,
                        initial_noise=deterministic_noise(
                            predecessor["xs"],
                            preference_keys,
                            repeat_index=-1,
                            window_index=predecessor_index,
                        ),
                        branch_mode=(
                            getattr(
                                self,
                                "rollout_preference_predecessor_branch_mode",
                                "full",
                            )
                        ),
                    )
                    generated_fractions.append(use_generated.float().mean())
                    generated_probabilities.append(generated_probability)
                state, observation = self._encode_rollout_state(
                    generated_previous,
                    predecessor["conditions"],
                    previous_state=state,
                )
            if state is None or observation is None or generated_previous is None:
                raise RuntimeError("Rollout did not produce a predecessor world state")
            mixed_context, use_generated, generated_probability = (
                self._mix_rollout_context(target["xs"], generated_previous)
            )
            generated_fractions.append(use_generated.float().mean())
            generated_probabilities.append(generated_probability)
            mixed_xs = target["xs"].clone()
            mixed_xs[:, :context_frames] = mixed_context
            context_rmse = (
                generated_previous[:, -context_frames:].float()
                - target["xs"][:, :context_frames].float()
            ).square().mean().sqrt().to(target["xs"])

        if state is None or observation is None:
            raise RuntimeError("Rollout target has no world-state observation")
        query_anchors = self._world_state_query_anchors(
            target["conditions"],
            state,
        )
        if target_index == 0:
            target_probability = self.rollout_preference_context_init_probability
        else:
            target_probability = (
                1.0
                - getattr(
                    self,
                    "rollout_preference_context_init_probability",
                    0.0,
                )
            ) * self.rollout_horizon_probabilities[target_index - 1]
        metrics = {
            "valid": target["xs"].new_ones(()),
            "generated_probability": target["xs"].new_tensor(
                sum(generated_probabilities) / len(generated_probabilities)
            ),
            "generated_fraction": torch.stack(generated_fractions).mean().to(
                target["xs"]
            ),
            "context_rmse": context_rmse,
            "state_span_code_agreement": observation[
                "span_code_agreement"
            ].detach().to(target["xs"]),
            "target_window_index": target["xs"].new_tensor(float(target_index)),
            "target_horizon_probability": target["xs"].new_tensor(
                target_probability
            ),
            "target_end_frame": target["xs"].new_tensor(
                float(target_index * (self.max_tokens - context_frames) + self.max_tokens)
            ),
            "state_confidence": state["confidence"].detach().mean().to(target["xs"]),
            "state_memory_tokens": target["xs"].new_tensor(
                float(state["physical"].shape[1])
            ),
        }
        return (
            mixed_xs,
            target["conditions"],
            target["masks"],
            state,
            query_anchors,
            metrics,
            target["xs"],
            {
                "windows": windows,
                "target_index": target_index,
                "sample_keys": preference_keys,
            },
        )

    def _rollout_boundary_loss(
        self,
        xs_pred: Tensor,
        conditioned_xs: Tensor,
        target_xs: Tensor,
        logsnr: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        boundary = self.rollout_overlap_frames
        pred_future = xs_pred[:, boundary:].float()
        target_future = target_xs[:, boundary:].detach().float()
        if pred_future.shape[1] < 1:
            raise ValueError("Rollout loss requires at least one future frame")

        pred_previous = torch.cat(
            (
                conditioned_xs[:, boundary - 1 : boundary].detach().float(),
                pred_future[:, :-1],
            ),
            dim=1,
        )
        target_previous = target_xs[:, boundary - 1 : -1].detach().float()
        pred_delta = (pred_future - pred_previous).flatten(2)
        target_delta = (target_future - target_previous).flatten(2)
        pred_delta_rms = pred_delta.square().mean(dim=-1).sqrt()
        target_delta_rms = target_delta.square().mean(dim=-1).sqrt()
        temporal_direction = 1.0 - F.cosine_similarity(
            pred_delta,
            target_delta,
            dim=-1,
            eps=1e-6,
        )
        temporal_magnitude = (
            torch.log(pred_delta_rms + 1e-4)
            - torch.log(target_delta_rms + 1e-4)
        ).abs().clamp_max(5.0)
        pred_origin = conditioned_xs[:, boundary - 1 : boundary].detach().float()
        target_origin = target_xs[:, boundary - 1 : boundary].detach().float()
        pred_trajectory = (pred_future - pred_origin).flatten(2)
        target_trajectory = (target_future - target_origin).flatten(2)
        pred_trajectory_rms = pred_trajectory.square().mean(dim=-1).sqrt()
        target_trajectory_rms = target_trajectory.square().mean(dim=-1).sqrt()
        trajectory_direction = 1.0 - F.cosine_similarity(
            pred_trajectory,
            target_trajectory,
            dim=-1,
            eps=1e-6,
        )
        trajectory_magnitude = (
            torch.log(pred_trajectory_rms + 1e-4)
            - torch.log(target_trajectory_rms + 1e-4)
        ).abs().clamp_max(5.0)
        pred_mean = pred_future.mean(dim=(-2, -1))
        target_mean = target_future.mean(dim=(-2, -1))
        pred_std = pred_future.std(dim=(-2, -1), unbiased=False)
        target_std = target_future.std(dim=(-2, -1), unbiased=False)
        color_map = F.smooth_l1_loss(
            pred_mean,
            target_mean,
            beta=0.05,
            reduction="none",
        ).mean(dim=-1) + F.smooth_l1_loss(
            pred_std,
            target_std,
            beta=0.05,
            reduction="none",
        ).mean(dim=-1)
        if logsnr.shape != conditioned_xs.shape[:2]:
            raise ValueError(
                "Boundary logSNR shape mismatch: "
                f"{tuple(logsnr.shape)} != {tuple(conditioned_xs.shape[:2])}"
            )
        signal = torch.sigmoid(logsnr[:, boundary:].float())
        valid = signal >= self.rollout_boundary_signal_threshold
        horizon = torch.linspace(
            1.0,
            2.0,
            pred_future.shape[1],
            device=signal.device,
            dtype=signal.dtype,
        ).pow(self.rollout_horizon_power)
        horizon = horizon / horizon.mean().clamp_min(1e-6)
        sample_weight = signal * valid.float() * horizon[None]

        def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
            denominator = weights.sum().clamp_min(1e-6)
            return (values * weights).sum() / denominator

        motion_mask = (target_delta_rms > 1e-3).to(sample_weight)
        trajectory_mask = (target_trajectory_rms > 1e-3).to(sample_weight)
        temporal = weighted_mean(
            temporal_direction,
            sample_weight * motion_mask,
        ) + 0.1 * weighted_mean(temporal_magnitude, sample_weight)
        trajectory = weighted_mean(
            trajectory_direction,
            sample_weight * trajectory_mask,
        ) + 0.1 * weighted_mean(trajectory_magnitude, sample_weight)
        color = weighted_mean(color_map, sample_weight)
        weighted = (
            self.rollout_boundary_temporal_weight * temporal
            + self.rollout_boundary_color_weight * color
            + self.rollout_trajectory_weight * trajectory
        )
        if not bool(valid.any()):
            weighted = xs_pred.sum() * 0.0
        return weighted, {
            "temporal": temporal.detach().to(xs_pred),
            "trajectory": trajectory.detach().to(xs_pred),
            "color": color.detach().to(xs_pred),
            "signal": signal.mean().detach().to(xs_pred),
            "signal_valid_fraction": valid.float().mean().detach().to(xs_pred),
            "future_frames": xs_pred.new_tensor(float(pred_future.shape[1])),
        }

    def _rollout_no_state_diffusion_loss_map(
        self,
        denoising_terms: dict[str, Tensor | None],
        processed_conditions: Tensor,
        state: dict[str, Tensor],
        query_anchors: Tensor | None,
    ) -> Tensor:
        """Replay the rollout denoising task without memory as a fixed baseline."""
        x_t = denoising_terms.get("x_t")
        noise = denoising_terms.get("noise")
        logsnr = denoising_terms.get("logsnr")
        if x_t is None or noise is None or logsnr is None:
            raise RuntimeError(
                "State-regret supervision requires x_t, noise, and logSNR"
            )

        backbone = self._world_state_backbone()
        adapter = backbone.world_state_adapter
        backbone_was_training = backbone.training
        adapter_was_training = adapter.training
        backbone.clear_world_state_condition()
        backbone.eval()
        try:
            with torch.no_grad():
                baseline_v = backbone(
                    x_t,
                    self.diffusion_model.precond_scale * logsnr,
                    processed_conditions,
                    external_cond_mask=denoising_terms.get("external_cond_mask"),
                )
                alpha = self.diffusion_model.add_shape_channels(
                    torch.sigmoid(logsnr).sqrt()
                )
                sigma = self.diffusion_model.add_shape_channels(
                    torch.sigmoid(-logsnr).sqrt()
                )
                baseline_noise = alpha * baseline_v + sigma * x_t
                sigmoid_bias = denoising_terms.get("sigmoid_bias")
                if sigmoid_bias is None:
                    sigmoid_bias = torch.as_tensor(
                        float(self.diffusion_model.sigmoid_bias),
                        device=logsnr.device,
                        dtype=logsnr.dtype,
                    )
                loss_weight = self.diffusion_model.add_shape_channels(
                    torch.sigmoid(sigmoid_bias - logsnr)
                )
                baseline_loss = (baseline_noise - noise).square() * loss_weight
        finally:
            backbone.train(backbone_was_training)
            adapter.train(adapter_was_training)
            backbone.set_world_state_condition(
                state,
                query_anchors=query_anchors,
                scale=self.world_state_runtime_scale,
                detach=False,
                branch_mode=getattr(
                    self, "world_state_runtime_branch_mode", "adaptive"
                ),
            )
        return baseline_loss.detach()

    @staticmethod
    def _rollout_state_regret_loss(
        state_loss_map: Tensor,
        baseline_loss_map: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Penalize memory only where it underperforms the no-memory branch."""
        if state_loss_map.shape != baseline_loss_map.shape:
            raise ValueError(
                "State and no-state diffusion loss maps must match, got "
                f"{tuple(state_loss_map.shape)} and {tuple(baseline_loss_map.shape)}"
            )
        if masks.shape != state_loss_map.shape[:2]:
            raise ValueError(
                "State-regret masks must match the batch/time axes, got "
                f"{tuple(masks.shape)} and {tuple(state_loss_map.shape[:2])}"
            )
        state_frame = state_loss_map.float().flatten(2).mean(dim=-1)
        baseline_frame = baseline_loss_map.detach().float().flatten(2).mean(dim=-1)
        valid = masks.float()
        valid_count = valid.sum().clamp_min(1.0)
        regret_map = F.relu(state_frame - baseline_frame)
        # Match DFoT's loss normalization: masked frames contribute zero while
        # the denominator remains B*T. This keeps the configured weight stable.
        regret = (regret_map * valid).mean()
        state_mean = (state_frame.detach() * valid).sum() / valid_count
        baseline_mean = (baseline_frame * valid).sum() / valid_count
        improvement = baseline_mean - state_mean
        win_fraction = (
            ((state_frame.detach() <= baseline_frame).float() * valid).sum()
            / valid_count
        )
        return regret, {
            "state_regret": regret.detach().to(state_loss_map),
            "state_diffusion_loss": state_mean.to(state_loss_map),
            "no_state_diffusion_loss": baseline_mean.to(state_loss_map),
            "state_improvement": improvement.to(state_loss_map),
            "state_win_fraction": win_fraction.to(state_loss_map),
        }

    @torch.no_grad()
    def _encode_preference_physical(
        self,
        video: Tensor,
        conditions: Tensor,
    ) -> Tensor:
        """Encode one clip in the frozen UPCAST physical-transition space."""
        backbone = self._world_state_backbone()
        backbone.clear_world_state_condition()
        latents, clean_logsnr = (
            self.diffusion_model.extract_clean_alignment_latents(
                video,
                self._process_conditions(conditions),
            )
        )
        observation = self.vggt_alignment_loss.encode_video_world_observation(
            latents,
            clean_logsnr,
        )
        backbone.clear_world_state_condition()
        return observation["physical_spans"].float()

    @torch.no_grad()
    def _preference_physical_scores(
        self,
        branch_videos: dict[str, Tensor],
        conditions: Tensor,
        *,
        target_physical: Tensor,
    ) -> dict[str, Tensor]:
        """Compare video-side UPCAST transitions to one cached GT transition."""
        target_physical = target_physical.detach().float()
        scores = {}
        for mode, video in branch_videos.items():
            branch_physical = self._encode_preference_physical(video, conditions)
            if branch_physical.shape != target_physical.shape:
                raise RuntimeError(
                    "Preference physical transition shape mismatch: "
                    f"{tuple(branch_physical.shape)} != "
                    f"{tuple(target_physical.shape)}"
                )
            scores[mode] = (
                1.0
                - F.cosine_similarity(
                    branch_physical,
                    target_physical,
                    dim=-1,
                    eps=1e-6,
                )
            ).flatten(1).mean(dim=1)
        return scores

    @staticmethod
    def _preference_state_scores(
        branch_states: dict[str, dict[str, Tensor]],
        target_state: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Score the accumulated physical memory, not only one local window."""

        def summarize(state: dict[str, Tensor]) -> Tensor:
            physical = state["physical"].float()
            confidence = state.get("confidence")
            if confidence is None:
                confidence = physical.new_ones(*physical.shape[:2], 1)
            confidence = confidence.float().clamp(0.0, 1.0)
            summary = (physical * confidence).sum(dim=1) / confidence.sum(
                dim=1
            ).clamp_min(1.0e-6)
            return F.layer_norm(summary, (summary.shape[-1],))

        target_summary = summarize(target_state).detach()
        return {
            mode: 1.0
            - F.cosine_similarity(
                summarize(state),
                target_summary,
                dim=-1,
                eps=1.0e-6,
            )
            for mode, state in branch_states.items()
        }

    @staticmethod
    def _preference_soft_targets(
        branch_scores: dict[str, dict[str, Tensor]],
        component_weights: dict[str, float],
        temperature: float,
        relative_floor: float = 1.0e-3,
        absolute_floor: float = 1.0e-6,
        confidence_temperature: float = 0.25,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Convert lower-is-better counterfactuals into calibrated targets.

        Absolute score normalization can hide small branch differences when
        the scores themselves are O(1). Each component is instead normalized
        by its paired branch spread.
        A score-relative floor and reliability term prevent numerical noise
        from becoming a confident preference when all branches are equal.
        """
        factorial = "appearance" in branch_scores
        required_modes = (
            ("none", "physical", "appearance", "full")
            if factorial
            else ("none", "physical", "full")
        )
        for mode in required_modes:
            if mode not in branch_scores:
                raise ValueError(f"Missing sampled preference mode {mode!r}")
        weight_sum = sum(float(value) for value in component_weights.values())
        if weight_sum <= 0:
            raise ValueError("Preference component weights must have positive mass")
        combined = {
            mode: next(iter(branch_scores[mode].values())).new_zeros(
                next(iter(branch_scores[mode].values())).shape
            )
            for mode in required_modes
        }
        diagnostics: dict[str, Tensor] = {}
        for component, raw_weight in component_weights.items():
            values = torch.stack(
                [branch_scores[mode][component].float() for mode in required_modes],
                dim=0,
            )
            score_level = values.abs().mean(dim=0)
            spread = values.amax(dim=0) - values.amin(dim=0)
            floor = torch.maximum(
                score_level * float(relative_floor),
                values.new_full(score_level.shape, float(absolute_floor)),
            )
            scale = torch.maximum(0.5 * spread, floor)
            reliability = spread / (spread + 2.0 * floor)
            centered = values - values.mean(dim=0, keepdim=True)
            normalized = centered / scale
            weight = float(raw_weight) / weight_sum
            for index, mode in enumerate(required_modes):
                combined[mode] = (
                    combined[mode]
                    + weight * reliability * normalized[index]
                )
            diagnostics[f"{component}_spread"] = spread
            diagnostics[f"{component}_reliability"] = reliability
        if factorial:
            # Average each branch's marginal utility with the other branch off
            # and on. This is the two-player Shapley value and does not assume
            # that physical conditioning precedes appearance conditioning.
            physical_advantage = 0.5 * (
                combined["none"]
                - combined["physical"]
                + combined["appearance"]
                - combined["full"]
            )
            appearance_advantage = 0.5 * (
                combined["none"]
                - combined["appearance"]
                + combined["physical"]
                - combined["full"]
            )
            diagnostics["branch_interaction"] = (
                combined["physical"]
                + combined["appearance"]
                - combined["none"]
                - combined["full"]
            )
        else:
            # Backward-compatible target construction for isolated unit tests
            # and analysis artifacts created before appearance-only rollouts.
            physical_advantage = combined["none"] - combined["physical"]
            appearance_advantage = combined["physical"] - combined["full"]
        physical_target = torch.sigmoid(physical_advantage / float(temperature))
        appearance_target = torch.sigmoid(
            appearance_advantage / float(temperature)
        )
        physical_confidence = torch.tanh(
            physical_advantage.abs() / float(confidence_temperature)
        )
        appearance_confidence = torch.tanh(
            appearance_advantage.abs() / float(confidence_temperature)
        )
        return (
            physical_target,
            appearance_target,
            physical_advantage,
            appearance_advantage,
            physical_confidence,
            appearance_confidence,
            diagnostics,
        )

    def _preference_target_spec(self) -> tuple[dict[str, Any], str]:
        """Describe every choice that can change cached utility labels."""
        adapter = self._world_state_backbone().world_state_adapter
        payload = {
            "schema": 1,
            "source_id": self.rollout_preference_source_id,
            "predecessor_branch_mode": (
                self.rollout_preference_predecessor_branch_mode
            ),
            "sampling_steps": self.rollout_sampling_steps,
            "num_windows": self.rollout_sampled_preference_num_windows,
            "overlap_frames": self.rollout_overlap_frames,
            "initial_context_frames": self.rollout_initial_context_frames,
            "generated_mix_min": self.rollout_generated_mix_min_prob,
            "generated_mix_max": self.rollout_generated_mix_max_prob,
            "horizon_power": self.rollout_horizon_power,
            "runtime_scale": self.world_state_runtime_scale,
            "initialize_from_context": self.world_state_initialize_from_context,
            "context_min_frames": self.world_state_context_min_frames,
            "context_max_frames": self.world_state_context_max_frames,
            "memory_grid_size": adapter.memory_grid_size,
            "max_state_tokens": adapter.max_state_tokens,
            "visibility_threshold": adapter.visibility_threshold,
            "visibility_temperature": adapter.visibility_temperature,
            "state_weight": self.rollout_preference_state_weight,
            "dino_weight": self.rollout_preference_dino_weight,
            "vjepa_temporal_weight": (
                self.rollout_preference_vjepa_temporal_weight
            ),
            "vjepa_spatial_weight": self.rollout_preference_vjepa_spatial_weight,
            "physical_weight": self.rollout_preference_physical_weight,
            "temperature": self.rollout_sampled_preference_temperature,
            "confidence_temperature": (
                self.rollout_preference_confidence_temperature
            ),
            "relative_floor": self.rollout_preference_relative_floor,
            "absolute_floor": self.rollout_preference_absolute_floor,
            "max_correction": adapter.preference_max_correction,
            "dino_model": getattr(
                self.vggt_alignment_loss,
                "_preference_dino_model_name",
                "unknown",
            ),
            "video_model": getattr(
                self.vggt_alignment_loss,
                "_preference_video_model_name",
                "unknown",
            ),
            "video_num_frames": getattr(
                self.vggt_alignment_loss,
                "_preference_video_num_frames",
                self.max_tokens,
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload, signature

    @torch.no_grad()
    def _prepare_counterfactual_preference_targets(
        self,
        target_xs: Tensor,
        conditions: Tensor,
        state: dict[str, Tensor],
        preference_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute repeat-invariant GT scorer features exactly once."""
        windows = preference_context["windows"]
        target_index = int(preference_context["target_index"])
        prepared_windows = []
        target_state = state
        for offset in range(self.rollout_sampled_preference_num_windows):
            window_index = target_index + offset
            window = windows[window_index]
            window_target = target_xs if offset == 0 else window["xs"]
            window_conditions = conditions if offset == 0 else window["conditions"]
            target_rgb = self._unnormalize_x(window_target).float().clamp(0.0, 1.0)
            prepared_windows.append(
                {
                    "visual": self.vggt_alignment_loss.prepare_preference_targets(
                        target_rgb
                    ),
                    "physical": self._encode_preference_physical(
                        window_target,
                        window_conditions,
                    ).detach(),
                }
            )
            target_state, _ = self._encode_rollout_state(
                window_target,
                window_conditions,
                previous_state=target_state,
            )
        return {
            "windows": prepared_windows,
            "final_state": target_state,
        }

    @torch.no_grad()
    def _counterfactual_preference_scores(
        self,
        conditioned_xs: Tensor,
        conditions: Tensor,
        state: dict[str, Tensor],
        preference_context: dict[str, Any],
        prepared_targets: dict[str, Any],
        *,
        repeat_index: int,
    ) -> dict[str, dict[str, Tensor]]:
        """Evaluate all factorial branches with deterministic per-sample noise."""
        windows = preference_context["windows"]
        target_index = int(preference_context["target_index"])
        sample_keys = list(preference_context["sample_keys"])
        available_windows = len(windows) - target_index
        num_windows = min(
            self.rollout_sampled_preference_num_windows,
            available_windows,
        )
        if num_windows != self.rollout_sampled_preference_num_windows:
            raise RuntimeError(
                "Cumulative preference target does not have enough windows: "
                f"need {self.rollout_sampled_preference_num_windows}, "
                f"have {available_windows} from target {target_index}"
            )
        modes = ("none", "physical", "appearance", "full")
        branch_states = {mode: state for mode in modes}
        target_state = prepared_targets["final_state"]
        previous_videos: dict[str, Tensor] | None = None
        branch_scores: dict[str, dict[str, Tensor]] = {
            mode: {} for mode in modes
        }
        total_horizon_weight = 0.0
        for offset in range(num_windows):
            window_index = target_index + offset
            window = windows[window_index]
            window_conditions = conditions if offset == 0 else window["conditions"]
            template = conditioned_xs if offset == 0 else window["xs"]
            shared_noise = deterministic_noise(
                template,
                sample_keys,
                repeat_index=repeat_index,
                window_index=window_index,
            )
            sampled: dict[str, Tensor] = {}
            for mode in modes:
                if offset == 0:
                    context_source = conditioned_xs[
                        :, : self.rollout_overlap_frames
                    ]
                else:
                    context_source = previous_videos[mode][
                        :, -self.rollout_overlap_frames :
                    ]
                sampled[mode] = self._sample_rollout_window(
                    template,
                    window_conditions,
                    context_source=context_source,
                    state=branch_states[mode],
                    initial_noise=shared_noise,
                    branch_mode=mode,
                )

            sampled_rgb = {
                mode: self._unnormalize_x(video).float().clamp(0.0, 1.0)
                for mode, video in sampled.items()
            }
            preference_targets = prepared_targets["windows"][offset]["visual"]
            horizon_weight = float(offset + 1) ** self.rollout_horizon_power
            total_horizon_weight += horizon_weight
            for mode, video in sampled_rgb.items():
                components = self.vggt_alignment_loss.compute_preference_components(
                    video,
                    preference_targets,
                    future_start=self.rollout_overlap_frames,
                )
                for component, score in components.items():
                    branch_scores[mode][component] = (
                        branch_scores[mode].get(
                            component,
                            torch.zeros_like(score),
                        )
                        + horizon_weight * score
                    )
            physical_scores = self._preference_physical_scores(
                sampled,
                window_conditions,
                target_physical=prepared_targets["windows"][offset]["physical"],
            )
            for mode, score in physical_scores.items():
                branch_scores[mode]["physical"] = (
                    branch_scores[mode].get(
                        "physical",
                        torch.zeros_like(score),
                    )
                    + horizon_weight * score
                )

            for mode in modes:
                branch_states[mode], _ = self._encode_rollout_state(
                    sampled[mode],
                    window_conditions,
                    previous_state=branch_states[mode],
                )
            previous_videos = sampled

        for mode in modes:
            for component in tuple(branch_scores[mode]):
                branch_scores[mode][component] = (
                    branch_scores[mode][component] / total_horizon_weight
                )
        if self.rollout_preference_state_weight > 0:
            state_scores = self._preference_state_scores(
                branch_states,
                target_state,
            )
            state_weight = self.rollout_preference_state_weight
            for mode, score in state_scores.items():
                branch_scores[mode]["physical_state"] = score.to(conditioned_xs)
                branch_scores[mode]["physical"] = (
                    branch_scores[mode]["physical"]
                    + state_weight * score.to(conditioned_xs)
                ) / (1.0 + state_weight)
        return branch_scores

    def _rollout_sampled_preference_loss(
        self,
        conditioned_xs: Tensor,
        target_xs: Tensor,
        conditions: Tensor,
        state: dict[str, Tensor],
        query_anchors: Tensor | None,
        gate_probabilities: dict[str, Tensor],
        preference_context: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Regress a stable window action from repeated cached utilities."""
        physical_probability = gate_probabilities.get("physical")
        appearance_probability = gate_probabilities.get("appearance")
        if physical_probability is None or appearance_probability is None:
            raise RuntimeError(
                "Stable preference supervision requires the live window action"
            )
        if physical_probability.ndim != 1 or appearance_probability.ndim != 1:
            raise RuntimeError("Stable preference probabilities must be [B]")
        if physical_probability.shape[0] != conditioned_xs.shape[0]:
            raise RuntimeError("Stable preference batch does not match rollout batch")

        adapter = self._world_state_backbone().world_state_adapter
        policy_inputs = adapter.preference_cache_inputs
        if not policy_inputs:
            raise RuntimeError(
                "Stable preference policy inputs were not captured for this window"
            )
        sample_keys = list(preference_context["sample_keys"])
        if len(sample_keys) != conditioned_xs.shape[0]:
            raise RuntimeError("Preference cache key batch mismatch")

        target_spec, target_signature = self._preference_target_spec()
        current_policy_spec = policy_spec_dict(adapter.preference_policy.spec)
        cache_entries: list[dict[str, Any] | None] = [None] * len(sample_keys)
        can_read = self.rollout_preference_cache_mode in {"read", "read_write"}
        if can_read and self._preference_cache is not None:
            cache_entries = [self._preference_cache.load(key) for key in sample_keys]
            cache_entries = [
                entry
                if entry is not None
                and int(entry.get("noise_repeats", 0))
                >= self.rollout_preference_noise_repeats
                and int(entry.get("sampling_steps", -1))
                == self.rollout_sampling_steps
                and entry.get("target_signature") == target_signature
                and entry.get("policy_spec") == current_policy_spec
                else None
                for entry in cache_entries
            ]
        cache_hits = sum(entry is not None for entry in cache_entries)
        if self.rollout_preference_cache_mode == "read" and cache_hits != len(
            cache_entries
        ):
            missing = [
                key for key, entry in zip(sample_keys, cache_entries) if entry is None
            ]
            raise RuntimeError(
                "Read-only preference cache is missing compatible entries: "
                + ", ".join(missing[:3])
            )

        target_diagnostics: dict[str, Tensor] = {}
        mean_branch_scores: dict[str, dict[str, Tensor]] = {}
        if cache_hits == len(cache_entries) and cache_entries:
            target_names = (
                "physical_target_correction",
                "physical_confidence",
                "physical_advantage_mean",
                "physical_advantage_std",
                "appearance_target_correction",
                "appearance_confidence",
                "appearance_advantage_mean",
                "appearance_advantage_std",
            )
            targets = {
                name: torch.stack(
                    [torch.as_tensor(entry["targets"][name]) for entry in cache_entries]
                ).to(conditioned_xs)
                for name in target_names
            }
            splits = [str(entry["split"]) for entry in cache_entries]
        else:
            component_weights = {
                "dino": self.rollout_preference_dino_weight,
                "vjepa_temporal": self.rollout_preference_vjepa_temporal_weight,
                "vjepa_spatial": self.rollout_preference_vjepa_spatial_weight,
                "physical": self.rollout_preference_physical_weight,
            }
            physical_advantages = []
            appearance_advantages = []
            repeated_scores = []
            repeated_diagnostics = []
            prepared_targets = self._prepare_counterfactual_preference_targets(
                target_xs,
                conditions,
                state,
                preference_context,
            )
            for repeat_index in range(self.rollout_preference_noise_repeats):
                scores = self._counterfactual_preference_scores(
                    conditioned_xs,
                    conditions,
                    state,
                    preference_context,
                    prepared_targets,
                    repeat_index=repeat_index,
                )
                repeated_scores.append(scores)
                target = self._preference_soft_targets(
                    scores,
                    component_weights,
                    self.rollout_sampled_preference_temperature,
                    self.rollout_preference_relative_floor,
                    self.rollout_preference_absolute_floor,
                    self.rollout_preference_confidence_temperature,
                )
                physical_advantages.append(target[2])
                appearance_advantages.append(target[3])
                repeated_diagnostics.append(target[-1])
            physical_advantages_tensor = torch.stack(physical_advantages)
            appearance_advantages_tensor = torch.stack(appearance_advantages)
            targets = continuous_targets_from_repeats(
                physical_advantages_tensor,
                appearance_advantages_tensor,
                temperature=self.rollout_sampled_preference_temperature,
                confidence_temperature=self.rollout_preference_confidence_temperature,
                max_correction=adapter.preference_max_correction,
            )
            target_diagnostics = {
                name: torch.stack(
                    [diagnostics[name] for diagnostics in repeated_diagnostics]
                ).mean(dim=0)
                for name in repeated_diagnostics[0]
            }
            for mode in repeated_scores[0]:
                mean_branch_scores[mode] = {}
                for component in repeated_scores[0][mode]:
                    mean_branch_scores[mode][component] = torch.stack(
                        [scores[mode][component] for scores in repeated_scores]
                    ).mean(dim=0)
            splits = [
                split_for_key(key, self.rollout_preference_holdout_fraction)
                for key in sample_keys
            ]
            can_write = self.rollout_preference_cache_mode in {
                "write",
                "read_write",
            }
            if can_write and self._preference_cache is not None:
                for index, key in enumerate(sample_keys):
                    payload_targets = {
                        name: value[index].detach().float().cpu()
                        for name, value in targets.items()
                    }
                    self._preference_cache.save(
                        key,
                        {
                            "split": splits[index],
                            "noise_repeats": self.rollout_preference_noise_repeats,
                            "sampling_steps": self.rollout_sampling_steps,
                            "num_windows": self.rollout_sampled_preference_num_windows,
                            "target_index": int(preference_context["target_index"]),
                            "target_spec": target_spec,
                            "target_signature": target_signature,
                            "policy_spec": current_policy_spec,
                            "policy_inputs": detached_policy_inputs(
                                policy_inputs,
                                index,
                            ),
                            "targets": payload_targets,
                        },
                    )

        predictions = {
            "physical_correction": adapter.preference_max_correction
            * (2.0 * physical_probability.float() - 1.0),
            "appearance_correction": adapter.preference_max_correction
            * (2.0 * appearance_probability.float() - 1.0),
        }
        train_mask = torch.tensor(
            [split == "train" for split in splits],
            device=conditioned_xs.device,
            dtype=torch.bool,
        )
        loss, loss_metrics = continuous_preference_loss(
            predictions,
            targets,
            beta=self.rollout_preference_regression_beta,
            identity_weight=self.rollout_preference_identity_weight,
            sample_mask=train_mask,
        )

        self._world_state_backbone().set_world_state_condition(
            state,
            query_anchors=query_anchors,
            scale=self.world_state_runtime_scale,
            detach=False,
            branch_mode=self.world_state_runtime_branch_mode,
        )
        metrics = {
            "preference_loss": loss.detach().to(conditioned_xs),
            "preference_regression_loss": loss_metrics["regression"].to(conditioned_xs),
            "preference_identity_loss": loss_metrics["identity"].to(conditioned_xs),
            "preference_physical_regression": loss_metrics[
                "physical_regression"
            ].to(conditioned_xs),
            "preference_appearance_regression": loss_metrics[
                "appearance_regression"
            ].to(conditioned_xs),
            "preference_physical_target_correction": targets[
                "physical_target_correction"
            ].mean().to(conditioned_xs),
            "preference_appearance_target_correction": targets[
                "appearance_target_correction"
            ].mean().to(conditioned_xs),
            "preference_physical_prediction_correction": predictions[
                "physical_correction"
            ].detach().mean().to(conditioned_xs),
            "preference_appearance_prediction_correction": predictions[
                "appearance_correction"
            ].detach().mean().to(conditioned_xs),
            "preference_physical_advantage": targets[
                "physical_advantage_mean"
            ].mean().to(conditioned_xs),
            "preference_appearance_advantage": targets[
                "appearance_advantage_mean"
            ].mean().to(conditioned_xs),
            "preference_physical_advantage_std": targets[
                "physical_advantage_std"
            ].mean().to(conditioned_xs),
            "preference_appearance_advantage_std": targets[
                "appearance_advantage_std"
            ].mean().to(conditioned_xs),
            "preference_physical_confidence": targets[
                "physical_confidence"
            ].mean().to(conditioned_xs),
            "preference_appearance_confidence": targets[
                "appearance_confidence"
            ].mean().to(conditioned_xs),
            "preference_cache_hit_fraction": conditioned_xs.new_tensor(
                cache_hits / max(len(cache_entries), 1)
            ),
            "preference_train_fraction": train_mask.float().mean(),
            "preference_holdout_fraction": (~train_mask).float().mean(),
            "preference_noise_repeats": conditioned_xs.new_tensor(
                float(self.rollout_preference_noise_repeats)
            ),
            "preference_num_windows": conditioned_xs.new_tensor(
                float(self.rollout_sampled_preference_num_windows)
            ),
        }
        for name, value in target_diagnostics.items():
            metrics[f"preference_{name}"] = value.detach().mean().to(conditioned_xs)
        for mode, components in mean_branch_scores.items():
            for component, score in components.items():
                metrics[f"preference_{mode}_{component}"] = (
                    score.detach().mean().to(conditioned_xs)
                )
        return loss, metrics

    def _cap_auxiliary_adapter_gradient(
        self,
        auxiliary_loss: Tensor,
        reference_loss: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = auxiliary_loss.detach().new_zeros(())
        metrics = {
            "gradient_scale": auxiliary_loss.detach().new_ones(()),
            "gradient_ratio_before_cap": zero,
            "gradient_reference_norm": zero,
            "gradient_auxiliary_norm": zero,
            "gradient_probe_scale": auxiliary_loss.detach().new_tensor(
                self.rollout_gradient_probe_scale
            ),
            "gradient_reference_valid": zero,
        }
        if (
            self.rollout_boundary_max_grad_ratio <= 0
            or not auxiliary_loss.requires_grad
            or not reference_loss.requires_grad
        ):
            if self.rollout_boundary_max_grad_ratio <= 0:
                metrics["gradient_scale"] = zero
                return auxiliary_loss * 0.0, metrics
            return auxiliary_loss, metrics
        adapter = self._world_state_backbone().world_state_adapter
        parameters = [
            parameter for parameter in adapter.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(
                "Rollout gradient cap found no trainable world-state adapter parameters"
            )
        # Lightning's GradScaler scales the final backward pass, but this
        # diagnostic autograd call runs before that happens.  With fp16
        # activations, the small diffusion gradient reaching the zero-init
        # adapter can therefore underflow to zero and incorrectly suppress the
        # entire auxiliary objective.  Probe both objectives at the same AMP-
        # sized scale; their ratio is unchanged after dividing the norms back.
        probe_scale = self.rollout_gradient_probe_scale
        reference_grads = torch.autograd.grad(
            reference_loss.float() * probe_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_grads = torch.autograd.grad(
            auxiliary_loss.float() * probe_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        def norm(grads) -> Tensor:
            terms = [
                gradient.detach().float().square().sum()
                for gradient in grads
                if gradient is not None
            ]
            if not terms:
                return auxiliary_loss.detach().float().new_zeros(())
            return torch.stack(terms).sum().sqrt()

        reference_norm = norm(reference_grads) / probe_scale
        auxiliary_norm = norm(auxiliary_grads) / probe_scale
        if not bool(torch.isfinite(reference_norm)) or not bool(
            torch.isfinite(auxiliary_norm)
        ):
            raise RuntimeError(
                "Non-finite rollout adapter gradient probe: "
                f"reference={reference_norm.item():.6g}, "
                f"auxiliary={auxiliary_norm.item():.6g}."
            )
        reference_valid = reference_norm > 1e-12
        auxiliary_valid = auxiliary_norm > 1e-12
        if bool(auxiliary_valid) and not bool(reference_valid):
            message = (
                "Rollout auxiliary gradient is non-zero but the diffusion "
                "reference gradient does not reach the world-state adapter "
                f"after a {probe_scale:g}x AMP-safe probe."
            )
            if self.strict_rollout_training:
                raise RuntimeError(message)
            metrics.update(
                {
                    "gradient_scale": zero,
                    "gradient_reference_norm": reference_norm.detach(),
                    "gradient_auxiliary_norm": auxiliary_norm.detach(),
                }
            )
            return auxiliary_loss * 0.0, metrics
        ratio = auxiliary_norm / reference_norm.clamp_min(1e-12)
        scale = (
            self.rollout_boundary_max_grad_ratio
            * reference_norm
            / auxiliary_norm.clamp_min(1e-12)
        ).clamp(max=1.0).detach()
        metrics = {
            "gradient_scale": scale,
            "gradient_ratio_before_cap": ratio.detach(),
            "gradient_reference_norm": reference_norm.detach(),
            "gradient_auxiliary_norm": auxiliary_norm.detach(),
            "gradient_probe_scale": auxiliary_loss.detach().new_tensor(
                probe_scale
            ),
            "gradient_reference_valid": reference_valid.detach().to(
                auxiliary_loss
            ),
        }
        return auxiliary_loss * scale, metrics

    @staticmethod
    def _detach_tail_state(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, list):
            return [DFoTUPCAST._detach_tail_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                DFoTUPCAST._detach_tail_state(item) for item in value
            )
        if isinstance(value, dict):
            return {
                key: DFoTUPCAST._detach_tail_state(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _gather_video_frame(value: Tensor, indices: Tensor) -> Tensor:
        if value.ndim < 2 or value.shape[0] != indices.shape[0]:
            raise ValueError(
                f"Cannot gather frames {tuple(indices.shape)} from {tuple(value.shape)}"
            )
        batch = torch.arange(value.shape[0], device=value.device)
        return value[batch, indices]

    def _generative_fm_error_per_sample(
        self,
        v_pred: Tensor,
        denoising_terms: dict[str, Tensor | None],
        future: Tensor,
    ) -> Tensor:
        x_t = denoising_terms.get("x_t")
        noise = denoising_terms.get("noise")
        logsnr = denoising_terms.get("logsnr")
        if x_t is None or noise is None or logsnr is None:
            raise RuntimeError(
                "Physical conditioner requires x_t, noise, and logSNR"
            )
        frame_v = self._gather_video_frame(v_pred, future).float()
        frame_x_t = self._gather_video_frame(x_t, future).float()
        frame_noise = self._gather_video_frame(noise, future).float()
        frame_logsnr = self._gather_video_frame(logsnr, future).float()
        alpha = torch.sigmoid(frame_logsnr).sqrt()
        sigma = torch.sigmoid(-frame_logsnr).sqrt()
        while alpha.ndim < frame_v.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        noise_pred = alpha * frame_v + sigma * frame_x_t
        error = (noise_pred - frame_noise).square().flatten(1).mean(dim=1)
        sigmoid_bias = float(
            torch.as_tensor(
                denoising_terms.get("sigmoid_bias", -1.0)
            ).item()
        )
        weight = torch.sigmoid(sigmoid_bias - frame_logsnr)
        return error * weight

    @staticmethod
    def _masked_sample_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _initialize_generative_teacher_gate(self) -> None:
        if (
            self.generative_unit_mode != "conditional_teacher"
            or self._generative_teacher_gate_initialized
        ):
            return
        conditioner = self._generative_unit_backbone().generative_unit_conditioner
        target = float(self.generative_teacher_initial_gate)
        with torch.no_grad():
            current = float(torch.tanh(conditioner.read_gate.float()))
            if current < target:
                target_tensor = conditioner.read_gate.new_tensor(target).clamp(
                    -1.0 + 1.0e-6,
                    1.0 - 1.0e-6,
                )
                conditioner.read_gate.copy_(torch.atanh(target_tensor))
        self._generative_teacher_gate_initialized = True
        rank_zero_print(
            "[DFoTUPCAST] Conditional teacher read scale initialized "
            f"to {conditioner.max_read_scale * max(current, target):.6f}."
        )

    @staticmethod
    def _shuffle_generative_physical_tokens(tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "Generative physical tokens must have shape [B,N,C]"
            )
        if tokens.shape[0] > 1:
            return tokens.roll(shifts=1, dims=0)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                # Formal training uses one sample per rank. A local spatial
                # roll retains exactly the same token set and is nearly
                # invisible to global cross-attention. Gather detached tokens
                # and use the next rank's scene as a true counterfactual.
                gathered = [torch.empty_like(tokens) for _ in range(world_size)]
                dist.all_gather(gathered, tokens.detach())
                negative_rank = (dist.get_rank() + 1) % world_size
                return gathered[negative_rank]
        token_count = int(tokens.shape[1])
        grid = int(round(token_count**0.5))
        if grid * grid == token_count and grid > 1:
            spatial = tokens.reshape(1, grid, grid, tokens.shape[-1])
            return spatial.roll(shifts=max(1, grid // 2), dims=2).reshape_as(tokens)
        return tokens.roll(shifts=max(1, token_count // 2), dims=1)

    @staticmethod
    def _generative_pair_tokens(
        tokens,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the aligned video/fused UPCAST pair in the injection dtype."""
        video = tokens.video.detach().to(reference)
        fused = tokens.fused.detach().to(reference)
        if video.shape != fused.shape:
            raise RuntimeError(
                "Physical conditioner video/fused token shapes differ: "
                f"{tuple(video.shape)} vs {tuple(fused.shape)}"
            )
        return video, fused

    @staticmethod
    def _generative_camera_centers(conditions: Tensor | None) -> Tensor | None:
        """Recover camera centers from raw RE10K poses or bare extrinsics."""
        if (
            not torch.is_tensor(conditions)
            or conditions.ndim != 3
            or conditions.shape[-1] not in {12, 16}
        ):
            return None
        raw = conditions.detach().float()
        if raw.shape[-1] == 16:
            # Match the model's camera conditioning convention exactly:
            # [fx, fy, px, py, flattened world-to-camera extrinsics].
            poses = CameraPose.from_vectors(raw).extrinsics(flatten=False)
        else:
            poses = raw.reshape(raw.shape[0], raw.shape[1], 3, 4)
        rotation = poses[..., :3]
        translation = poses[..., 3]
        centers = -torch.matmul(
            rotation.transpose(-1, -2),
            translation.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.isfinite(centers).all()):
            return None
        return centers

    def _generative_coarse_latents(self, values: Tensor) -> Tensor:
        if values.ndim != 5:
            raise ValueError(
                "Generative motion expects [B,T,C,H,W], got "
                f"{tuple(values.shape)}"
            )
        batch, frames, channels = values.shape[:3]
        pooled = F.adaptive_avg_pool2d(
            values.float().reshape(batch * frames, channels, *values.shape[-2:]),
            output_size=(
                self.generative_motion_spatial_grid,
                self.generative_motion_spatial_grid,
            ),
        )
        return pooled.reshape(
            batch,
            frames,
            channels,
            self.generative_motion_spatial_grid,
            self.generative_motion_spatial_grid,
        )

    def _generative_pair_motion_salience(
        self,
        clean_xs: Tensor,
        conditions: Tensor | None,
        current: Tensor,
        future: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Measure how much scene/camera change the sampled UPCAST pair carries."""
        coarse = self._generative_coarse_latents(clean_xs.detach())
        batch = torch.arange(coarse.shape[0], device=coarse.device)
        pair_delta = coarse[batch, future] - coarse[batch, current]
        latent_rms = pair_delta.flatten(1).square().mean(dim=1).sqrt()
        adjacent_rms = (
            (coarse[:, 1:] - coarse[:, :-1])
            .flatten(2)
            .square()
            .mean(dim=-1)
            .sqrt()
            .mean(dim=1)
        )
        latent_score = latent_rms / (latent_rms + adjacent_rms + 1.0e-6)

        camera_score = latent_score.new_zeros(latent_score.shape)
        camera_centers = self._generative_camera_centers(conditions)
        if camera_centers is not None and camera_centers.shape[:2] == coarse.shape[:2]:
            camera_centers = camera_centers.to(coarse)
            camera_delta = (
                camera_centers[batch, future] - camera_centers[batch, current]
            ).norm(dim=-1)
            adjacent_camera = (
                camera_centers[:, 1:] - camera_centers[:, :-1]
            ).norm(dim=-1).mean(dim=1)
            camera_score = camera_delta / (
                camera_delta + adjacent_camera + 1.0e-6
            )
        salience = torch.maximum(latent_score, camera_score).clamp(0.0, 1.0)
        return salience.detach(), {
            "generative_pair_latent_motion_rms": latent_rms.mean().detach(),
            "generative_pair_latent_motion_score": latent_score.mean().detach(),
            "generative_pair_camera_motion_score": camera_score.mean().detach(),
            "generative_pair_motion_salience": salience.mean().detach(),
        }

    @staticmethod
    def _generative_frame_supervision_mask(
        masks: Tensor,
        batch_size: int,
        num_frames: int,
    ) -> Tensor:
        if masks.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                "Generative motion mask shape mismatch: "
                f"{tuple(masks.shape)} vs {(batch_size, num_frames)}"
            )
        if masks.ndim == 2:
            return masks.bool()
        return masks.reshape(batch_size, num_frames, -1).any(dim=-1)

    def _generative_motion_consistency_loss(
        self,
        xs_pred: Tensor,
        xs_target: Tensor,
        logsnr: Tensor,
        masks: Tensor,
        conditions: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Align coarse multi-span motion without regressing texture values."""
        if xs_pred.shape != xs_target.shape:
            raise ValueError(
                "Generative motion prediction/target shape mismatch: "
                f"{tuple(xs_pred.shape)} vs {tuple(xs_target.shape)}"
            )
        batch_size, num_frames = xs_pred.shape[:2]
        if logsnr.shape != (batch_size, num_frames):
            raise ValueError(
                "Generative motion logSNR shape mismatch: "
                f"{tuple(logsnr.shape)} vs {(batch_size, num_frames)}"
            )
        predicted = self._generative_coarse_latents(xs_pred)
        target = self._generative_coarse_latents(xs_target.detach())
        supervised = self._generative_frame_supervision_mask(
            masks,
            batch_size,
            num_frames,
        )
        signal = torch.sigmoid(logsnr.detach().float())
        camera_centers = self._generative_camera_centers(conditions)
        if camera_centers is not None:
            camera_centers = camera_centers.to(predicted)

        zero = xs_pred.sum() * 0.0
        direction_numerator = zero.float()
        direction_denominator = zero.detach().float()
        magnitude_numerator = zero.float()
        magnitude_denominator = zero.detach().float()
        ratio_numerator = zero.float()
        valid_count = zero.detach().float()
        possible_count = zero.detach().float()
        camera_weight_numerator = zero.detach().float()
        camera_score_numerator = zero.detach().float()
        target_rms_numerator = zero.detach().float()
        predicted_rms_numerator = zero.detach().float()

        for stride in self.generative_motion_strides:
            if stride >= num_frames:
                continue
            predicted_delta = predicted[:, stride:] - predicted[:, :-stride]
            target_delta = target[:, stride:] - target[:, :-stride]
            predicted_flat = predicted_delta.flatten(2)
            target_flat = target_delta.flatten(2)
            predicted_rms = (
                predicted_flat.square().mean(dim=-1) + 1.0e-12
            ).sqrt()
            target_rms = (
                target_flat.square().mean(dim=-1) + 1.0e-12
            ).sqrt()
            pair_signal = (
                signal[:, stride:] * signal[:, :-stride]
            ).clamp_min(0.0).sqrt()
            valid = (
                supervised[:, stride:]
                & (pair_signal >= self.generative_motion_signal_threshold)
            )
            possible_count = possible_count + valid.new_tensor(
                float(valid.numel()),
                dtype=torch.float32,
            )
            if not bool(valid.any()):
                continue

            reference_scale = target_rms.detach().mean(
                dim=1,
                keepdim=True,
            )
            latent_salience = target_rms.detach() / (
                target_rms.detach() + reference_scale + 1.0e-6
            )
            camera_score = target_rms.new_zeros(target_rms.shape)
            if camera_centers is not None:
                camera_delta = (
                    camera_centers[:, stride:]
                    - camera_centers[:, :-stride]
                ).norm(dim=-1)
                camera_reference = camera_delta.mean(dim=1, keepdim=True)
                camera_score = camera_delta / (
                    camera_delta + camera_reference + 1.0e-6
                )
            camera_weight = 1.0 + self.generative_motion_camera_boost * camera_score
            stride_weight = math.sqrt(float(stride))
            weights = (
                pair_signal
                * valid.float()
                * (0.5 + 0.5 * latent_salience)
                * camera_weight
                * stride_weight
            )
            motion_weights = weights * (target_rms.detach() > 1.0e-4).float()
            direction = 1.0 - F.cosine_similarity(
                predicted_flat,
                target_flat,
                dim=-1,
                eps=1.0e-6,
            )
            log_ratio = (
                (predicted_rms + 1.0e-4).log()
                - (target_rms.detach() + 1.0e-4).log()
            ).clamp(-5.0, 5.0)
            magnitude = (
                self.generative_motion_under_weight * F.relu(-log_ratio)
                + F.relu(log_ratio)
            )
            direction_numerator = direction_numerator + (
                direction * motion_weights
            ).sum()
            direction_denominator = direction_denominator + motion_weights.sum()
            magnitude_numerator = magnitude_numerator + (
                magnitude * weights
            ).sum()
            magnitude_denominator = magnitude_denominator + weights.sum()
            ratio_numerator = ratio_numerator + (
                (predicted_rms.detach() / (target_rms.detach() + 1.0e-4))
                .clamp_max(10.0)
                * weights.detach()
            ).sum()
            camera_weight_numerator = camera_weight_numerator + (
                camera_weight.detach() * weights.detach()
            ).sum()
            camera_score_numerator = camera_score_numerator + (
                camera_score.detach() * weights.detach()
            ).sum()
            target_rms_numerator = target_rms_numerator + (
                target_rms.detach() * weights.detach()
            ).sum()
            predicted_rms_numerator = predicted_rms_numerator + (
                predicted_rms.detach() * weights.detach()
            ).sum()
            valid_count = valid_count + valid.float().sum()

        direction_loss = direction_numerator / direction_denominator.clamp_min(1.0e-6)
        magnitude_loss = magnitude_numerator / magnitude_denominator.clamp_min(1.0e-6)
        loss = direction_loss + (
            self.generative_motion_magnitude_weight * magnitude_loss
        )
        has_valid = magnitude_denominator > 0
        if not bool(has_valid):
            loss = zero
        metric_denominator = magnitude_denominator.detach().clamp_min(1.0e-6)
        return loss, {
            "generative_motion_direction_loss": direction_loss.detach().to(xs_pred),
            "generative_motion_magnitude_loss": magnitude_loss.detach().to(xs_pred),
            "generative_motion_speed_ratio": (
                ratio_numerator / metric_denominator
            ).detach().to(xs_pred),
            "generative_motion_target_rms": (
                target_rms_numerator / metric_denominator
            ).detach().to(xs_pred),
            "generative_motion_predicted_rms": (
                predicted_rms_numerator / metric_denominator
            ).detach().to(xs_pred),
            "generative_motion_camera_weight": (
                camera_weight_numerator / metric_denominator
            ).detach().to(xs_pred),
            "generative_motion_camera_score": (
                camera_score_numerator / metric_denominator
            ).detach().to(xs_pred),
            "generative_motion_camera_valid": xs_pred.new_tensor(
                float(camera_centers is not None)
            ),
            "generative_motion_valid_fraction": (
                valid_count / possible_count.clamp_min(1.0)
            ).detach().to(xs_pred),
        }

    def _cap_generative_motion_gradient(
        self,
        auxiliary_loss: Tensor,
        reference_loss: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = auxiliary_loss.detach().new_zeros(())
        metrics = {
            "generative_motion_gradient_scale": auxiliary_loss.detach().new_ones(()),
            "generative_motion_gradient_ratio_before_cap": zero,
            "generative_motion_gradient_reference_norm": zero,
            "generative_motion_gradient_auxiliary_norm": zero,
            "generative_motion_gradient_reference_valid": zero,
        }
        if (
            self.generative_motion_max_grad_ratio <= 0
            or not auxiliary_loss.requires_grad
            or not reference_loss.requires_grad
        ):
            if self.generative_motion_max_grad_ratio <= 0:
                metrics["generative_motion_gradient_scale"] = zero
                return auxiliary_loss * 0.0, metrics
            return auxiliary_loss, metrics
        _, diffusion_parameters = self._generative_parameter_groups()
        parameters = [
            parameter
            for parameter in diffusion_parameters
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(
                "Generative motion gradient cap found no trainable suffix"
            )
        probe_scale = self.generative_motion_gradient_probe_scale
        reference_grads = torch.autograd.grad(
            reference_loss.float() * probe_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_grads = torch.autograd.grad(
            auxiliary_loss.float() * probe_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        def gradient_norm(gradients) -> Tensor:
            terms = [
                gradient.detach().float().square().sum()
                for gradient in gradients
                if gradient is not None
            ]
            if not terms:
                return auxiliary_loss.detach().float().new_zeros(())
            return torch.stack(terms).sum().sqrt() / probe_scale

        reference_norm = gradient_norm(reference_grads)
        auxiliary_norm = gradient_norm(auxiliary_grads)
        if not bool(torch.isfinite(reference_norm)) or not bool(
            torch.isfinite(auxiliary_norm)
        ):
            raise RuntimeError(
                "Non-finite Physical conditioner motion gradient probe: "
                f"reference={reference_norm.item():.6g}, "
                f"auxiliary={auxiliary_norm.item():.6g}"
            )
        reference_valid = reference_norm > 1.0e-12
        auxiliary_valid = auxiliary_norm > 1.0e-12
        if bool(auxiliary_valid) and not bool(reference_valid):
            message = (
                "Generative motion gradient is non-zero but the diffusion "
                "reference does not reach the trainable suffix"
            )
            if self.strict_generative_motion:
                raise RuntimeError(message)
            metrics.update(
                {
                    "generative_motion_gradient_scale": zero,
                    "generative_motion_gradient_reference_norm": reference_norm,
                    "generative_motion_gradient_auxiliary_norm": auxiliary_norm,
                }
            )
            return auxiliary_loss * 0.0, metrics
        ratio = auxiliary_norm / reference_norm.clamp_min(1.0e-12)
        scale = (
            self.generative_motion_max_grad_ratio
            * reference_norm
            / auxiliary_norm.clamp_min(1.0e-12)
        ).clamp(max=1.0).detach()
        metrics.update(
            {
                "generative_motion_gradient_scale": scale,
                "generative_motion_gradient_ratio_before_cap": ratio.detach(),
                "generative_motion_gradient_reference_norm": reference_norm.detach(),
                "generative_motion_gradient_auxiliary_norm": auxiliary_norm.detach(),
                "generative_motion_gradient_reference_valid": (
                    reference_valid.detach().to(auxiliary_loss)
                ),
            }
        )
        return auxiliary_loss * scale, metrics

    def _generative_correction_energy_per_sample(
        self,
        corrected_v: Tensor,
        base_v: Tensor,
        denoising_terms: dict[str, Tensor | None],
        future: Tensor,
    ) -> Tensor:
        """Measure a condition-induced correction in the FM noise space."""
        logsnr = denoising_terms.get("logsnr")
        if logsnr is None:
            raise RuntimeError("Physical conditioner correction requires logSNR")
        frame_delta = (
            self._gather_video_frame(corrected_v, future).float()
            - self._gather_video_frame(base_v, future).float()
        )
        frame_logsnr = self._gather_video_frame(logsnr, future).float()
        alpha = torch.sigmoid(frame_logsnr).sqrt()
        while alpha.ndim < frame_delta.ndim:
            alpha = alpha.unsqueeze(-1)
        noise_delta = alpha * frame_delta
        sigmoid_bias = float(
            torch.as_tensor(
                denoising_terms.get("sigmoid_bias", -1.0)
            ).item()
        )
        weight = torch.sigmoid(sigmoid_bias - frame_logsnr)
        return noise_delta.square().flatten(1).mean(dim=1) * weight

    @staticmethod
    def _generative_relative_rms(
        energy: Tensor,
        denominator: Tensor,
        *,
        rms_floor: float = 1.0e-3,
    ) -> Tensor:
        """Convert relative correction energy to a non-vanishing RMS penalty."""
        relative_energy = energy / denominator
        floor = relative_energy.new_tensor(float(rms_floor))
        return (relative_energy.clamp_min(0.0) + floor.square()).sqrt() - floor

    @staticmethod
    def _generative_pair_gate_calibration_per_sample(
        correct_gate: Tensor,
        shuffled_gate: Tensor,
        *,
        prior: float,
    ) -> Tensor:
        """Calibrate aligned pairs toward one and counterfactual pairs to zero."""
        if correct_gate.shape != shuffled_gate.shape:
            raise ValueError(
                "Correct and shuffled pair gates must have identical shapes"
            )
        eps = 1.0e-4
        correct_probability = (correct_gate.float() * float(prior)).clamp(
            eps,
            1.0 - eps,
        )
        shuffled_probability = (shuffled_gate.float() * float(prior)).clamp(
            eps,
            1.0 - eps,
        )
        correct_loss = -correct_probability.log()
        shuffled_loss = -(1.0 - shuffled_probability).log()
        return 0.5 * (correct_loss + shuffled_loss).flatten(1).mean(dim=1)

    def _update_generative_teacher_quality(
        self,
        relative_gain: Tensor,
        shuffle_gain: Tensor,
        sample_mask: Tensor,
    ) -> None:
        valid = sample_mask.bool()
        valid_float = valid.to(device=relative_gain.device, dtype=torch.float32)
        statistics = torch.stack(
            (
                valid_float.sum(),
                (relative_gain.detach().float() * valid_float).sum(),
                (shuffle_gain.detach().float() * valid_float).sum(),
                ((relative_gain > 0).float() * valid_float).sum(),
                ((shuffle_gain > 0).float() * valid_float).sum(),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        count = float(statistics[0].item())
        if count <= 0:
            return
        gain_mean = float(statistics[1].item()) / count
        shuffle_mean = float(statistics[2].item()) / count
        win_mean = float(statistics[3].item()) / count
        condition_win_mean = float(statistics[4].item()) / count
        quality = self._generative_teacher_quality
        quality["samples"] += count
        quality["relative_gain_sum"] += gain_mean * count
        quality["teacher_wins"] += win_mean * count
        quality["shuffle_gain_sum"] += shuffle_mean * count
        quality["condition_wins"] += condition_win_mean * count
        decay = float(self.generative_teacher_quality_ema_decay)
        if not quality["ema_initialized"]:
            quality["ema_relative_gain"] = gain_mean
            quality["ema_win_fraction"] = win_mean
            quality["ema_shuffle_gain"] = shuffle_mean
            quality["ema_condition_win_fraction"] = condition_win_mean
            quality["ema_initialized"] = True
            return
        keep = decay
        update = 1.0 - decay
        quality["ema_relative_gain"] = (
            keep * quality["ema_relative_gain"] + update * gain_mean
        )
        quality["ema_win_fraction"] = (
            keep * quality["ema_win_fraction"] + update * win_mean
        )
        quality["ema_shuffle_gain"] = (
            keep * quality["ema_shuffle_gain"] + update * shuffle_mean
        )
        quality["ema_condition_win_fraction"] = (
            keep * quality["ema_condition_win_fraction"]
            + update * condition_win_mean
        )

    def _generative_teacher_quality_snapshot(self) -> dict[str, float | bool]:
        quality = dict(self._generative_teacher_quality)
        samples = max(float(quality["samples"]), 1.0)
        quality.update(
            {
                "relative_gain": float(quality["relative_gain_sum"]) / samples,
                "win_fraction": float(quality["teacher_wins"]) / samples,
                "shuffle_gain": float(quality["shuffle_gain_sum"]) / samples,
                "condition_win_fraction": (
                    float(quality["condition_wins"]) / samples
                ),
                "min_relative_gain": self.generative_teacher_min_relative_gain,
                "min_win_fraction": self.generative_teacher_min_win_fraction,
                "min_shuffle_gain": self.generative_teacher_min_shuffle_gain,
                "min_condition_win_fraction": (
                    self.generative_teacher_min_condition_win_fraction
                ),
                "min_samples": self.generative_teacher_min_samples,
            }
        )
        quality["qualified"] = bool(
            quality["samples"] >= self.generative_teacher_min_samples
            and quality["ema_relative_gain"]
            >= self.generative_teacher_min_relative_gain
            and quality["ema_win_fraction"]
            >= self.generative_teacher_min_win_fraction
            and quality["ema_shuffle_gain"]
            >= self.generative_teacher_min_shuffle_gain
            and quality["ema_condition_win_fraction"]
            >= self.generative_teacher_min_condition_win_fraction
        )
        return quality

    def _validate_conditional_teacher_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        if int(metadata.get("training_contract_version", -1)) != 6:
            raise ValueError(
                "Student distillation requires a calibrated paired conditional-"
                "teacher checkpoint with training contract version 6"
            )
        if str(metadata.get("mode")) != "conditional_teacher":
            raise ValueError(
                "Student distillation requires a conditional_teacher checkpoint"
            )
        if str(metadata.get("objective_contract")) != (
            "calibrated_gate_rms_identity"
        ):
            raise ValueError(
                "Student distillation requires the calibrated gate/RMS "
                "conditional-teacher objective"
            )
        quality = metadata.get("teacher_quality")
        if not isinstance(quality, dict):
            raise ValueError("Conditional teacher checkpoint has no quality audit")
        if self.generative_require_qualified_teacher and not bool(
            quality.get("qualified", False)
        ):
            raise ValueError(
                "Conditional teacher did not pass the checkpoint quality gate: "
                f"relative_gain={quality.get('ema_relative_gain')}, "
                f"win_fraction={quality.get('ema_win_fraction')}, "
                f"shuffle_gain={quality.get('ema_shuffle_gain')}, "
                "condition_win_fraction="
                f"{quality.get('ema_condition_win_fraction')}"
            )

    def _ensure_generative_teacher(self):
        if self.generative_unit_mode != "student_distill":
            return None, None
        tail = self._generative_teacher_tail
        conditioner = self._generative_teacher_conditioner
        if tail is not None and conditioner is not None:
            return tail, conditioner
        if self._generative_teacher_failed:
            raise RuntimeError("Generative teacher construction previously failed")
        try:
            checkpoint_path = Path(self.generative_teacher_checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Generative teacher checkpoint does not exist: {checkpoint_path}"
                )
            try:
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(checkpoint_path, map_location="cpu")
            metadata = payload.get("generative_unit")
            if metadata is None or int(metadata.get("version", -1)) != 4:
                raise ValueError(
                    "Generative teacher checkpoint is missing architecture "
                    "metadata version 4"
                )
            self._validate_conditional_teacher_metadata(metadata)
            current_conditioner = (
                self._generative_unit_backbone().generative_unit_conditioner
            )
            expected_spec = {
                "physical_dim": int(current_conditioner.physical_dim),
                "spatial_grid": int(current_conditioner.spatial_grid),
                "num_heads": int(current_conditioner.cross_attention.num_heads),
                "max_read_scale": float(current_conditioner.max_read_scale),
                "local_read_fraction": float(
                    current_conditioner.local_read_fraction
                ),
                "local_read_radius": int(
                    current_conditioner.local_read_radius
                ),
                "pair_encoder_layers": int(
                    current_conditioner.pair_encoder_layers
                ),
                "pair_gate_prior": float(
                    current_conditioner.pair_gate_prior
                ),
                "pair_update_scale": float(
                    current_conditioner.pair_update_scale
                ),
                "condition_interface": "paired_video_fused",
                "objective_contract": "calibrated_gate_rms_identity",
            }
            checkpoint_spec = {
                key: metadata.get(key)
                for key in expected_spec
            }
            if checkpoint_spec != expected_spec:
                raise ValueError(
                    "Generative teacher behavior contract differs from the "
                    f"student configuration: checkpoint={checkpoint_spec}, "
                    f"configured={expected_spec}"
                )
            state = payload.get("state_dict", payload)
            backbone_state = {}
            for key, value in state.items():
                for prefix in (
                    "diffusion_model.model.",
                    "diffusion_model._orig_mod.model.",
                ):
                    if key.startswith(prefix):
                        backbone_state[key[len(prefix) :]] = value
                        break
            if not backbone_state:
                raise RuntimeError(
                    "Generative teacher checkpoint has no DFoT backbone state"
                )
            backbone = self._generative_unit_backbone()
            tail = backbone.build_frozen_tail(activation_checkpointing=True)
            tail.load_from_backbone_state_dict(backbone_state)
            conditioner = copy.deepcopy(backbone.generative_unit_conditioner)
            conditioner_prefix = "generative_unit_conditioner."
            conditioner_state = {
                key[len(conditioner_prefix) :]: value
                for key, value in backbone_state.items()
                if key.startswith(conditioner_prefix)
            }
            conditioner.load_state_dict(conditioner_state, strict=True)
            device = next(backbone.parameters()).device
            tail.to(device=device).eval().requires_grad_(False)
            conditioner.to(device=device).eval().requires_grad_(False)
            object.__setattr__(self, "_generative_teacher_tail", tail)
            object.__setattr__(
                self,
                "_generative_teacher_conditioner",
                conditioner,
            )
            del payload, state, backbone_state, conditioner_state
            rank_zero_print(
                "[DFoTUPCAST] Loaded immutable Physical conditioner teacher "
                f"from {checkpoint_path}."
            )
            return tail, conditioner
        except Exception:
            self._generative_teacher_failed = True
            raise

    def _compute_generative_unit_loss(
        self,
        latents_list: list[Tensor],
        denoising_terms: dict[str, Tensor | None],
        prepared_context,
        batch_idx: int,
        clean_xs: Tensor,
        conditions: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if len(latents_list) != 1:
            raise RuntimeError(
                "Physical conditioner requires exactly one captured DFoT feature"
            )
        tail_state = denoising_terms.get("tail_state")
        student_v = denoising_terms.get("v_pred")
        if tail_state is None or student_v is None:
            raise RuntimeError(
                "Physical conditioner requires captured tail state and base v prediction"
            )
        backbone = self._generative_unit_backbone()
        # PAGE, fusion, and RVQ are an immutable oracle in all continuation
        # phases. No graph through the large UPCAST model is retained.
        with torch.no_grad():
            tokens = self.vggt_alignment_loss.encode_generative_physical_tokens(
                [latents_list[0].detach()],
                denoising_terms,
                prepared_context,
            )
        clean_physical = tokens.fused.detach().float()
        physical_noise = torch.randn_like(clean_physical).clamp(-4.0, 4.0)
        physical_logsnr = tokens.future_logsnr.float()
        physical_alpha = torch.sigmoid(physical_logsnr).sqrt()
        physical_sigma = torch.sigmoid(-physical_logsnr).sqrt()
        expanded_alpha = physical_alpha[:, None, None]
        expanded_sigma = physical_sigma[:, None, None]
        noisy_physical = (
            expanded_alpha * clean_physical
            + expanded_sigma * physical_noise
        ).to(latents_list[0])
        physical_velocity_target = (
            expanded_alpha * physical_noise
            - expanded_sigma * clean_physical
        ).to(latents_list[0])
        video_physical, fused_physical = self._generative_pair_tokens(
            tokens,
            noisy_physical,
        )
        condition_physical = fused_physical - video_physical
        for tensor_name, tensor_value in (
            ("clean_physical", clean_physical),
            ("noisy_physical", noisy_physical),
            ("video_physical", video_physical),
            ("fused_physical", fused_physical),
            ("condition_physical", condition_physical),
            ("physical_logsnr", physical_logsnr),
        ):
            if not bool(torch.isfinite(tensor_value).all()):
                nonfinite = int((~torch.isfinite(tensor_value)).sum().item())
                raise RuntimeError(
                    "Physical conditioner received a non-finite teacher input: "
                    f"name={tensor_name}, nonfinite={nonfinite}, "
                    f"shape={tuple(tensor_value.shape)}"
                )
        detached_tail_state = self._detach_tail_state(tail_state)
        captured_latent = latents_list[0].detach()
        raw_sample_mask = tokens.sample_mask.bool()
        # Batch-one DDP ranks can occasionally receive no transition above the
        # conservative UPCAST signal threshold. A zero objective would make the
        # whole accumulation window look like a broken graph. Use that local
        # sample as a low-confidence fallback while retaining the raw fraction
        # in diagnostics.
        mask_fallback = ~raw_sample_mask.any()
        sample_mask = (
            torch.ones_like(raw_sample_mask)
            if bool(mask_fallback)
            else raw_sample_mask
        )
        zero = captured_latent.sum() * 0.0
        metrics: dict[str, Tensor] = {
            "generative_unit_valid_fraction": raw_sample_mask.float().mean(),
            "generative_unit_mask_fallback": mask_fallback.float(),
            "generative_unit_code_agreement": (
                tokens.video_indices == tokens.fused_indices
            ).float().mean(),
            "generative_unit_future_frame": tokens.future.float().mean(),
            "generative_unit_pair_delta": tokens.delta.float().mean(),
            "generative_unit_physical_signal": torch.sigmoid(
                physical_logsnr
            ).mean(),
            "generative_unit_clean_physical_rms": (
                clean_physical.square().mean().sqrt()
            ),
            "generative_unit_video_physical_rms": (
                tokens.video.detach().float().square().mean().sqrt()
            ),
            "generative_unit_innovation_rms": (
                (tokens.fused.detach().float() - tokens.video.detach().float())
                .square()
                .mean()
                .sqrt()
            ),
            "generative_unit_condition_rms": (
                condition_physical.float().square().mean().sqrt()
            ),
            "generative_unit_velocity_target_rms": (
                physical_velocity_target.float().square().mean().sqrt()
            ),
        }
        for level in range(tokens.fused_indices.shape[-1]):
            level_indices = tokens.fused_indices[..., level].reshape(-1)
            counts = torch.bincount(
                level_indices,
                minlength=self.vggt_alignment_loss.unit_quantizer.codebook_size,
            ).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            metrics[f"generative_unit_fused_active_q{level}"] = (
                counts.gt(0).float().mean()
            )
            metrics[f"generative_unit_fused_perplexity_q{level}"] = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )

        if self.generative_unit_mode in {
            "adapter_warmup",
            "conditional_teacher",
        }:
            self._initialize_generative_teacher_gate()
            (
                conditioned,
                physical_velocity,
                correct_pair_gate,
            ) = backbone.condition_generative_unit(
                captured_latent,
                noisy_physical,
                tokens.future,
                physical_logsnr,
                video_physical=video_physical,
                fused_physical=fused_physical,
            )
            correct_diagnostics = {
                name: value.detach()
                for name, value in backbone.generative_unit_diagnostics.items()
            }
            joint_v = backbone.forward_tail_from_state(
                conditioned,
                detached_tail_state,
            )
            if not self._generative_tail_replay_validated:
                gate = backbone.generative_unit_conditioner.read_gate.detach()
                if float(gate.abs()) <= 1e-12:
                    replay_error = (
                        joint_v.detach().float() - student_v.detach().float()
                    ).abs()
                    replay_mean = float(replay_error.mean())
                    replay_max = float(replay_error.max())
                    if replay_mean > 2e-4 or replay_max > 5e-3:
                        raise RuntimeError(
                            "Live Physical conditioner suffix does not reproduce the "
                            "base DFoT output at the zero gate: "
                            f"mean_error={replay_mean:.8g}, "
                            f"max_error={replay_max:.8g}"
                        )
                    self._generative_tail_replay_mean_error = replay_mean
                    self._generative_tail_replay_max_error = replay_max
                    self._generative_tail_replay_validated = True
            joint_error = self._generative_fm_error_per_sample(
                joint_v,
                denoising_terms,
                tokens.future,
            )
            student_error = self._generative_fm_error_per_sample(
                student_v.detach(),
                denoising_terms,
                tokens.future,
            )
            video_fm = self._masked_sample_mean(joint_error, sample_mask)
            physical_per_sample = F.smooth_l1_loss(
                physical_velocity.float(),
                physical_velocity_target.float(),
                reduction="none",
                beta=0.1,
            ).flatten(1).mean(dim=1)
            physical_velocity_loss = self._masked_sample_mean(
                physical_per_sample,
                sample_mask,
            )
            denominator = student_error.detach().clamp_min(
                self.generative_fm_error_floor
            )
            relative_gain = (
                student_error.detach() - joint_error.detach()
            ) / denominator
            relative_fm = self._masked_sample_mean(
                joint_error / denominator,
                sample_mask,
            )
            baseline_rank = self._masked_sample_mean(
                F.relu(
                    joint_error
                    - student_error.detach()
                    + self.generative_rank_margin * denominator
                )
                / denominator,
                sample_mask,
            )
            shuffled_error = None
            shuffle_gain = None
            shuffle_rank = zero
            shuffle_identity = zero
            pair_gate_rank = zero
            pair_gate_calibration = zero
            shuffled_pair_gate = None
            shuffled_diagnostics: dict[str, Tensor] = {}
            if self.generative_unit_mode == "conditional_teacher":
                # Keep q_v fixed and corrupt only q_f. The conditioner must
                # therefore recognize a mismatched physical future for this
                # exact video, not merely react to a different residual set.
                shuffled_fused = self._shuffle_generative_physical_tokens(
                    fused_physical
                )
                (
                    shuffled_latent,
                    _,
                    shuffled_pair_gate,
                ) = backbone.condition_generative_unit(
                    captured_latent,
                    noisy_physical,
                    tokens.future,
                    physical_logsnr,
                    video_physical=video_physical,
                    fused_physical=shuffled_fused,
                )
                shuffled_diagnostics = {
                    name: value.detach()
                    for name, value in backbone.generative_unit_diagnostics.items()
                }
                shuffled_v = backbone.forward_tail_from_state(
                    shuffled_latent,
                    detached_tail_state,
                )
                shuffled_error = self._generative_fm_error_per_sample(
                    shuffled_v,
                    denoising_terms,
                    tokens.future,
                )
                shuffle_gain = (
                    shuffled_error.detach() - joint_error.detach()
                ) / denominator
                # The corrupted branch is a fixed comparison target. Detaching
                # it prevents the conditioner from satisfying the ranking loss
                # by deliberately making wrong conditions worse.
                shuffle_rank = self._masked_sample_mean(
                    F.relu(
                        joint_error
                        - shuffled_error.detach()
                        + self.generative_rank_margin * denominator
                    )
                    / denominator,
                    sample_mask,
                )
                shuffle_identity_per_sample = self._generative_relative_rms(
                    self._generative_correction_energy_per_sample(
                        shuffled_v,
                        student_v.detach(),
                        denoising_terms,
                        tokens.future,
                    ),
                    denominator,
                )
                shuffle_identity = self._masked_sample_mean(
                    shuffle_identity_per_sample,
                    sample_mask,
                )
                correct_gate_per_sample = correct_pair_gate.float().mean(
                    dim=(1, 2)
                )
                shuffled_gate_per_sample = shuffled_pair_gate.float().mean(
                    dim=(1, 2)
                )
                pair_gate_rank = self._masked_sample_mean(
                    F.relu(
                        self.generative_pair_gate_margin
                        - correct_gate_per_sample
                        + shuffled_gate_per_sample
                    ),
                    sample_mask,
                )
                pair_gate_calibration = self._masked_sample_mean(
                    self._generative_pair_gate_calibration_per_sample(
                        correct_pair_gate,
                        shuffled_pair_gate,
                        prior=backbone.generative_unit_conditioner.pair_gate_prior,
                    ),
                    sample_mask,
                )
                loss = (
                    self.generative_conditional_fm_weight * relative_fm
                    + self.generative_baseline_rank_weight * baseline_rank
                    + self.generative_shuffle_rank_weight * shuffle_rank
                    + self.generative_shuffle_identity_weight
                    * shuffle_identity
                    + self.generative_pair_gate_rank_weight
                    * pair_gate_rank
                    + self.generative_pair_gate_calibration_weight
                    * pair_gate_calibration
                    + self.generative_physical_velocity_weight
                    * physical_velocity_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "Physical conditioner conditional-teacher forward loss is "
                        "non-finite before backward: "
                        f"relative_fm={float(relative_fm.detach())}, "
                        f"baseline_rank={float(baseline_rank.detach())}, "
                        f"shuffle_rank={float(shuffle_rank.detach())}, "
                        f"shuffle_identity={float(shuffle_identity.detach())}, "
                        f"pair_gate_rank={float(pair_gate_rank.detach())}, "
                        "pair_gate_calibration="
                        f"{float(pair_gate_calibration.detach())}, "
                        "physical_velocity="
                        f"{float(physical_velocity_loss.detach())}"
                    )
                teacher_signal = torch.sigmoid(physical_logsnr)
                teacher_quality_mask = sample_mask & (
                    teacher_signal >= self.generative_distill_signal_min
                ) & (
                    teacher_signal <= self.generative_distill_signal_max
                )
                self._update_generative_teacher_quality(
                    relative_gain,
                    shuffle_gain,
                    teacher_quality_mask,
                )
            else:
                loss = (
                    self.generative_video_fm_weight * video_fm
                    + self.generative_physical_velocity_weight
                    * physical_velocity_loss
                )
            metrics.update(
                {
                    "generative_joint_video_fm": video_fm.detach(),
                    "generative_student_video_fm": self._masked_sample_mean(
                        student_error,
                        sample_mask,
                    ).detach(),
                    "generative_physical_velocity_loss": (
                        physical_velocity_loss.detach()
                    ),
                    "generative_tail_replay_mean_error": zero.new_tensor(
                        self._generative_tail_replay_mean_error
                    ),
                    "generative_tail_replay_max_error": zero.new_tensor(
                        self._generative_tail_replay_max_error
                    ),
                    "generative_joint_fm_gain": self._masked_sample_mean(
                        student_error - joint_error.detach(),
                        sample_mask,
                    ),
                    "generative_correct_relative_gain": (
                        self._masked_sample_mean(
                            relative_gain,
                            sample_mask,
                        ).detach()
                    ),
                    "generative_conditional_relative_fm": relative_fm.detach(),
                    "generative_baseline_rank_loss": baseline_rank.detach(),
                    "generative_shuffle_rank_loss": shuffle_rank.detach(),
                    "generative_shuffle_identity_rms_loss": (
                        shuffle_identity.detach()
                    ),
                    "generative_pair_gate_rank_loss": pair_gate_rank.detach(),
                    "generative_pair_gate_calibration_loss": (
                        pair_gate_calibration.detach()
                    ),
                    "generative_correct_pair_gate": correct_pair_gate.float()
                    .mean()
                    .detach(),
                    "generative_shuffled_pair_gate": (
                        zero
                        if shuffled_pair_gate is None
                        else shuffled_pair_gate.float().mean().detach()
                    ),
                    "generative_teacher_win_fraction": (
                        (joint_error < student_error) & sample_mask
                    ).float().sum()
                    / sample_mask.float().sum().clamp_min(1.0),
                }
            )
            metrics.update(
                {
                    name: value.detach()
                    for name, value in correct_diagnostics.items()
                }
            )
            metrics.update(
                {
                    "generative_shuffled_"
                    + name.removeprefix("generative_unit_"): value
                    for name, value in shuffled_diagnostics.items()
                }
            )
            if shuffled_error is not None and shuffle_gain is not None:
                quality = self._generative_teacher_quality_snapshot()
                metrics.update(
                    {
                        "generative_shuffled_video_fm": self._masked_sample_mean(
                            shuffled_error.detach(),
                            sample_mask,
                        ),
                        "generative_shuffle_fm_gap": self._masked_sample_mean(
                            shuffled_error.detach() - joint_error.detach(),
                            sample_mask,
                        ),
                        "generative_shuffle_relative_gain": (
                            self._masked_sample_mean(
                                shuffle_gain,
                                sample_mask,
                            ).detach()
                        ),
                        "generative_condition_win_fraction": (
                            ((joint_error < shuffled_error.detach()) & sample_mask)
                            .float()
                            .sum()
                            / sample_mask.float().sum().clamp_min(1.0)
                        ),
                        "generative_teacher_quality_ema_gain": zero.new_tensor(
                            quality["ema_relative_gain"]
                        ),
                        "generative_teacher_quality_ema_win": zero.new_tensor(
                            quality["ema_win_fraction"]
                        ),
                        "generative_teacher_quality_ema_shuffle_gain": (
                            zero.new_tensor(quality["ema_shuffle_gain"])
                        ),
                        "generative_teacher_quality_ema_condition_win": (
                            zero.new_tensor(
                                quality["ema_condition_win_fraction"]
                            )
                        ),
                        "generative_teacher_quality_valid_fraction": (
                            teacher_quality_mask.float().mean()
                        ),
                        "generative_teacher_quality_samples": zero.new_tensor(
                            quality["samples"]
                        ),
                        "generative_shuffle_computed": zero.new_ones(()),
                    }
                )
            elif (
                self.generative_shuffle_diagnostic_freq > 0
                and int(batch_idx) % self.generative_shuffle_diagnostic_freq == 0
            ):
                with torch.no_grad():
                    shuffled_fused = self._shuffle_generative_physical_tokens(
                        fused_physical
                    )
                    shuffled_latent, _, _ = backbone.condition_generative_unit(
                        captured_latent,
                        noisy_physical,
                        tokens.future,
                        physical_logsnr,
                        video_physical=video_physical,
                        fused_physical=shuffled_fused,
                    )
                    shuffled_v = backbone.forward_tail_from_state(
                        shuffled_latent,
                        detached_tail_state,
                    )
                    shuffled_error = self._generative_fm_error_per_sample(
                        shuffled_v,
                        denoising_terms,
                        tokens.future,
                    )
                    metrics["generative_shuffle_fm_gap"] = (
                        self._masked_sample_mean(shuffled_error, sample_mask)
                        - video_fm.detach()
                    )
                    metrics["generative_shuffle_computed"] = zero.new_ones(())
            else:
                metrics["generative_shuffle_fm_gap"] = zero.detach()
                metrics["generative_shuffle_computed"] = zero.detach()
            return loss, metrics

        teacher_tail, teacher_conditioner = self._ensure_generative_teacher()
        with torch.no_grad():
            teacher_latent, _, _ = teacher_conditioner(
                captured_latent,
                noisy_physical,
                tokens.future,
                physical_logsnr,
                video_physical=video_physical,
                fused_physical=fused_physical,
            )
            teacher_v = teacher_tail(
                teacher_latent,
                detached_tail_state,
            )
            teacher_error = self._generative_fm_error_per_sample(
                teacher_v,
                denoising_terms,
                tokens.future,
            )
        student_error = self._generative_fm_error_per_sample(
            student_v,
            denoising_terms,
            tokens.future,
        )
        signal = torch.sigmoid(physical_logsnr)
        medium_snr = (
            (signal >= self.generative_distill_signal_min)
            & (signal <= self.generative_distill_signal_max)
        )
        valid = sample_mask & medium_snr
        advantage = (
            (student_error.detach() - teacher_error)
            / student_error.detach().clamp_min(self.generative_fm_error_floor)
        ).clamp(0.0, self.generative_advantage_max)
        motion_salience, motion_metrics = self._generative_pair_motion_salience(
            clean_xs,
            conditions,
            tokens.current,
            tokens.future,
        )
        motion_weight = 1.0 + (
            self.generative_distill_motion_boost * motion_salience
        )
        # Redistribute a fixed expected distillation budget toward transitions
        # with real latent/camera displacement. This avoids increasing the
        # global auxiliary strength while static clips dominate the dataset.
        motion_normalizer = self._masked_sample_mean(
            motion_weight,
            valid,
        ).detach()
        if not bool(valid.any()):
            motion_normalizer = motion_normalizer.new_ones(())
        motion_weight = motion_weight / motion_normalizer.clamp_min(1.0e-6)
        frame_student = self._gather_video_frame(student_v, tokens.future)
        frame_teacher = self._gather_video_frame(teacher_v, tokens.future)
        # RMS output matching retains a useful gradient when the qualified
        # teacher differs from the student by only a small corrective velocity.
        # The previous beta=0.1 SmoothL1 made this term quadratic and effectively
        # zero at precisely the small residuals we want to distill.
        distill_per_sample = (
            (frame_student.float() - frame_teacher.float())
            .square()
            .flatten(1)
            .mean(dim=1)
            .add(1.0e-12)
            .sqrt()
        )
        distill_loss = self._masked_sample_mean(
            distill_per_sample * advantage * motion_weight,
            valid,
        )
        loss = self.generative_advantage_distill_weight * distill_loss
        metrics.update(motion_metrics)
        metrics.update(
            {
                "generative_student_video_fm": self._masked_sample_mean(
                    student_error.detach(),
                    valid,
                ),
                "generative_joint_video_fm": self._masked_sample_mean(
                    teacher_error.detach(),
                    valid,
                ),
                "generative_joint_fm_gain": self._masked_sample_mean(
                    student_error.detach() - teacher_error.detach(),
                    valid,
                ),
                "generative_advantage_mean": self._masked_sample_mean(
                    advantage,
                    valid,
                ),
                "generative_distill_motion_weight": self._masked_sample_mean(
                    motion_weight.detach(),
                    valid,
                ),
                "generative_distill_weighted_motion_salience": (
                    self._masked_sample_mean(
                        motion_salience * motion_weight.detach(),
                        valid,
                    )
                ),
                "generative_positive_advantage_fraction": (
                    ((advantage > 0) & valid).float().sum()
                    / valid.float().sum().clamp_min(1.0)
                ),
                "generative_teacher_win_fraction": (
                    ((teacher_error < student_error.detach()) & valid)
                    .float()
                    .sum()
                    / valid.float().sum().clamp_min(1.0)
                ),
                "generative_distill_valid_fraction": (
                    valid.float().mean()
                ),
                "generative_distill_mask_fallback": zero.detach(),
                "generative_distill_output_rms": self._masked_sample_mean(
                    distill_per_sample.detach(),
                    valid,
                ),
                "generative_advantage_distill_loss": distill_loss.detach(),
                "generative_shuffle_fm_gap": zero.detach(),
                "generative_shuffle_computed": zero.detach(),
            }
        )
        return loss, metrics

    def training_step(self, batch, batch_idx, namespace="training"):
        """Training step"""
        xs, conditions, masks, *_ = batch
        rollout_batch = batch[4] if len(batch) > 4 and isinstance(batch[4], dict) else None
        rollout_active = self._is_rollout_training_step(
            batch_idx,
            namespace,
            rollout_batch,
        )
        rollout_target_xs = None
        rollout_state = None
        rollout_query_anchors = None
        rollout_preference_context: dict[str, Any] | None = None
        rollout_gate_probabilities: dict[str, Tensor] = {}
        rollout_metrics = {
            "valid": xs.new_zeros(()),
            "generated_probability": xs.new_zeros(()),
            "generated_fraction": xs.new_zeros(()),
            "context_rmse": xs.new_zeros(()),
            "state_span_code_agreement": xs.new_zeros(()),
            "target_window_index": xs.new_zeros(()),
            "target_horizon_probability": xs.new_zeros(()),
            "target_end_frame": xs.new_zeros(()),
            "state_confidence": xs.new_zeros(()),
            "state_memory_tokens": xs.new_zeros(()),
        }
        if self.unit_inference_enable:
            self._world_state_backbone().clear_world_state_condition()
        if rollout_active:
            try:
                (
                    xs,
                    conditions,
                    masks,
                    rollout_state,
                    rollout_query_anchors,
                    rollout_metrics,
                    rollout_target_xs,
                    rollout_preference_context,
                ) = self._prepare_real_rollout_training_batch(rollout_batch)
                self._world_state_backbone().set_world_state_condition(
                    rollout_state,
                    query_anchors=rollout_query_anchors,
                    scale=self.world_state_runtime_scale,
                    detach=False,
                    branch_mode=self.world_state_runtime_branch_mode,
                )
            except Exception as exc:
                self._world_state_backbone().clear_world_state_condition()
                if self.strict_rollout_training:
                    raise RuntimeError("Real recurrent rollout training failed") from exc
                rollout_active = False
                rank_zero_print(
                    "[DFoTUPCAST] WARNING: rollout training failed and "
                    f"was skipped for this batch: {exc}"
                )
        self._suppress_selective_diffusion_gradients = bool(
            rollout_active
            and self.world_state_training_mode == "selective_joint"
            and self.rollout_adapter_only_gradients
        )
        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        if rollout_active:
            context_frames = self.rollout_overlap_frames
            noise_levels[:, :context_frames] = 0
            masks[:, :context_frames] = False
            future_mask = masks[:, context_frames:]
            if not bool(future_mask.any()):
                raise RuntimeError(
                    "Recurrent rollout batch has no future supervision after "
                    "excluding overlap context"
                )
            rollout_metrics["future_supervision_fraction"] = (
                future_mask.float().mean().to(xs)
            )
        processed_conditions = self._process_conditions(conditions)
        denoising_terms = None
        diffusion_loss_map = None
        geometry_tokenizer_only = bool(
            namespace == "training"
            and self._diffusion_frozen
            and self.alignment_coeff > 0
            and hasattr(self, "vggt_alignment_loss")
            and self.vggt_alignment_loss.unit_phase == "geometry"
        )
        should_compute_generated_context = bool(
            namespace == "training"
            and not self._diffusion_frozen
            and self.generated_context_exposure_weight > 0
            and self.generated_context_exposure_start_step >= 0
            and int(self.global_step)
            >= self.generated_context_exposure_start_step
            and self._is_frequency_step(
                batch_idx,
                self.generated_context_exposure_freq,
                self.generated_context_exposure_phase,
            )
        )
        skip_alignment_for_context = bool(
            should_compute_generated_context
            and self.generated_context_skip_alignment
        )
        generative_enabled_this_batch = bool(
            self.generative_conditioning_enabled and not skip_alignment_for_context
        )
        should_compute_generative_motion = bool(
            namespace == "training"
            and self.generative_unit_mode in {
                "student_distill",
                "motion_calibration",
            }
            and self.generative_motion_loss_weight > 0
            and not should_compute_generated_context
            and self._is_frequency_step(
                batch_idx,
                self.generative_motion_loss_freq,
                self.generative_motion_loss_phase,
            )
        )
        alignment_enabled_this_batch = bool(
            self.alignment_coeff > 0
            and not skip_alignment_for_context
            and not (rollout_active and self.rollout_skip_alignment)
        )
        skip_diffusion_for_context = bool(
            should_compute_generated_context
            and self.generated_context_skip_base_diffusion
        )
        alignment_sampling_scale = 1.0
        diffusion_sampling_scale = 1.0
        generative_sampling_scale = 1.0
        exposure_curriculum_active = bool(
            namespace == "training"
            and not self._diffusion_frozen
            and self.generated_context_exposure_weight > 0
            and self.generated_context_exposure_start_step >= 0
            and int(self.global_step)
            >= self.generated_context_exposure_start_step
        )
        if (
            exposure_curriculum_active
            and self.generated_context_skip_alignment
            and not skip_alignment_for_context
        ):
            alignment_sampling_scale = (
                self.generated_context_exposure_freq
                / (self.generated_context_exposure_freq - 1)
            )
            if self.generative_unit_enabled:
                generative_sampling_scale = alignment_sampling_scale
        rollout_curriculum_active = bool(
            namespace == "training"
            and self.rollout_training_enable
            and self.rollout_training_start_step >= 0
            and int(self.global_step) >= self.rollout_training_start_step
            and self.rollout_skip_alignment
            and self.rollout_training_freq > 1
        )
        if rollout_curriculum_active and not rollout_active:
            alignment_sampling_scale *= (
                self.rollout_training_freq
                / (self.rollout_training_freq - 1)
            )
        if (
            exposure_curriculum_active
            and self.generated_context_skip_base_diffusion
            and not skip_diffusion_for_context
        ):
            diffusion_sampling_scale = (
                self.generated_context_exposure_freq
                / (self.generated_context_exposure_freq - 1)
            )
        alignment_requires_denoising_terms = bool(
            generative_enabled_this_batch
            or (
                alignment_enabled_this_batch
                and hasattr(self, "vggt_alignment_loss")
                and self.vggt_alignment_loss is not None
                and getattr(
                    self.vggt_alignment_loss,
                    "requires_denoising_terms",
                    False,
                )
                and not geometry_tokenizer_only
            )
        )
        alignment_context_images = None
        prepared_teacher_context = None
        prepared_physical_context = None
        prepared_generation_targets = None
        prepared_denoising_terms = None
        prepared_prior_target = None
        prepared_prior_valid = None
        should_compute_generation_loss = False
        should_compute_prior = False
        if self.alignment_coeff > 0 or generative_enabled_this_batch:
            alignment_context_images = self._unnormalize_x(xs)
        # The frozen-base prior is independent of geometry alignment.  In
        # motion_calibration, alignment_coeff is deliberately zero, so tying
        # this path to alignment_context_images silently disabled the prior.
        if not self._diffusion_frozen:
            should_compute_prior = self._should_compute_video_prior(
                namespace,
                batch_idx,
            )
            if should_compute_prior:
                prepare_denoising = getattr(
                    self.diffusion_model,
                    "prepare_training_denoising",
                    None,
                )
                if prepare_denoising is None:
                    raise RuntimeError(
                        "Joint frozen-base prior requires "
                        "prepare_training_denoising"
                    )
                prepared_denoising_terms = prepare_denoising(
                    xs,
                    noise_levels,
                )
                (
                    prepared_prior_target,
                    prepared_prior_valid,
                ) = self._compute_video_prior_target(
                    prepared_denoising_terms,
                    processed_conditions,
                )
        if (
            alignment_context_images is not None
            and not self._diffusion_frozen
            and hasattr(self, "vggt_alignment_loss")
        ):
            should_compute_generation_loss = bool(
                self.compute_generation_loss
                and not rollout_active
                and (
                    namespace != "training"
                    or self._is_frequency_step(
                        batch_idx,
                        self.generation_loss_freq,
                        self.generation_loss_phase,
                    )
                )
            )
            if alignment_enabled_this_batch:
                prepared_teacher_context = (
                    self.vggt_alignment_loss.prepare_teacher_context(
                        alignment_context_images,
                    )
                )
            if should_compute_generation_loss:
                prepared_generation_targets = (
                    self.vggt_alignment_loss.prepare_generation_targets(
                        alignment_context_images,
                    )
                )
        if generative_enabled_this_batch:
            if alignment_context_images is None:
                raise RuntimeError(
                    "Physical conditioner could not prepare aligned RGB frames"
                )
            prepared_physical_context = (
                self.vggt_alignment_loss.prepare_physical_teacher_context(
                    alignment_context_images,
                    training=namespace == "training",
                )
            )
        if geometry_tokenizer_only:
            # The geometry teacher phase has no dependency on DFoT features or
            # noise. Skipping the 458M-parameter video model materially reduces
            # both wall time and peak memory without changing its objective.
            xs_pred = xs.detach()
            latents_list = []
            diffusion_loss_mean = torch.zeros(
                (),
                device=xs.device,
                dtype=xs.dtype,
            )
        elif self._diffusion_frozen and self.alignment_coeff > 0 and hasattr(
            self.diffusion_model, "forward_with_alignment_latents_only"
        ):
            with torch.no_grad():
                if alignment_requires_denoising_terms:
                    xs_pred, latents_list, denoising_terms = (
                        self.diffusion_model.forward_with_alignment_latents_only(
                            xs,
                            processed_conditions,
                            k=noise_levels,
                            return_denoising_terms=True,
                            return_tail_state=True,
                        )
                    )
                else:
                    xs_pred, latents_list = (
                        self.diffusion_model.forward_with_alignment_latents_only(
                            xs,
                            processed_conditions,
                            k=noise_levels,
                        )
                    )
            diffusion_loss_mean = torch.zeros((), device=xs.device, dtype=xs.dtype)
        else:
            want_denoising_terms = (
                self.video_prior_preservation_weight > 0
                or alignment_requires_denoising_terms
                or should_compute_generative_motion
                or should_compute_generated_context
                or rollout_active
            )
            if want_denoising_terms:
                try:
                    forward_context = (
                        torch.no_grad()
                        if skip_diffusion_for_context
                        else contextlib.nullcontext()
                    )
                    with forward_context:
                        xs_pred, loss, latents_list, denoising_terms = (
                            self.diffusion_model.forward_with_alignment_loss(
                                xs,
                                processed_conditions,
                                k=noise_levels,
                                return_denoising_terms=True,
                                return_tail_state=alignment_requires_denoising_terms,
                                return_latents=(
                                    alignment_enabled_this_batch
                                    or generative_enabled_this_batch
                                ),
                                prepared_denoising_terms=prepared_denoising_terms,
                            )
                        )
                except TypeError:
                    if prepared_denoising_terms is not None:
                        raise
                    xs_pred, loss, latents_list = self.diffusion_model.forward_with_alignment_loss(
                        xs,
                        processed_conditions,
                        k=noise_levels,
                    )
                    denoising_terms = None
            else:
                xs_pred, loss, latents_list = self.diffusion_model.forward_with_alignment_loss(
                    xs,
                    processed_conditions,
                    k=noise_levels,
                )
            if skip_diffusion_for_context:
                diffusion_loss_mean = torch.zeros(
                    (), device=xs.device, dtype=xs.dtype
                )
            else:
                diffusion_loss_map = loss
                diffusion_loss_mean = (
                    self._reweight_loss(loss, masks)
                    * diffusion_sampling_scale
                )
        if (
            rollout_active
            and self.world_state_training_mode in {"adapter_only", "gate_only"}
            and not diffusion_loss_mean.requires_grad
        ):
            raise RuntimeError(
                "Frozen diffusion loss is detached from the world-state adapter"
            )
        if rollout_active:
            rollout_gate_probabilities = dict(
                self._world_state_backbone().world_state_gate_probabilities
            )
        optimization_diffusion_loss = diffusion_loss_mean
        if (
            rollout_active
            and self.world_state_training_mode in {"adapter_only", "gate_only"}
        ):
            optimization_diffusion_loss = (
                self.rollout_adapter_diffusion_weight * diffusion_loss_mean
            )

        generative_unit_loss = xs.new_zeros(())
        generative_unit_metrics: dict[str, Tensor] = {}
        if generative_enabled_this_batch:
            if denoising_terms is None or prepared_physical_context is None:
                raise RuntimeError(
                    "Physical conditioner is missing denoising or PAGE context"
                )
            generative_unit_loss, generative_unit_metrics = (
                self._compute_generative_unit_loss(
                    latents_list,
                    denoising_terms,
                    prepared_physical_context,
                    batch_idx,
                    xs,
                    conditions,
                )
            )
            generative_unit_metrics["generative_sampling_scale"] = (
                xs.new_tensor(generative_sampling_scale)
            )
            generative_unit_loss = (
                generative_unit_loss * generative_sampling_scale
            )
            self._latest_generative_unit_metrics = {
                name: value.detach()
                for name, value in generative_unit_metrics.items()
            }

        generative_motion_loss = xs.new_zeros(())
        generative_motion_loss_expected = xs.new_zeros(())
        generative_motion_metrics: dict[str, Tensor] = {
            "generative_motion_computed": xs.new_zeros(()),
            "generative_motion_direction_loss": xs.new_zeros(()),
            "generative_motion_magnitude_loss": xs.new_zeros(()),
            "generative_motion_speed_ratio": xs.new_zeros(()),
            "generative_motion_valid_fraction": xs.new_zeros(()),
            "generative_motion_camera_valid": xs.new_zeros(()),
            "generative_motion_camera_score": xs.new_zeros(()),
            "generative_motion_camera_weight": xs.new_zeros(()),
            "generative_motion_gradient_scale": xs.new_zeros(()),
            "generative_motion_gradient_ratio_before_cap": xs.new_zeros(()),
        }
        if should_compute_generative_motion:
            try:
                if denoising_terms is None or denoising_terms.get("logsnr") is None:
                    raise RuntimeError(
                        "Generative motion calibration requires frame logSNR"
                    )
                (
                    generative_motion_raw,
                    generative_motion_metrics,
                ) = self._generative_motion_consistency_loss(
                    xs_pred,
                    xs,
                    denoising_terms["logsnr"],
                    masks,
                    conditions,
                )
                generative_motion_metrics["generative_motion_computed"] = (
                    xs.new_ones(())
                )
                generative_motion_loss_expected = (
                    self.generative_motion_loss_weight * generative_motion_raw
                )
                generative_motion_loss = generative_motion_loss_expected
                if self.generative_motion_frequency_compensation:
                    generative_motion_loss = (
                        generative_motion_loss * self.generative_motion_loss_freq
                    )
                (
                    generative_motion_loss,
                    gradient_metrics,
                ) = self._cap_generative_motion_gradient(
                    generative_motion_loss,
                    optimization_diffusion_loss,
                )
                generative_motion_metrics.update(gradient_metrics)
            except Exception as exc:
                if self.strict_generative_motion:
                    raise RuntimeError(
                        "Generative motion calibration failed"
                    ) from exc
                rank_zero_print(
                    "[DFoTUPCAST] WARNING: generative motion "
                    f"calibration was skipped: {exc}"
                )

        generated_context_loss = torch.zeros((), device=xs.device, dtype=xs.dtype)
        generated_context_loss_expected = torch.zeros(
            (), device=xs.device, dtype=xs.dtype
        )
        generated_context_metrics = {
            "valid": torch.zeros((), device=xs.device, dtype=xs.dtype),
            "active_fraction": torch.zeros((), device=xs.device, dtype=xs.dtype),
            "context_residual_rms": torch.zeros(
                (), device=xs.device, dtype=xs.dtype
            ),
        }
        if should_compute_generated_context:
            try:
                (
                    generated_context_loss,
                    generated_context_metrics,
                ) = self._compute_generated_context_exposure_loss(
                    xs,
                    xs_pred,
                    masks,
                    denoising_terms,
                    processed_conditions,
                )
                generated_context_loss_expected = generated_context_loss
                self._latest_generated_context_expected = (
                    generated_context_loss_expected.detach()
                )
                self._latest_generated_context_active_fraction = (
                    generated_context_metrics["active_fraction"].detach()
                )
                self._latest_generated_context_residual_rms = (
                    generated_context_metrics["context_residual_rms"].detach()
                )
                if self.generated_context_frequency_compensation:
                    generated_context_loss = (
                        generated_context_loss
                        * self.generated_context_exposure_freq
                    )
            except Exception as exc:
                if self.strict_generated_context_exposure:
                    raise RuntimeError(
                        "Generated-context exposure failed"
                    ) from exc
                if not self._warned_generated_context_exposure_failed:
                    rank_zero_print(
                        "[DFoTUPCAST] WARNING: generated-context "
                        f"exposure failed and will be skipped. Error: {exc}"
                    )
                    self._warned_generated_context_exposure_failed = True

        generated_context_loss_weighted = (
            self.generated_context_exposure_weight * generated_context_loss
        )
        rollout_state_regret_loss = xs.new_zeros(())
        rollout_state_regret_weighted = xs.new_zeros(())
        rollout_state_regret_metrics = {
            "state_regret": xs.new_zeros(()),
            "state_diffusion_loss": xs.new_zeros(()),
            "no_state_diffusion_loss": xs.new_zeros(()),
            "state_improvement": xs.new_zeros(()),
            "state_win_fraction": xs.new_zeros(()),
        }
        rollout_forward_diagnostics = (
            {
                key: value.detach()
                for key, value in self._world_state_backbone().world_state_diagnostics.items()
            }
            if rollout_active
            else {}
        )
        if rollout_active and self.rollout_state_regret_weight > 0:
            if (
                diffusion_loss_map is None
                or denoising_terms is None
                or rollout_state is None
            ):
                raise RuntimeError(
                    "State-regret rollout is missing its denoising task or world state"
                )
            baseline_loss_map = self._rollout_no_state_diffusion_loss_map(
                denoising_terms,
                processed_conditions,
                rollout_state,
                rollout_query_anchors,
            )
            (
                rollout_state_regret_loss,
                rollout_state_regret_metrics,
            ) = self._rollout_state_regret_loss(
                diffusion_loss_map,
                baseline_loss_map,
                masks,
            )
            rollout_state_regret_weighted = (
                self.rollout_state_regret_weight * rollout_state_regret_loss
            )
        rollout_preference_loss = xs.new_zeros(())
        rollout_preference_expected = xs.new_zeros(())
        rollout_preference_weighted = xs.new_zeros(())
        rollout_preference_metrics: dict[str, Tensor] = {
            "preference_valid": xs.new_zeros(()),
        }
        should_compute_rollout_preference = bool(
            rollout_active
            and namespace == "training"
            and self.rollout_sampled_preference_enable
            and self._is_frequency_step(
                batch_idx,
                self.rollout_sampled_preference_freq,
                self.rollout_sampled_preference_phase,
            )
        )
        if should_compute_rollout_preference:
            if (
                rollout_state is None
                or rollout_target_xs is None
                or rollout_preference_context is None
            ):
                raise RuntimeError(
                    "Sampled preference is missing rollout state, GT target, "
                    "or cumulative context"
                )
            try:
                (
                    rollout_preference_loss,
                    computed_preference_metrics,
                ) = self._rollout_sampled_preference_loss(
                    xs,
                    rollout_target_xs,
                    conditions,
                    rollout_state,
                    rollout_query_anchors,
                    rollout_gate_probabilities,
                    rollout_preference_context,
                )
                rollout_preference_expected = rollout_preference_loss
                rollout_preference_metrics = {
                    **computed_preference_metrics,
                    "preference_valid": xs.new_ones(()),
                }
                if self.rollout_sampled_preference_frequency_compensation:
                    rollout_preference_loss = (
                        rollout_preference_loss
                        * self.rollout_sampled_preference_freq
                    )
            except Exception as exc:
                self._world_state_backbone().set_world_state_condition(
                    rollout_state,
                    query_anchors=rollout_query_anchors,
                    scale=self.world_state_runtime_scale,
                    detach=False,
                    branch_mode=self.world_state_runtime_branch_mode,
                )
                if self.strict_rollout_sampled_preference:
                    raise RuntimeError(
                        "Same-noise sampled state preference failed"
                    ) from exc
                rank_zero_print(
                    "[DFoTUPCAST] WARNING: sampled state preference "
                    f"was skipped: {exc}"
                )
        rollout_preference_weighted = (
            self.rollout_sampled_preference_weight * rollout_preference_loss
        )
        rollout_boundary_loss = xs.new_zeros(())
        rollout_boundary_expected = xs.new_zeros(())
        rollout_boundary_components = {
            "temporal": xs.new_zeros(()),
            "trajectory": xs.new_zeros(()),
            "color": xs.new_zeros(()),
            "signal": xs.new_zeros(()),
            "signal_valid_fraction": xs.new_zeros(()),
            "future_frames": xs.new_zeros(()),
            "gradient_scale": xs.new_ones(()),
            "gradient_ratio_before_cap": xs.new_zeros(()),
            "gradient_reference_norm": xs.new_zeros(()),
            "gradient_auxiliary_norm": xs.new_zeros(()),
            "gradient_probe_scale": xs.new_tensor(
                self.rollout_gradient_probe_scale
            ),
            "gradient_reference_valid": xs.new_zeros(()),
        }
        boundary_objective_enabled = any(
            weight > 0
            for weight in (
                self.rollout_boundary_temporal_weight,
                self.rollout_boundary_color_weight,
                self.rollout_trajectory_weight,
            )
        )
        if rollout_active and boundary_objective_enabled:
            if rollout_target_xs is None:
                raise RuntimeError("Rollout target window was not retained")
            (
                rollout_boundary_loss,
                computed_boundary_components,
            ) = self._rollout_boundary_loss(
                xs_pred,
                xs,
                rollout_target_xs,
                denoising_terms["logsnr"],
            )
            # Keep zero/default gradient diagnostics when all optional raw-RGB
            # boundary objectives are disabled.
            rollout_boundary_components.update(computed_boundary_components)
            rollout_boundary_expected = rollout_boundary_loss
            if self.rollout_frequency_compensation:
                rollout_boundary_loss = (
                    rollout_boundary_loss * self.rollout_training_freq
                )
            if self.world_state_training_mode in {
                "adapter_only",
                "gate_only",
                "selective_joint",
            }:
                (
                    rollout_boundary_loss,
                    gradient_metrics,
                ) = self._cap_auxiliary_adapter_gradient(
                    rollout_boundary_loss,
                    diffusion_loss_mean,
                )
                rollout_boundary_components.update(gradient_metrics)
        elif rollout_active:
            rollout_boundary_components["future_frames"] = xs.new_tensor(
                float(max(int(xs.shape[1]) - self.rollout_overlap_frames, 0))
            )
        if rollout_active:
            latest_rollout_metrics = {
                **{key: value.detach() for key, value in rollout_metrics.items()},
                **{
                    key: value.detach()
                    for key, value in rollout_state_regret_metrics.items()
                },
                **{
                    key: value.detach()
                    for key, value in rollout_preference_metrics.items()
                },
                "state_diffusion_weighted": optimization_diffusion_loss.detach(),
                "preference_expected": rollout_preference_expected.detach(),
                "preference_weighted": rollout_preference_weighted.detach(),
                "future_frames": rollout_boundary_components[
                    "future_frames"
                ].detach(),
            }
            if boundary_objective_enabled:
                latest_rollout_metrics.update(
                    {
                        "boundary_expected": rollout_boundary_expected.detach(),
                        "boundary_temporal": rollout_boundary_components[
                            "temporal"
                        ].detach(),
                        "boundary_trajectory": rollout_boundary_components[
                            "trajectory"
                        ].detach(),
                        "boundary_color": rollout_boundary_components[
                            "color"
                        ].detach(),
                        "boundary_signal": rollout_boundary_components[
                            "signal"
                        ].detach(),
                        "boundary_signal_valid_fraction": rollout_boundary_components[
                            "signal_valid_fraction"
                        ].detach(),
                        "boundary_gradient_scale": rollout_boundary_components[
                            "gradient_scale"
                        ].detach(),
                        "boundary_gradient_ratio_before_cap": rollout_boundary_components[
                            "gradient_ratio_before_cap"
                        ].detach(),
                        "boundary_gradient_reference_norm": rollout_boundary_components[
                            "gradient_reference_norm"
                        ].detach(),
                        "boundary_gradient_auxiliary_norm": rollout_boundary_components[
                            "gradient_auxiliary_norm"
                        ].detach(),
                        "boundary_gradient_probe_scale": rollout_boundary_components[
                            "gradient_probe_scale"
                        ].detach(),
                        "boundary_gradient_reference_valid": rollout_boundary_components[
                            "gradient_reference_valid"
                        ].detach(),
                    }
                )
            if self.world_state_training_mode == "selective_joint":
                latest_rollout_metrics["adapter_only_gradients"] = xs.new_tensor(
                    float(self._suppress_selective_diffusion_gradients)
                )
            latest_rollout_metrics.update(rollout_forward_diagnostics)
            self._latest_rollout_metrics = latest_rollout_metrics
            self._accumulate_rollout_horizon_metrics(latest_rollout_metrics)
        preservation_loss_mean = torch.zeros((), device=xs.device, dtype=xs.dtype)
        preservation_loss_expected = torch.zeros((), device=xs.device, dtype=xs.dtype)
        preservation_loss_valid = torch.tensor(0.0, device=xs.device, dtype=xs.dtype)
        preservation_loss_weighted = torch.zeros((), device=xs.device, dtype=xs.dtype)
        alignment_generation_loss = torch.zeros((), device=xs.device, dtype=xs.dtype)
        alignment_generation_loss_expected = torch.zeros((), device=xs.device, dtype=xs.dtype)
        alignment_generation_loss_unweighted = torch.zeros((), device=xs.device, dtype=xs.dtype)
        alignment_generation_loss_valid = torch.tensor(0.0, device=xs.device)
        if should_compute_prior:
            preservation_loss_mean, preservation_loss_valid = (
                self._compute_video_prior_preservation_loss(
                    denoising_terms=denoising_terms,
                    processed_conditions=processed_conditions,
                    prepared_target=prepared_prior_target,
                    prepared_valid=prepared_prior_valid,
                )
            )
            preservation_loss_expected = preservation_loss_mean
            if namespace == "training":
                self._latest_video_prior_expected = preservation_loss_expected.detach()
                self._latest_video_prior_expected_weighted = (
                    float(self.video_prior_preservation_weight)
                    * preservation_loss_expected.detach()
                )
                self._latest_video_prior_valid = preservation_loss_valid.detach()
            if namespace == "training" and self.video_prior_frequency_compensation:
                preservation_loss_mean = (
                    preservation_loss_mean * self.video_prior_preservation_freq
                )
        preservation_loss_weighted = (
            float(self.video_prior_preservation_weight) * preservation_loss_mean
        )
        if should_compute_generation_loss:
            try:
                xs_pred_unnorm_for_generation = self._unnormalize_x(xs_pred)
                (
                    alignment_generation_loss,
                    alignment_generation_loss_unweighted,
                ) = self.vggt_alignment_loss.compute_alignment_generation_loss(
                    xs_pred=xs_pred_unnorm_for_generation,
                    xs_target=alignment_context_images,
                    logsnr=(
                        None
                        if denoising_terms is None
                        else denoising_terms.get("logsnr")
                    ),
                    prepared_targets=prepared_generation_targets,
                )
                alignment_generation_loss_expected = alignment_generation_loss
                if namespace == "training":
                    self._latest_generation_expected = (
                        alignment_generation_loss_expected.detach()
                    )
                if (
                    namespace == "training"
                    and self.generation_loss_frequency_compensation
                ):
                    alignment_generation_loss = (
                        alignment_generation_loss * self.generation_loss_freq
                    )
                alignment_generation_loss_valid = xs.new_ones(())
            except Exception as exc:
                if self.strict_generation_protection:
                    raise RuntimeError(
                        "Output-level generation protection failed"
                    ) from exc
                if not self._warned_alignment_generation_loss_failed:
                    rank_zero_print(
                        "[DFoTUPCAST] WARNING: alignment generation "
                        f"loss failed and will be skipped. Error: {exc}"
                    )
                    self._warned_alignment_generation_loss_failed = True

        shared_video_auxiliary_loss = (
            generative_unit_loss
            + generative_motion_loss
            + preservation_loss_weighted
            + alignment_generation_loss
            + generated_context_loss_weighted
            + rollout_state_regret_weighted
            + rollout_preference_weighted
            + rollout_boundary_loss
        )
        alignment_loss_mean = torch.zeros((), device=xs.device, dtype=xs.dtype)
        alignment_loss_unweighted_mean = torch.zeros(
            (), device=xs.device, dtype=xs.dtype
        )
        geometry_loss_mean = torch.zeros((), device=xs.device, dtype=xs.dtype)
        if alignment_enabled_this_batch:
            if alignment_context_images is None:
                raise RuntimeError("Alignment context images were not prepared")

            alignment_loss = self.alignment_loss(
                latents_list,
                alignment_context_images,
                denoising_terms=denoising_terms,
                batch_idx=batch_idx,
                prepared_teacher_context=prepared_teacher_context,
            )
            alignment_loss_mean = alignment_loss.mean()
            alignment_loss_unweighted_mean = alignment_loss_mean
            if hasattr(self.vggt_alignment_loss, "last_metrics"):
                alignment_loss_unweighted_mean = self.vggt_alignment_loss.last_metrics.get(
                    "alignment_loss_unweighted",
                    alignment_loss_mean.detach(),
                )

            alignment_coeff = self._alignment_coeff_tensor(alignment_loss_mean)
            geometry_loss_mean = (
                alignment_loss_mean
                * alignment_coeff
                * alignment_sampling_scale
            )

            if self._diffusion_frozen:
                # Learn the transition representation with a frozen backbone.
                loss = geometry_loss_mean
                loss_unweighted = alignment_loss_unweighted_mean * alignment_coeff
            else:
                # Combine representation and video objectives for a trainable backbone.
                if not should_compute_generation_loss and hasattr(self.vggt_alignment_loss, "last_generation_metrics"):
                    self.vggt_alignment_loss.last_generation_metrics = {
                        "alignment_generation_dino_patch_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_spatial_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_temporal_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_temporal_direction_component": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_temporal_magnitude_component": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_global_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_color_moment_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_rgb_temporal_loss_component": alignment_generation_loss.detach(),
                        "alignment_generation_rgb_temporal_direction_component": alignment_generation_loss.detach(),
                        "alignment_generation_rgb_temporal_magnitude_component": alignment_generation_loss.detach(),
                        "alignment_generation_dino_patch_loss_weighted": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_spatial_loss_weighted": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_temporal_loss_weighted": alignment_generation_loss.detach(),
                        "alignment_generation_vjepa_global_loss_weighted": alignment_generation_loss.detach(),
                        "alignment_generation_color_moment_loss_weighted": alignment_generation_loss.detach(),
                        "alignment_generation_rgb_temporal_loss_weighted": alignment_generation_loss.detach(),
                    }
                loss = (
                    optimization_diffusion_loss
                    + geometry_loss_mean
                    + shared_video_auxiliary_loss
                )
                loss_unweighted = (
                    optimization_diffusion_loss
                    + alignment_loss_unweighted_mean
                    * alignment_coeff
                    * alignment_sampling_scale
                    + shared_video_auxiliary_loss
                )
        elif self.alignment_coeff > 0 and not self._diffusion_frozen:
            # Exposure microbatches use the same Video-DiT parameters but avoid
            # retaining PAGE/UPCAST/tail graphs at the same time as a second full
            # DFoT pass. Non-exposure microbatches are frequency compensated,
            # preserving the expected UPCAST objective over each schedule cycle.
            loss = (
                optimization_diffusion_loss
                + shared_video_auxiliary_loss
            )
            loss_unweighted = loss
        else:
            loss = (
                optimization_diffusion_loss
                + shared_video_auxiliary_loss
            )
            loss_unweighted = loss

        if batch_idx % self.cfg.logging.loss_freq == 0:
            eta_hours = None
            if self._fit_start_walltime is not None:
                elapsed = max(0.0, time.time() - self._fit_start_walltime)
                done_steps = max(1, int(self.global_step) - int(self._fit_start_step) + 1)
                step_time = elapsed / done_steps
                total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
                if total_steps is not None and total_steps > 0:
                    remaining_steps = max(int(total_steps) - int(self.global_step), 0)
                    eta_hours = remaining_steps * step_time / 3600.0

            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                f"{namespace}/loss_unweighted",
                loss_unweighted,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            self.log(
                f"{namespace}/diffusion_loss",
                diffusion_loss_mean,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            self.log(
                f"{namespace}/diffusion_loss_computed",
                float(not skip_diffusion_for_context),
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            self.log(
                f"{namespace}/diffusion_sampling_scale",
                float(diffusion_sampling_scale),
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )
            if self.generative_unit_enabled:
                self.log(
                    f"{namespace}/generative_unit_loss",
                    generative_unit_loss.detach(),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    prog_bar=True,
                )
                self.log(
                    f"{namespace}/generative_unit_mode_id",
                    float(
                        {
                            "adapter_warmup": 1,
                            "conditional_teacher": 2,
                            "student_distill": 3,
                            "motion_calibration": 4,
                        }[self.generative_unit_mode]
                    ),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=False,
                )
                self.log(
                    f"{namespace}/generative_conditioning_computed",
                    float(generative_enabled_this_batch),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                for metric_name in sorted(generative_unit_metrics):
                    self.log(
                        f"{namespace}/{metric_name}",
                        generative_unit_metrics[metric_name],
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
            if self.generative_motion_loss_weight > 0:
                self.log(
                    f"{namespace}/generative_motion_loss_expected_weighted",
                    generative_motion_loss_expected.detach(),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/generative_motion_loss_applied",
                    generative_motion_loss.detach(),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                for metric_name in sorted(generative_motion_metrics):
                    self.log(
                        f"{namespace}/{metric_name}",
                        generative_motion_metrics[metric_name],
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
            if namespace != "training":
                self.log(
                    f"{namespace}/weights_source_ema",
                    float(getattr(self, "loaded_weights_source", "raw") == "ema"),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            if self.video_prior_preservation_weight > 0:
                latest_prior_expected = (
                    self._latest_video_prior_expected
                    if namespace == "training"
                    else preservation_loss_expected.detach()
                )
                latest_prior_expected_weighted = (
                    self._latest_video_prior_expected_weighted
                    if namespace == "training"
                    else float(self.video_prior_preservation_weight)
                    * preservation_loss_expected.detach()
                )
                if latest_prior_expected is None:
                    latest_prior_expected = preservation_loss_expected.detach()
                if latest_prior_expected_weighted is None:
                    latest_prior_expected_weighted = (
                        float(self.video_prior_preservation_weight)
                        * preservation_loss_expected.detach()
                    )
                latest_prior_valid = (
                    self._latest_video_prior_valid
                    if namespace == "training"
                    else preservation_loss_valid.detach()
                )
                if latest_prior_valid is None:
                    latest_prior_valid = preservation_loss_valid.detach()
                self.log(
                    f"{namespace}/video_prior_preservation_loss",
                    preservation_loss_mean,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/video_prior_preservation_loss_expected",
                    latest_prior_expected,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/video_prior_preservation_loss_expected_weighted",
                    latest_prior_expected_weighted,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/video_prior_preservation_loss_weighted",
                    float(self.video_prior_preservation_weight) * preservation_loss_mean,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/video_prior_preservation_valid",
                    latest_prior_valid,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                latest_generation_expected = (
                    self._latest_generation_expected
                    if namespace == "training"
                    else alignment_generation_loss_expected.detach()
                )
                if latest_generation_expected is not None:
                    self.log(
                        f"{namespace}/generation_to_prior_expected_ratio",
                        latest_generation_expected
                        / latest_prior_expected_weighted.clamp_min(1e-12),
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
            if self.generated_context_exposure_weight > 0:
                latest_exposure = (
                    self._latest_generated_context_expected
                    if namespace == "training"
                    else generated_context_loss_expected.detach()
                )
                latest_active = (
                    self._latest_generated_context_active_fraction
                    if namespace == "training"
                    else generated_context_metrics["active_fraction"].detach()
                )
                latest_residual = (
                    self._latest_generated_context_residual_rms
                    if namespace == "training"
                    else generated_context_metrics[
                        "context_residual_rms"
                    ].detach()
                )
                if latest_exposure is None:
                    latest_exposure = generated_context_loss_expected.detach()
                if latest_active is None:
                    latest_active = generated_context_metrics[
                        "active_fraction"
                    ].detach()
                if latest_residual is None:
                    latest_residual = generated_context_metrics[
                        "context_residual_rms"
                    ].detach()
                self.log(
                    f"{namespace}/generated_context_exposure_loss_expected",
                    latest_exposure,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/generated_context_exposure_loss_expected_weighted",
                    self.generated_context_exposure_weight * latest_exposure,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/generated_context_active_fraction",
                    latest_active,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/generated_context_residual_rms",
                    latest_residual,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/generated_context_exposure_computed",
                    generated_context_metrics["valid"],
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
            if self.rollout_training_enable:
                if namespace == "training" and self._latest_rollout_metrics:
                    rollout_log_metrics = self._latest_rollout_metrics
                else:
                    rollout_log_metrics = {
                        **rollout_metrics,
                        "future_frames": rollout_boundary_components[
                            "future_frames"
                        ].detach(),
                    }
                    if boundary_objective_enabled:
                        rollout_log_metrics.update(
                            {
                                "boundary_expected": rollout_boundary_expected.detach(),
                                "boundary_temporal": rollout_boundary_components[
                                    "temporal"
                                ].detach(),
                                "boundary_trajectory": rollout_boundary_components[
                                    "trajectory"
                                ].detach(),
                                "boundary_color": rollout_boundary_components[
                                    "color"
                                ].detach(),
                            }
                        )
                    if self.world_state_training_mode == "selective_joint":
                        rollout_log_metrics["adapter_only_gradients"] = xs.new_tensor(
                            float(self._suppress_selective_diffusion_gradients)
                        )
                for metric_name, metric_value in rollout_log_metrics.items():
                    self.log(
                        f"{namespace}/rollout_{metric_name}",
                        metric_value,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
            if namespace == "training" and eta_hours is not None:
                self.log(
                    f"{namespace}/eta_hours",
                    float(eta_hours),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
            if self.alignment_coeff > 0:
                self.log(
                    f"{namespace}/alignment_computed",
                    float(alignment_enabled_this_batch),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/alignment_sampling_scale",
                    float(alignment_sampling_scale),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/alignment_coeff_effective",
                    self._alignment_coeff_tensor(alignment_loss_mean).detach(),
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                self.log(
                    f"{namespace}/alignment_loss",
                    alignment_loss_mean,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    prog_bar=True,
                )
                self.log(
                    f"{namespace}/alignment_loss_unweighted",
                    alignment_loss_unweighted_mean,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                )
                if (
                    alignment_enabled_this_batch
                    and hasattr(self.vggt_alignment_loss, "last_metrics")
                ):
                    metrics = self.vggt_alignment_loss.last_metrics
                    metrics["alignment_unit_health_unhealthy_steps"] = (
                        alignment_loss_mean.new_tensor(
                            float(self._unit_health_unhealthy_steps)
                        )
                    )
                    for metric_name in sorted(metrics.keys()):
                        if metric_name in {"alignment_loss", "alignment_loss_unweighted"}:
                            continue
                        self.log(
                            f"{namespace}/{metric_name}",
                            metrics[metric_name],
                            on_step=namespace == "training",
                            on_epoch=namespace != "training",
                            sync_dist=True,
                        )

                # Stage-2 only: output-space video/semantic protection.
                if not self._diffusion_frozen:
                    self.log(
                        f"{namespace}/alignment_generation_loss",
                        alignment_generation_loss,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
                    self.log(
                        f"{namespace}/alignment_generation_loss_unweighted",
                        alignment_generation_loss_unweighted,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
                    self.log(
                        f"{namespace}/alignment_generation_loss_expected",
                        alignment_generation_loss_expected,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )
                    if hasattr(self.vggt_alignment_loss, "last_generation_metrics"):
                        for metric_name in sorted(self.vggt_alignment_loss.last_generation_metrics.keys()):
                            self.log(
                                f"{namespace}/{metric_name}",
                                self.vggt_alignment_loss.last_generation_metrics[metric_name],
                                on_step=namespace == "training",
                                on_epoch=namespace != "training",
                                sync_dist=True,
                            )
                    self.log(
                        f"{namespace}/alignment_generation_loss_valid",
                        alignment_generation_loss_valid,
                        on_step=namespace == "training",
                        on_epoch=namespace != "training",
                        sync_dist=True,
                    )


        if self.unit_inference_enable:
            self._world_state_backbone().clear_world_state_condition()
        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))
        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        return output_dict
