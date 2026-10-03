"""Action-conditioned DFoT adaptation to the learned unified physical space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from upcast.transition_learning import FactorizedTransitionLearning
from utils.distributed_utils import rank_zero_print

from .dfot_video import DFoTVideo


class DFoTActionUPCAST(DFoTVideo):
    """Transfer the RE10K UPCAST teacher to an action-conditioned DiT.

    The first phase registers one middle DiT feature with the fixed geometry
    code space. The second phase freezes UPCAST and distils that shared physical
    transition representation into the action-conditioned video model. UPCAST is
    a training-only teacher; sampling remains the original DFoT path.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        if not self.is_latent_diffusion:
            raise ValueError("Action UPCAST evaluation expects the official latent DFoT")
        alignment = OmegaConf.to_container(cfg.alignment, resolve=True)
        if not isinstance(alignment, dict):
            raise TypeError("alignment must resolve to a mapping")

        self.bootstrap_steps = int(alignment.pop("unit_bootstrap_steps"))
        self.bootstrap_alignment_coeff = float(
            alignment.pop("bootstrap_alignment_coeff")
        )
        self.joint_alignment_coeff = float(alignment.pop("alignment_coeff"))
        self.diffusion_lr = float(alignment.pop("diffusion_lr"))
        self.unit_lr = float(alignment.pop("unit_lr"))
        init_checkpoint = alignment.pop("unit_init_checkpoint", None)
        self.unit_init_min_fraction = float(
            alignment.pop("unit_init_min_fraction", 0.95)
        )
        if self.bootstrap_steps < 0:
            raise ValueError("unit_bootstrap_steps must be non-negative")
        if self.bootstrap_alignment_coeff <= 0 or self.joint_alignment_coeff <= 0:
            raise ValueError("Both bootstrap and joint alignment coefficients must be positive")

        self.vggt_alignment_loss = FactorizedTransitionLearning(**alignment)
        self.current_alignment_coeff = self.bootstrap_alignment_coeff
        self._action_unit_phase = "registration"
        self.requires_find_unused_parameters = True
        # These names are consumed by the generic checkpoint contract.
        self.unit_geometry_tokenizer_steps = 0
        self.unit_joint_ramp_steps = 0
        self.unit_inference_enable = False
        self.unit_video_only_inference = False
        self.unit_allow_curriculum_override = False

        if init_checkpoint not in (None, "", "null"):
            self._load_unit_initialization(str(init_checkpoint))

    def _load_unit_initialization(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"UPCAST initialization checkpoint is missing: {path}")
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        source = payload.get("state_dict", payload)
        prefix = "vggt_alignment_loss."
        target = self.vggt_alignment_loss.state_dict()
        matched: dict[str, torch.Tensor] = {}
        mismatched: list[str] = []
        for key, value in source.items():
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix) :]
            if local_key not in target:
                continue
            if target[local_key].shape != value.shape:
                mismatched.append(local_key)
                continue
            matched[local_key] = value

        # Frozen PAGE/DINO backbone tensors are intentionally absent from UPCAST
        # checkpoints and loaded from their own official assets.
        transferable = {
            key: value
            for key, value in target.items()
            if not key.startswith(
                (
                    "geometry_teacher.model.",
                    "dino_extractor.model.",
                    "generation_video_extractor.model.",
                )
            )
        }
        matched_numel = sum(target[key].numel() for key in matched)
        transferable_numel = sum(value.numel() for value in transferable.values())
        fraction = matched_numel / max(transferable_numel, 1)
        if fraction < self.unit_init_min_fraction:
            raise RuntimeError(
                "UPCAST transfer coverage is too low: "
                f"{fraction:.3%} < {self.unit_init_min_fraction:.3%}; "
                f"matched={len(matched)}, shape_mismatches={mismatched[:8]}"
            )
        incompatible = self.vggt_alignment_loss.load_state_dict(matched, strict=False)
        rank_zero_print(
            "[DFoTActionUPCAST] Loaded shared UPCAST initialization "
            f"from {path}: parameter coverage={fraction:.2%}, "
            f"matched_tensors={len(matched)}, shape_mismatches={len(mismatched)}, "
            f"missing_tensors={len(incompatible.missing_keys)}."
        )
        del payload, source, target, transferable, matched

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        desired = "alignment" if int(self.global_step) < self.bootstrap_steps else "joint"
        if desired == self._action_unit_phase:
            return
        self.vggt_alignment_loss.set_unit_phase(desired)
        self._action_unit_phase = desired
        if desired == "alignment":
            self.current_alignment_coeff = self.bootstrap_alignment_coeff
            self.diffusion_model.eval()
            rank_zero_print(
                "[DFoTActionUPCAST] Registering the middle action-DiT feature "
                f"for {self.bootstrap_steps} optimizer steps."
            )
        else:
            self.current_alignment_coeff = self.joint_alignment_coeff
            self.diffusion_model.train()
            rank_zero_print(
                "[DFoTActionUPCAST] Froze UPCAST and entered action-DiT joint distillation."
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for parameter in self.diffusion_model.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": self.diffusion_lr,
                },
                {
                    "params": [
                        parameter
                        for parameter in self.vggt_alignment_loss.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": self.unit_lr,
                },
            ],
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        frequency = max(int(getattr(self.cfg, "lr_scheduler_frequency", 1) or 1), 1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(
                    optimizer=optimizer,
                    **self.cfg.lr_scheduler,
                ),
                "interval": "step",
                "frequency": frequency,
            },
        }

    def training_step(self, batch, batch_idx, namespace="training"):
        xs, conditions, masks, gt_videos = batch
        if gt_videos is None:
            raise RuntimeError(
                "Action UPCAST training requires latent.load_video_during_training=true"
            )
        if xs.shape[1] != 16 or gt_videos.shape[1] != 16:
            raise RuntimeError(
                "Action UPCAST uses exactly 16 aligned frames, got "
                f"latents={xs.shape[1]}, RGB={gt_videos.shape[1]}"
            )

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        processed_conditions = self._process_conditions(conditions, noise_levels)
        bootstrap = self._action_unit_phase == "alignment"
        context = torch.no_grad() if bootstrap else torch.enable_grad()
        with context:
            xs_pred, diffusion_loss_map, latents, denoising_terms = (
                self.diffusion_model.forward_with_alignment_loss(
                    xs,
                    processed_conditions,
                    k=noise_levels,
                    return_denoising_terms=True,
                )
            )
        if len(latents) != 1 or latents[0].shape[1:] != (16, 768, 16, 16):
            raise RuntimeError(
                "Expected one middle DiT feature [B,16,768,16,16], got "
                f"{[tuple(value.shape) for value in latents]}"
            )

        alignment_loss = self.vggt_alignment_loss(
            latents,
            gt_videos.clamp(0.0, 1.0),
            frozen_diffusion=bootstrap,
            denoising_terms=denoising_terms,
            batch_idx=batch_idx,
        )
        diffusion_loss = self._reweight_loss(diffusion_loss_map, masks)
        total_loss = (
            self.current_alignment_coeff * alignment_loss
            if bootstrap
            else diffusion_loss + self.current_alignment_coeff * alignment_loss
        )

        if batch_idx % self.cfg.logging.loss_freq == 0:
            log_kwargs = {
                "on_step": namespace == "training",
                "on_epoch": namespace != "training",
                "sync_dist": True,
                "batch_size": xs.shape[0],
            }
            self.log(f"{namespace}/loss", total_loss, prog_bar=True, **log_kwargs)
            self.log(f"{namespace}/diffusion_loss", diffusion_loss, **log_kwargs)
            self.log(
                f"{namespace}/unit_weighted_loss",
                self.current_alignment_coeff * alignment_loss,
                **log_kwargs,
            )
            self.log(
                f"{namespace}/unit_phase_alignment",
                xs.new_tensor(float(bootstrap)),
                **log_kwargs,
            )
            selected = {
                name: value
                for name, value in self.vggt_alignment_loss.last_metrics.items()
                if name
                in {
                    "alignment_unit_visual_cross_loss",
                    "alignment_unit_geometry_cross_loss",
                    "alignment_unit_video_geometry_code_agreement",
                    "alignment_unit_physical_retrieval_top1",
                    "alignment_unit_appearance_retrieval_top1",
                    "alignment_unit_signal_valid_fraction",
                }
            }
            self.log_dict(
                {f"{namespace}/{name}": value for name, value in selected.items()},
                **log_kwargs,
            )

        xs_out, xs_pred_out = map(self._unnormalize_x, (xs, xs_pred))
        return {
            "loss": total_loss,
            "xs_pred": xs_pred_out,
            "xs": xs_out,
        }
