"""DFoT fine-tuning with REPA or VideoREPA supervision."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from upcast.representation_alignment import RepresentationAlignmentLoss

from .dfot_video_pose import DFoTVideoPose


class DFoTRepresentationAlignment(DFoTVideoPose):
    """A clean same-backbone implementation of semantic/video alignment.

    This class intentionally does not inherit from ``DFoTUPCAST``:
    REPA baselines should not instantiate UPCAST, PAGE-4D, memory, generation
    protection, or any of their optimizer policies.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        alignment = OmegaConf.to_container(cfg.representation_alignment, resolve=True)
        if not isinstance(alignment, dict):
            raise TypeError("representation_alignment must resolve to a mapping")
        self.representation_alignment_coeff = float(
            alignment.pop("alignment_coeff", 0.5)
        )
        if self.representation_alignment_coeff <= 0:
            raise ValueError("representation alignment coefficient must be positive")
        self.representation_alignment = RepresentationAlignmentLoss(**alignment)

    def _should_include_in_checkpoint(self, key: str) -> bool:
        """Persist the trainable projector while excluding the frozen teacher."""
        return super()._should_include_in_checkpoint(key) or key.startswith(
            "representation_alignment.projector."
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["representation_alignment_metadata"] = {
            "version": 1,
            "method": self.representation_alignment.method,
            "latent_return_mode": str(self.cfg.backbone.latent_return_mode),
            "latent_return_stage": str(self.cfg.backbone.latent_return_stage),
            "latent_return_level": int(self.cfg.backbone.latent_return_level),
            "latent_return_blocks": [
                int(value) for value in self.cfg.backbone.latent_return_blocks
            ],
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        metadata = checkpoint.get("representation_alignment_metadata")
        if not metadata:
            return
        expected = {
            "method": self.representation_alignment.method,
            "latent_return_mode": str(self.cfg.backbone.latent_return_mode),
            "latent_return_stage": str(self.cfg.backbone.latent_return_stage),
            "latent_return_level": int(self.cfg.backbone.latent_return_level),
            "latent_return_blocks": [
                int(value) for value in self.cfg.backbone.latent_return_blocks
            ],
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Representation-alignment checkpoint/config mismatch: "
                f"{mismatches}"
            )

    def configure_optimizers(self):
        parameters = [
            parameter
            for parameter in (
                list(self.diffusion_model.parameters())
                + list(self.representation_alignment.projector.parameters())
            )
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        frequency = max(
            int(getattr(self.cfg, "lr_scheduler_frequency", 1) or 1), 1
        )
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
        xs, conditions, masks, *_ = batch
        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        processed_conditions = self._process_conditions(conditions, noise_levels)
        xs_pred, diffusion_loss_map, latents = (
            self.diffusion_model.forward_with_alignment_loss(
                xs,
                processed_conditions,
                k=noise_levels,
            )
        )
        if len(latents) != 1:
            raise RuntimeError(
                "REPA/VideoREPA requires exactly one aligned DFoT feature, got "
                f"{len(latents)}"
            )
        diffusion_loss = self._reweight_loss(diffusion_loss_map, masks)
        clean_video = self._unnormalize_x(xs).clamp(0.0, 1.0)
        alignment_loss, alignment_metrics = self.representation_alignment(
            latents[0], clean_video
        )
        total_loss = (
            diffusion_loss
            + self.representation_alignment_coeff * alignment_loss
        )

        if batch_idx % self.cfg.logging.loss_freq == 0:
            log_kwargs = {
                "on_step": namespace == "training",
                "on_epoch": namespace != "training",
                "sync_dist": True,
                "batch_size": xs.shape[0],
            }
            self.log(f"{namespace}/loss", total_loss, prog_bar=True, **log_kwargs)
            self.log(
                f"{namespace}/diffusion_loss", diffusion_loss, **log_kwargs
            )
            self.log(
                f"{namespace}/representation_weighted_loss",
                self.representation_alignment_coeff * alignment_loss,
                **log_kwargs,
            )
            self.log_dict(
                {
                    f"{namespace}/{name}": value
                    for name, value in alignment_metrics.items()
                },
                **log_kwargs,
            )

        xs_out, xs_pred_out = map(self._unnormalize_x, (xs, xs_pred))
        return {
            "loss": total_loss,
            "xs_pred": xs_pred_out,
            "xs": xs_out,
        }
