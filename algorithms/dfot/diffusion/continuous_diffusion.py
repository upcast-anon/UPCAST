"""
A very minimal implementation of continuous-time diffusion models. For compatibility with other modules,
sampling schedules are still implemented in discrete time.
"""

from abc import ABC, abstractmethod
from typing import Optional
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from .discrete_diffusion import DiscreteDiffusion, ModelPrediction


class ContinuousNoiseSchedule(nn.Module, ABC):
    """
    An abstract class for continuous noise schedule that is compatible with continuous-time diffusion models.
    """

    @classmethod
    def from_config(cls, cfg: DictConfig):
        match cfg.name:
            case "cosine":
                return CosineNoiseSchedule(cfg)
            case _:
                raise ValueError(f"unknown noise schedule {cfg.name}")

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Given the timestep t within [0, 1], return the logSNR value at that timestep."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_logsnr(self) -> torch.Tensor:
        """Return the maximum logSNR value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def min_logsnr(self) -> torch.Tensor:
        """Return the minimum logSNR value."""
        raise NotImplementedError


class CosineNoiseSchedule(ContinuousNoiseSchedule):
    """
    Cosine noise schedule that can be shifted from base resolution to target resolution,
    proposed in Simple Diffusion (2023, https://arxiv.org/abs/2301.11093).
    Here, `shift` should be set to `base_resolution / target_resolution`.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        logsnr_min, logsnr_max = cfg.get("logsnr_min", -15.0), cfg.get(
            "logsnr_max", 15.0
        )
        shift = cfg.get("shift", 1.0)
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "shift",
            2 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(0.0, dtype=torch.float32, device=self.shift.device)
        )

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(1.0, dtype=torch.float32, device=self.shift.device)
        )


class ContinuousDiffusion(DiscreteDiffusion):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
    ):
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens, external_cond_dim)
        assert (
            self.objective == "pred_v" and self.loss_weighting.strategy == "sigmoid"
        ), "ContinuousDiffusion only supports 'pred_v' objective and 'sigmoid' loss weighting"
        self.precond_scale = cfg.precond_scale
        self.sigmoid_bias = cfg.loss_weighting.sigmoid_bias

    def _build_buffer(self):
        super()._build_buffer()
        self.training_schedule = ContinuousNoiseSchedule.from_config(
            self.cfg.training_schedule
        )

    def model_predictions(self, x, k, external_cond=None, external_cond_mask=None,**kwargs):
        model_output = self.model(
            x, self.precond_scale * self.logsnr[k], external_cond, external_cond_mask,**kwargs
        )

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        **kwargs
    ):
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        # v-prediction
        v_pred = self.model(x_t, self.precond_scale * logsnr, external_cond,**kwargs)
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        # sigmoid loss weighting
        # proposed by Kingma & Gao (2023, https://arxiv.org/abs/2303.00848)
        # further studied in Simple Diffusion 2 (2024, https://arxiv.org/abs/2410.19324)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    def forward_with_alignment_loss(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        return_denoising_terms: bool = False,
        return_tail_state: bool = False,
        return_latents: bool = True,
        prepared_denoising_terms: Optional[dict[str, torch.Tensor | None]] = None,

    ):
        if prepared_denoising_terms is None:
            prepared_denoising_terms = self.prepare_training_denoising(x, k)
        logsnr = prepared_denoising_terms["logsnr"]
        noise = prepared_denoising_terms["noise"]
        x_t = prepared_denoising_terms["x_t"]
        external_cond_mask = prepared_denoising_terms.get("external_cond_mask")
        if logsnr is None or noise is None or x_t is None:
            raise RuntimeError("Prepared denoising terms are incomplete")
        if x_t.shape != x.shape or noise.shape != x.shape:
            raise ValueError(
                "Prepared denoising tensors must match the clean input: "
                f"x={tuple(x.shape)}, x_t={tuple(x_t.shape)}, "
                f"noise={tuple(noise.shape)}"
            )
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        # v-prediction
        # import pdb; pdb.set_trace()
        ## START modification for alignment loss
        if return_latents:
            model_out = self.model(
                x_t,
                self.precond_scale * logsnr,
                external_cond,
                external_cond_mask=external_cond_mask,
                return_latents=True,
                return_tail_state=return_tail_state,
            )
            tail_state = None
            if isinstance(model_out, tuple) and len(model_out) == 3:
                v_pred, latents_list, tail_state = model_out
            else:
                v_pred, latents_list = model_out
        else:
            v_pred = self.model(
                x_t,
                self.precond_scale * logsnr,
                external_cond,
                external_cond_mask=external_cond_mask,
            )
            latents_list = []
            tail_state = None
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred
        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        # sigmoid loss weighting
        # proposed by Kingma & Gao (2023, https://arxiv.org/abs/2303.00848)
        # further studied in Simple Diffusion 2 (2024, https://arxiv.org/abs/2410.19324)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        if return_denoising_terms:
            return x_pred, loss, latents_list, {
                "v_pred": v_pred,
                "x_t": x_t.detach(),
                "noise": noise.detach(),
                "logsnr": logsnr.detach(),
                "tail_state": tail_state,
                "external_cond_mask": external_cond_mask,
                "sigmoid_bias": torch.as_tensor(
                    float(self.sigmoid_bias),
                    device=logsnr.device,
                    dtype=logsnr.dtype,
                ),
            }
        return x_pred, loss , latents_list

    def prepare_training_denoising(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Sample one denoising task that current and frozen models can replay.

        Keeping this operation separate lets a frozen prior run before the
        trainable DFoT autograd graph is built, which materially lowers peak
        memory during joint training.
        """
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        conditioning_dropout = float(
            getattr(self.model, "conditioning_dropout", 0.0) or 0.0
        )
        external_cond_mask = None
        if self.model.training and conditioning_dropout > 0:
            external_cond_mask = (
                torch.rand(x.shape[0], device=x.device) < conditioning_dropout
            )
        return {
            "x_t": x_t.detach(),
            "noise": noise.detach(),
            "logsnr": logsnr.detach(),
            "external_cond_mask": external_cond_mask,
        }

    def context_exposure_loss(
        self,
        external_cond: Optional[torch.Tensor],
        prepared_denoising_terms: dict[str, torch.Tensor | None],
        context_estimate: torch.Tensor,
        context_mask: torch.Tensor,
        activation_checkpointing: bool = True,
    ) -> torch.Tensor:
        """Denoise future frames while conditioning on generated-like history.

        Context frames are replaced by detached estimates and marked as clean.
        Their targets are intentionally invalid and must be masked by the caller;
        every non-context frame retains the original, exact diffusion task.
        """
        x_t = prepared_denoising_terms.get("x_t")
        noise = prepared_denoising_terms.get("noise")
        logsnr = prepared_denoising_terms.get("logsnr")
        if x_t is None or noise is None or logsnr is None:
            raise RuntimeError(
                "Context exposure requires prepared x_t, noise, and logSNR"
            )
        if context_estimate.shape != x_t.shape:
            raise ValueError(
                "Context estimate must match x_t, got "
                f"{tuple(context_estimate.shape)} and {tuple(x_t.shape)}"
            )
        if context_mask.shape != x_t.shape[:2]:
            raise ValueError(
                "Context mask must have shape [B,T], got "
                f"{tuple(context_mask.shape)} for x_t={tuple(x_t.shape)}"
            )
        if context_mask.dtype != torch.bool:
            context_mask = context_mask.bool()

        expanded_mask = context_mask
        while expanded_mask.ndim < x_t.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        exposed_x_t = torch.where(
            expanded_mask,
            context_estimate.detach().to(x_t),
            x_t,
        )
        clean_logsnr = self.training_schedule.max_logsnr.to(logsnr)
        exposed_logsnr = torch.where(
            context_mask,
            clean_logsnr.expand_as(logsnr),
            logsnr,
        )
        alpha_t = self.add_shape_channels(torch.sigmoid(exposed_logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-exposed_logsnr).sqrt())
        scaled_logsnr = self.precond_scale * exposed_logsnr
        external_cond_mask = prepared_denoising_terms.get("external_cond_mask")

        def model_forward(model_x_t, model_logsnr):
            return self.model(
                model_x_t,
                model_logsnr,
                external_cond,
                external_cond_mask=external_cond_mask,
            )

        should_checkpoint = bool(
            activation_checkpointing
            and torch.is_grad_enabled()
            and any(parameter.requires_grad for parameter in self.model.parameters())
        )
        if should_checkpoint:
            v_pred = activation_checkpoint(
                model_forward,
                exposed_x_t,
                scaled_logsnr,
                use_reentrant=False,
                # DFoT contains dropout. Backward recomputation must replay the
                # same masks or the checkpointed gradient is not the gradient
                # of the forward loss.
                preserve_rng_state=True,
            )
        else:
            v_pred = model_forward(exposed_x_t, scaled_logsnr)
        noise_pred = alpha_t * v_pred + sigma_t * exposed_x_t
        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")
        loss_weight = torch.sigmoid(self.sigmoid_bias - exposed_logsnr)
        return loss * self.add_shape_channels(loss_weight)

    def forward_with_alignment_latents_only(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
        return_denoising_terms: bool = False,
        return_tail_state: bool = False,
    ):
        """
        Forward pass used by frozen-diffusion stage:
        - keeps the same noisy denoising path to get intermediate latents
        - skips diffusion loss computation to save compute and avoid unused branches
        """
        logsnr = self.training_schedule(k)
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        model_out = self.model(
            x_t,
            self.precond_scale * logsnr,
            external_cond,
            return_latents=True,
            return_tail_state=return_tail_state,
        )
        tail_state = None
        if isinstance(model_out, tuple) and len(model_out) == 3:
            v_pred, latents_list, tail_state = model_out
        else:
            v_pred, latents_list = model_out
        x_pred = alpha_t * x_t - sigma_t * v_pred
        if return_denoising_terms:
            return x_pred, latents_list, {
                "v_pred": v_pred,
                "x_t": x_t.detach(),
                "noise": noise.detach(),
                "logsnr": logsnr.detach(),
                "tail_state": tail_state,
                "sigmoid_bias": torch.as_tensor(
                    float(self.sigmoid_bias),
                    device=logsnr.device,
                    dtype=logsnr.dtype,
                ),
            }
        return x_pred, latents_list

    @torch.no_grad()
    def extract_clean_alignment_latents(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Capture DFoT features from a finalized clean autoregressive window."""
        clean_logsnr = self.training_schedule.max_logsnr.to(x)
        logsnr = clean_logsnr.expand(x.shape[0], x.shape[1])
        model_out = self.model(
            x,
            self.precond_scale * logsnr,
            external_cond,
            external_cond_mask=None,
            return_latents=True,
            return_tail_state=False,
        )
        if not isinstance(model_out, tuple) or len(model_out) < 2:
            raise RuntimeError(
                "Backbone did not return alignment latents for world-state update"
            )
        return list(model_out[1]), logsnr
