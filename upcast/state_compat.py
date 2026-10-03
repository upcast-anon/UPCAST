"""World-state utility cache and checkpoint-compatible policy components.

The policy is intentionally independent of the diffusion timestep.  It sees
one compact summary of the current recurrent state and the requested camera
trajectory, then emits a pair of bounded residual read scales for the entire
autoregressive window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


PREFERENCE_CACHE_VERSION = 2
PREFERENCE_POLICY_VERSION = 3


@dataclass(frozen=True)
class PreferencePolicySpec:
    hidden_dim: int
    content_dim: int = 32
    gate_hidden_dim: int = 64
    max_correction: float = 0.5
    metadata_dim: int = 6
    relation_dim: int = 8

    def validate(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.content_dim < 4:
            raise ValueError("content_dim must be at least 4")
        if self.gate_hidden_dim < 8:
            raise ValueError("gate_hidden_dim must be at least 8")
        if not 0.0 < self.max_correction <= 1.0:
            raise ValueError("max_correction must be in (0,1]")
        if self.metadata_dim < 1 or self.relation_dim < 1:
            raise ValueError("metadata and relation dimensions must be positive")


class StableWorldStatePreferencePolicy(nn.Module):
    """Predict one bounded physical/appearance action per rollout window."""

    def __init__(self, spec: PreferencePolicySpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.query_projection = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.content_dim),
            nn.SiLU(),
        )
        self.physical_projection = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.content_dim),
            nn.SiLU(),
        )
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.content_dim),
            nn.SiLU(),
        )
        input_dim = (
            spec.metadata_dim
            + spec.relation_dim
            + 9 * spec.content_dim
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, spec.gate_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(spec.gate_hidden_dim * 2, spec.gate_hidden_dim),
            nn.SiLU(),
        )
        self.physical_head = nn.Linear(spec.gate_hidden_dim, 1)
        self.appearance_head = nn.Linear(spec.gate_hidden_dim, 1)
        nn.init.zeros_(self.physical_head.weight)
        nn.init.zeros_(self.physical_head.bias)
        nn.init.zeros_(self.appearance_head.weight)
        nn.init.zeros_(self.appearance_head.bias)

    @staticmethod
    def _temporal_summary(frames: Tensor, projection: nn.Module) -> Tensor:
        if frames.ndim != 3:
            raise ValueError(
                f"Preference frame summaries must be [B,T,C], got {tuple(frames.shape)}"
            )
        projected = projection(frames.float())
        return torch.cat(
            (
                projected.mean(dim=1),
                projected[:, -1] - projected[:, 0],
                projected.std(dim=1, unbiased=False),
            ),
            dim=-1,
        )

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        required = (
            "query_frames",
            "physical_frames",
            "appearance_frames",
            "metadata",
            "relations",
        )
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ValueError(f"Missing stable preference inputs: {missing}")
        query = inputs["query_frames"]
        physical = inputs["physical_frames"]
        appearance = inputs["appearance_frames"]
        if query.shape != physical.shape or query.shape != appearance.shape:
            raise ValueError(
                "Stable preference frame tensors must have matching shapes, got "
                f"{tuple(query.shape)}, {tuple(physical.shape)}, "
                f"{tuple(appearance.shape)}"
            )
        metadata = inputs["metadata"].float()
        relations = inputs["relations"].float()
        batch_size = query.shape[0]
        if metadata.shape != (batch_size, self.spec.metadata_dim):
            raise ValueError(
                "Preference metadata shape mismatch: "
                f"{tuple(metadata.shape)} != {(batch_size, self.spec.metadata_dim)}"
            )
        if relations.shape != (batch_size, self.spec.relation_dim):
            raise ValueError(
                "Preference relation shape mismatch: "
                f"{tuple(relations.shape)} != {(batch_size, self.spec.relation_dim)}"
            )

        features = torch.cat(
            (
                metadata,
                self._temporal_summary(query, self.query_projection),
                self._temporal_summary(physical, self.physical_projection),
                self._temporal_summary(appearance, self.appearance_projection),
                relations,
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        physical_logit = self.physical_head(hidden).squeeze(-1)
        appearance_logit = self.appearance_head(hidden).squeeze(-1)
        # Factorial targets observe each branch at scales 0 and 1 only. A
        # positive marginal utility supports retaining scale 1, not
        # extrapolating beyond it. Clamp the residual to attenuation-only so a
        # policy can suppress harmful memory but can never invent an unmeasured
        # >1 gain. Use a straight-through clamp so an accidentally positive
        # logit is still recoverable during fitting instead of entering a dead
        # zero-gradient region.
        physical_raw = self.spec.max_correction * torch.tanh(physical_logit)
        appearance_raw = self.spec.max_correction * torch.tanh(appearance_logit)
        physical_correction = physical_raw + (
            physical_raw.clamp_max(0.0) - physical_raw
        ).detach()
        appearance_correction = appearance_raw + (
            appearance_raw.clamp_max(0.0) - appearance_raw
        ).detach()
        return {
            "physical_logit": physical_logit,
            "appearance_logit": appearance_logit,
            "physical_correction": physical_correction,
            "appearance_correction": appearance_correction,
            # Retain probability diagnostics with 0.5 meaning the full-state
            # identity action. These are not trained with classification BCE.
            "physical_probability": 0.5
            * (physical_correction / self.spec.max_correction + 1.0),
            "appearance_probability": 0.5
            * (appearance_correction / self.spec.max_correction + 1.0),
        }


def continuous_targets_from_repeats(
    physical_advantages: Tensor,
    appearance_advantages: Tensor,
    *,
    temperature: float,
    confidence_temperature: float,
    max_correction: float,
) -> dict[str, Tensor]:
    """Aggregate repeated counterfactual utilities into stable actions.

    Inputs are ``[R,B]`` normalized Shapley advantages where positive means a
    branch reduced the lower-is-better utility. Confidence combines effect
    size with cross-seed sign stability.
    """
    if physical_advantages.ndim != 2 or (
        appearance_advantages.shape != physical_advantages.shape
    ):
        raise ValueError("Repeated preference advantages must be matching [R,B]")
    if physical_advantages.shape[0] < 1:
        raise ValueError("At least one counterfactual noise repeat is required")
    if temperature <= 0 or confidence_temperature <= 0:
        raise ValueError("Preference temperatures must be positive")

    def aggregate(values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = values.float()
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        target = -float(max_correction) * torch.tanh(
            F.relu(-mean) / (2.0 * float(temperature))
        )
        effect_confidence = torch.tanh(
            mean.abs() / float(confidence_temperature)
        )
        if values.shape[0] == 1:
            stability = torch.ones_like(mean)
        else:
            stability = mean.abs() / (mean.abs() + std + 1.0e-6)
        confidence = (effect_confidence * stability).clamp(0.0, 1.0)
        return target, confidence, mean, std

    physical = aggregate(physical_advantages)
    appearance = aggregate(appearance_advantages)
    return {
        "physical_target_correction": physical[0],
        "physical_confidence": physical[1],
        "physical_advantage_mean": physical[2],
        "physical_advantage_std": physical[3],
        "appearance_target_correction": appearance[0],
        "appearance_confidence": appearance[1],
        "appearance_advantage_mean": appearance[2],
        "appearance_advantage_std": appearance[3],
    }


def continuous_preference_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    beta: float,
    identity_weight: float,
    sample_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Confidence-weighted robust regression with a low-confidence identity prior."""
    if beta <= 0:
        raise ValueError("Preference regression beta must be positive")
    if identity_weight < 0:
        raise ValueError("Preference identity weight must be non-negative")
    branch_losses = []
    identity_losses = []
    metrics: dict[str, Tensor] = {}
    for branch in ("physical", "appearance"):
        prediction = predictions[f"{branch}_correction"].float()
        target = targets[f"{branch}_target_correction"].detach().float()
        confidence = targets[f"{branch}_confidence"].detach().float()
        if sample_mask is not None:
            mask = sample_mask.to(device=prediction.device, dtype=torch.bool)
            prediction = prediction[mask]
            target = target[mask]
            confidence = confidence[mask]
        if prediction.numel() == 0:
            # Keep a graph-connected zero for DDP batches containing only
            # held-out samples.
            branch_loss = predictions[f"{branch}_correction"].sum() * 0.0
            identity = branch_loss
            mae = branch_loss.detach()
        else:
            per_sample = F.smooth_l1_loss(
                prediction,
                target,
                beta=float(beta),
                reduction="none",
            )
            weight_sum = confidence.sum()
            if float(weight_sum.detach()) > 1.0e-6:
                branch_loss = (per_sample * confidence).sum() / weight_sum
            else:
                branch_loss = per_sample.sum() * 0.0
            identity = (
                prediction.square() * (1.0 - confidence)
            ).mean()
            mae = (prediction.detach() - target).abs().mean()
        branch_losses.append(branch_loss)
        identity_losses.append(identity)
        metrics[f"{branch}_regression"] = branch_loss.detach()
        metrics[f"{branch}_mae"] = mae.detach()
    regression = 0.5 * sum(branch_losses)
    identity = 0.5 * sum(identity_losses)
    loss = regression + float(identity_weight) * identity
    metrics.update(
        {
            "regression": regression.detach(),
            "identity": identity.detach(),
            "loss": loss.detach(),
        }
    )
    return loss, metrics


def preference_sample_keys(
    target_xs: Tensor,
    conditions: Tensor,
    *,
    target_index: int,
    namespace: str,
) -> list[str]:
    """Build batch/order-independent keys from pose and coarse video content."""
    if target_xs.ndim != 5 or conditions.ndim < 3:
        raise ValueError("Preference cache keys require video and condition batches")
    if target_xs.shape[0] != conditions.shape[0]:
        raise ValueError("Preference cache video/condition batches must match")
    pooled = F.adaptive_avg_pool3d(
        target_xs.detach().float().permute(0, 2, 1, 3, 4),
        (min(4, target_xs.shape[1]), 8, 8),
    )
    pooled = ((pooled.clamp(-4.0, 4.0) + 4.0) * 31.875).round().byte().cpu()
    quantized_conditions = (
        conditions.detach().float().mul(1.0e4).round().to(torch.int32).cpu()
    )
    keys = []
    for index in range(target_xs.shape[0]):
        digest = hashlib.sha256()
        digest.update(f"{namespace}|{int(target_index)}|".encode("utf-8"))
        digest.update(pooled[index].numpy().tobytes())
        digest.update(quantized_conditions[index].numpy().tobytes())
        keys.append(digest.hexdigest())
    return keys


def deterministic_noise(
    template: Tensor,
    sample_keys: list[str],
    *,
    repeat_index: int,
    window_index: int,
) -> Tensor:
    """Generate identical per-sample noise independent of DDP/batch layout."""
    if template.shape[0] != len(sample_keys):
        raise ValueError("Noise template batch does not match preference keys")
    samples = []
    for key in sample_keys:
        seed_digest = hashlib.sha256(
            f"{key}|{repeat_index}|{window_index}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(seed_digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=template.device).manual_seed(seed)
        samples.append(
            torch.randn(
                (1, *template.shape[1:]),
                device=template.device,
                dtype=template.dtype,
                generator=generator,
            )
        )
    return torch.cat(samples, dim=0)


def split_for_key(key: str, holdout_fraction: float) -> str:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0,1)")
    value = int(key[:16], 16) / float(16**16 - 1)
    return "holdout" if value < holdout_fraction else "train"


class CounterfactualUtilityCache:
    """One atomic ``.pt`` shard per clip/window target."""

    def __init__(self, root: str | os.PathLike[str], namespace: str):
        self.root = Path(root).expanduser().resolve()
        self.namespace = str(namespace)
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.root / "metadata.json"
        expected = {
            "cache_version": PREFERENCE_CACHE_VERSION,
            "namespace": self.namespace,
        }
        if not metadata_path.exists():
            # Every DDP rank constructs the cache. Link a fully written,
            # process-unique temporary file into place so no rank can observe a
            # partially written metadata document or race on the same temp name.
            temporary = metadata_path.with_name(
                f".{metadata_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(expected, indent=2, sort_keys=True))
            try:
                os.link(temporary, metadata_path)
            except FileExistsError:
                pass
            finally:
                temporary.unlink(missing_ok=True)
        actual = json.loads(metadata_path.read_text())
        if actual != expected:
            raise ValueError(
                f"Preference cache metadata mismatch: {actual} != {expected}"
            )

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pt"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("cache_version") != PREFERENCE_CACHE_VERSION:
            raise ValueError(f"Unsupported preference cache shard: {path}")
        if payload.get("namespace") != self.namespace or payload.get("key") != key:
            raise ValueError(f"Corrupt preference cache identity: {path}")
        return payload

    def save(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        complete = {
            **payload,
            "cache_version": PREFERENCE_CACHE_VERSION,
            "namespace": self.namespace,
            "key": key,
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.tmp"
        )
        torch.save(complete, temporary)
        os.replace(temporary, path)
        return path

    def shards(self) -> list[Path]:
        return sorted(self.root.glob("??/*.pt"))


class PreferenceCacheDataset(Dataset):
    def __init__(
        self,
        cache: CounterfactualUtilityCache,
        split: str,
    ):
        if split not in {"train", "holdout"}:
            raise ValueError("Preference cache split must be train or holdout")
        self.cache = cache
        self.split = split
        self.paths = []
        for path in cache.shards():
            payload = cache.load(path.stem)
            if payload is None:
                raise RuntimeError(f"Preference cache shard disappeared: {path}")
            if payload.get("split") == split:
                self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        payload = self.cache.load(path.stem)
        if payload is None:
            raise RuntimeError(f"Preference cache shard disappeared: {path}")
        return payload


def detached_policy_inputs(
    inputs: dict[str, Tensor],
    index: int,
) -> dict[str, Tensor]:
    return {
        name: value[index].detach().to(device="cpu", dtype=torch.float16)
        for name, value in inputs.items()
    }


def policy_spec_dict(spec: PreferencePolicySpec) -> dict[str, Any]:
    return asdict(spec)


def pearson_correlation(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.detach().float().flatten()
    target = target.detach().float().flatten()
    if prediction.numel() < 2:
        return float("nan")
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float((prediction * target).sum() / denominator)


def weighted_pearson_correlation(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
) -> float:
    """Confidence-weighted Pearson correlation with finite zero fallback."""
    prediction = prediction.detach().float().flatten()
    target = target.detach().float().flatten()
    weight = weight.detach().float().flatten().clamp_min(0.0)
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("Weighted correlation tensors must have matching shapes")
    total = weight.sum()
    if prediction.numel() < 2 or float(total) <= 1.0e-8:
        return 0.0
    weight = weight / total
    prediction = prediction - (weight * prediction).sum()
    target = target - (weight * target).sum()
    covariance = (weight * prediction * target).sum()
    denominator = (
        (weight * prediction.square()).sum().sqrt()
        * (weight * target.square()).sum().sqrt()
    )
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float(covariance / denominator)


def collate_policy_inputs(
    batch: Iterable[dict[str, Any]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    examples = list(batch)
    inputs = {
        name: torch.stack([example["policy_inputs"][name] for example in examples])
        for name in (
            "query_frames",
            "physical_frames",
            "appearance_frames",
            "metadata",
            "relations",
        )
    }
    targets = {
        name: torch.stack(
            [torch.as_tensor(example["targets"][name]) for example in examples]
        ).float()
        for name in (
            "physical_target_correction",
            "physical_confidence",
            "physical_advantage_mean",
            "appearance_target_correction",
            "appearance_confidence",
            "appearance_advantage_mean",
        )
    }
    return inputs, targets
