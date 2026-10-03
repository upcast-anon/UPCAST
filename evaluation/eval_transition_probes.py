#!/usr/bin/env python3
"""Probe the learned factorized shared-physical/private-visual space.

The probe uses the exact training configuration stored beside a checkpoint and
does not update any model parameter.  It evaluates fixed temporal spans so that
video, PAGE, DINO, and fused branches always describe the same transition.
Wrong clip, wrong span, reversed-time, photometric, and decoder-component
controls make the resulting numbers harder to explain by scene identity alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import roma
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import build_experiment
from upcast.transition_learning import PreparedPhysicalTeacherContext
from utils.geometry_utils import CameraPose


DEFAULT_CKPT = "checkpoints/upcast_acquired.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output-dir",
        default="output/unified_space_probe",
    )
    parser.add_argument("--num-clips", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--spans", default="0-3,0-7,0-15")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-photometric-control", action="store_true")
    parser.add_argument("--skip-decoder-ablation", action="store_true")
    return parser.parse_args()


def infer_config_path(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find .hydra/config.yaml above checkpoint {checkpoint}"
    )


def parse_spans(value: str, num_frames: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for item in value.split(","):
        fields = item.strip().split("-")
        if len(fields) != 2:
            raise ValueError(f"Invalid span {item!r}; expected current-future")
        current, future = (int(field) for field in fields)
        if not 0 <= current < future < num_frames:
            raise ValueError(
                f"Span {(current, future)} is outside [0,{num_frames - 1}]"
            )
        spans.append((current, future))
    if len(spans) < 2:
        raise ValueError("At least two spans are required for wrong-span controls")
    if len(set(spans)) != len(spans):
        raise ValueError("Probe spans must be unique")
    return tuple(spans)


def load_probe_config(args: argparse.Namespace):
    checkpoint = Path(args.ckpt).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else infer_config_path(checkpoint)
    )
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.experiment._name = "video_generation"
        cfg.dataset._name = "realestate10k"
        cfg.algorithm._name = "dfot_upcast"
        cfg.wandb.mode = "disabled"
        cfg.experiment.tasks = ["training"]
        cfg.experiment.training.batch_size = args.batch_size
        cfg.experiment.training.data.num_workers = args.num_workers
        cfg.experiment.training.data.shuffle = False
        cfg.experiment.validation.batch_size = args.batch_size
        cfg.experiment.validation.data.num_workers = args.num_workers
        cfg.experiment.validation.data.shuffle = False
        cfg.experiment.validation.limit_batch = 0.0
        cfg.algorithm.compile = False
        cfg.algorithm.checkpoint.strict = True
        cfg.algorithm.checkpoint.weights_source = "raw"
        # These branches are training-time regularizers and are not needed to
        # read the already learned UPCAST representations.
        cfg.algorithm.alignment.video_prior_preservation_weight = 0.0
        cfg.algorithm.alignment.video_prior_mode = "full_model"
        cfg.algorithm.alignment.compute_generation_loss = False
        cfg.algorithm.alignment.unit_inference_enable = False
        cfg.algorithm.alignment.rollout_training_enable = False
        cfg.algorithm.alignment.unit_physical_only_continuation = False
        cfg.algorithm.alignment.unit_video_only_inference = False
        # The V8 source predates the later preference-policy memory head. The
        # factorization probe reads UPCAST encoders/decoders only, so do not
        # instantiate or migrate an unrelated runtime memory adapter.
        cfg.algorithm.backbone.world_state.enable = False
        cfg.algorithm.alignment.world_state_training_mode = "joint"
    return cfg, checkpoint, config_path


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def deterministic_photometric_transform(images: torch.Tensor) -> torch.Tensor:
    """Change appearance while preserving the image-plane scene layout."""
    gains = images.new_tensor((1.28, 0.76, 1.12)).view(1, 1, 3, 1, 1)
    transformed = images.clamp(0.0, 1.0).pow(0.72) * gains + 0.04
    luminance = transformed.mean(dim=2, keepdim=True)
    transformed = luminance + 1.22 * (transformed - luminance)
    return transformed.clamp(0.0, 1.0)


def descriptor(tokens: torch.Tensor) -> torch.Tensor:
    tokens = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    return torch.cat((tokens.flatten(1), tokens.mean(dim=1), tokens.std(dim=1)), dim=-1)


def feature_map_descriptor(features: torch.Tensor) -> torch.Tensor:
    values = features.float().flatten(2)
    return torch.cat((values.mean(dim=2), values.std(dim=2)), dim=1)


def gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Batch mismatch while gathering time: {values.shape[0]} != {indices.shape[0]}"
        )
    return values[torch.arange(values.shape[0], device=values.device), indices]


def appearance_statistics(
    images: torch.Tensor,
    current: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    """A geometry-light target describing color and contrast changes."""

    def statistics(frames: torch.Tensor) -> torch.Tensor:
        flat = frames.float().flatten(2)
        quantiles = torch.quantile(
            flat,
            frames.new_tensor((0.1, 0.5, 0.9), dtype=torch.float32),
            dim=2,
        ).permute(1, 0, 2).flatten(1)
        maximum = frames.float().amax(dim=1)
        minimum = frames.float().amin(dim=1)
        saturation = (maximum - minimum).flatten(1)
        luminance = (
            0.2126 * frames[:, 0].float()
            + 0.7152 * frames[:, 1].float()
            + 0.0722 * frames[:, 2].float()
        ).flatten(1)
        return torch.cat(
            (
                flat.mean(dim=2),
                flat.std(dim=2),
                quantiles,
                saturation.mean(dim=1, keepdim=True),
                saturation.std(dim=1, keepdim=True),
                luminance.mean(dim=1, keepdim=True),
                luminance.std(dim=1, keepdim=True),
            ),
            dim=1,
        )

    current_stats = statistics(gather_time(images, current))
    future_stats = statistics(gather_time(images, future))
    return torch.cat((future_stats, future_stats - current_stats), dim=1)


def representation_sensitivity(
    clean: torch.Tensor,
    changed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    clean_float = clean.float()
    changed_float = changed.float()
    cosine = cosine_rows(descriptor(clean_float), descriptor(changed_float))
    relative_l2 = (
        (changed_float - clean_float).flatten(1).norm(dim=1)
        / clean_float.flatten(1).norm(dim=1).clamp_min(1e-8)
    )
    return {"cosine": cosine, "relative_l2": relative_l2}


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (F.normalize(left.float(), dim=-1) * F.normalize(right.float(), dim=-1)).sum(-1)


def retrieval(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    similarity = left @ right.t()
    count = similarity.shape[0]
    target = torch.arange(count)
    diagonal = similarity.diagonal()
    wrong = similarity.masked_fill(torch.eye(count, dtype=torch.bool), -torch.inf)
    hardest = wrong.max(dim=1).values
    ranks_lr = (similarity.argsort(dim=1, descending=True) == target[:, None]).nonzero()[:, 1]
    ranks_rl = (similarity.t().argsort(dim=1, descending=True) == target[:, None]).nonzero()[:, 1]
    return {
        "count": count,
        "chance_top1": 1.0 / max(count, 1),
        "top1_left_to_right": float((ranks_lr == 0).float().mean()),
        "top1_right_to_left": float((ranks_rl == 0).float().mean()),
        "top5_left_to_right": float((ranks_lr < min(5, count)).float().mean()),
        "top5_right_to_left": float((ranks_rl < min(5, count)).float().mean()),
        "mean_rank_left_to_right": float(ranks_lr.float().mean() + 1.0),
        "mean_rank_right_to_left": float(ranks_rl.float().mean() + 1.0),
        "paired_cosine": float(diagonal.mean()),
        "paired_hard_negative_margin": float((diagonal - hardest).mean()),
    }


def grouped_span_retrieval(
    left: torch.Tensor,
    right: torch.Tensor,
    clip_ids: torch.Tensor,
) -> dict[str, float]:
    """Retrieve the correct transition span while holding scene identity fixed."""
    top1: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for clip_id in clip_ids.unique(sorted=True):
        keep = clip_ids == clip_id
        left_clip = F.normalize(left[keep].float(), dim=-1)
        right_clip = F.normalize(right[keep].float(), dim=-1)
        similarity = left_clip @ right_clip.t()
        target = torch.arange(similarity.shape[0])
        diagonal = similarity.diagonal()
        wrong = similarity.masked_fill(
            torch.eye(similarity.shape[0], dtype=torch.bool),
            -torch.inf,
        )
        top1.append((similarity.argmax(dim=1) == target).float())
        margins.append(diagonal - wrong.max(dim=1).values)
    return {
        "top1": float(torch.cat(top1).mean()),
        "chance_top1": 1.0 / int((clip_ids == clip_ids[0]).sum()),
        "paired_wrong_span_margin": float(torch.cat(margins).mean()),
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean(dim=0, keepdim=True)
    right = right.double() - right.double().mean(dim=0, keepdim=True)
    # The Gram form is exactly equivalent to ||X^T Y||_F^2 but scales with
    # the number of transitions rather than the flattened token dimension.
    left_gram = left @ left.t()
    right_gram = right @ right.t()
    numerator = (left_gram * right_gram).sum()
    denominator = (
        left_gram.square().sum().sqrt()
        * right_gram.square().sum().sqrt()
    ).clamp_min(1e-12)
    return float(numerator / denominator)


def code_metrics(
    indices: dict[str, torch.Tensor],
    codebook_size: int,
) -> dict[str, Any]:
    def pair_statistics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        left_flat = left.reshape(-1).long()
        right_flat = right.reshape(-1).long()
        left_count = torch.bincount(left_flat, minlength=codebook_size).double()
        right_count = torch.bincount(right_flat, minlength=codebook_size).double()
        left_prob = left_count / left_count.sum().clamp_min(1.0)
        right_prob = right_count / right_count.sum().clamp_min(1.0)
        joint = torch.bincount(
            left_flat * codebook_size + right_flat,
            minlength=codebook_size**2,
        ).double().reshape(codebook_size, codebook_size)
        joint /= joint.sum().clamp_min(1.0)
        independent = left_prob[:, None] * right_prob[None]
        valid = joint > 0
        mutual_information = (
            joint[valid]
            * (joint[valid] / independent[valid].clamp_min(1e-12)).log()
        ).sum()
        entropy_left = -(
            left_prob[left_prob > 0] * left_prob[left_prob > 0].log()
        ).sum()
        entropy_right = -(
            right_prob[right_prob > 0] * right_prob[right_prob > 0].log()
        ).sum()
        return {
            "exact_token_agreement": float((left_flat == right_flat).double().mean()),
            "histogram_overlap": float(torch.minimum(left_prob, right_prob).sum()),
            "normalized_mutual_information": float(
                mutual_information
                / (entropy_left * entropy_right).sqrt().clamp_min(1e-12)
            ),
        }

    output: dict[str, Any] = {"branches": {}, "pairs": {}}
    for name, values in indices.items():
        output["branches"][name] = {}
        for level in range(values.shape[-1]):
            flat = values[..., level].reshape(-1).long()
            counts = torch.bincount(flat, minlength=codebook_size).double()
            probability = counts / counts.sum().clamp_min(1.0)
            entropy = -(probability[probability > 0] * probability[probability > 0].log()).sum()
            output["branches"][name][f"q{level}"] = {
                "active_codes": int((counts > 0).sum()),
                "active_ratio": float((counts > 0).double().mean()),
                "perplexity": float(entropy.exp()),
            }
    for left_name, right_name in (("video", "geometry"), ("video", "fused"), ("geometry", "fused")):
        left = indices[left_name]
        right = indices[right_name]
        levels: dict[str, Any] = {}
        for level in range(left.shape[-1]):
            paired = pair_statistics(left[..., level], right[..., level])
            shuffle_shift = max(1, right.shape[0] // 2)
            shuffled = pair_statistics(
                left[..., level],
                right[..., level].roll(shuffle_shift, dims=0),
            )
            levels[f"q{level}"] = {
                **paired,
                "shuffled_exact_token_agreement": shuffled[
                    "exact_token_agreement"
                ],
                "shuffled_normalized_mutual_information": shuffled[
                    "normalized_mutual_information"
                ],
                "paired_minus_shuffled_agreement": paired[
                    "exact_token_agreement"
                ]
                - shuffled["exact_token_agreement"],
                "paired_minus_shuffled_nmi": paired[
                    "normalized_mutual_information"
                ]
                - shuffled["normalized_mutual_information"],
            }
        output["pairs"][f"{left_name}_{right_name}"] = levels
    return output


def ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    clip_ids: torch.Tensor,
    ridge: float,
) -> dict[str, float]:
    unique = clip_ids.unique(sorted=True)
    split = max(2, min(len(unique) - 1, int(math.floor(0.75 * len(unique)))))
    train_ids = unique[:split]
    train_mask = torch.isin(clip_ids, train_ids)
    test_mask = ~train_mask
    x_train = features[train_mask].double()
    x_test = features[test_mask].double()
    y_train = targets[train_mask].double()
    y_test = targets[test_mask].double()
    x_mean = x_train.mean(0, keepdim=True)
    x_std = x_train.std(0, keepdim=True).clamp_min(1e-5)
    y_mean = y_train.mean(0, keepdim=True)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    centered = y_train - y_mean
    gram = x_train @ x_train.t() / x_train.shape[1]
    scale = gram.diagonal().mean().clamp_min(1e-8)
    dual = torch.linalg.solve(
        gram + ridge * scale * torch.eye(gram.shape[0], dtype=gram.dtype),
        centered,
    )
    weight = x_train.t() @ dual / x_train.shape[1]
    prediction = x_test @ weight + y_mean
    residual = (prediction - y_test).square().sum()
    total = (y_test - y_test.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
    return {
        "train_clips": int(split),
        "test_clips": int(len(unique) - split),
        "rmse": float((prediction - y_test).square().mean().sqrt()),
        "r2": float(1.0 - residual / total),
        "cosine": float(cosine_rows(prediction, y_test).mean()),
    }


def relative_camera_target(
    conditions: torch.Tensor,
    current: int,
    future: int,
) -> torch.Tensor:
    pose = CameraPose.from_vectors(conditions.float())
    pose.normalize_by_first()
    extrinsics = pose.extrinsics()
    rotation_current = extrinsics[:, current, :3, :3]
    rotation_future = extrinsics[:, future, :3, :3]
    translation_current = extrinsics[:, current, :3, 3]
    translation_future = extrinsics[:, future, :3, 3]
    relative_rotation = rotation_future @ rotation_current.transpose(-1, -2)
    relative_translation = (
        translation_future
        - (relative_rotation @ translation_current.unsqueeze(-1)).squeeze(-1)
    )
    return torch.cat((roma.rotmat_to_rotvec(relative_rotation), relative_translation), dim=-1)


@torch.no_grad()
def encode_spans(
    module,
    latents: list[torch.Tensor] | torch.Tensor,
    denoising_terms: dict[str, Any],
    teacher,
    dino_features: torch.Tensor,
    spans: tuple[tuple[int, int], ...],
) -> list[dict[str, torch.Tensor]]:
    outputs: list[dict[str, torch.Tensor]] = []
    batch_size = teacher.target.shape[0]
    source_latents = module._select_source_latents(latents)
    num_frames = source_latents[0].shape[1]
    logsnr = module._frame_logsnr(
        denoising_terms,
        batch_size,
        num_frames,
        source_latents[0].device,
    )
    for current_index, future_index in spans:
        current = torch.full(
            (batch_size,), current_index, device=teacher.target.device, dtype=torch.long
        )
        future = torch.full_like(current, future_index)
        context = PreparedPhysicalTeacherContext(
            teacher=teacher,
            current=current,
            future=future,
            delta=future - current,
        )
        tokens = module.encode_generative_physical_tokens(
            latents,
            denoising_terms,
            context,
        )
        page_features = torch.nan_to_num(teacher.features)
        points = torch.nan_to_num(teacher.target)
        geometry_current, geometry_future = module.geometry_pair_projector(
            page_features,
            points,
            current,
            future,
        )
        physical_geometry = module.geometry_transition_encoder(
            geometry_current,
            geometry_future,
            future - current,
        )
        geometry_up, _, geometry_metrics = module.unit_quantizer(
            module.geometry_vq_adapter(physical_geometry),
            freeze_codebook=True,
        )
        video_current, video_future = module.video_pair_projector(
            source_latents,
            current,
            future,
        )
        current_logsnr = logsnr.gather(1, current[:, None]).squeeze(1)
        future_logsnr = logsnr.gather(1, future[:, None]).squeeze(1)
        appearance_video = module.video_appearance_encoder(
            video_current,
            video_future,
            future - current,
            current_logsnr,
            future_logsnr,
        )
        pair_dino = torch.stack(
            (
                gather_time(dino_features, current),
                gather_time(dino_features, future),
            ),
            dim=1,
        )
        pair_current = torch.zeros_like(current)
        pair_future = torch.ones_like(future)
        semantic_current, semantic_future = module.semantic_pair_projector(
            pair_dino,
            pair_current,
            pair_future,
        )
        appearance_semantic = module.semantic_transition_encoder(
            semantic_current,
            semantic_future,
            future - current,
        )
        appearance_fused = module.appearance_fusion_encoder(
            appearance_video,
            appearance_semantic,
        )
        outputs.append(
            {
                "video": tokens.video,
                "geometry": geometry_metrics["native_quantized"],
                "fused": tokens.fused,
                "video_world_physical": module.unit_quantizer.project_up(
                    tokens.video
                ),
                "geometry_world_physical": geometry_up,
                "fused_world_physical": module.unit_quantizer.project_up(
                    tokens.fused
                ),
                "appearance_video": appearance_video,
                "appearance_semantic": appearance_semantic,
                "appearance_fused": appearance_fused,
                "video_indices": tokens.video_indices,
                "geometry_indices": geometry_metrics["indices"],
                "fused_indices": tokens.fused_indices,
                "physical_geometry": physical_geometry,
                "current": current,
                "future": future,
                "delta": future - current,
                "current_dino": pair_dino[:, 0],
                "target_dino": pair_dino[:, 1],
            }
        )
    return outputs


@torch.no_grad()
def encode_video_only_spans(
    module,
    latents: list[torch.Tensor] | torch.Tensor,
    denoising_terms: dict[str, Any],
    spans: tuple[tuple[int, int], ...],
) -> list[dict[str, torch.Tensor]]:
    source_latents = module._select_source_latents(latents)
    batch_size, num_frames = source_latents[0].shape[:2]
    logsnr = module._frame_logsnr(
        denoising_terms,
        batch_size,
        num_frames,
        source_latents[0].device,
    )
    outputs = []
    for current_index, future_index in spans:
        current = torch.full(
            (batch_size,), current_index, device=logsnr.device, dtype=torch.long
        )
        future = torch.full_like(current, future_index)
        current_map, future_map = module.video_pair_projector(
            source_latents,
            current,
            future,
        )
        current_logsnr = logsnr.gather(1, current[:, None]).squeeze(1)
        future_logsnr = logsnr.gather(1, future[:, None]).squeeze(1)
        physical = module.video_transition_encoder(
            current_map,
            future_map,
            future - current,
            current_logsnr,
            future_logsnr,
        )
        appearance = module.video_appearance_encoder(
            current_map,
            future_map,
            future - current,
            current_logsnr,
            future_logsnr,
        )
        _, _, metrics = module.unit_quantizer(
            module.video_vq_adapter(physical), freeze_codebook=True
        )
        outputs.append(
            {
                "physical": metrics["native_quantized"],
                "appearance": appearance,
                "indices": metrics["indices"],
            }
        )
    return outputs


@torch.no_grad()
def decoder_component_ablation(
    module,
    encoded: list[dict[str, torch.Tensor]],
    images: torch.Tensor,
    teacher,
) -> dict[str, torch.Tensor]:
    """Intervene on one long-span world token without retraining the decoder.

    The longest configured span is the target and the preceding span supplies
    a same-scene wrong-transition control.  Zeroing tests component necessity;
    wrong-span replacement tests whether a loss increase is caused by temporal
    content rather than only by an out-of-distribution zero vector.
    """
    if len(encoded) < 2:
        raise ValueError("Decoder component ablation requires at least two spans")
    target = encoded[-1]
    wrong = encoded[-2]
    physical = target["video_world_physical"]
    appearance = target["appearance_video"]
    zero_physical = torch.zeros_like(physical)
    zero_appearance = torch.zeros_like(appearance)
    visual_anchor = module.visual_anchor_encoder(
        gather_time(images, target["current"]),
        target["current_dino"],
    )
    visual_variants = {
        "full": module.world_token_fusion(physical, appearance),
        "physical_only": module.world_token_fusion(physical, zero_appearance),
        "appearance_only": module.world_token_fusion(zero_physical, appearance),
        "neither": module.world_token_fusion(zero_physical, zero_appearance),
        "wrong_physical": module.world_token_fusion(
            wrong["video_world_physical"], appearance
        ),
        "wrong_appearance": module.world_token_fusion(
            physical,
            wrong["appearance_video"],
        ),
    }
    output: dict[str, torch.Tensor] = {}
    for name, world_tokens in visual_variants.items():
        pred_dino, _ = module.visual_decoder(
            world_tokens,
            visual_anchor,
            target["delta"],
        )
        _, per_sample = module._visual_loss(
            pred_dino,
            target["target_dino"],
            None,
            return_per_sample=True,
        )
        output[f"visual_conditioned_{name}"] = per_sample
    zero_visual_anchor = torch.zeros_like(visual_anchor)
    for name in ("full", "physical_only", "appearance_only", "neither"):
        pred_dino, _ = module.visual_decoder(
            visual_variants[name],
            zero_visual_anchor,
            target["delta"],
        )
        _, per_sample = module._visual_loss(
            pred_dino,
            target["target_dino"],
            None,
            return_per_sample=True,
        )
        output[f"visual_zero_anchor_{name}"] = per_sample

    normalized_confidence = module._normalize_teacher_confidence(
        teacher.confidence,
        teacher.target,
    )
    target_page = gather_time(teacher.features, target["future"])
    target_points = gather_time(teacher.target, target["future"])
    target_confidence = gather_time(
        normalized_confidence,
        target["future"],
    )
    current_page, current_points = module._extract_geometry_anchor(
        images,
        target["current"],
    )
    geometry_anchor = module.geometry_anchor_encoder(current_page, current_points)
    geometry_variants = {
        "physical": physical,
        "neither": zero_physical,
        "wrong_physical": wrong["video_world_physical"],
    }
    for name, physical_tokens in geometry_variants.items():
        pred_page, pred_points, pred_confidence = module.geometry_decoder(
            physical_tokens,
            geometry_anchor,
            target["delta"],
        )
        _, _, per_sample = module._geometry_loss(
            pred_page,
            pred_points,
            pred_confidence,
            target_page,
            target_points,
            target_confidence,
            None,
            return_per_sample=True,
        )
        output[f"geometry_conditioned_{name}"] = per_sample
    zero_geometry_anchor = torch.zeros_like(geometry_anchor)
    for name in ("physical", "neither"):
        pred_page, pred_points, pred_confidence = module.geometry_decoder(
            geometry_variants[name],
            zero_geometry_anchor,
            target["delta"],
        )
        _, _, per_sample = module._geometry_loss(
            pred_page,
            pred_points,
            pred_confidence,
            target_page,
            target_points,
            target_confidence,
            None,
            return_per_sample=True,
        )
        output[f"geometry_zero_anchor_{name}"] = per_sample
    return output


def summarize_component_ablation(
    values: dict[str, torch.Tensor],
) -> dict[str, Any]:
    means = {name: float(value.float().mean()) for name, value in values.items()}

    def delta(name: str, baseline: str) -> dict[str, float]:
        absolute = means[name] - means[baseline]
        return {
            "loss": means[name],
            "absolute_increase": absolute,
            "relative_increase": absolute / max(abs(means[baseline]), 1e-8),
            "worse_fraction": float(
                (values[name].float() > values[baseline].float()).float().mean()
            ),
        }

    visual_baseline = "visual_conditioned_full"
    geometry_baseline = "geometry_conditioned_physical"
    visual_zero_baseline = "visual_zero_anchor_full"
    geometry_zero_baseline = "geometry_zero_anchor_physical"
    return {
        "num_samples": int(values[visual_baseline].numel()),
        "visual_baseline": means[visual_baseline],
        "visual": {
            name.removeprefix("visual_conditioned_"): delta(name, visual_baseline)
            for name in values
            if name.startswith("visual_conditioned_") and name != visual_baseline
        },
        "visual_zero_anchor_baseline": means[visual_zero_baseline],
        "visual_zero_anchor": {
            name.removeprefix("visual_zero_anchor_"): delta(
                name,
                visual_zero_baseline,
            )
            for name in values
            if name.startswith("visual_zero_anchor_")
            and name != visual_zero_baseline
        },
        "geometry_baseline": means[geometry_baseline],
        "geometry": {
            name.removeprefix("geometry_conditioned_"): delta(
                name,
                geometry_baseline,
            )
            for name in values
            if name.startswith("geometry_conditioned_")
            and name != geometry_baseline
        },
        "geometry_zero_anchor_baseline": means[geometry_zero_baseline],
        "geometry_zero_anchor": {
            name.removeprefix("geometry_zero_anchor_"): delta(
                name,
                geometry_zero_baseline,
            )
            for name in values
            if name.startswith("geometry_zero_anchor_")
            and name != geometry_zero_baseline
        },
    }


def make_visualizations(
    output_dir: Path,
    descriptors: dict[str, torch.Tensor],
    indices: dict[str, torch.Tensor],
    codebook_size: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = ("video", "geometry", "fused")
    joined = torch.cat([descriptors[name] for name in names], dim=0).float()
    joined = joined - joined.mean(0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(joined, q=2, center=False)
    projection = (joined @ vectors[:, :2]).numpy()
    count = descriptors[names[0]].shape[0]
    fig, axis = plt.subplots(figsize=(7, 6))
    for offset, name, color in zip(range(3), names, ("#377eb8", "#e41a1c", "#4daf4a")):
        points = projection[offset * count : (offset + 1) * count]
        axis.scatter(points[:, 0], points[:, 1], s=14, alpha=0.65, label=name, color=color)
    axis.set_title("Unified physical transition tokens (PCA)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "unified_physical_pca.png", dpi=180)
    plt.close(fig)

    private_names = (
        "appearance_video",
        "appearance_semantic",
        "appearance_fused",
    )
    if all(name in descriptors for name in private_names):
        private_joined = torch.cat(
            [descriptors[name] for name in private_names], dim=0
        ).float()
        private_joined = private_joined - private_joined.mean(0, keepdim=True)
        _, _, vectors = torch.pca_lowrank(private_joined, q=2, center=False)
        projection = (private_joined @ vectors[:, :2]).numpy()
        fig, axis = plt.subplots(figsize=(7, 6))
        for offset, name, color in zip(
            range(3),
            private_names,
            ("#377eb8", "#984ea3", "#ff7f00"),
        ):
            points = projection[offset * count : (offset + 1) * count]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=14,
                alpha=0.65,
                label=name,
                color=color,
            )
        axis.set_title("Private visual transition tokens (PCA)")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "private_visual_pca.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    bins = np.arange(codebook_size + 1)
    for row, name in enumerate(names):
        for level in range(2):
            values = indices[name][..., level].reshape(-1).numpy()
            counts, _ = np.histogram(values, bins=bins)
            axes[row, level].plot(counts, linewidth=0.8)
            axes[row, level].set_title(f"{name}, quantizer {level}")
            axes[row, level].set_ylabel("count")
    axes[-1, 0].set_xlabel("code")
    axes[-1, 1].set_xlabel("code")
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_usage.png", dpi=180)
    plt.close(fig)

    video = F.normalize(descriptors["video"].float(), dim=-1)
    geometry = F.normalize(descriptors["geometry"].float(), dim=-1)
    similarity = (video @ geometry.t()).numpy()
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_title("Video-to-PAGE transition similarity")
    axis.set_xlabel("PAGE transition index")
    axis.set_ylabel("Video transition index")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "video_geometry_similarity.png", dpi=180)
    plt.close(fig)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Factorized Unified-Space Probe",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Clips: {report['num_clips']}",
        f"- Fixed spans: `{report['spans']}`",
        f"- Noise level: `{report['noise_level']}`",
        "",
        "## Shared physical correspondence",
        "",
        "| Pair | clip top-1 | chance | top-5 | paired cosine | hard-negative margin | CKA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, values in report["physical_retrieval"].items():
        lines.append(
            f"| {pair} | {values['top1_left_to_right']:.4f} | "
            f"{values['chance_top1']:.4f} | {values['top5_left_to_right']:.4f} | "
            f"{values['paired_cosine']:.4f} | "
            f"{values['paired_hard_negative_margin']:.4f} | "
            f"{report['physical_cka'][pair]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Private visual correspondence",
            "",
            "| Pair | clip top-1 | chance | top-5 | paired cosine | hard-negative margin | CKA |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair, values in report["private_retrieval"].items():
        lines.append(
            f"| {pair} | {values['top1_left_to_right']:.4f} | "
            f"{values['chance_top1']:.4f} | {values['top5_left_to_right']:.4f} | "
            f"{values['paired_cosine']:.4f} | "
            f"{values['paired_hard_negative_margin']:.4f} | "
            f"{report['private_cka'][pair]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Same-scene temporal-span controls",
            "",
            "| Pair | correct-span top-1 | chance | correct-vs-wrong margin |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair, values in {
        **report["physical_span_retrieval"],
        **report["private_span_retrieval"],
    }.items():
        lines.append(
            f"| {pair} | {values['top1']:.4f} | {values['chance_top1']:.4f} | "
            f"{values['paired_wrong_span_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Temporal and appearance controls",
            "",
            f"- Forward vs reversed physical cosine: `{report['controls']['forward_reverse_physical_cosine']:.4f}` "
            "(lower means direction is distinguished).",
        ]
    )
    photo = report["controls"].get("photometric")
    if photo:
        lines.extend(
            [
                f"- Physical cosine after photometric change: `{photo['physical_cosine']:.4f}`.",
                f"- Physical relative L2 change: `{photo['physical_relative_l2']:.4f}`.",
                f"- Physical code stability: `{photo['physical_code_stability']:.4f}`.",
                f"- Private appearance cosine after photometric change: `{photo['appearance_cosine']:.4f}`.",
                f"- Private appearance relative L2 change: `{photo['appearance_relative_l2']:.4f}`.",
                f"- Physical-minus-appearance invariance gap: `{photo['invariance_gap']:.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Shared/private specialization",
            "",
            "| Source | Camera R2 | Appearance R2 | DINO R2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, values in report["camera_probe"].items():
        lines.append(
            f"| {source} | {values['r2']:.4f} | "
            f"{report['appearance_probe'][source]['r2']:.4f} | "
            f"{report['semantic_probe'][source]['r2']:.4f} |"
        )
    lines.extend(["", "Cross-space CKA (lower indicates more distinct organization):"])
    for pair, value in report["cross_space_cka"].items():
        lines.append(f"- {pair}: `{value:.4f}`.")

    codebook_pairs = report["codebook"]["pairs"]
    lines.extend(
        [
            "",
            "## Shared-code paired controls",
            "",
            "| Pair / level | agreement | shuffled | delta | NMI | shuffled | delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair, levels in codebook_pairs.items():
        for level, values in levels.items():
            lines.append(
                f"| {pair}/{level} | {values['exact_token_agreement']:.4f} | "
                f"{values['shuffled_exact_token_agreement']:.4f} | "
                f"{values['paired_minus_shuffled_agreement']:.4f} | "
                f"{values['normalized_mutual_information']:.4f} | "
                f"{values['shuffled_normalized_mutual_information']:.4f} | "
                f"{values['paired_minus_shuffled_nmi']:.4f} |"
            )

    component = report.get("component_ablation")
    if component:
        lines.extend(
            [
                "",
                "## Frozen-decoder component intervention",
                "",
                f"The visual baseline loss is `{component['visual_baseline']:.4f}`.",
                "",
                "| Visual intervention | loss | relative increase | worse fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in component["visual"].items():
            lines.append(
                f"| {name} | {values['loss']:.4f} | "
                f"{values['relative_increase']:.4f} | "
                f"{values['worse_fraction']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"With the current-frame anchor removed, the full visual loss is `{component['visual_zero_anchor_baseline']:.4f}`.",
                "",
                "| Zero-anchor visual intervention | loss | relative increase | worse fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in component["visual_zero_anchor"].items():
            lines.append(
                f"| {name} | {values['loss']:.4f} | "
                f"{values['relative_increase']:.4f} | "
                f"{values['worse_fraction']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"The geometry baseline loss is `{component['geometry_baseline']:.4f}`.",
                "",
                "| Geometry intervention | loss | relative increase | worse fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in component["geometry"].items():
            lines.append(
                f"| {name} | {values['loss']:.4f} | "
                f"{values['relative_increase']:.4f} | "
                f"{values['worse_fraction']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"With the geometry anchor removed, the physical-token loss is `{component['geometry_zero_anchor_baseline']:.4f}`.",
                "",
                "| Zero-anchor geometry intervention | loss | relative increase | worse fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in component["geometry_zero_anchor"].items():
            lines.append(
                f"| {name} | {values['loss']:.4f} | "
                f"{values['relative_increase']:.4f} | "
                f"{values['worse_fraction']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Physical video/PAGE retrieval and paired-vs-shuffled code statistics test whether the shared space is genuinely cross-modal.",
            "- Camera versus appearance/semantic probes test specialization; no single metric alone establishes factorization.",
            "- Wrong-span and component interventions test whether each token group contributes transition-specific information beyond the current-frame anchor.",
            "- The photometric control tests differential sensitivity without changing image-plane layout.",
            "- PAGE is a frozen teacher, so these probes measure teacher-relative physical organization rather than sensor-ground-truth 3D accuracy.",
            "",
            "Figures: `unified_physical_pca.png`, `private_visual_pca.png`, `video_geometry_similarity.png`, and `codebook_usage.png`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.num_clips < 4:
        raise ValueError("--num-clips must be at least 4")
    if not 0.0 <= args.noise_level <= 1.0:
        raise ValueError("--noise-level must be in [0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg, checkpoint, config_path = load_probe_config(args)
    spans = parse_spans(args.spans, int(cfg.dataset.max_frames))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment = build_experiment(cfg, logger=None, ckpt_path=checkpoint)
    experiment.algo = experiment._build_algo()
    experiment._load_weights_only_checkpoint(checkpoint)
    algo = experiment.algo.to(device).eval()
    algo._set_diffusion_trainable(False)
    module = algo.vggt_alignment_loss.eval()
    if int(module.unit_arch_version_marker.item()) != 7:
        raise RuntimeError("The paper probe requires UPCAST architecture version 7")
    required_components = (
        "world_token_fusion",
        "semantic_pair_projector",
        "semantic_transition_encoder",
        "appearance_fusion_encoder",
        "visual_decoder",
        "geometry_decoder",
    )
    missing_components = [
        name for name in required_components if not hasattr(module, name)
    ]
    if missing_components:
        raise RuntimeError(
            "Factorization verification requires the complete UPCAST source "
            "checkpoint, not a physical-only continuation checkpoint. Missing: "
            + ", ".join(missing_components)
        )
    if module.dino_extractor is None:
        raise RuntimeError("Factorization verification requires the frozen DINO teacher")
    if getattr(experiment.data_module, "trainer", None) is None:
        experiment.data_module.trainer = SimpleNamespace(current_epoch=0)
    dataloader = experiment.data_module.val_dataloader()

    storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    index_storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    camera_storage: list[torch.Tensor] = []
    appearance_target_storage: list[torch.Tensor] = []
    semantic_target_storage: list[torch.Tensor] = []
    clip_id_storage: list[torch.Tensor] = []
    reverse_cosines: list[torch.Tensor] = []
    photo_physical_sensitivity: dict[str, list[torch.Tensor]] = defaultdict(list)
    photo_appearance_sensitivity: dict[str, list[torch.Tensor]] = defaultdict(list)
    photo_code_stability: list[torch.Tensor] = []
    component_storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    processed = 0
    autocast = device.type == "cuda"

    for batch_index, batch in enumerate(dataloader):
        if processed >= args.num_clips:
            break
        batch = move_to_device(batch, device)
        xs, conditions, masks, *_ = algo.on_after_batch_transfer(batch, 0)
        keep = min(xs.shape[0], args.num_clips - processed)
        xs = xs[:keep]
        conditions = conditions[:keep]
        noise_levels = torch.full(
            (keep, xs.shape[1]),
            args.noise_level,
            device=device,
            dtype=xs.dtype,
        )
        processed_conditions = algo._process_conditions(conditions)

        seed = args.seed + batch_index
        torch.manual_seed(seed)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
            _, latents, denoising_terms = algo.diffusion_model.forward_with_alignment_latents_only(
                xs,
                processed_conditions,
                k=noise_levels,
                return_denoising_terms=True,
                return_tail_state=False,
            )
            images = algo._unnormalize_x(xs).clamp(0.0, 1.0)
            teacher = module._extract_geometry_output(images)
            dino_features = torch.nan_to_num(
                module._extract_dino(images),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            encoded = encode_spans(
                module,
                latents,
                denoising_terms,
                teacher,
                dino_features,
                spans,
            )

        for (current, future), values in zip(spans, encoded):
            for branch in (
                "video",
                "geometry",
                "fused",
                "video_world_physical",
                "geometry_world_physical",
                "fused_world_physical",
                "appearance_video",
                "appearance_semantic",
                "appearance_fused",
            ):
                storage[branch].append(descriptor(values[branch]).cpu())
            for branch in ("video", "geometry", "fused"):
                index_storage[branch].append(values[f"{branch}_indices"].cpu())
            camera_storage.append(relative_camera_target(conditions, current, future).cpu())
            appearance_target_storage.append(
                appearance_statistics(
                    images,
                    values["current"],
                    values["future"],
                ).cpu()
            )
            semantic_target_storage.append(
                feature_map_descriptor(values["target_dino"]).cpu()
            )
            clip_id_storage.append(torch.arange(processed, processed + keep))

        reversed_spans = tuple((future, current) for current, future in spans)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
            reversed_video = encode_video_only_spans(
                module,
                latents,
                denoising_terms,
                reversed_spans,
            )
        for forward, backward in zip(encoded, reversed_video):
            reverse_cosines.append(
                cosine_rows(descriptor(forward["video"]), descriptor(backward["physical"])).cpu()
            )

        if not args.no_photometric_control:
            perturbed_images = deterministic_photometric_transform(images)
            perturbed_xs = algo._normalize_x(perturbed_images)
            torch.manual_seed(seed)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
                _, perturbed_latents, perturbed_terms = algo.diffusion_model.forward_with_alignment_latents_only(
                    perturbed_xs,
                    processed_conditions,
                    k=noise_levels,
                    return_denoising_terms=True,
                    return_tail_state=False,
                )
                clean_video = encode_video_only_spans(module, latents, denoising_terms, spans)
                changed_video = encode_video_only_spans(
                    module,
                    perturbed_latents,
                    perturbed_terms,
                    spans,
                )
            for clean, changed in zip(clean_video, changed_video):
                physical_sensitivity = representation_sensitivity(
                    clean["physical"],
                    changed["physical"],
                )
                appearance_sensitivity = representation_sensitivity(
                    clean["appearance"],
                    changed["appearance"],
                )
                for name, values in physical_sensitivity.items():
                    photo_physical_sensitivity[name].append(values.cpu())
                for name, values in appearance_sensitivity.items():
                    photo_appearance_sensitivity[name].append(values.cpu())
                photo_code_stability.append(
                    (clean["indices"] == changed["indices"]).float().mean(dim=(1, 2)).cpu()
                )

        if not args.skip_decoder_ablation:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast,
            ):
                component_values = decoder_component_ablation(
                    module,
                    encoded,
                    images,
                    teacher,
                )
            for name, values in component_values.items():
                component_storage[name].append(values.detach().cpu())

        processed += keep
        print(f"processed {processed}/{args.num_clips} clips", flush=True)
        del latents, denoising_terms, teacher, dino_features, encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    descriptors = {
        name: torch.cat(chunks, dim=0) for name, chunks in storage.items()
    }
    indices = {
        name: torch.cat(chunks, dim=0) for name, chunks in index_storage.items()
    }
    camera_targets = torch.cat(camera_storage, dim=0)
    appearance_targets = torch.cat(appearance_target_storage, dim=0)
    semantic_targets = torch.cat(semantic_target_storage, dim=0)
    clip_ids = torch.cat(clip_id_storage, dim=0)

    physical_pairs = (
        ("video", "geometry"),
        ("video", "fused"),
        ("geometry", "fused"),
    )
    private_pairs = (
        ("appearance_video", "appearance_semantic"),
        ("appearance_video", "appearance_fused"),
        ("appearance_semantic", "appearance_fused"),
    )
    physical_retrieval = {
        f"{left}_{right}": retrieval(descriptors[left], descriptors[right])
        for left, right in physical_pairs
    }
    private_retrieval = {
        f"{left}_{right}": retrieval(descriptors[left], descriptors[right])
        for left, right in private_pairs
    }
    physical_span_report = {
        f"{left}_{right}": grouped_span_retrieval(
            descriptors[left], descriptors[right], clip_ids
        )
        for left, right in physical_pairs
    }
    private_span_report = {
        f"{left}_{right}": grouped_span_retrieval(
            descriptors[left], descriptors[right], clip_ids
        )
        for left, right in private_pairs
    }
    physical_cka = {
        f"{left}_{right}": linear_cka(descriptors[left], descriptors[right])
        for left, right in physical_pairs
    }
    private_cka = {
        f"{left}_{right}": linear_cka(descriptors[left], descriptors[right])
        for left, right in private_pairs
    }
    cross_space_cka = {
        "video_physical_vs_private": linear_cka(
            descriptors["video_world_physical"],
            descriptors["appearance_video"],
        ),
        "teacher_physical_vs_private": linear_cka(
            descriptors["geometry_world_physical"],
            descriptors["appearance_semantic"],
        ),
        "fused_physical_vs_private": linear_cka(
            descriptors["fused_world_physical"],
            descriptors["appearance_fused"],
        ),
    }
    controls: dict[str, Any] = {
        "forward_reverse_physical_cosine": float(torch.cat(reverse_cosines).mean())
    }
    if photo_physical_sensitivity:
        physical_cosine = float(
            torch.cat(photo_physical_sensitivity["cosine"]).mean()
        )
        appearance_cosine = float(
            torch.cat(photo_appearance_sensitivity["cosine"]).mean()
        )
        controls["photometric"] = {
            "physical_cosine": physical_cosine,
            "physical_relative_l2": float(
                torch.cat(photo_physical_sensitivity["relative_l2"]).mean()
            ),
            "appearance_cosine": appearance_cosine,
            "appearance_relative_l2": float(
                torch.cat(photo_appearance_sensitivity["relative_l2"]).mean()
            ),
            "physical_code_stability": float(torch.cat(photo_code_stability).mean()),
            "invariance_gap": physical_cosine - appearance_cosine,
        }

    probe_features = {
        "video_physical": F.normalize(
            descriptors["video_world_physical"].float(), dim=-1
        ),
        "video_private": F.normalize(
            descriptors["appearance_video"].float(), dim=-1
        ),
        "teacher_physical": F.normalize(
            descriptors["geometry_world_physical"].float(), dim=-1
        ),
        "teacher_private": F.normalize(
            descriptors["appearance_semantic"].float(), dim=-1
        ),
        "fused_physical": F.normalize(
            descriptors["fused_world_physical"].float(), dim=-1
        ),
        "fused_private": F.normalize(
            descriptors["appearance_fused"].float(), dim=-1
        ),
    }
    camera_probe = {
        name: ridge_probe(features, camera_targets, clip_ids, args.ridge)
        for name, features in probe_features.items()
    }
    appearance_probe = {
        name: ridge_probe(features, appearance_targets, clip_ids, args.ridge)
        for name, features in probe_features.items()
    }
    semantic_probe = {
        name: ridge_probe(features, semantic_targets, clip_ids, args.ridge)
        for name, features in probe_features.items()
    }
    report = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "unit_arch_version": 7,
        "num_clips": processed,
        "num_transitions": int(descriptors["video"].shape[0]),
        "spans": [list(span) for span in spans],
        "noise_level": args.noise_level,
        "physical_retrieval": physical_retrieval,
        "private_retrieval": private_retrieval,
        "physical_span_retrieval": physical_span_report,
        "private_span_retrieval": private_span_report,
        "physical_cka": physical_cka,
        "private_cka": private_cka,
        "cross_space_cka": cross_space_cka,
        "codebook": code_metrics(indices, module.unit_quantizer.codebook_size),
        "controls": controls,
        "camera_probe": camera_probe,
        "appearance_probe": appearance_probe,
        "semantic_probe": semantic_probe,
    }
    if component_storage:
        report["component_ablation"] = summarize_component_ablation(
            {
                name: torch.cat(chunks, dim=0)
                for name, chunks in component_storage.items()
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_markdown(report))
    make_visualizations(output_dir, descriptors, indices, module.unit_quantizer.codebook_size)
    print(render_markdown(report), flush=True)


if __name__ == "__main__":
    main()
