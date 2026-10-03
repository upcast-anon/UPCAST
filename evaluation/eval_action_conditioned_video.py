#!/usr/bin/env python3
"""Evaluate action-conditioned Minecraft video generation.

The protocol uses held-out trajectories and identical sample indices/noise for
all compared runs. An inverse-dynamics probe is trained only on ground-truth
training trajectories and must pass a held-out balanced-accuracy gate before
its generated-video action scores are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common.metrics.video.shared_registry import SharedVideoMetricModelRegistry
from algorithms.common.metrics.video.video_metric import VideoMetric


MANEUVER_NAMES = ("forward", "left", "right")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH")
    return name, Path(path).expanduser().resolve()


def parse_named_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=VALUE")
    name, result = value.split("=", 1)
    return name.strip(), result.strip()


def parse_causal_pair(value: str) -> tuple[str, tuple[str, str]]:
    name, runs = parse_named_value(value)
    parts = [part.strip() for part in runs.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("Expected NAME=CORRECT_RUN,INTERVENTION_RUN")
    return name, (parts[0], parts[1])


def decode_video(path: Path) -> torch.Tensor:
    frames: list[np.ndarray] = []
    with av.open(str(path), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def maneuver_from_actions(actions: Iterable[int]) -> int:
    values = set(int(value) for value in actions)
    if 2 in values and 3 not in values:
        return 1
    if 3 in values and 2 not in values:
        return 2
    return 0


def probe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(target.tolist(), prediction.tolist()):
        confusion[int(truth), int(pred)] += 1
    recall = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(3, dtype=np.float64),
        where=confusion.sum(axis=1) > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(3, dtype=np.float64),
        where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    result = {
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "num_transitions": int(confusion.sum()),
    }
    for index, name in enumerate(MANEUVER_NAMES):
        result[f"recall_{name}"] = float(recall[index])
        result[f"support_{name}"] = int(confusion[index].sum())
    return result


class GroundTruthTransitionDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.video_paths = sorted((root / split).glob("**/*.mp4"), key=str)
        if not self.video_paths:
            raise FileNotFoundError(f"No {split} videos under {root}")
        self.videos = [decode_video(path) for path in self.video_paths]
        self.actions = [
            np.load(path.with_suffix(".npz"))["actions"].astype(np.int64)
            for path in self.video_paths
        ]
        self.samples: list[tuple[int, int, int]] = []
        for video_index, (video, actions) in enumerate(zip(self.videos, self.actions)):
            # The official loader exposes 300 frames even though these MP4/NPZ
            # files contain a final codec boundary sample. Use every stride-2
            # transition at both start parities to cover all evaluation clips.
            end = min(len(video), len(actions), 300)
            for future in range(2, end):
                label = maneuver_from_actions(actions[future - 1 : future + 1])
                self.samples.append((video_index, future, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        video_index, future, label = self.samples[index]
        video = self.videos[video_index]
        return video[future - 2], video[future], label


class InverseDynamicsProbe(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            9,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3].copy_(0.5 * old_conv.weight)
            self.backbone.conv1.weight[:, 3:6].copy_(0.5 * old_conv.weight)
            self.backbone.conv1.weight[:, 6:].zero_()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        current = F.interpolate(current.float(), size=(128, 128), mode="bilinear", align_corners=False)
        future = F.interpolate(future.float(), size=(128, 128), mode="bilinear", align_corners=False)
        if current.max() > 1.5:
            current = current / 255.0
            future = future / 255.0
        mean = self.mean.to(dtype=current.dtype)
        std = self.std.to(dtype=current.dtype)
        current = (current - mean) / std
        future = (future - mean) / std
        difference = future - current
        return self.backbone(torch.cat((current, future, difference), dim=1))


@torch.no_grad()
def evaluate_probe(
    model: InverseDynamicsProbe,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    for current, future, target in loader:
        current = current.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(current, future)
        losses.append(float(F.cross_entropy(logits, target).item()))
        targets.append(target.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    result = probe_metrics(np.concatenate(targets), np.concatenate(predictions))
    result["cross_entropy"] = float(np.mean(losses))
    return result


def train_or_load_probe(args: argparse.Namespace, device: torch.device) -> tuple[InverseDynamicsProbe, dict[str, float]]:
    checkpoint = args.probe_checkpoint
    validation_data = GroundTruthTransitionDataset(args.dataset_root, "validation")
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if checkpoint.is_file() and not args.force_train_probe:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = InverseDynamicsProbe(pretrained=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        validation = evaluate_probe(model, validation_loader, device)
        return model, validation

    training_data = GroundTruthTransitionDataset(args.dataset_root, "training")
    labels = np.asarray([sample[2] for sample in training_data.samples])
    counts = np.bincount(labels, minlength=3)
    class_weights = counts.sum() / np.maximum(3 * counts, 1)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_data,
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = InverseDynamicsProbe(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.probe_epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        for current, future, target in training_loader:
            current = current.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(current, future), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            batches += 1
        scheduler.step()
        validation = evaluate_probe(model, validation_loader, device)
        validation["epoch"] = epoch + 1
        validation["train_loss"] = train_loss / max(batches, 1)
        history.append(validation)
        if validation["balanced_accuracy"] > best_score:
            best_score = validation["balanced_accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"probe epoch {epoch + 1:02d}: train={validation['train_loss']:.4f}, "
            f"val_bal_acc={validation['balanced_accuracy']:.4f}, val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("Inverse-dynamics probe did not complete an epoch")
    model.load_state_dict(best_state, strict=True)
    validation = evaluate_probe(model, validation_loader, device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "validation": validation,
            "history": history,
            "class_counts": counts.tolist(),
            "protocol": "stride-2 aggregate maneuver: forward/left/right",
        },
        checkpoint,
    )
    return model, validation


@dataclass(frozen=True)
class ClipManifest:
    sample_index: int
    trajectory_index: int
    trajectory: str
    start_frame: int
    correct_labels: tuple[int, ...]


def build_manifests(dataset_root: Path, num_samples: int, output_frames: int = 16, frame_skip: int = 2) -> tuple[list[ClipManifest], list[Path], list[np.ndarray]]:
    paths = sorted((dataset_root / "validation").glob("**/*.mp4"), key=str)
    actions = [np.load(path.with_suffix(".npz"))["actions"].astype(np.int64) for path in paths]
    source_frames = 1 + (output_frames - 1) * frame_skip
    clip_counts = [max(min(len(action), 300) - source_frames + 1, 1) for action in actions]
    cumulative = np.cumsum(clip_counts).tolist()
    remap = list(range(cumulative[-1]))
    random.seed(0)
    random.shuffle(remap)
    manifests: list[ClipManifest] = []
    for sample_index, absolute_index in enumerate(remap[:num_samples]):
        trajectory_index = int(np.searchsorted(cumulative, absolute_index, side="right"))
        previous = 0 if trajectory_index == 0 else cumulative[trajectory_index - 1]
        start = absolute_index - previous
        sequence = actions[trajectory_index]
        labels = tuple(
            maneuver_from_actions(sequence[start + frame_skip * step - frame_skip + 1 : start + frame_skip * step + 1])
            for step in range(1, output_frames)
        )
        manifests.append(
            ClipManifest(
                sample_index=sample_index,
                trajectory_index=trajectory_index,
                trajectory=str(paths[trajectory_index].relative_to(dataset_root)),
                start_frame=int(start),
                correct_labels=labels,
            )
        )
    return manifests, paths, actions


def conditioned_labels(
    manifest: ClipManifest,
    mode: str,
    actions: list[np.ndarray],
    donor_offset: int,
    temporal_shift: int,
    frame_skip: int = 2,
) -> np.ndarray:
    sequence = actions[manifest.trajectory_index]
    if mode == "donor":
        sequence = actions[(manifest.trajectory_index + donor_offset) % len(actions)]
    elif mode == "temporal_shift":
        sequence = np.roll(sequence, temporal_shift)
    elif mode == "swap_turns":
        sequence = sequence.copy()
        left = sequence == 2
        right = sequence == 3
        sequence[left] = 3
        sequence[right] = 2
    elif mode != "correct":
        raise ValueError(f"Unsupported action mode: {mode}")
    start = manifest.start_frame
    return np.asarray(
        [
            maneuver_from_actions(sequence[start + frame_skip * step - frame_skip + 1 : start + frame_skip * step + 1])
            for step in range(1, 16)
        ],
        dtype=np.int64,
    )


def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(
        raw_dir.glob("*/data.npz"),
        key=lambda path: int(path.parent.name),
    )
    if not files:
        raise FileNotFoundError(f"No raw data.npz files under {raw_dir}")
    expected = list(range(len(files)))
    actual = [int(path.parent.name) for path in files]
    if actual != expected:
        raise RuntimeError(f"Raw sample indices must be contiguous: {actual[:8]} ...")
    return files


def load_pair(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as payload:
        prediction = torch.from_numpy(payload["gen"])
        target = torch.from_numpy(payload["gt"])
    if prediction.ndim != 4 or prediction.shape[1] not in {1, 3}:
        raise RuntimeError(f"Unexpected generated-video shape in {path}: {prediction.shape}")
    return prediction, target


@torch.no_grad()
def evaluate_generated_actions(
    model: InverseDynamicsProbe,
    files: list[Path],
    labels: list[np.ndarray],
    correct_labels: list[np.ndarray],
    device: torch.device,
    batch_videos: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    intended: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for offset in range(0, len(files), batch_videos):
        videos = torch.stack([load_pair(path)[0] for path in files[offset : offset + batch_videos]]).to(device)
        current = videos[:, :-1].flatten(0, 1)
        future = videos[:, 1:].flatten(0, 1)
        logits = model(current, future)
        prob = logits.softmax(dim=1).cpu().numpy()
        probabilities.append(prob)
        predictions.append(prob.argmax(axis=1))
        intended.append(np.concatenate(labels[offset : offset + batch_videos]))
        correct.append(np.concatenate(correct_labels[offset : offset + batch_videos]))
    probability = np.concatenate(probabilities)
    prediction = np.concatenate(predictions)
    intended_array = np.concatenate(intended)
    correct_array = np.concatenate(correct)
    result = probe_metrics(intended_array, prediction)
    result["nll_to_condition"] = float(
        -np.log(np.clip(probability[np.arange(len(probability)), intended_array], 1e-8, 1.0)).mean()
    )
    correct_result = probe_metrics(correct_array, prediction)
    result.update({f"to_recorded_{key}": value for key, value in correct_result.items()})
    return result, {
        "probability": probability,
        "prediction": prediction,
        "intended": intended_array,
        "correct": correct_array,
    }


@torch.no_grad()
def evaluate_quality(
    files: list[Path],
    metric_names: list[str],
    device: torch.device,
    batch_videos: int,
) -> dict[str, float]:
    registry = SharedVideoMetricModelRegistry()
    metric = VideoMetric(registry, metric_names, split_batch_size=batch_videos).to(device).eval()
    context_mask = torch.zeros(16, dtype=torch.bool, device=device)
    context_mask[0] = True
    for offset in range(0, len(files), batch_videos):
        pairs = [load_pair(path) for path in files[offset : offset + batch_videos]]
        predictions = torch.stack([pair[0] for pair in pairs]).to(device).float() / 255.0
        targets = torch.stack([pair[1] for pair in pairs]).to(device).float() / 255.0
        metric(predictions, targets, context_mask=context_mask)
    output: dict[str, float] = {}
    for metric_type, module in metric.items():
        value = module.compute()
        if isinstance(value, tuple):
            value = value[0]
        output[metric_type.value] = float(torch.as_tensor(value).detach().cpu().item())
    return output


def paired_pixel_difference(first: list[Path], second: list[Path]) -> float:
    if len(first) != len(second):
        raise ValueError("Causal runs have different sample counts")
    values = []
    for first_path, second_path in zip(first, second):
        first_video = load_pair(first_path)[0][1:].float() / 255.0
        second_video = load_pair(second_path)[0][1:].float() / 255.0
        values.append(float((first_video - second_video).abs().mean().item()))
    return float(np.mean(values))


def causal_metrics(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    first_files: list[Path],
    second_files: list[Path],
) -> dict[str, float]:
    changed_intent = first["intended"] != second["intended"]
    count = int(changed_intent.sum())
    result = {
        "changed_intent_transitions": count,
        "paired_generated_l1": paired_pixel_difference(first_files, second_files),
    }
    if count == 0:
        result.update(
            {
                "prediction_response_rate": 0.0,
                "first_intent_accuracy_on_changed": 0.0,
                "second_intent_accuracy_on_changed": 0.0,
                "dual_compliance_rate": 0.0,
            }
        )
        return result
    first_match = first["prediction"][changed_intent] == first["intended"][changed_intent]
    second_match = second["prediction"][changed_intent] == second["intended"][changed_intent]
    result.update(
        {
            "prediction_response_rate": float(
                np.mean(first["prediction"][changed_intent] != second["prediction"][changed_intent])
            ),
            "first_intent_accuracy_on_changed": float(first_match.mean()),
            "second_intent_accuracy_on_changed": float(second_match.mean()),
            "dual_compliance_rate": float(np.mean(first_match & second_match)),
        }
    )
    return result


def write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "action_conditioned_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    rows = []
    for name, result in report["runs"].items():
        quality = result["quality"]
        action = result["action"]
        rows.append(
            {
                "run": name,
                "action_mode": result["action_mode"],
                **quality,
                "action_balanced_accuracy": action["balanced_accuracy"],
                "action_macro_f1": action["macro_f1"],
                "action_nll": action["nll_to_condition"],
            }
        )
    columns = list(rows[0])
    with (output_dir / "action_conditioned_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metric_order = ["fvd", "fid", "lpips", "ssim", "psnr", "mse"]
    lines = [
        "# Action-Conditioned Minecraft Evaluation",
        "",
        f"Held-out inverse-dynamics probe balanced accuracy: {report['probe_validation']['balanced_accuracy']:.4f}",
        "",
        "| Run | Condition | "
        + " | ".join(name.upper() for name in metric_order)
        + " | Action balanced acc. | Action macro-F1 |",
        "|---|---|" + "---:|" * (len(metric_order) + 2),
    ]
    for row in rows:
        values = [f"{row.get(metric, float('nan')):.4f}" for metric in metric_order]
        lines.append(
            f"| {row['run']} | {row['action_mode']} | "
            + " | ".join(values)
            + f" | {row['action_balanced_accuracy']:.4f} | {row['action_macro_f1']:.4f} |"
        )
    if report["causal_pairs"]:
        lines.extend(["", "## Causal Action Interventions", ""])
        for name, result in report["causal_pairs"].items():
            lines.append(
                f"- **{name}:** changed intents={result['changed_intent_transitions']}, "
                f"prediction response={result['prediction_response_rate']:.4f}, "
                f"paired generated L1={result['paired_generated_l1']:.6f}."
            )
    lines.extend(
        [
            "",
            "This is the official DFoT quickstart mini split (10 train and 10 validation trajectories). "
            "It is a paired pilot protocol, not the full 210 GB Minecraft benchmark.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_qualitative_grids(
    files: dict[str, list[Path]],
    output_dir: Path,
    max_samples: int = 8,
) -> None:
    frame_indices = (0, 5, 10, 15)
    row_names = ["ground truth", *files.keys()]
    label_width = 180
    for sample_index in range(min(max_samples, min(map(len, files.values())))):
        first_target = load_pair(next(iter(files.values()))[sample_index])[1]
        rows = [first_target]
        for name in files:
            rows.append(load_pair(files[name][sample_index])[0])
        height, width = int(rows[0].shape[-2]), int(rows[0].shape[-1])
        canvas = Image.new(
            "RGB",
            (label_width + len(frame_indices) * width, len(rows) * height),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, (name, video) in enumerate(zip(row_names, rows)):
            y = row_index * height
            draw.text((8, y + 8), name, fill=(0, 0, 0))
            for column, frame_index in enumerate(frame_indices):
                frame = video[frame_index].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                canvas.paste(Image.fromarray(frame), (label_width + column * width, y))
        canvas.save(output_dir / f"qualitative_{sample_index:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_named_path)
    parser.add_argument("--run-mode", action="append", default=[], type=parse_named_value)
    parser.add_argument("--causal-pair", action="append", default=[], type=parse_causal_pair)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./action_eval_assets/data/minecraft"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./action_eval_assets/evaluations/action_conditioned_minecraft_16f/report"),
    )
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--force-train-probe", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-lr", type=float, default=3e-4)
    parser.add_argument("--min-probe-balanced-accuracy", type=float, default=0.45)
    parser.add_argument("--metric-batch-size", type=int, default=2)
    parser.add_argument("--metrics", default="fvd,fid,lpips,mse,ssim,psnr")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--donor-offset", type=int, default=1)
    parser.add_argument("--temporal-shift", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.probe_checkpoint is None:
        args.probe_checkpoint = args.output_dir.parent / "inverse_dynamics_probe.pt"
    else:
        args.probe_checkpoint = args.probe_checkpoint.expanduser().resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    runs = dict(args.run)
    run_modes = dict(args.run_mode)
    unknown_modes = set(run_modes) - set(runs)
    if unknown_modes:
        raise ValueError(f"Modes supplied for unknown runs: {sorted(unknown_modes)}")
    files = {name: raw_files(path) for name, path in runs.items()}
    sample_counts = {name: len(value) for name, value in files.items()}
    if len(set(sample_counts.values())) != 1:
        raise RuntimeError(f"Runs have different sample counts: {sample_counts}")
    num_samples = next(iter(sample_counts.values()))
    manifests, _, action_sequences = build_manifests(args.dataset_root, num_samples)
    correct_labels = [np.asarray(manifest.correct_labels, dtype=np.int64) for manifest in manifests]

    model, probe_validation = train_or_load_probe(args, device)
    if probe_validation["balanced_accuracy"] < args.min_probe_balanced_accuracy:
        raise RuntimeError(
            "Held-out inverse-dynamics probe is below the reliability gate: "
            f"{probe_validation['balanced_accuracy']:.4f} < {args.min_probe_balanced_accuracy:.4f}"
        )
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    report: dict = {
        "protocol": {
            "dataset": "official DFoT Minecraft quickstart mini split",
            "num_samples": num_samples,
            "frames": 16,
            "context_frames": 1,
            "frame_skip": 2,
            "seed": args.seed,
            "metrics": metric_names,
        },
        "probe_validation": probe_validation,
        "runs": {},
        "causal_pairs": {},
    }
    action_details: dict[str, dict[str, np.ndarray]] = {}
    for name in runs:
        mode = run_modes.get(name, "correct")
        labels = [
            conditioned_labels(
                manifest,
                mode,
                action_sequences,
                args.donor_offset,
                args.temporal_shift,
            )
            for manifest in manifests
        ]
        print(f"Evaluating {name}: {len(files[name])} clips, action mode={mode}", flush=True)
        action_result, details = evaluate_generated_actions(
            model,
            files[name],
            labels,
            correct_labels,
            device,
            args.metric_batch_size,
        )
        quality_result = evaluate_quality(
            files[name], metric_names, device, args.metric_batch_size
        )
        action_details[name] = details
        report["runs"][name] = {
            "path": str(runs[name]),
            "action_mode": mode,
            "quality": quality_result,
            "action": action_result,
        }
    for pair_name, (first_name, second_name) in args.causal_pair:
        if first_name not in runs or second_name not in runs:
            raise ValueError(f"Unknown run in causal pair {pair_name}: {first_name}, {second_name}")
        report["causal_pairs"][pair_name] = causal_metrics(
            action_details[first_name],
            action_details[second_name],
            files[first_name],
            files[second_name],
        )
    write_report(report, args.output_dir)
    write_qualitative_grids(files, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
