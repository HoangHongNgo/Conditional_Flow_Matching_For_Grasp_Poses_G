#!/usr/bin/env python3
"""Survey grasp configurations around economic_graspable seed points."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# utils.arguments parses sys.argv at import time. Keep this script's CLI private.
_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from dataset.graspnet_dataset import GraspNetDataset, collate_fn  # noqa: E402
from flow.models.grasp_cfm import economic_graspable  # noqa: E402
from libs.knn.knn_modules import knn  # noqa: E402
from utils.arguments import cfgs  # noqa: E402
from utils.loss_utils import generate_grasp_views, transform_point_cloud  # noqa: E402
sys.argv = _ORIGINAL_ARGV


@dataclass
class SeedSurveyRow:
    """Store count and score statistics for one seed point."""

    scene_name: str
    frame_id: int
    seed_index: int
    count_positive: int
    count_score_ge_03: int
    count_score_ge_05: int
    count_score_ge_07: int
    count_score_ge_09: int
    count_selected: int
    selected_unique_rotation_count: int
    selected_dominant_rotation_class: int | None
    selected_dominant_rotation_fraction: float | None
    selected_rotation_entropy: float | None
    selected_rotation_entropy_normalized: float | None
    score_mean: float | None
    score_std: float | None
    score_min: float | None
    score_p25: float | None
    score_p50: float | None
    score_p75: float | None
    score_max: float | None


@dataclass
class SceneSurveyRow:
    """Store aggregate neighborhood statistics for one scene/frame."""

    scene_name: str
    frame_id: int
    seed_count: int
    seed_with_positive_count: int
    mean_positive_count_per_seed: float
    median_positive_count_per_seed: float
    max_positive_count_per_seed: int
    total_positive_configs: int
    total_selected_configs: int
    selected_seed_count: int
    selected_mean_unique_rotation_count: float
    selected_mean_dominant_rotation_fraction: float | None
    selected_mean_rotation_entropy_normalized: float | None
    score_mean: float | None
    score_p50: float | None
    score_p75: float | None
    score_p90: float | None
    score_max: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the seed neighborhood survey."""
    parser = argparse.ArgumentParser(
        description=(
            "Survey GT grasp configurations within a radius around seed points "
            "selected by frozen economic_graspable."
        )
    )
    parser.add_argument("--dataset_root", default=cfgs.dataset_root)
    parser.add_argument("--checkpoint", default="checkpoints/economicgrasp_realsense.tar")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "eval", "test", "test_seen", "test_similar", "test_novel"],
    )
    parser.add_argument("--num_scenes", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.005)
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_points", type=int, default=cfgs.num_point)
    parser.add_argument("--voxel_size", type=float, default=cfgs.voxel_size)
    parser.add_argument(
        "--output_dir",
        default="flow/analysis/results/seed_neighborhoods",
        help="Directory for JSON and CSV outputs.",
    )
    return parser.parse_args()


def move_batch_to_device(batch_data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a GraspNet batch produced by collate_fn to the requested device."""
    for key, value in batch_data.items():
        if "list" in key:
            for batch_idx in range(len(value)):
                for obj_idx in range(len(value[batch_idx])):
                    value[batch_idx][obj_idx] = value[batch_idx][obj_idx].to(device)
        elif isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
    return batch_data


def load_frozen_graspable(checkpoint_path: str, device: torch.device) -> economic_graspable:
    """Load the frozen economic_graspable module from an EconomicGrasp checkpoint."""
    if not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("CUDA is required because economic_graspable currently calls .cuda() internally.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = economic_graspable(seed_feat_dim=512, is_training=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    mapped_state = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.startswith("backbone.") or clean_key.startswith("graspable."):
            mapped_state[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(mapped_state, strict=False)
    print(f"Loaded {len(mapped_state)} checkpoint tensors into economic_graspable.")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def merge_scene_grasp_labels(
    end_points: dict[str, Any],
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge object-level grasp labels into scene coordinates for one batch item.

    Returns:
        tuple:
            - grasp_points: Tensor with shape [M, 3].
            - grasp_scores: Tensor with shape [M, V].
            - grasp_rotations: Tensor with shape [M, V].
            - top_view_index: Tensor with shape [M, V].
    """
    device = end_points["point_clouds"].device
    poses = end_points["object_poses_list"][batch_index]

    grasp_points_merged = []
    grasp_scores_merged = []
    grasp_rotations_merged = []
    top_view_index_merged = []

    for obj_idx, pose in enumerate(poses):
        grasp_points = end_points["grasp_points_list"][batch_index][obj_idx]  # [P, 3]
        grasp_scores = end_points["grasp_scores_list"][batch_index][obj_idx]  # [P, V]
        grasp_rotations = end_points["grasp_rotations_list"][batch_index][obj_idx]  # [P, V]
        top_view_index = end_points["top_view_index_list"][batch_index][obj_idx]  # [P, V]
        num_grasp_points = grasp_points.size(0)

        grasp_views = generate_grasp_views(cfgs.num_view).to(device)  # [300, 3]
        grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")  # [P, 3]
        grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], "3x3")  # [300, 3]

        # Align transformed views back to canonical template indices.
        grasp_views_query = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, 300]
        grasp_views_transposed = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, 300]
        view_inds = knn(grasp_views_transposed, grasp_views_query, k=1).squeeze() - 1  # [300]

        # Invalid slots remain -1 when multiple transformed views map to the same template view.
        top_view_index_trans = -1 * torch.ones(
            (num_grasp_points, top_view_index.shape[1]),
            dtype=torch.long,
            device=device,
        )  # [P, V]
        point_idx, view_idx, transformed_idx = torch.where(view_inds == top_view_index.unsqueeze(-1))
        top_view_index_trans[point_idx, view_idx] = transformed_idx

        grasp_points_merged.append(grasp_points_trans)
        grasp_scores_merged.append(grasp_scores)
        grasp_rotations_merged.append(grasp_rotations)
        top_view_index_merged.append(top_view_index_trans)

    if not grasp_points_merged:
        empty_points = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_scores = torch.empty((0, cfgs.num_view), dtype=torch.float32, device=device)
        empty_rotations = torch.empty((0, cfgs.num_view), dtype=torch.long, device=device)
        empty_indices = torch.empty((0, cfgs.num_view), dtype=torch.long, device=device)
        return empty_points, empty_scores, empty_rotations, empty_indices

    return (
        torch.cat(grasp_points_merged, dim=0),  # [M, 3]
        torch.cat(grasp_scores_merged, dim=0),  # [M, V]
        torch.cat(grasp_rotations_merged, dim=0),  # [M, V]
        torch.cat(top_view_index_merged, dim=0),  # [M, V]
    )


def summarize_scores(scores: torch.Tensor) -> dict[str, float | None]:
    """Compute scalar summary statistics for a one-dimensional score tensor."""
    if scores.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    scores_cpu = scores.detach().float().cpu()
    return {
        "mean": float(scores_cpu.mean().item()),
        "std": float(scores_cpu.std(unbiased=False).item()),
        "min": float(scores_cpu.min().item()),
        "p25": float(torch.quantile(scores_cpu, 0.25).item()),
        "p50": float(torch.quantile(scores_cpu, 0.50).item()),
        "p75": float(torch.quantile(scores_cpu, 0.75).item()),
        "p90": float(torch.quantile(scores_cpu, 0.90).item()),
        "max": float(scores_cpu.max().item()),
    }


def summarize_rotation_classes(rotation_classes: torch.Tensor) -> dict[str, float | int | list[int] | None]:
    """Summarize concentration of discrete grasp rotation classes."""
    if rotation_classes.numel() == 0:
        return {
            "histogram": [0 for _ in range(cfgs.num_angle)],
            "unique_count": 0,
            "dominant_class": None,
            "dominant_fraction": None,
            "entropy": None,
            "entropy_normalized": None,
        }

    hist = torch.bincount(rotation_classes.detach().long().cpu(), minlength=cfgs.num_angle)[:cfgs.num_angle]
    probs = hist.float() / hist.sum().float()
    nonzero_probs = probs[probs > 0]
    entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs)).item()
    entropy_normalized = entropy / float(np.log(cfgs.num_angle))
    dominant_count, dominant_class = torch.max(hist, dim=0)
    return {
        "histogram": [int(value) for value in hist.tolist()],
        "unique_count": int((hist > 0).sum().item()),
        "dominant_class": int(dominant_class.item()),
        "dominant_fraction": float(dominant_count.item() / hist.sum().item()),
        "entropy": float(entropy),
        "entropy_normalized": float(entropy_normalized),
    }


def survey_scene(
    end_points: dict[str, Any],
    scene_name: str,
    frame_id: int,
    radius: float,
    score_threshold: float,
) -> tuple[SceneSurveyRow, list[SeedSurveyRow], torch.Tensor]:
    """Survey all selected seed points in one scene/frame."""
    seed_xyz = end_points["xyz_graspable"][0]  # [1024, 3]
    grasp_points, grasp_scores, grasp_rotations, top_view_index = merge_scene_grasp_labels(end_points, 0)

    if grasp_points.numel() == 0:
        empty_counts = torch.zeros(seed_xyz.shape[0], dtype=torch.long)
        scene_row = SceneSurveyRow(
            scene_name, frame_id, seed_xyz.shape[0], 0, 0.0, 0.0, 0, 0, 0, 0, 0.0, None, None, None, None, None, None, None
        )
        return scene_row, [], empty_counts

    distances = torch.cdist(seed_xyz.unsqueeze(0), grasp_points.unsqueeze(0)).squeeze(0)  # [1024, M]
    seed_rows = []
    all_positive_scores = []
    all_selected_rotations = []
    positive_counts = []
    selected_counts = []
    selected_unique_rotation_counts = []
    selected_dominant_fractions = []
    selected_entropy_normalized = []

    for seed_index in range(seed_xyz.shape[0]):
        in_radius = distances[seed_index] <= radius  # [M]
        if in_radius.any():
            nearby_scores = grasp_scores[in_radius]  # [K, V]
            nearby_rotations = grasp_rotations[in_radius]  # [K, V]
            nearby_top_view_index = top_view_index[in_radius]  # [K, V]
            valid_config_mask = (nearby_scores > 0.0) & (nearby_top_view_index >= 0)  # [K, V]
            selected_config_mask = (nearby_scores > score_threshold) & (nearby_top_view_index >= 0)  # [K, V]
            positive_scores = nearby_scores[valid_config_mask]  # [C]
            selected_rotations = nearby_rotations[selected_config_mask]  # [S]
        else:
            positive_scores = grasp_scores.new_empty((0,))
            selected_rotations = grasp_rotations.new_empty((0,))

        stats = summarize_scores(positive_scores)
        rotation_stats = summarize_rotation_classes(selected_rotations)
        positive_count = int(positive_scores.numel())
        selected_count = int(selected_rotations.numel())
        positive_counts.append(positive_count)
        selected_counts.append(selected_count)
        if positive_count > 0:
            all_positive_scores.append(positive_scores.detach())
        if selected_count > 0:
            all_selected_rotations.append(selected_rotations.detach())
            selected_unique_rotation_counts.append(int(rotation_stats["unique_count"]))
            selected_dominant_fractions.append(float(rotation_stats["dominant_fraction"]))
            selected_entropy_normalized.append(float(rotation_stats["entropy_normalized"]))

        seed_rows.append(
            SeedSurveyRow(
                scene_name=scene_name,
                frame_id=frame_id,
                seed_index=seed_index,
                count_positive=positive_count,
                count_score_ge_03=int((positive_scores >= 0.3).sum().item()),
                count_score_ge_05=int((positive_scores >= 0.5).sum().item()),
                count_score_ge_07=int((positive_scores >= 0.7).sum().item()),
                count_score_ge_09=int((positive_scores >= 0.9).sum().item()),
                count_selected=selected_count,
                selected_unique_rotation_count=int(rotation_stats["unique_count"]),
                selected_dominant_rotation_class=rotation_stats["dominant_class"],
                selected_dominant_rotation_fraction=rotation_stats["dominant_fraction"],
                selected_rotation_entropy=rotation_stats["entropy"],
                selected_rotation_entropy_normalized=rotation_stats["entropy_normalized"],
                score_mean=stats["mean"],
                score_std=stats["std"],
                score_min=stats["min"],
                score_p25=stats["p25"],
                score_p50=stats["p50"],
                score_p75=stats["p75"],
                score_max=stats["max"],
            )
        )

    counts_tensor = torch.tensor(positive_counts, dtype=torch.float32)  # [1024]
    selected_counts_tensor = torch.tensor(selected_counts, dtype=torch.float32)  # [1024]
    scene_scores = torch.cat(all_positive_scores, dim=0) if all_positive_scores else seed_xyz.new_empty((0,))
    scene_score_stats = summarize_scores(scene_scores)
    unique_counts_tensor = torch.tensor(selected_unique_rotation_counts, dtype=torch.float32)
    dominant_tensor = torch.tensor(selected_dominant_fractions, dtype=torch.float32)
    entropy_tensor = torch.tensor(selected_entropy_normalized, dtype=torch.float32)
    scene_row = SceneSurveyRow(
        scene_name=scene_name,
        frame_id=frame_id,
        seed_count=int(seed_xyz.shape[0]),
        seed_with_positive_count=int((counts_tensor > 0).sum().item()),
        mean_positive_count_per_seed=float(counts_tensor.mean().item()),
        median_positive_count_per_seed=float(torch.quantile(counts_tensor, 0.5).item()),
        max_positive_count_per_seed=int(counts_tensor.max().item()),
        total_positive_configs=int(counts_tensor.sum().item()),
        total_selected_configs=int(selected_counts_tensor.sum().item()),
        selected_seed_count=int((selected_counts_tensor > 0).sum().item()),
        selected_mean_unique_rotation_count=float(unique_counts_tensor.mean().item()) if unique_counts_tensor.numel() else 0.0,
        selected_mean_dominant_rotation_fraction=float(dominant_tensor.mean().item()) if dominant_tensor.numel() else None,
        selected_mean_rotation_entropy_normalized=float(entropy_tensor.mean().item()) if entropy_tensor.numel() else None,
        score_mean=scene_score_stats["mean"],
        score_p50=scene_score_stats["p50"],
        score_p75=scene_score_stats["p75"],
        score_p90=scene_score_stats["p90"],
        score_max=scene_score_stats["max"],
    )
    return scene_row, seed_rows, counts_tensor.long()


def write_csv(path: Path, rows: list[Any]) -> None:
    """Write dataclass rows to a CSV file."""
    if not rows:
        return
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def sample_scene_frame_indices(dataset: GraspNetDataset, num_scenes: int, seed: int) -> list[int]:
    """Sample one random frame index from each of several unique scenes."""
    rng = random.Random(seed)
    scene_to_indices: dict[str, list[int]] = {}
    for index, scene_name in enumerate(dataset.scenename):
        scene_to_indices.setdefault(scene_name, []).append(index)

    scene_names = sorted(scene_to_indices)
    sampled_scene_names = rng.sample(scene_names, min(num_scenes, len(scene_names)))
    sampled_indices = [rng.choice(scene_to_indices[scene_name]) for scene_name in sampled_scene_names]
    return sampled_indices


def build_summary(
    args: argparse.Namespace,
    sampled_indices: list[int],
    scene_rows: list[SceneSurveyRow],
    seed_rows: list[SeedSurveyRow],
) -> dict[str, Any]:
    """Build a JSON-serializable aggregate survey summary."""
    counts = torch.tensor([row.count_positive for row in seed_rows], dtype=torch.float32)  # [num_scenes * 1024]
    selected_counts = torch.tensor([row.count_selected for row in seed_rows], dtype=torch.float32)
    nonzero_counts = counts[counts > 0]
    score_values = [row.score_mean for row in seed_rows if row.score_mean is not None]
    dominant_values = [
        row.selected_dominant_rotation_fraction
        for row in seed_rows
        if row.selected_dominant_rotation_fraction is not None
    ]
    entropy_values = [
        row.selected_rotation_entropy_normalized
        for row in seed_rows
        if row.selected_rotation_entropy_normalized is not None
    ]
    unique_values = [
        row.selected_unique_rotation_count
        for row in seed_rows
        if row.count_selected > 0
    ]

    return {
        "config": vars(args),
        "sampled_dataset_indices": sampled_indices,
        "num_scene_frames": len(scene_rows),
        "num_seed_points": len(seed_rows),
        "seed_positive_count": {
            "mean": float(counts.mean().item()) if counts.numel() else 0.0,
            "median": float(torch.quantile(counts, 0.5).item()) if counts.numel() else 0.0,
            "p75": float(torch.quantile(counts, 0.75).item()) if counts.numel() else 0.0,
            "p90": float(torch.quantile(counts, 0.90).item()) if counts.numel() else 0.0,
            "p95": float(torch.quantile(counts, 0.95).item()) if counts.numel() else 0.0,
            "max": int(counts.max().item()) if counts.numel() else 0,
            "nonzero_mean": float(nonzero_counts.mean().item()) if nonzero_counts.numel() else 0.0,
            "nonzero_seed_fraction": float((counts > 0).float().mean().item()) if counts.numel() else 0.0,
        },
        "seed_selected_count": {
            "score_threshold": args.score_threshold,
            "mean": float(selected_counts.mean().item()) if selected_counts.numel() else 0.0,
            "median": float(torch.quantile(selected_counts, 0.5).item()) if selected_counts.numel() else 0.0,
            "p75": float(torch.quantile(selected_counts, 0.75).item()) if selected_counts.numel() else 0.0,
            "p90": float(torch.quantile(selected_counts, 0.90).item()) if selected_counts.numel() else 0.0,
            "p95": float(torch.quantile(selected_counts, 0.95).item()) if selected_counts.numel() else 0.0,
            "max": int(selected_counts.max().item()) if selected_counts.numel() else 0,
            "nonzero_seed_fraction": float((selected_counts > 0).float().mean().item()) if selected_counts.numel() else 0.0,
        },
        "selected_rotation_concentration": {
            "mean_unique_rotation_count": float(np.mean(unique_values)) if unique_values else 0.0,
            "mean_dominant_rotation_fraction": float(np.mean(dominant_values)) if dominant_values else None,
            "median_dominant_rotation_fraction": float(np.median(dominant_values)) if dominant_values else None,
            "mean_entropy_normalized": float(np.mean(entropy_values)) if entropy_values else None,
            "median_entropy_normalized": float(np.median(entropy_values)) if entropy_values else None,
        },
        "per_seed_mean_score": summarize_scores(torch.tensor(score_values, dtype=torch.float32)),
        "scenes": [asdict(row) for row in scene_rows],
    }


def main() -> None:
    """Run the seed neighborhood survey and write summary files."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GraspNetDataset(
        args.dataset_root,
        camera=args.camera,
        split=args.split,
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        remove_outlier=True,
        augment=False,
        load_label=True,
    )

    sampled_indices = sample_scene_frame_indices(dataset, args.num_scenes, args.seed)
    sample_count = len(sampled_indices)
    dataloader = DataLoader(
        Subset(dataset, sampled_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = load_frozen_graspable(args.checkpoint, device)
    scene_rows = []
    seed_rows = []

    with torch.no_grad():
        for local_idx, batch_data in enumerate(dataloader):
            dataset_idx = sampled_indices[local_idx]
            scene_name = dataset.scenename[dataset_idx]
            frame_id = int(dataset.frameid[dataset_idx])
            print(f"Surveying {scene_name} frame {frame_id:04d} ({local_idx + 1}/{sample_count})")

            batch_data = move_batch_to_device(batch_data, device)
            end_points = model(batch_data)
            scene_row, per_seed_rows, _ = survey_scene(
                end_points,
                scene_name,
                frame_id,
                args.radius,
                args.score_threshold,
            )
            scene_rows.append(scene_row)
            seed_rows.extend(per_seed_rows)

    write_csv(output_dir / "per_scene_summary.csv", scene_rows)
    write_csv(output_dir / "per_seed_summary.csv", seed_rows)

    summary = build_summary(args, sampled_indices, scene_rows, seed_rows)
    with (output_dir / "survey_summary.json").open("w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print(f"Wrote survey outputs to: {output_dir}")
    print(json.dumps(summary["seed_positive_count"], indent=2))
    print(json.dumps(summary["selected_rotation_concentration"], indent=2))


if __name__ == "__main__":
    main()
