#!/usr/bin/env python3
"""Survey valid grasp-pool sizes per seed to choose a practical padded max K."""

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

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from dataset.graspnet_dataset import GraspNetDataset, collate_fn  # noqa: E402
from flow.models.grasp_cfm import economic_graspable  # noqa: E402
from flow.utils.cfm_label_generation import process_grasp_labels  # noqa: E402
from utils.arguments import cfgs  # noqa: E402
sys.argv = _ORIGINAL_ARGV


@dataclass
class ScenePoolSummary:
    """Store pool-size statistics for one sampled scene."""

    scene_name: str
    frame_id: int
    seed_count: int
    valid_seed_count: int
    valid_seed_fraction: float
    mean_k_all: float
    mean_k_valid: float | None
    p50_k_valid: float | None
    p75_k_valid: float | None
    p90_k_valid: float | None
    p95_k_valid: float | None
    p99_k_valid: float | None
    max_k_valid: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the max-K survey."""
    parser = argparse.ArgumentParser(
        description="Survey valid grasp-pool sizes per seed for padded max-K design."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate_k",
        type=int,
        nargs="+",
        default=[32, 64, 96, 128, 160, 192, 256, 320, 384],
        help="Candidate padded max-K values to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="flow/analysis/results/seed_pool_size",
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
    """Load the frozen economic_graspable module from a checkpoint."""
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

    model.load_state_dict(mapped_state, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    print(f"Loaded {len(mapped_state)} checkpoint tensors into economic_graspable.")
    return model


def summarize_valid_counts(valid_counts: torch.Tensor) -> dict[str, float | int | None]:
    """Compute percentile statistics for valid seed pool sizes."""
    if valid_counts.numel() == 0:
        return {
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": 0,
        }

    valid_counts = valid_counts.float().cpu()
    return {
        "mean": float(valid_counts.mean().item()),
        "p50": float(torch.quantile(valid_counts, 0.50).item()),
        "p75": float(torch.quantile(valid_counts, 0.75).item()),
        "p90": float(torch.quantile(valid_counts, 0.90).item()),
        "p95": float(torch.quantile(valid_counts, 0.95).item()),
        "p99": float(torch.quantile(valid_counts, 0.99).item()),
        "max": int(valid_counts.max().item()),
    }


def build_k_coverage(counts: torch.Tensor, candidate_k: list[int]) -> list[dict[str, float | int]]:
    """Measure how much truncation each padded K would introduce."""
    counts_cpu = counts.cpu()
    valid_counts = counts_cpu[counts_cpu > 0]
    total_seed = int(counts_cpu.numel())
    total_valid_seed = int(valid_counts.numel())
    total_valid_grasps = int(valid_counts.sum().item())

    rows = []
    for k in candidate_k:
        truncated_mask = counts_cpu > k
        truncated_valid_mask = valid_counts > k
        kept = torch.clamp(counts_cpu, max=k)
        valid_kept = torch.clamp(valid_counts, max=k)

        row = {
            "k": int(k),
            "seed_truncated_count": int(truncated_mask.sum().item()),
            "seed_truncated_fraction_all": float(truncated_mask.float().mean().item()),
            "seed_truncated_fraction_valid": (
                float(truncated_valid_mask.float().mean().item()) if total_valid_seed > 0 else 0.0
            ),
            "grasp_kept_fraction": (
                float(kept.sum().item() / max(total_valid_grasps, 1)) if total_valid_grasps > 0 else 0.0
            ),
            "mean_padding_waste_valid": (
                float((k - valid_kept).float().mean().item()) if total_valid_seed > 0 else float(k)
            ),
        }
        rows.append(row)
    return rows


def main() -> None:
    """Run the survey and write summary files."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GraspNetDataset(
        args.dataset_root,
        camera=args.camera,
        split=args.split,
        voxel_size=cfgs.voxel_size,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
    )

    sampled_indices = random.sample(range(len(dataset)), k=min(args.num_scenes, len(dataset)))
    subset = Subset(dataset, sampled_indices)
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = load_frozen_graspable(args.checkpoint, device)

    all_counts = []
    per_scene_rows: list[ScenePoolSummary] = []
    with torch.no_grad():
        for subset_idx, batch_data in enumerate(dataloader):
            dataset_idx = sampled_indices[subset_idx]
            batch_data = move_batch_to_device(batch_data, device)
            end_points = model(batch_data)
            end_points = process_grasp_labels(end_points)

            if "seed_grasp_count" in end_points:
                count_tensor = end_points["seed_grasp_count"][0].to(torch.int32)  # [1024]
            else:
                score_list = end_points["seed_grasp_score_list"][0]
                count_tensor = torch.tensor(
                    [len(seed_scores) for seed_scores in score_list],
                    dtype=torch.int32,
                    device=device,
                )  # [1024]
            valid_counts = count_tensor[count_tensor > 0]  # [num_valid_seed]
            summary = summarize_valid_counts(valid_counts)

            all_counts.append(count_tensor.cpu())
            per_scene_rows.append(
                ScenePoolSummary(
                    scene_name=dataset.scenename[dataset_idx],
                    frame_id=int(dataset.frameid[dataset_idx]),
                    seed_count=int(count_tensor.numel()),
                    valid_seed_count=int((count_tensor > 0).sum().item()),
                    valid_seed_fraction=float((count_tensor > 0).float().mean().item()),
                    mean_k_all=float(count_tensor.float().mean().item()),
                    mean_k_valid=summary["mean"],
                    p50_k_valid=summary["p50"],
                    p75_k_valid=summary["p75"],
                    p90_k_valid=summary["p90"],
                    p95_k_valid=summary["p95"],
                    p99_k_valid=summary["p99"],
                    max_k_valid=int(summary["max"]),
                )
            )

    if not all_counts:
        raise RuntimeError("No scenes were surveyed.")

    all_counts_tensor = torch.cat(all_counts, dim=0)  # [num_scenes * 1024]
    valid_counts_tensor = all_counts_tensor[all_counts_tensor > 0]
    valid_summary = summarize_valid_counts(valid_counts_tensor)
    k_coverage = build_k_coverage(all_counts_tensor, args.candidate_k)

    best_k = None
    for row in k_coverage:
        if row["seed_truncated_fraction_valid"] <= 0.10:
            best_k = row["k"]
            break
    if best_k is None:
        best_k = args.candidate_k[-1]

    summary = {
        "config": {
            "dataset_root": args.dataset_root,
            "checkpoint": args.checkpoint,
            "camera": args.camera,
            "split": args.split,
            "num_scenes": len(per_scene_rows),
            "seed": args.seed,
            "candidate_k": args.candidate_k,
        },
        "aggregate": {
            "total_seed_count": int(all_counts_tensor.numel()),
            "valid_seed_count": int(valid_counts_tensor.numel()),
            "valid_seed_fraction": float((all_counts_tensor > 0).float().mean().item()),
            "mean_k_all": float(all_counts_tensor.float().mean().item()),
            "mean_k_valid": valid_summary["mean"],
            "p50_k_valid": valid_summary["p50"],
            "p75_k_valid": valid_summary["p75"],
            "p90_k_valid": valid_summary["p90"],
            "p95_k_valid": valid_summary["p95"],
            "p99_k_valid": valid_summary["p99"],
            "max_k_valid": int(valid_summary["max"]),
        },
        "candidate_k_summary": k_coverage,
        "recommended_k": int(best_k),
        "recommendation_rule": "Smallest K with valid-seed truncation fraction <= 10%.",
    }

    summary_path = output_dir / "pool_size_summary.json"
    with summary_path.open("w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    scene_csv_path = output_dir / "per_scene_pool_size.csv"
    with scene_csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(per_scene_rows[0]).keys()))
        writer.writeheader()
        for row in per_scene_rows:
            writer.writerow(asdict(row))

    candidate_csv_path = output_dir / "candidate_k_summary.csv"
    with candidate_csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(k_coverage[0].keys()))
        writer.writeheader()
        for row in k_coverage:
            writer.writerow(row)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Recommended K: {best_k}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
