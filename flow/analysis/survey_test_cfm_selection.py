#!/usr/bin/env python3
"""Survey how many `test_cfm.py`-selected seeds are invalid on a small scene subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from graspnetAPI import GraspGroup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from dataset.graspnet_dataset import GraspNetDataset, collate_fn  # noqa: E402
from models.economicgrasp import economicgrasp  # noqa: E402
from utils.arguments import cfgs  # noqa: E402
from utils.collision_detector import ModelFreeCollisionDetector  # noqa: E402
from flow.models.grasp_cfm import GraspVelocityMLP  # noqa: E402
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction  # noqa: E402
from flow.test_cfm import extract_scene_inputs, resolve_cfm_checkpoint  # noqa: E402
from flow.utils.cfm_label_generation import process_grasp_labels  # noqa: E402
from flow.utils.cfm_solver import euler_solve  # noqa: E402
from flow.utils.lie import exp_so3  # noqa: E402
sys.argv = _ORIGINAL_ARGV


@dataclass
class FrameSurveyRow:
    """Store invalid-seed statistics for one surveyed frame."""

    split: str
    scene_name: str
    frame_id: int
    total_seed_count: int
    post_collision_count: int
    post_top10_per_object_count: int
    final_top50_count: int
    selected_valid_seed_count: int
    selected_invalid_seed_count: int
    selected_invalid_fraction: float


@dataclass
class SplitSurveySummary:
    """Store aggregate invalid-seed statistics for one split."""

    split: str
    frame_count: int
    scene_count: int
    total_seed_count: int
    post_collision_count: int
    post_top10_per_object_count: int
    final_top50_count: int
    selected_valid_seed_count: int
    selected_invalid_seed_count: int
    selected_invalid_fraction: float
    mean_final_top50_count: float
    mean_selected_invalid_count: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the seed-validity survey."""
    parser = argparse.ArgumentParser(
        description=(
            "Survey how many seeds selected by the `test_cfm.py` inference policy "
            "are invalid according to `process_grasp_labels`."
        )
    )
    parser.add_argument(
        "--dataset_root",
        default=cfgs.dataset_root,
        help="GraspNet dataset root.",
    )
    parser.add_argument(
        "--camera",
        default="realsense",
        choices=["realsense", "kinect"],
        help="Camera stream to survey.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/economicgrasp_realsense.tar",
        help="Base EconomicGrasp checkpoint path.",
    )
    parser.add_argument(
        "--cfm_checkpoint_path",
        default="",
        help="Optional CFM checkpoint path. Empty uses the newest flow/results/*/flowgrasp_latest.tar.",
    )
    parser.add_argument(
        "--stats_path",
        default="/media/dsp520/Grasp_2T/graspnet/cfm_seed5d_norm_stats.pt",
        help="Path to CFM normalization metadata.",
    )
    parser.add_argument("--n_steps", type=int, default=20, help="Euler solver steps.")
    parser.add_argument("--nsample", type=int, default=32, help="Neighborhood size for seed conditioning.")
    parser.add_argument("--sphere_radius", type=float, default=0.005, help="Seed grouping radius in meters.")
    parser.add_argument("--collision_thresh", type=float, default=0.01, help="Collision threshold for filtering.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_seen"],
        choices=["test_seen", "test_similar", "test_novel"],
        help="Evaluation splits to survey.",
    )
    parser.add_argument(
        "--scene_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit GraspNet scene ids, for example `100 101`.",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="Number of scenes to survey when `--scene_ids` is not provided.",
    )
    parser.add_argument(
        "--frames_per_scene",
        type=int,
        default=4,
        help="Maximum number of frames to survey per scene.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional hard cap on surveyed frames per split after scene filtering.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional JSON path for writing summaries and per-frame rows.",
    )
    return parser.parse_args()


def move_batch_to_device(batch_data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a GraspNet batch produced by `collate_fn` onto one device."""
    for key, value in batch_data.items():
        if "list" in key:
            for batch_idx in range(len(value)):
                for item_idx in range(len(value[batch_idx])):
                    if isinstance(value[batch_idx][item_idx], torch.Tensor):
                        value[batch_idx][item_idx] = value[batch_idx][item_idx].to(device)
        elif isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
    return batch_data


def load_base_net(checkpoint_path: str, device: torch.device) -> economicgrasp:
    """Load the frozen EconomicGrasp network used to reproduce test-time seed selection."""
    model = economicgrasp(seed_feat_dim=512, is_training=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_cfm_modules(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GraspVelocityMLP, Sphere_Grouping_Global_Interaction, dict[str, Any]]:
    """Load the frozen CFM modules and normalization metadata."""
    cfm_checkpoint_path = resolve_cfm_checkpoint(args.cfm_checkpoint_path)
    print(f"Surveying CFM checkpoint: {cfm_checkpoint_path}")

    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=args.nsample,
        seed_feature_dim=512,
        sphere_radius=args.sphere_radius,
    ).to(device)
    mlp = GraspVelocityMLP(grasp_dim=5, cond_dim=128).to(device)

    cfm_checkpoint = torch.load(cfm_checkpoint_path, map_location=device)
    seed_conditioner.load_state_dict(cfm_checkpoint["seed_conditioner_state_dict"])
    mlp.load_state_dict(cfm_checkpoint["mlp_state_dict"])
    seed_conditioner.eval()
    mlp.eval()

    norm_metadata = torch.load(args.stats_path, map_location=device)
    return mlp, seed_conditioner, norm_metadata


def scene_id_from_name(scene_name: str) -> int:
    """Extract the integer GraspNet scene id from a `scene_XXXX` name."""
    return int(scene_name.split("_")[-1])


def choose_scene_ids(dataset: GraspNetDataset, args: argparse.Namespace) -> list[int]:
    """Choose a small deterministic scene subset to survey for one split."""
    if args.scene_ids:
        return sorted(set(args.scene_ids))

    discovered_scene_ids = []
    seen_scene_ids = set()
    for scene_name in dataset.scenename:
        scene_id = scene_id_from_name(scene_name)
        if scene_id in seen_scene_ids:
            continue
        discovered_scene_ids.append(scene_id)
        seen_scene_ids.add(scene_id)
        if len(discovered_scene_ids) >= args.num_scenes:
            break
    return discovered_scene_ids


def collect_target_indices(dataset: GraspNetDataset, args: argparse.Namespace) -> list[int]:
    """Collect dataset indices for the requested scene subset and frame budget."""
    selected_scene_ids = set(choose_scene_ids(dataset, args))
    frames_per_scene = defaultdict(int)
    target_indices = []

    for dataset_idx, scene_name in enumerate(dataset.scenename):
        scene_id = scene_id_from_name(scene_name)
        if scene_id not in selected_scene_ids:
            continue
        if frames_per_scene[scene_id] >= args.frames_per_scene:
            continue

        target_indices.append(dataset_idx)
        frames_per_scene[scene_id] += 1

        if args.max_frames is not None and len(target_indices) >= args.max_frames:
            break

    return target_indices


def select_candidate_seed_indices(
    seed_graspness: torch.Tensor,
    seed_object_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    max_per_object: int = 10,
) -> torch.Tensor:
    """Keep the highest-graspness collision-free seeds for each detected object."""
    candidate_indices = []
    valid_object_mask = valid_mask & (seed_object_ids > 0)

    for object_id in torch.unique(seed_object_ids[valid_object_mask]):
        object_indices = torch.nonzero(
            valid_mask & (seed_object_ids == object_id),
            as_tuple=False,
        ).squeeze(1)
        if object_indices.numel() == 0:
            continue

        object_scores = seed_graspness[object_indices]  # Shape: [num_object_seed]
        top_k = min(max_per_object, object_indices.numel())
        top_local = torch.topk(object_scores, k=top_k, largest=True, sorted=True).indices
        candidate_indices.append(object_indices[top_local])

    if not candidate_indices:
        return torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    return torch.cat(candidate_indices, dim=0)


def select_top_seed_indices(
    seed_graspness: torch.Tensor,
    seed_object_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    max_per_object: int = 10,
    total_top_k: int = 50,
) -> torch.Tensor:
    """Match the final top-k scene-level seed selection used during CFM testing."""
    candidate_indices = select_candidate_seed_indices(
        seed_graspness,
        seed_object_ids,
        valid_mask,
        max_per_object=max_per_object,
    )
    if candidate_indices.numel() == 0:
        return candidate_indices

    candidate_scores = seed_graspness[candidate_indices]  # Shape: [num_candidate_seed]
    final_k = min(total_top_k, candidate_indices.numel())
    top_global = torch.topk(candidate_scores, k=final_k, largest=True, sorted=True).indices
    return candidate_indices[top_global]


def build_grasp_group(
    pred_grasps: torch.Tensor,
    seed_xyz: torch.Tensor,
    seed_graspness: torch.Tensor,
) -> GraspGroup:
    """Convert predicted 5D grasp parameters into a `GraspGroup` for collision checks."""
    num_seed = seed_xyz.shape[0]
    omega = pred_grasps[:, :3]  # Shape: [N, 3]
    width = pred_grasps[:, 3]  # Shape: [N]
    depth = pred_grasps[:, 4]  # Shape: [N]
    rot_matrices = exp_so3(omega)  # Shape: [N, 3, 3]
    rot_flat = rot_matrices.reshape(num_seed, 9)  # Shape: [N, 9]

    score = seed_graspness.view(-1, 1)  # Shape: [N, 1]
    grasp_width = torch.clamp(width, min=0.0, max=0.1).view(-1, 1)  # Shape: [N, 1]
    grasp_depth = torch.clamp(depth, min=0.01, max=0.04).view(-1, 1)  # Shape: [N, 1]
    grasp_height = 0.02 * torch.ones_like(score)  # Shape: [N, 1]
    obj_ids = -1 * torch.ones_like(score)  # Shape: [N, 1]

    gg_preds = torch.cat(
        [
            score,
            grasp_width,
            grasp_height,
            grasp_depth,
            rot_flat,
            seed_xyz,
            obj_ids,
        ],
        dim=-1,
    ).cpu().numpy()
    return GraspGroup(gg_preds)


def build_collision_keep_mask(
    args: argparse.Namespace,
    raw_dataset: GraspNetDataset,
    dataset_idx: int,
    gg_preds: GraspGroup,
    num_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the collision-free mask for all seed-conditioned grasps in one frame."""
    keep_mask = torch.ones(num_seed, dtype=torch.bool, device=device)
    if args.collision_thresh <= 0:
        return keep_mask

    cloud, _ = raw_dataset.get_data(dataset_idx, return_raw_cloud=True)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(
        gg_preds,
        approach_dist=0.05,
        collision_thresh=args.collision_thresh,
    )
    return torch.from_numpy(~collision_mask).to(device=device, dtype=torch.bool)


def survey_frame(
    split: str,
    args: argparse.Namespace,
    base_net: economicgrasp,
    raw_dataset: GraspNetDataset,
    dataset_idx: int,
    mlp: GraspVelocityMLP,
    seed_conditioner: Sphere_Grouping_Global_Interaction,
    norm_metadata: dict[str, Any],
    device: torch.device,
) -> FrameSurveyRow:
    """Survey one frame and count invalid seeds after `test_cfm.py`-style selection."""
    raw_sample = raw_dataset[dataset_idx]
    batch_data = move_batch_to_device(collate_fn([raw_sample]), device)

    with torch.no_grad():
        seed_xyz, seed_feats, seed_graspness, seed_object_ids = extract_scene_inputs(base_net, batch_data)
        batch_data["xyz_graspable"] = seed_xyz
        batch_data = process_grasp_labels(batch_data)
        seed_valid_mask = batch_data["seed_valid_mask"].squeeze(0)  # Shape: [1024]

        x0 = torch.randn(1, seed_xyz.shape[1], 5, device=device)  # Shape: [1, 1024, 5]
        x_pred = euler_solve(
            seed_conditioner,
            mlp,
            x0,
            seed_xyz,
            seed_feats,
            norm_metadata,
            n_steps=args.n_steps,
        )

        gg_preds = build_grasp_group(
            x_pred[0],
            seed_xyz[0],
            seed_graspness[0],
        )
        keep_mask = build_collision_keep_mask(
            args,
            raw_dataset,
            dataset_idx,
            gg_preds,
            seed_xyz.shape[1],
            device,
        )
        candidate_indices = select_candidate_seed_indices(
            seed_graspness[0],
            seed_object_ids[0],
            keep_mask,
        )
        selected_indices = select_top_seed_indices(
            seed_graspness[0],
            seed_object_ids[0],
            keep_mask,
        )

    selected_valid_mask = seed_valid_mask[selected_indices]  # Shape: [K]
    selected_valid_seed_count = int(selected_valid_mask.sum().item())
    selected_invalid_seed_count = int((~selected_valid_mask).sum().item())
    selected_count = int(selected_indices.numel())

    return FrameSurveyRow(
        split=split,
        scene_name=raw_dataset.scenename[dataset_idx],
        frame_id=int(raw_dataset.frameid[dataset_idx]),
        total_seed_count=int(seed_xyz.shape[1]),
        post_collision_count=int(keep_mask.sum().item()),
        post_top10_per_object_count=int(candidate_indices.numel()),
        final_top50_count=selected_count,
        selected_valid_seed_count=selected_valid_seed_count,
        selected_invalid_seed_count=selected_invalid_seed_count,
        selected_invalid_fraction=(
            selected_invalid_seed_count / selected_count if selected_count > 0 else 0.0
        ),
    )


def summarize_split(split: str, frame_rows: list[FrameSurveyRow]) -> SplitSurveySummary:
    """Aggregate per-frame survey rows into one split-level summary."""
    frame_count = len(frame_rows)
    scene_count = len({row.scene_name for row in frame_rows})
    total_seed_count = sum(row.total_seed_count for row in frame_rows)
    post_collision_count = sum(row.post_collision_count for row in frame_rows)
    post_top10_per_object_count = sum(row.post_top10_per_object_count for row in frame_rows)
    final_top50_count = sum(row.final_top50_count for row in frame_rows)
    selected_valid_seed_count = sum(row.selected_valid_seed_count for row in frame_rows)
    selected_invalid_seed_count = sum(row.selected_invalid_seed_count for row in frame_rows)

    return SplitSurveySummary(
        split=split,
        frame_count=frame_count,
        scene_count=scene_count,
        total_seed_count=total_seed_count,
        post_collision_count=post_collision_count,
        post_top10_per_object_count=post_top10_per_object_count,
        final_top50_count=final_top50_count,
        selected_valid_seed_count=selected_valid_seed_count,
        selected_invalid_seed_count=selected_invalid_seed_count,
        selected_invalid_fraction=(
            selected_invalid_seed_count / final_top50_count if final_top50_count > 0 else 0.0
        ),
        mean_final_top50_count=(final_top50_count / frame_count) if frame_count > 0 else 0.0,
        mean_selected_invalid_count=(selected_invalid_seed_count / frame_count) if frame_count > 0 else 0.0,
    )


def survey_split(
    args: argparse.Namespace,
    split: str,
    base_net: economicgrasp,
    mlp: GraspVelocityMLP,
    seed_conditioner: Sphere_Grouping_Global_Interaction,
    norm_metadata: dict[str, Any],
    device: torch.device,
) -> tuple[SplitSurveySummary, list[FrameSurveyRow]]:
    """Survey one split on a small scene subset."""
    raw_dataset = GraspNetDataset(
        args.dataset_root,
        split=split,
        camera=args.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        load_label=True,
        augment=False,
    )
    target_indices = collect_target_indices(raw_dataset, args)
    print(
        f"[{split}] Surveying {len(target_indices)} frames from "
        f"{len({raw_dataset.scenename[idx] for idx in target_indices})} scenes."
    )

    frame_rows = []
    start_time = time.time()
    for dataset_idx in target_indices:
        frame_row = survey_frame(
            split,
            args,
            base_net,
            raw_dataset,
            dataset_idx,
            mlp,
            seed_conditioner,
            norm_metadata,
            device,
        )
        frame_rows.append(frame_row)
        print(
            f"[{split}] {frame_row.scene_name} frame {frame_row.frame_id}: "
            f"selected={frame_row.final_top50_count}, "
            f"invalid={frame_row.selected_invalid_seed_count}, "
            f"invalid_fraction={frame_row.selected_invalid_fraction:.3f}"
        )

    elapsed = time.time() - start_time
    print(f"[{split}] Finished in {elapsed:.2f}s")
    return summarize_split(split, frame_rows), frame_rows


def main() -> None:
    """Run the seed-validity survey and optionally write results to JSON."""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Base checkpoint: {args.checkpoint_path}")

    base_net = load_base_net(args.checkpoint_path, device)
    mlp, seed_conditioner, norm_metadata = load_cfm_modules(args, device)

    summaries = []
    frame_rows = []
    for split in args.splits:
        summary, split_rows = survey_split(
            args,
            split,
            base_net,
            mlp,
            seed_conditioner,
            norm_metadata,
            device,
        )
        summaries.append(summary)
        frame_rows.extend(split_rows)
        print(
            f"[{split}] invalid_selected={summary.selected_invalid_seed_count}/"
            f"{summary.final_top50_count} "
            f"({summary.selected_invalid_fraction:.3%}), "
            f"mean_invalid_per_frame={summary.mean_selected_invalid_count:.2f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summaries": [asdict(summary) for summary in summaries],
                    "frames": [asdict(row) for row in frame_rows],
                },
                handle,
                indent=2,
            )
        print(f"Wrote survey summary to {output_path}")


if __name__ == "__main__":
    main()
