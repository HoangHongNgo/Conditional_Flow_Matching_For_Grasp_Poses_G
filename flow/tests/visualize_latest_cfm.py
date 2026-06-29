import os
import sys
import argparse
import glob
import numpy as np
import torch
import open3d as o3d

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graspnetAPI import GraspGroup
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp
from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs
from flow.models.grasp_cfm import GraspVelocityMLP
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction
from flow.utils.cfm_solver import euler_solve
from flow.utils.cfm_label_generation import process_grasp_labels


def extract_scene_inputs(base_net, batch_data):
    """Extract graspable FPS seeds, features, scores, and object ids for one batch."""
    import MinkowskiEngine as ME
    from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

    seed_xyz = batch_data['point_clouds']  # [B, N, 3]
    B, point_num, _ = seed_xyz.shape

    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [coord for coord in batch_data['coordinates_for_voxel']],
        [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)]
    )
    coordinates_batch, features_batch, _, batch_data['quantize2original'] = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True
    )

    device = seed_xyz.device
    coordinates_batch = coordinates_batch.to(device)
    features_batch = features_batch.to(device)

    mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
    seed_features = base_net.backbone(mink_input).F
    seed_features = seed_features[batch_data['quantize2original']].view(
        B, point_num, -1
    ).transpose(1, 2)

    batch_data = base_net.graspable(seed_features, batch_data)
    seed_features_flipped = seed_features.transpose(1, 2)  # [B, N, 512]
    objectness_score = batch_data['objectness_score']
    graspness_score = batch_data['graspness_score'].squeeze(1)
    segmentation_label = batch_data.get('segmentation_label')
    if segmentation_label is not None:
        segmentation_label = segmentation_label.long()  # [B, N]
    objectness_pred = torch.argmax(objectness_score, 1)
    objectness_mask = (objectness_pred == 1)
    graspness_mask = graspness_score > cfgs.graspness_threshold
    graspable_mask = objectness_mask & graspness_mask

    seed_features_graspable = []
    seed_xyz_graspable = []
    seed_graspness_graspable = []
    seed_object_ids_graspable = []
    for i in range(B):
        cur_mask = graspable_mask[i]
        if cur_mask.sum() == 0:
            cur_mask = torch.ones_like(cur_mask, dtype=torch.bool)

        cur_feat = seed_features_flipped[i][cur_mask]
        cur_seed_xyz = seed_xyz[i][cur_mask]
        cur_graspness = graspness_score[i][cur_mask]
        if segmentation_label is None:
            cur_object_ids = torch.full_like(cur_graspness, fill_value=-1, dtype=torch.long)
        else:
            cur_object_ids = segmentation_label[i][cur_mask]

        cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
        fps_idxs = furthest_point_sample(cur_seed_xyz, base_net.M_points)
        cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
        cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(
            1, 2
        ).squeeze(0).contiguous()
        cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
        cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()
        cur_graspness = cur_graspness[fps_idxs.squeeze(0).long()].contiguous()
        cur_object_ids = cur_object_ids[fps_idxs.squeeze(0).long()].contiguous()

        seed_features_graspable.append(cur_feat)
        seed_xyz_graspable.append(cur_seed_xyz)
        seed_graspness_graspable.append(cur_graspness)
        seed_object_ids_graspable.append(cur_object_ids)

    seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # [B, 1024, 3]
    seed_features_graspable = torch.stack(seed_features_graspable)  # [B, 512, 1024]
    seed_graspness_graspable = torch.stack(seed_graspness_graspable, 0)  # [B, 1024]
    seed_object_ids_graspable = torch.stack(seed_object_ids_graspable, 0)  # [B, 1024]
    return (
        seed_xyz_graspable,
        seed_features_graspable,
        seed_graspness_graspable,
        seed_object_ids_graspable,
    )


def resolve_latest_checkpoint(checkpoint_path):
    """Resolve a checkpoint path or auto-discover the newest flowgrasp_latest.tar."""
    if checkpoint_path:
        return checkpoint_path

    candidates = glob.glob("flow/results/*/flowgrasp_latest.tar")
    if not candidates:
        raise FileNotFoundError("No flowgrasp_latest.tar found under flow/results/*/")

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def expected_split_for_scene(scene_id):
    """Return the GraspNet split that contains a scene id."""
    if 0 <= scene_id < 100:
        return 'train'
    if 100 <= scene_id < 130:
        return 'test_seen'
    if 130 <= scene_id < 160:
        return 'test_similar'
    if 160 <= scene_id < 190:
        return 'test_novel'
    return None


def to_device(batch_data, device):
    """Move a GraspNet batch to the target device."""
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def graspgroup_from_5d(x_denorm, seed_xyz, score_values=None):
    """Convert denormalized 5D grasps into a GraspGroup with explicit scores."""
    from flow.utils.lie import exp_so3

    num_grasp = x_denorm.shape[0]
    p = seed_xyz[:num_grasp].cpu()
    omega = x_denorm[:, :3].cpu()
    w = x_denorm[:, 3].cpu()
    d = x_denorm[:, 4].cpu()

    rot_flat = exp_so3(omega).reshape(num_grasp, 9).cpu()
    if score_values is None:
        score = torch.linspace(0.99, 0.50, steps=num_grasp).view(-1, 1)
    else:
        score = score_values[:num_grasp].detach().cpu().view(-1, 1)
    grasp_width = torch.clamp(w, min=0.0, max=0.1).view(-1, 1)
    grasp_depth = torch.clamp(d, min=0.01, max=0.04).view(-1, 1)
    grasp_height = 0.02 * torch.ones_like(score)
    obj_ids = -1 * torch.ones_like(score)

    gg_preds = torch.cat([
        score,
        grasp_width,
        grasp_height,
        grasp_depth,
        rot_flat,
        p,
        obj_ids,
    ], dim=-1).numpy()
    return GraspGroup(gg_preds)


def select_top_seed_indices(seed_graspness, seed_object_ids, valid_mask, max_per_object=10, total_top_k=50):
    """Select top seed indices after collision filtering using per-object then scene-level graspness."""
    valid_object_mask = valid_mask & (seed_object_ids > 0)
    candidate_indices = []

    for object_id in torch.unique(seed_object_ids[valid_object_mask]):
        object_indices = torch.nonzero(
            valid_mask & (seed_object_ids == object_id),
            as_tuple=False
        ).squeeze(1)
        if object_indices.numel() == 0:
            continue

        object_scores = seed_graspness[object_indices]
        top_k = min(max_per_object, object_indices.numel())
        top_local = torch.topk(object_scores, k=top_k, largest=True, sorted=True).indices
        candidate_indices.append(object_indices[top_local])

    if not candidate_indices:
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        fallback_k = min(total_top_k, valid_indices.numel())
        if fallback_k == 0:
            return valid_indices
        valid_scores = seed_graspness[valid_indices]
        top_global = torch.topk(valid_scores, k=fallback_k, largest=True, sorted=True).indices
        return valid_indices[top_global]

    candidate_indices = torch.cat(candidate_indices, dim=0)
    candidate_scores = seed_graspness[candidate_indices]
    final_k = min(total_top_k, candidate_indices.numel())
    top_global = torch.topk(candidate_scores, k=final_k, largest=True, sorted=True).indices
    return candidate_indices[top_global]


def build_collision_keep_mask(x_pred, seed_xyz, cloud_points, device, collision_thresh):
    """Return a boolean mask for seed grasps that pass model-free collision detection."""
    num_seed = x_pred.shape[0]
    keep_mask = torch.ones(num_seed, dtype=torch.bool, device=device)  # [1024]
    if collision_thresh <= 0:
        return keep_mask

    all_seed_gg = graspgroup_from_5d(x_pred, seed_xyz)
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(
        all_seed_gg,
        approach_dist=0.05,
        collision_thresh=collision_thresh,
    )
    return torch.from_numpy(~collision_mask).to(device=device, dtype=torch.bool)


def scores_to_colors(score_values):
    """Map grasp scores to RGB colors, from red (low) to blue (high)."""
    score_np = score_values.detach().cpu().numpy().astype(np.float32)
    if score_np.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    score_min = float(score_np.min())
    score_max = float(score_np.max())
    if score_max - score_min < 1e-8:
        normalized = np.full_like(score_np, 0.5, dtype=np.float32)
    else:
        normalized = (score_np - score_min) / (score_max - score_min)

    red = 1.0 - normalized
    green = 0.25 * (1.0 - np.abs(2.0 * normalized - 1.0))
    blue = normalized
    return np.stack([red, green, blue], axis=1)


def build_best_gt_graspgroup(end_points, selected_indices, seed_xyz):
    """Build one best-scoring ground-truth grasp per selected seed point."""
    gt_end_points = process_grasp_labels(end_points.copy())
    gt_scores = gt_end_points['seed_grasp_score'][0, selected_indices]  # [K, max_k]
    gt_slot_mask = gt_end_points['seed_grasp_slot_mask'][0, selected_indices]  # [K, max_k]
    gt_rot_lie = gt_end_points['seed_grasp_rot_lie'][0, selected_indices]  # [K, max_k, 3]
    gt_width = gt_end_points['seed_grasp_width'][0, selected_indices]  # [K, max_k]
    gt_depth = gt_end_points['seed_grasp_depth'][0, selected_indices]  # [K, max_k]

    invalid_fill = torch.full_like(gt_scores, -1e9)
    masked_scores = torch.where(gt_slot_mask, gt_scores, invalid_fill)
    best_slot = torch.argmax(masked_scores, dim=1)  # [K]
    valid_seed_mask = gt_slot_mask.any(dim=1)  # [K]
    if not valid_seed_mask.any():
        return None, None

    valid_seed_indices = torch.nonzero(valid_seed_mask, as_tuple=False).squeeze(1)
    best_slot_valid = best_slot[valid_seed_indices]

    best_rot_lie = gt_rot_lie[valid_seed_indices, best_slot_valid]  # [K_valid, 3]
    best_width = gt_width[valid_seed_indices, best_slot_valid]  # [K_valid]
    best_depth = gt_depth[valid_seed_indices, best_slot_valid]  # [K_valid]
    best_score = gt_scores[valid_seed_indices, best_slot_valid]  # [K_valid]
    best_seed_xyz = seed_xyz[0, selected_indices[valid_seed_indices]]  # [K_valid, 3]

    gt_x = torch.cat(
        [
            best_rot_lie,
            best_width.unsqueeze(1),
            best_depth.unsqueeze(1),
        ],
        dim=1,
    )  # [K_valid, 5]
    gt_gg = graspgroup_from_5d(gt_x, best_seed_xyz, best_score)
    return gt_gg, best_score


def build_best_gt_graspgroup_for_all_seeds(gt_end_points, seed_xyz):
    """Build one best-scoring ground-truth grasp for each valid seed point."""
    gt_scores = gt_end_points['seed_grasp_score'][0]  # [N, max_k]
    gt_slot_mask = gt_end_points['seed_grasp_slot_mask'][0]  # [N, max_k]
    gt_rot_lie = gt_end_points['seed_grasp_rot_lie'][0]  # [N, max_k, 3]
    gt_width = gt_end_points['seed_grasp_width'][0]  # [N, max_k]
    gt_depth = gt_end_points['seed_grasp_depth'][0]  # [N, max_k]

    valid_seed_mask = gt_slot_mask.any(dim=1)  # [N]
    if not valid_seed_mask.any():
        return None, None

    invalid_fill = torch.full_like(gt_scores, -1e9)
    masked_scores = torch.where(gt_slot_mask, gt_scores, invalid_fill)  # [N, max_k]
    best_slot = torch.argmax(masked_scores, dim=1)  # [N]
    seed_indices = torch.nonzero(valid_seed_mask, as_tuple=False).squeeze(1)  # [N_valid]
    slot_indices = best_slot[seed_indices]  # [N_valid]

    gt_x = torch.cat(
        [
            gt_rot_lie[seed_indices, slot_indices],  # [N_valid, 3]
            gt_width[seed_indices, slot_indices].unsqueeze(1),  # [N_valid, 1]
            gt_depth[seed_indices, slot_indices].unsqueeze(1),  # [N_valid, 1]
        ],
        dim=1,
    )  # [N_valid, 5]
    gt_seed_xyz = seed_xyz[0, seed_indices]  # [N_valid, 3]
    gt_scores_flat = gt_scores[seed_indices, slot_indices]  # [N_valid]
    return graspgroup_from_5d(gt_x, gt_seed_xyz, gt_scores_flat), gt_scores_flat


def survey_selected_valid_points(end_points, selected_indices):
    """Count selected seed points that have at least one valid ground-truth grasp."""
    gt_end_points = process_grasp_labels(end_points.copy())
    return survey_selected_valid_points_from_labels(gt_end_points, selected_indices)


def survey_selected_valid_points_from_labels(gt_end_points, selected_indices):
    """Count selected seed validity from precomputed CFM ground-truth labels."""
    selected_valid_mask = gt_end_points['seed_valid_mask'][0, selected_indices].bool()  # [K]
    selected_grasp_counts = gt_end_points['seed_grasp_count'][0, selected_indices].long()  # [K]

    selected_count = int(selected_indices.numel())
    valid_count = int(selected_valid_mask.sum().item())
    invalid_count = selected_count - valid_count
    valid_fraction = valid_count / selected_count if selected_count > 0 else 0.0

    valid_grasp_counts = selected_grasp_counts[selected_valid_mask]  # [K_valid]
    if valid_grasp_counts.numel() == 0:
        min_count = 0
        max_count = 0
        mean_count = 0.0
    else:
        min_count = int(valid_grasp_counts.min().item())
        max_count = int(valid_grasp_counts.max().item())
        mean_count = float(valid_grasp_counts.float().mean().item())

    return {
        'selected_count': selected_count,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'valid_fraction': valid_fraction,
        'min_valid_grasps': min_count,
        'max_valid_grasps': max_count,
        'mean_valid_grasps': mean_count,
    }


def make_open3d_cloud(points, colors=None):
    """Build an Open3D point cloud from numpy point and color arrays."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def main():
    parser = argparse.ArgumentParser(description="Visualize the latest CFM checkpoint from train_cfm.py")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet',
                        help='GraspNet dataset root')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'],
                        help='Camera type')
    parser.add_argument('--split', type=str, default='test_seen',
                        choices=['test_seen', 'test_similar', 'test_novel', 'train'],
                        help='Dataset split used to pick one scene')
    parser.add_argument('--scene_id', type=int, default=0, help='Scene id')
    parser.add_argument('--ann_id', type=int, default=0, help='Annotation/frame id')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to flowgrasp_latest.tar. If empty, auto-detect latest under flow/results/*/')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_seed5d_norm_stats.pt',
                        help='Path to fixed normalization metadata')
    parser.add_argument('--base_checkpoint_path', type=str,
                        default='checkpoints/economicgrasp_realsense.tar',
                        help='Path to base economicgrasp checkpoint')
    parser.add_argument('--n_steps', type=int, default=20, help='Euler steps for inference')
    parser.add_argument('--nsample', type=int, default=32, help='Neighborhood size for seed conditioner')
    parser.add_argument('--sphere_radius', type=float, default=0.005, help='Seed grouping radius in meters')
    parser.add_argument(
        '--collision_thresh',
        type=float,
        default=cfgs.collision_thresh,
        help='Collision threshold before graspness filtering. Use 0.01 to enable; <=0 disables it.'
    )
    parser.add_argument('--vis_every', type=int, default=0,
                        help='Deprecated; step-by-step trajectory visualization is skipped.')
    parser.add_argument('--smoke_test', action='store_true', help='Only run inference and skip Open3D window')
    args = parser.parse_args()

    checkpoint_path = resolve_latest_checkpoint(args.checkpoint_path)
    print(f"Using CFM checkpoint: {checkpoint_path}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading dataset split {args.split}...", flush=True)
    dataset = GraspNetDataset(
        args.dataset_root,
        split=args.split,
        camera=args.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        load_label=True,
        augment=False
    )

    scene_idx = None
    for i, scene_name in enumerate(dataset.sceneIds):
        if scene_name == f"scene_{args.scene_id:04d}":
            scene_idx = i
            break
    if scene_idx is None:
        expected_split = expected_split_for_scene(args.scene_id)
        available = f"{dataset.sceneIds[0]}..{dataset.sceneIds[-1]}"
        if expected_split is None:
            raise FileNotFoundError(f"scene_{args.scene_id:04d} is outside GraspNet scene range [0000, 0189].")
        raise FileNotFoundError(
            f"scene_{args.scene_id:04d} is in split '{expected_split}', not '{args.split}'. "
            f"Current split contains {available}."
        )
    if not (0 <= args.ann_id < 256):
        raise ValueError("--ann_id must be in [0, 255]")
    data_idx = scene_idx * 256 + args.ann_id
    print(f"Resolved dataset index: {data_idx} (scene_idx={scene_idx}, frame={args.ann_id})", flush=True)

    print("Loading scene point cloud and raw batch...", flush=True)
    raw_data = dataset[data_idx]
    batch_data = collate_fn([raw_data])
    batch_data = to_device(batch_data, device)

    cloud_points, cloud_colors = dataset.get_data(data_idx, return_raw_cloud=True)
    cloud = make_open3d_cloud(cloud_points, cloud_colors)
    geoms = [cloud]

    print(f"Loading base checkpoint: {args.base_checkpoint_path}", flush=True)
    base_net = economicgrasp(seed_feat_dim=512, is_training=False).to(device)
    base_ckpt = torch.load(args.base_checkpoint_path, map_location=device)
    base_net.load_state_dict(base_ckpt['model_state_dict'], strict=False)
    base_net.eval()

    print("Extracting seed features...", flush=True)
    with torch.no_grad():
        seed_xyz, seed_feats, seed_graspness, seed_object_ids = extract_scene_inputs(base_net, batch_data)

    print("Loading latest CFM checkpoint weights...", flush=True)
    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=args.nsample,
        seed_feature_dim=512,
        sphere_radius=args.sphere_radius,
    ).to(device)
    mlp = GraspVelocityMLP(grasp_dim=5, cond_dim=128).to(device)
    cfm_ckpt = torch.load(checkpoint_path, map_location=device)
    seed_conditioner.load_state_dict(cfm_ckpt['seed_conditioner_state_dict'])
    mlp.load_state_dict(cfm_ckpt['mlp_state_dict'])
    seed_conditioner.eval()
    mlp.eval()

    norm_metadata = torch.load(args.stats_path, map_location=device)

    num_seed = seed_xyz.shape[1]
    x0 = torch.randn(1, num_seed, 5, device=device)
    print(f"Running Euler solver for {args.n_steps} steps...", flush=True)
    with torch.no_grad():
        x_pred = euler_solve(
            seed_conditioner, mlp, x0, seed_xyz, seed_feats,
            norm_metadata, n_steps=args.n_steps,
        )  # [B, N, 5]

    keep_mask = build_collision_keep_mask(
        x_pred[0],
        seed_xyz[0],
        cloud_points,
        device,
        args.collision_thresh,
    )
    if args.collision_thresh > 0:
        print(
            f"Collision filtering kept {int(keep_mask.sum().item())}/{keep_mask.numel()} seed grasps "
            f"(threshold={args.collision_thresh}).",
            flush=True,
        )

    collision_free_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # [N_keep]
    collision_free_scores = seed_graspness[0, collision_free_indices]  # [N_keep]
    collision_free_geoms = []
    if collision_free_indices.numel() > 0:
        collision_free_gg = graspgroup_from_5d(
            x_pred[0, collision_free_indices],
            seed_xyz[0, collision_free_indices],
            collision_free_scores,
        )
        collision_free_geoms = collision_free_gg.to_open3d_geometry_list()
        collision_free_colors = scores_to_colors(collision_free_scores)
        for geom, color in zip(collision_free_geoms, collision_free_colors):
            geom.paint_uniform_color(color.tolist())
        print(
            f"Prepared collision-free all-seed visualization for {len(collision_free_geoms)} grasps "
            f"with score range "
            f"[{float(collision_free_scores.min().item()):.4f}, "
            f"{float(collision_free_scores.max().item()):.4f}].",
            flush=True,
        )
    else:
        print("Prepared collision-free all-seed visualization with 0 grasps.", flush=True)

    batch_data['xyz_graspable'] = seed_xyz
    gt_end_points = process_grasp_labels(batch_data.copy())
    best_gt_all_seed_gg, best_gt_all_seed_scores = build_best_gt_graspgroup_for_all_seeds(
        gt_end_points,
        seed_xyz,
    )
    best_gt_all_seed_geoms = []
    if best_gt_all_seed_gg is None:
        print("Could not find any valid ground-truth grasps for the 1024 seed points.", flush=True)
    else:
        best_gt_all_seed_geoms = best_gt_all_seed_gg.to_open3d_geometry_list()
        best_gt_all_seed_colors = scores_to_colors(best_gt_all_seed_scores)
        print(
            f"Prepared best-GT-per-seed visualization for {len(best_gt_all_seed_geoms)} "
            f"valid seed points out of 1024 "
            f"with score range "
            f"[{float(best_gt_all_seed_scores.min().item()):.4f}, "
            f"{float(best_gt_all_seed_scores.max().item()):.4f}].",
            flush=True,
        )
        for geom, color in zip(best_gt_all_seed_geoms, best_gt_all_seed_colors):
            geom.paint_uniform_color(color.tolist())

    valid_object_count = torch.unique(seed_object_ids[0][keep_mask & (seed_object_ids[0] > 0)]).numel()
    selected_indices = select_top_seed_indices(
        seed_graspness[0],
        seed_object_ids[0],
        keep_mask,
        max_per_object=10,
        total_top_k=50,
    )
    if valid_object_count == 0:
        print(
            "Top-50 selection could not find valid object ids after collision filtering; "
            "falling back to global top grasps by graspness among collision-free seeds.",
            flush=True,
        )
    else:
        print(
            f"Top-50 selection kept {selected_indices.numel()} grasps from "
            f"{valid_object_count} objects after collision filtering.",
            flush=True,
        )

    if selected_indices.numel() == 0:
        print("No collision-free seed grasps remained for visualization.", flush=True)
        if not args.smoke_test:
            print("Opening collision-free all-seed predicted-grasp Open3D window...", flush=True)
            o3d.visualization.draw_geometries(
                geoms + collision_free_geoms,
                window_name=f"Collision-free CFM grasps for 1024 seeds - scene {args.scene_id:04d}"
            )
            print("Opening best-GT-per-seed Open3D window...", flush=True)
            o3d.visualization.draw_geometries(
                geoms + best_gt_all_seed_geoms,
                window_name=f"Best GT grasp per valid seed - scene {args.scene_id:04d}"
            )
        return

    selected_valid_stats = survey_selected_valid_points_from_labels(gt_end_points, selected_indices)
    print(
        "Selected seed validity survey: "
        f"valid_points={selected_valid_stats['valid_count']}/"
        f"{selected_valid_stats['selected_count']} "
        f"({selected_valid_stats['valid_fraction']:.2%}), "
        f"invalid_points={selected_valid_stats['invalid_count']}, "
        f"valid_grasps_per_valid_point="
        f"min/mean/max="
        f"{selected_valid_stats['min_valid_grasps']}/"
        f"{selected_valid_stats['mean_valid_grasps']:.2f}/"
        f"{selected_valid_stats['max_valid_grasps']}.",
        flush=True,
    )

    pred = x_pred[0, selected_indices]  # [K, 5]
    pred_scores = seed_graspness[0, selected_indices]  # [K]
    pred_gg = graspgroup_from_5d(pred, seed_xyz[0, selected_indices], pred_scores)  # [K, 3]
    pred_geoms = pred_gg.to_open3d_geometry_list()
    pred_colors = scores_to_colors(pred_scores)
    print(
        f"Visualizing {selected_indices.numel()} grasps with score range "
        f"[{float(pred_scores.min().item()):.4f}, {float(pred_scores.max().item()):.4f}].",
        flush=True,
    )
    for geom, color in zip(pred_geoms, pred_colors):
        geom.paint_uniform_color(color.tolist())

    gt_gg, gt_scores = build_best_gt_graspgroup(batch_data, selected_indices, seed_xyz)
    gt_geoms = []
    if gt_gg is None:
        print("Could not find valid ground-truth grasps for the selected seed points.", flush=True)
    else:
        gt_geoms = gt_gg.to_open3d_geometry_list()
        gt_colors = scores_to_colors(gt_scores)
        print(
            f"Ground-truth window uses {len(gt_geoms)} best seed-matched grasps with score range "
            f"[{float(gt_scores.min().item()):.4f}, {float(gt_scores.max().item()):.4f}].",
            flush=True,
        )
        for geom, color in zip(gt_geoms, gt_colors):
            geom.paint_uniform_color(color.tolist())

    if args.smoke_test:
        print(
            f"Smoke test done. Prepared {len(collision_free_geoms)} collision-free all-seed grasps, "
            f"{len(best_gt_all_seed_geoms)} best-GT-per-seed grasps, "
            f"predicted {len(pred_geoms)} selected grasps, and matched "
            f"{len(gt_geoms)} ground-truth grasps from {checkpoint_path}",
            flush=True,
        )
        return

    print("Opening collision-free all-seed predicted-grasp Open3D window...", flush=True)
    o3d.visualization.draw_geometries(
        geoms + collision_free_geoms,
        window_name=f"Collision-free CFM grasps for 1024 seeds - scene {args.scene_id:04d}"
    )

    print("Opening best-GT-per-seed Open3D window...", flush=True)
    o3d.visualization.draw_geometries(
        geoms + best_gt_all_seed_geoms,
        window_name=f"Best GT grasp per valid seed - scene {args.scene_id:04d}"
    )

    print("Opening predicted-grasp Open3D window...", flush=True)
    o3d.visualization.draw_geometries(
        geoms + pred_geoms,
        window_name=f"Predicted top-50 CFM grasps - scene {args.scene_id:04d}"
    )

    print("Opening ground-truth Open3D window...", flush=True)
    o3d.visualization.draw_geometries(
        geoms + gt_geoms,
        window_name=f"Best GT grasps for selected seeds - scene {args.scene_id:04d}"
    )


if __name__ == '__main__':
    main()
