"""
Survey how many high-quality grasps (score > 0.7) exist around each seed point,
and at what distances. This answers the feasibility question for the user's
desired selection strategy: "3 grasps with score > 0.7, nearest to seed, with randomness."

Scores in end_points['grasp_scores_list'] are already scaled to [0, 1]
(raw integer scores 0-10 divided by 10 in graspnet_dataset.py line 226).
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.loss_utils import generate_grasp_views, transform_point_cloud


NUM_FRAMES = 5
SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    """Survey high-score grasp availability around predicted seed points."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset and model
    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    subset = torch.utils.data.Subset(dataset, list(range(NUM_FRAMES)))
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    ckpt = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    # Per-threshold statistics collectors
    # For each threshold, store per-seed: (num_candidates, min_dist, mean_dist)
    stats = {t: {'count': [], 'min_dist_mm': [], 'mean_dist_mm': []} for t in SCORE_THRESHOLDS}
    # Also track: for 10mm and 20mm radius, how many candidates per threshold
    radius_stats = {}
    for r_mm in [5, 10, 20]:
        radius_stats[r_mm] = {t: [] for t in SCORE_THRESHOLDS}

    print(f"Surveying {NUM_FRAMES} frames ...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Frames")):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            seed_xyz = end_points['xyz_graspable'][0]  # [1024, 3]

            # Merge and transform GT grasp labels for this scene
            poses = batch_data['object_poses_list'][0]
            all_pts = []    # grasp point positions in scene frame
            all_scores = [] # best score per point (max over 60 view slots)

            for obj_idx, pose in enumerate(poses):
                pts = batch_data['grasp_points_list'][0][obj_idx]     # [P, 3]
                scs = batch_data['grasp_scores_list'][0][obj_idx]     # [P, 60], already 0-1
                # Transform points to scene frame
                pts_trans = torch.matmul(pts, pose[:3, :3].t()) + pose[:3, 3]  # [P, 3]
                # Best score across all 60 view slots for each point
                best_score = scs.max(dim=1)[0]  # [P]
                all_pts.append(pts_trans)
                all_scores.append(best_score)

            gt_pts = torch.cat(all_pts, dim=0)       # [M, 3]
            gt_best = torch.cat(all_scores, dim=0)    # [M]

            # Distance matrix: [1024, M]
            dists = torch.cdist(seed_xyz.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
            dists_mm = dists * 1000.0  # convert to mm

            for s_idx in range(seed_xyz.shape[0]):
                d = dists_mm[s_idx]  # [M] distances in mm

                for t in SCORE_THRESHOLDS:
                    mask = gt_best >= t  # [M]
                    count = mask.sum().item()
                    stats[t]['count'].append(count)

                    if count > 0:
                        d_valid = d[mask]
                        stats[t]['min_dist_mm'].append(d_valid.min().item())
                        stats[t]['mean_dist_mm'].append(d_valid.mean().item())
                    else:
                        stats[t]['min_dist_mm'].append(float('inf'))
                        stats[t]['mean_dist_mm'].append(float('inf'))

                    for r_mm in [5, 10, 20]:
                        in_radius = (mask & (d <= r_mm)).sum().item()
                        radius_stats[r_mm][t].append(in_radius)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("  HIGH-SCORE GRASP AVAILABILITY AROUND SEED POINTS")
    print("  (best score per GT point, across all 60 view slots)")
    print("=" * 70)

    for t in SCORE_THRESHOLDS:
        c = np.array(stats[t]['count'])
        md = np.array(stats[t]['min_dist_mm'])
        md_finite = md[np.isfinite(md)]

        print(f"\n--- Score threshold >= {t:.1f} ---")
        print(f"  GT points with best_score >= {t:.1f} (entire scene):")
        print(f"    Mean count : {c.mean():.1f},  Median: {np.median(c):.0f}")
        print(f"    Min / Max  : {c.min()} / {c.max()}")
        frac_any = (c > 0).mean() * 100
        print(f"    Seeds with at least 1: {frac_any:.1f}%")

        if len(md_finite) > 0:
            print(f"  Distance to nearest qualifying point:")
            print(f"    Mean: {md_finite.mean():.2f} mm,  Median: {np.median(md_finite):.2f} mm")
            print(f"    10% / 90%: {np.percentile(md_finite, 10):.2f} / {np.percentile(md_finite, 90):.2f} mm")

        for r_mm in [5, 10, 20]:
            rc = np.array(radius_stats[r_mm][t])
            frac_ge3 = (rc >= 3).mean() * 100
            print(f"  Within {r_mm:2d} mm: mean={rc.mean():.1f}, "
                  f"median={np.median(rc):.0f}, "
                  f">=3 candidates: {frac_ge3:.1f}%")

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)


if __name__ == '__main__':
    main()
