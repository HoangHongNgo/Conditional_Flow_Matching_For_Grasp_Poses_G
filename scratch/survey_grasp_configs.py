"""
Count individual grasp CONFIGURATIONS (point × view_slot) with score >= threshold
around each seed point. This is the correct unit for selecting grasps.

Each GT grasp point has 60 view slots. Each slot is a separate grasp configuration
with its own (view_direction, rotation, depth, width, score).
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

NUM_FRAMES = 5
SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    """Count per-view-slot grasp configurations above score thresholds."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # stats[threshold][radius_mm] = list of per-seed counts
    radius_list = [5, 10, 20]
    stats = {t: {r: [] for r in radius_list} for t in SCORE_THRESHOLDS}

    print(f"Surveying {NUM_FRAMES} frames — counting individual (point, view) configs ...")
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

            # Merge GT points and per-view scores from all objects
            poses = batch_data['object_poses_list'][0]
            all_pts = []
            all_scores = []  # keep full [P, 60] scores, not just max

            for obj_idx, pose in enumerate(poses):
                pts = batch_data['grasp_points_list'][0][obj_idx]   # [P, 3]
                scs = batch_data['grasp_scores_list'][0][obj_idx]   # [P, 60], scale 0-1
                pts_trans = torch.matmul(pts, pose[:3, :3].t()) + pose[:3, 3]  # [P, 3]
                all_pts.append(pts_trans)
                all_scores.append(scs)

            gt_pts = torch.cat(all_pts, dim=0)       # [M, 3]
            gt_scores = torch.cat(all_scores, dim=0)  # [M, 60]

            # Distance matrix: [1024, M]
            dists = torch.cdist(seed_xyz.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
            dists_mm = dists * 1000.0

            for s_idx in range(seed_xyz.shape[0]):
                d = dists_mm[s_idx]  # [M]

                for r_mm in radius_list:
                    in_radius = d <= r_mm  # [M] bool
                    if in_radius.sum() == 0:
                        for t in SCORE_THRESHOLDS:
                            stats[t][r_mm].append(0)
                        continue

                    # Gather scores of neighbor points: [num_neighbors, 60]
                    nb_scores = gt_scores[in_radius]

                    for t in SCORE_THRESHOLDS:
                        # Count individual (point, view_slot) configs with score >= t
                        num_configs = (nb_scores >= t).sum().item()
                        stats[t][r_mm].append(num_configs)

    # ---- Report ----
    print("\n" + "=" * 75)
    print("  GRASP CONFIGURATIONS (point × view_slot) WITH score >= threshold")
    print("  around each seed point (averaged over 5 frames × 1024 seeds)")
    print("=" * 75)

    for t in SCORE_THRESHOLDS:
        print(f"\n--- Score >= {t:.1f} ---")
        for r_mm in radius_list:
            c = np.array(stats[t][r_mm])
            frac_ge3 = (c >= 3).mean() * 100
            frac_ge10 = (c >= 10).mean() * 100
            print(f"  r={r_mm:2d}mm: mean={c.mean():7.1f}, "
                  f"median={np.median(c):5.0f}, "
                  f"10%/90%={np.percentile(c,10):5.0f}/{np.percentile(c,90):5.0f}, "
                  f">=3: {frac_ge3:5.1f}%, >=10: {frac_ge10:5.1f}%")


if __name__ == '__main__':
    main()
