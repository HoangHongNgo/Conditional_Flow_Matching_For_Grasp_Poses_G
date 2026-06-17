"""
Per-scene breakdown: how many grasp configs (point × view_slot) with score >= 0.7
exist within 5mm of each seed point.
Reports statistics for each scene separately.
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
RADIUS_MM = 5.0
SCORE_THRESH = 0.7


def main():
    """Per-scene report of grasp config counts within 5mm, score >= 0.7."""
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

    print(f"Radius = {RADIUS_MM} mm, Score threshold >= {SCORE_THRESH}")
    print(f"Surveying {NUM_FRAMES} scenes ...\n")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Scenes")):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            seed_xyz = end_points['xyz_graspable'][0]  # [1024, 3]
            scene_name = dataset.scenename[batch_idx]
            frame_id = dataset.frameid[batch_idx]

            # Merge GT labels
            poses = batch_data['object_poses_list'][0]
            all_pts, all_scores = [], []
            for obj_idx, pose in enumerate(poses):
                pts = batch_data['grasp_points_list'][0][obj_idx]   # [P, 3]
                scs = batch_data['grasp_scores_list'][0][obj_idx]   # [P, 60]
                pts_trans = torch.matmul(pts, pose[:3, :3].t()) + pose[:3, 3]
                all_pts.append(pts_trans)
                all_scores.append(scs)

            gt_pts = torch.cat(all_pts, dim=0)       # [M, 3]
            gt_scores = torch.cat(all_scores, dim=0)  # [M, 60]
            M = gt_pts.shape[0]

            # Distance: [1024, M]
            dists_mm = torch.cdist(seed_xyz.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0) * 1000.0

            counts = []  # per-seed config count
            for s_idx in range(seed_xyz.shape[0]):
                in_radius = dists_mm[s_idx] <= RADIUS_MM  # [M]
                if in_radius.sum() == 0:
                    counts.append(0)
                    continue
                nb_scores = gt_scores[in_radius]  # [num_nb, 60]
                num_configs = (nb_scores >= SCORE_THRESH).sum().item()
                counts.append(num_configs)

            c = np.array(counts)

            print(f"\n{'='*60}")
            print(f"  {scene_name} / frame {frame_id}")
            print(f"  Total GT grasp points in scene: {M}")
            print(f"  Seed points: {seed_xyz.shape[0]}")
            print(f"{'='*60}")
            print(f"  Grasp configs (point × view) with score >= {SCORE_THRESH} within {RADIUS_MM}mm:")
            print(f"    Mean   : {c.mean():.1f}")
            print(f"    Median : {np.median(c):.0f}")
            print(f"    Std    : {c.std():.1f}")
            print(f"    Min    : {c.min()}")
            print(f"    Max    : {c.max()}")
            print(f"    10%/25%/75%/90% : {np.percentile(c,10):.0f} / {np.percentile(c,25):.0f} / {np.percentile(c,75):.0f} / {np.percentile(c,90):.0f}")
            print(f"    Seeds with 0 configs   : {(c == 0).sum()} ({(c == 0).mean()*100:.1f}%)")
            print(f"    Seeds with >= 1 config : {(c >= 1).sum()} ({(c >= 1).mean()*100:.1f}%)")
            print(f"    Seeds with >= 3 configs: {(c >= 3).sum()} ({(c >= 3).mean()*100:.1f}%)")
            print(f"    Seeds with >= 10 configs: {(c >= 10).sum()} ({(c >= 10).mean()*100:.1f}%)")
            print(f"    Seeds with >= 30 configs: {(c >= 30).sum()} ({(c >= 30).mean()*100:.1f}%)")

            # Histogram buckets
            buckets = [0, 1, 3, 10, 30, 50, 100, 200, 500]
            print(f"    Distribution:")
            for b_idx in range(len(buckets) - 1):
                lo, hi = buckets[b_idx], buckets[b_idx + 1]
                frac = ((c >= lo) & (c < hi)).mean() * 100
                print(f"      [{lo:4d}, {hi:4d}): {frac:5.1f}%")
            frac_last = (c >= buckets[-1]).mean() * 100
            print(f"      [{buckets[-1]:4d},  inf): {frac_last:5.1f}%")


if __name__ == '__main__':
    main()
