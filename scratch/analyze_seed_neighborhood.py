import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.label_generation import process_grasp_labels

def main():
    print("=" * 80)
    print("  QUANTITATIVE DATASET ANALYSIS: SEALSENSE SEED POINT NEIGHBORHOOD")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = 'checkpoints/economicgrasp_realsense.tar'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        return

    # 1. Initialize dataset for realsense
    print("\n[1] Loading Realsense dataset...")
    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    
    # We will analyze 10 scenes to get robust statistics
    indices = list(range(min(10, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"    Loaded {len(subset)} training frames.")

    # 2. Load Base Model
    print("\n[2] Loading economicgrasp base model with realsense weights...")
    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("    Model loaded successfully.")

    # 3. Statistics collectors
    dist_list = []
    width_list = []
    depth_list = []
    score_list = []

    print("\n[3] Running frames and collecting distance statistics...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Analyzing Frames")):
            # Move data to device
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            # Get predicted seed points (xyz_graspable) and graspness score
            end_points = net(batch_data)
            xyz_graspable = end_points['xyz_graspable'][0]  # [1024, 3]
            
            # Let's extract the graspness score for the 1024 seeds
            graspness_score = end_points['graspness_score'][0]  # [20000]
            # Get seed indices to find their graspness
            # Since the FPS points are gathered, we can find the top 50 points directly
            # Or we can sort the 1024 seed points by their graspness to pick the top 50
            # To be simple and robust: we check all 1024 seed points
            
            # Load GT labels mapped to these 1024 seed points
            batch_grasp_views_rot, end_points = process_grasp_labels(end_points)
            
            # GT parameters for each of the 1024 seed points
            gt_points = end_points['batch_grasp_point'][0]  # [1024, 3] or similar (mapped GT point)
            gt_widths = end_points['batch_grasp_width'][0]   # [1024]
            gt_depths = end_points['batch_grasp_depth'][0]   # [1024]
            gt_scores = end_points['batch_grasp_score'][0]   # [1024]
            valid_mask = end_points['batch_valid_mask'][0]   # [1024]

            # We only care about points where valid_mask == 1
            valid_idx = torch.where(valid_mask == 1)[0]
            if len(valid_idx) == 0:
                continue

            # Gather valid data
            seed_xyz_valid = xyz_graspable[valid_idx] # [num_valid, 3]
            gt_points_valid = gt_points[valid_idx]     # [num_valid, 3]
            gt_widths_valid = gt_widths[valid_idx]     # [num_valid]
            gt_depths_valid = gt_depths[valid_idx]     # [num_valid]
            gt_scores_valid = gt_scores[valid_idx]     # [num_valid]

            # Compute L2 distance between predicted seed point and its mapped GT grasp center
            dists = torch.norm(seed_xyz_valid - gt_points_valid, p=2, dim=1) # [num_valid]

            dist_list.extend(dists.cpu().numpy().tolist())
            width_list.extend(gt_widths_valid.cpu().numpy().tolist())
            # Convert depth index (1-4) to meters: (idx + 1) * 0.01
            depths_m = (gt_depths_valid.to(torch.float32) + 1.0) * 0.01
            depth_list.extend(depths_m.cpu().numpy().tolist())
            score_list.extend(gt_scores_valid.cpu().numpy().tolist())

    # 4. Print results
    dist_np = np.array(dist_list)
    width_np = np.array(width_list)
    depth_np = np.array(depth_list)
    score_np = np.array(score_list)

    print("\n" + "=" * 50)
    print("  QUANTITATIVE STATISTICS REPORT")
    print("=" * 50)
    
    print(f"Total valid grasp points analyzed: {len(dist_np)}")
    
    print("\n[1] Translation Distance (Seed Point to GT Grasp Center):")
    print(f"    Mean ± Std : {np.mean(dist_np):.4f} ± {np.std(dist_np):.4f} meters")
    print(f"    Median     : {np.median(dist_np):.4f} meters")
    print(f"    90% / 95%  : {np.percentile(dist_np, 90):.4f} / {np.percentile(dist_np, 95):.4f} meters")
    print(f"    Min / Max  : {np.min(dist_np):.4f} / {np.max(dist_np):.4f} meters")
    print(f"    Variance   : {np.var(dist_np):.8f} (meters^2)")

    print("\n[2] Gripper Width (Valid Grasps):")
    print(f"    Mean ± Std : {np.mean(width_np):.4f} ± {np.std(width_np):.4f} meters")
    print(f"    Min / Max  : {np.min(width_np):.4f} / {np.max(width_np):.4f} meters")

    print("\n[3] Gripper Depth (Valid Grasps):")
    print(f"    Mean ± Std : {np.mean(depth_np):.4f} ± {np.std(depth_np):.4f} meters")
    print(f"    Min / Max  : {np.min(depth_np):.4f} / {np.max(depth_np):.4f} meters")

    print("\n[4] Grasp Score (Friction Coefficient Quality):")
    print(f"    Mean ± Std : {np.mean(score_np):.4f} ± {np.std(score_np):.4f}")
    print(f"    Min / Max  : {np.min(score_np):.4f} / {np.max(score_np):.4f}")

if __name__ == '__main__':
    main()
