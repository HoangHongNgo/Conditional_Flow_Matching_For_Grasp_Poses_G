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

def analyze_seed_neighborhoods():
    """
    Load a trained model and dataset to analyze the neighborhood of predicted seed points.
    Specifically, find the number of valid ground-truth grasp points and viewpoints
    within various radii of each seed point, as well as the distribution of their scores.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = 'checkpoints/economicgrasp_realsense.tar'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        return

    # 1. Initialize dataset
    print("Loading Realsense train dataset...")
    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    
    # We will analyze 5 frames to obtain reliable averages
    num_frames = min(5, len(dataset))
    indices = list(range(num_frames))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 2. Load Model
    print("Loading model...")
    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Radii to check (in meters)
    radii = [0.005, 0.01, 0.02, 0.05] # 5mm, 10mm, 20mm, 50mm

    # Statistics collectors
    # For each radius, store a list of lists/tensors
    neighbor_counts = {r: [] for r in radii}
    valid_grasp_counts = {r: [] for r in radii}
    grasp_scores_by_radius = {r: [] for r in radii}

    print("Analyzing seed neighborhoods...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            # Forward pass to get seed points
            end_points = net(batch_data)
            xyz_graspable = end_points['xyz_graspable'][0] # [1024, 3] (tensor)
            num_seeds = xyz_graspable.shape[0]

            # Reconstruct merged ground truth labels across all objects in this scene
            poses = batch_data['object_poses_list'][0]
            grasp_points_list = batch_data['grasp_points_list'][0]
            grasp_scores_list = batch_data['grasp_scores_list'][0]

            # Merge and transform points to camera frame
            gt_points_trans = []
            gt_scores_merged = []
            for obj_idx, pose in enumerate(poses):
                pts = grasp_points_list[obj_idx] # [N_obj, 3]
                scs = grasp_scores_list[obj_idx] # [N_obj, 60]
                
                # Transform to scene frame
                # R * X + t
                pts_trans = torch.matmul(pts, pose[:3, :3].t()) + pose[:3, 3] # [N_obj, 3]
                gt_points_trans.append(pts_trans)
                gt_scores_merged.append(scs)

            if len(gt_points_trans) == 0:
                continue

            gt_points = torch.cat(gt_points_trans, dim=0) # [M, 3]
            gt_scores = torch.cat(gt_scores_merged, dim=0) # [M, 60] (M is total gt grasp points in scene)

            # Compute distance matrix between all 1024 seeds and all M ground truth points
            # Shape: [1024, M]
            dists = torch.cdist(xyz_graspable.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)

            # Analyze for each seed point
            for i in range(num_seeds):
                seed_dists = dists[i] # [M]
                
                for r in radii:
                    # Find ground truth points within radius r
                    neighbor_mask = seed_dists <= r # [M]
                    num_neighbors = torch.sum(neighbor_mask).item()
                    neighbor_counts[r].append(num_neighbors)

                    if num_neighbors > 0:
                        # Fetch scores of these neighbor grasp points
                        neighbor_scores = gt_scores[neighbor_mask] # [num_neighbors, 60]
                        # A grasp is valid if score > 0
                        valid_mask = neighbor_scores > 0 # [num_neighbors, 60]
                        num_valid_grasps = torch.sum(valid_mask).item()
                        valid_grasp_counts[r].append(num_valid_grasps)

                        # Collect valid scores
                        valid_scores = neighbor_scores[valid_mask] # [num_valid_grasps]
                        if len(valid_scores) > 0:
                            grasp_scores_by_radius[r].extend(valid_scores.cpu().numpy().tolist())
                    else:
                        valid_grasp_counts[r].append(0)

    # Print Report
    print("\n" + "=" * 60)
    print("  SEED NEIGHBORHOOD ANALYSIS REPORT (Averaged over 5 Scenes)")
    print("=" * 60)
    
    for r in radii:
        r_mm = r * 1000.0
        n_pts = np.array(neighbor_counts[r])
        n_grasps = np.array(valid_grasp_counts[r])
        scores_arr = np.array(grasp_scores_by_radius[r])

        print(f"\n--- Radius: {r_mm:.1f} mm ---")
        print(f"  GT grasp points in neighborhood:")
        print(f"    Average: {np.mean(n_pts):.2f} points")
        print(f"    Min/Max: {np.min(n_pts)} / {np.max(n_pts)} points")
        print(f"  Valid grasps (point-view configurations, score > 0) in neighborhood:")
        print(f"    Average: {np.mean(n_grasps):.2f} grasps")
        print(f"    Min/Max: {np.min(n_grasps)} / {np.max(n_grasps)} grasps")
        
        if len(scores_arr) > 0:
            print(f"  Grasp Quality (Scores scaled 0.0 - 1.0):")
            print(f"    Mean ± Std : {np.mean(scores_arr):.4f} ± {np.std(scores_arr):.4f}")
            print(f"    Median     : {np.median(scores_arr):.4f}")
            print(f"    Min/Max    : {np.min(scores_arr):.4f} / {np.max(scores_arr):.4f}")
        else:
            print(f"  No valid grasps found in this radius range.")

if __name__ == '__main__':
    analyze_seed_neighborhoods()
