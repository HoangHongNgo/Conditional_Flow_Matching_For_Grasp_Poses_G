import sys
import os
import torch
import random
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp
from utils.arguments import cfgs
from torch.utils.data import DataLoader

def check_valid_mask():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize dataset
    print("Initializing dataset...")
    dataset = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                              voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, 
                              remove_outlier=True, augment=False)
    
    dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, shuffle=True, 
                            num_workers=2, collate_fn=collate_fn)

    # 2. Initialize model (must be in training mode to generate labels/valid_mask)
    print("Initializing model...")
    model = economicgrasp(seed_feat_dim=512, is_training=True).to(device)
    
    # Try to load pretrained weights to get realistic predictions
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints/economicgrasp_kinect.tar')
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
    else:
        print(f"Pretrained weights not found at {ckpt_path}. Model predictions will be random.")

    model.train() # Must be in train mode

    # 3. Check random scenes
    num_scenes_to_check = 3
    print(f"\nChecking valid_mask for {num_scenes_to_check} random batches...\n")

    for i, batch in enumerate(dataloader):
        if i >= num_scenes_to_check:
            break
            
        print(f"--- Batch {i+1} ---")
        
        # Move batch to device
        for key in batch:
            if 'list' in key:
                for b_idx in range(len(batch[key])):
                    for obj_idx in range(len(batch[key][b_idx])):
                        batch[key][b_idx][obj_idx] = batch[key][b_idx][obj_idx].to(device)
            else:
                batch[key] = batch[key].to(device)
                
        # Forward pass (this triggers process_grasp_labels and generates valid_mask)
        with torch.no_grad(): # We don't need gradients just to check the mask
            end_points = model(batch)
            
        valid_mask = end_points['batch_valid_mask'] # Shape: [B, 1024]
        valid_point_mask = end_points.get('batch_valid_point_mask', None)
        valid_view_mask = end_points.get('batch_valid_view_mask', None)
        
        batch_size, num_seed_points = valid_mask.shape
        
        for b in range(batch_size):
            mask_b = valid_mask[b]
            num_valid = torch.sum(mask_b).item()
            num_invalid = num_seed_points - num_valid
            percentage = (num_valid / num_seed_points) * 100
            
            print(f"  Scene {b+1} in batch:")
            print(f"    Total Seed Points: {num_seed_points}")
            print(f"    Valid Points     : {num_valid} ({percentage:.2f}%)")
            print(f"    Invalid Points   : {num_invalid}")
            
            if valid_point_mask is not None and valid_view_mask is not None:
                mask_p_b = valid_point_mask[b]
                mask_v_b = valid_view_mask[b]
                
                num_fail_dist = num_seed_points - torch.sum(mask_p_b).item()
                num_fail_view = num_seed_points - torch.sum(mask_v_b).item()
                num_fail_both = torch.sum(~mask_p_b & ~mask_v_b).item()
                num_fail_dist_only = num_fail_dist - num_fail_both
                num_fail_view_only = num_fail_view - num_fail_both
                
                print(f"      -> Fail due to DISTANCE (>5mm from GT): {num_fail_dist_only}")
                print(f"      -> Fail due to VIEW (Pred view not in Top 60): {num_fail_view_only}")
                print(f"      -> Fail due to BOTH (Distance & View): {num_fail_both}")
                
                batch_dist = end_points.get('batch_dist', None)
                if batch_dist is not None:
                    dist_b = batch_dist[b]
                    fail_dist_points = dist_b[~mask_p_b]
                    if len(fail_dist_points) > 0:
                        min_d = fail_dist_points.min().item() * 1000 # to mm
                        max_d = fail_dist_points.max().item() * 1000 # to mm
                        mean_d = fail_dist_points.mean().item() * 1000 # to mm
                        median_d = fail_dist_points.median().item() * 1000 # to mm
                        print(f"      => Distance Stats for invalid points: Min={min_d:.2f}mm, Max={max_d:.2f}mm, Mean={mean_d:.2f}mm, Median={median_d:.2f}mm")
            
            # Print breakdown of why they might be invalid (using valid points count from end_points)
            # D: Graspable Points is the ratio of graspable points over 20000
            # C: Valid Points is the ratio of valid points in the 1024 sample
            if b == 0:
                 print(f"    (Avg Valid points across batch reported by model: {end_points.get('C: Valid Points', 0):.2f})")
        print()

if __name__ == '__main__':
    check_valid_mask()
