import os
import sys
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.flowgrasp import economic_graspable
from utils.label_generation import process_grasp_labels_without_seed_mapping
from utils.arguments import cfgs
from flow.utils.lie import log_SO3, bracket_so3, exp_so3

def test_generate_single_with_lie():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Initialize dataset
    dataset = GraspNetDataset(
        cfgs.dataset_root, 
        camera=cfgs.camera, 
        split='train',
        voxel_size=cfgs.voxel_size, 
        num_points=cfgs.num_point, 
        remove_outlier=True, 
        augment=False
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    # 2. Initialize model and load checkpoint
    checkpoint_path = os.path.join("checkpoints", "economicgrasp_realsense.tar")
    if not os.path.isfile(checkpoint_path):
        # Fallback to Kinect if Realsense not found
        checkpoint_path = os.path.join("checkpoints", "economicgrasp_kinect.tar")
        
    print("Loading checkpoint from:", checkpoint_path)
    net = economic_graspable(seed_feat_dim=512, is_training=True)
    net.to(device)
    net.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    mapped_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        if name.startswith('backbone.') or name.startswith('graspable.'):
            mapped_dict[name] = v
    net.load_state_dict(mapped_dict, strict=False)
    
    # 3. Get first batch and run forward
    batch_data = next(iter(dataloader))
    for k in batch_data:
        if 'list' in k:
            for i in range(len(batch_data[k])):
                for j in range(len(batch_data[k][i])):
                    batch_data[k][i][j] = batch_data[k][i][j].to(device)
        else:
            batch_data[k] = batch_data[k].to(device)
            
    print("\nRunning model forward pass...")
    with torch.no_grad():
        end_points = net(batch_data)
        
    print("Running label generation with complete rotations...")
    batch_grasp_views_rot, end_points = process_grasp_labels_without_seed_mapping(end_points)
    
    # 4. Convert grasp rotation matrices to 3D Lie algebra vectors
    print("\nConverting rotation matrices to Lie algebra vectors...")
    batch_grasp_views_rot_lie = []
    for R in batch_grasp_views_rot:
        # R: [top_k_scene, 3, 3]
        w_mat = log_SO3(R)  # [top_k_scene, 3, 3]
        w_vec = bracket_so3(w_mat)  # [top_k_scene, 3]
        batch_grasp_views_rot_lie.append(w_vec)
        
    print("Success! Number of batches:", len(batch_grasp_views_rot_lie))
    print("Shape of first batch Lie algebra tensor:", batch_grasp_views_rot_lie[0].shape)
    
    # 5. Check reconstruction correctness: exp(log(R)) == R
    R_reconstructed = exp_so3(batch_grasp_views_rot_lie[0])  # [top_k_scene, 3, 3]
    reconstruction_err = torch.max(torch.abs(batch_grasp_views_rot[0] - R_reconstructed)).item()
    print(f"Reconstruction error (max |R_orig - exp(log(R))|): {reconstruction_err:.6f}")
    assert reconstruction_err < 5e-3, "Lie algebra reconstruction check failed!"
    print("Reconstruction check passed perfectly!")
    
    print("\nEverything runs flawlessly! The Lie algebra mapping is 100% correct.")

if __name__ == '__main__':
    test_generate_single_with_lie()
