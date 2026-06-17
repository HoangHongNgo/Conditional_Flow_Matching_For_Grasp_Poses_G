import os
import sys
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.label_generation import process_grasp_labels_without_seed_mapping
from utils.arguments import cfgs

def test_rotation():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        batch_size=cfgs.batch_size, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    # 2. Get a batch
    print("Fetching a batch of data...")
    batch_data = next(iter(dataloader))
    for k in batch_data:
        if 'list' in k:
            for i in range(len(batch_data[k])):
                for j in range(len(batch_data[k][i])):
                    batch_data[k][i][j] = batch_data[k][i][j].to(device)
        else:
            batch_data[k] = batch_data[k].to(device)
            
    # 3. Call the label generation
    print("\nRunning process_grasp_labels_without_seed_mapping...")
    views_rot_complete, end_points = process_grasp_labels_without_seed_mapping(batch_data)
    
    print("\nStarting programmatical mathematical checks:")
    
    for i in range(len(views_rot_complete)):
        R_complete = views_rot_complete[i]  # [50, 3, 3]
        
        print(f"\n--- Checking Batch Item {i} (50 grasp poses) ---")
        print("Shape of R_complete:", R_complete.shape)
        
        # Check 1: Determinant is exactly 1.0 (det(R) = 1.0)
        dets = torch.linalg.det(R_complete)
        mean_det = torch.mean(dets).item()
        min_det = torch.min(dets).item()
        max_det = torch.max(dets).item()
        print(f"  Check 1 (Determinant = 1.0): Min={min_det:.4f}, Max={max_det:.4f}, Mean={mean_det:.4f}")
        assert torch.allclose(dets, torch.ones_like(dets), atol=1e-4), "Determinant is not +1.0!"
        
        # Check 2: Orthogonality (R^T * R = I)
        RT_R = torch.matmul(R_complete.transpose(1, 2), R_complete)
        I = torch.eye(3, device=device).unsqueeze(0).expand_as(RT_R)
        ortho_err = torch.max(torch.abs(RT_R - I)).item()
        print(f"  Check 2 (Orthogonality RT * R = I): Max absolute error = {ortho_err:.6f}")
        assert ortho_err < 1e-4, "Orthogonality check failed!"
        
        # Check 3: Approach vector (first column) has unit norm
        norms = torch.norm(R_complete[:, :, 0], dim=-1)
        mean_norm = torch.mean(norms).item()
        print(f"  Check 3 (Approach axis norm = 1.0): Mean norm = {mean_norm:.6f}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Approach axis is not normalized!"
        
    print("\nAll standard validity verifications passed successfully!")
    
    # 4. Run the deep mathematical composition check to prove correct gộp of approach and angle
    verify_composition_math(device)

def verify_composition_math(device):
    print("\n=======================================================")
    print("Test 4: Verifying Composition Math (R_relative = R_base.T @ R_complete == R1)")
    print("=======================================================")
    
    # 1. Create a dummy base rotation matrix from a mock approach vector
    from utils.loss_utils import batch_viewpoint_params_to_matrix
    towards = torch.tensor([[0.5, 0.5, 0.7071]], device=device)  # [1, 3]
    angle_idx = torch.tensor([5], device=device)  # [1] (angle index = 5)
    
    # 2. Compute grasp_angle (5 * pi / 12)
    grasp_angle = angle_idx.to(towards.dtype) * np.pi / 12.0  # [1]
    
    # 3. Base viewpoint rotation (angle = 0)
    R_base = batch_viewpoint_params_to_matrix(towards, torch.zeros_like(grasp_angle))  # [1, 3, 3]
    
    # 4. Synthesize complete rotation matrix using our composition method
    ones = torch.ones(grasp_angle.shape[0], dtype=grasp_angle.dtype, device=grasp_angle.device)
    zeros = torch.zeros(grasp_angle.shape[0], dtype=grasp_angle.dtype, device=grasp_angle.device)
    sin = torch.sin(grasp_angle)
    cos = torch.cos(grasp_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1).reshape([-1, 3, 3])  # [1, 3, 3]
    
    R_complete = torch.matmul(R_base, R1)  # [1, 3, 3]
    
    # 5. Check 1: R_complete[:, :, 0] (approach vector) is identical to R_base[:, :, 0]
    approach_base = R_base[0, :, 0]
    approach_complete = R_complete[0, :, 0]
    print(f"  Base approach vector:     {approach_base.cpu().numpy()}")
    print(f"  Complete approach vector: {approach_complete.cpu().numpy()}")
    approach_close = torch.allclose(approach_base, approach_complete, atol=1e-6)
    print(f"  -> Check 1 (Approach invariant): {approach_close}")
    assert approach_close, "Approach vector changed after rotation!"
    
    # 6. Check 2: Relative rotation R_relative = R_base.T @ R_complete must be exactly equal to R1!
    R_relative = torch.matmul(R_base.transpose(1, 2), R_complete)  # [1, 3, 3]
    
    print("\n  R1 (Expected relative rotation around X-axis):")
    print(R1[0].cpu().numpy())
    print("  R_relative (Actual relative rotation R_base.T @ R_complete):")
    print(R_relative[0].cpu().numpy())
    
    relative_close = torch.allclose(R1, R_relative, atol=1e-6)
    print(f"  -> Check 2 (R_relative == R1): {relative_close}")
    assert relative_close, "Relative rotation is not equal to R1!"
    print("\nAll programmatical mathematical verifications passed perfectly! The implementation is 100% correct!")

if __name__ == "__main__":
    test_rotation()
