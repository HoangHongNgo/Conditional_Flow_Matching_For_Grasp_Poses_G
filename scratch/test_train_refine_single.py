import os
import torch
import torch.optim as optim
import time

import sys
import os

# Add the project root to sys.path so we can import from 'utils', 'models', etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp, pred_decode
from models.flowgrasp import ConditionalFlowMatching
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from train_refine import construct_pred_grasp_pose, construct_gt_grasp_pose

# Setup configs for single scene test
CHECKPOINT_PATH = 'checkpoints/economicgrasp_kinect.tar'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCENE_IDX = 0  # Test on the very first scene

def main():
    print("="*60)
    print("SINGLE SCENE DEBUG RUN - CONDITIONAL FLOW MATCHING")
    print("="*60)
    
    # 1. Load Dataset (1 scene only, no augment for deterministic debug)
    print(f"\n[1] Loading Scene {SCENE_IDX}...")
    dataset = GraspNetDataset(
        cfgs.dataset_root, camera=cfgs.camera, split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    data = dataset[SCENE_IDX]
    batch_data = collate_fn([data])
    
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(DEVICE)
        else:
            batch_data[key] = batch_data[key].to(DEVICE)
    print("    Scene loaded to device.")

    # 2. Load Base Model
    print("\n[2] Loading frozen base model...")
    base_net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
    base_net.to(DEVICE)
    base_net.eval()
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    for param in base_net.parameters():
        param.requires_grad = False
    print("    Base model loaded and frozen.")

    # 3. Load CFM Model
    print("\n[3] Initializing CFM Model...")
    cfm_net = ConditionalFlowMatching(in_dim=11, out_dim=5, cond_dim=256, hidden_dim=256, time_dim=64, num_blocks=4)
    cfm_net.to(DEVICE)
    cfm_net.train()
    
    optimizer = optim.Adam(cfm_net.parameters(), lr=0.001)
    
    # Print out projection weights before step to verify gradients
    param_sample = cfm_net.output_proj[-1].weight
    print(f"    Initial output projection norm: {param_sample.norm().item()}")

    # 4. Forward Base Model
    print("\n[4] Forwarding base model...")
    with torch.no_grad():
        end_points = base_net(batch_data)
        
    valid_mask = end_points['batch_valid_mask']
    print(f"    Total points: {valid_mask.numel()}")
    print(f"    Valid points for training: {valid_mask.sum().item()} ({(valid_mask.sum().float() / valid_mask.numel() * 100):.2f}%)")

    # 5. Extract x0 and x1
    print("\n[5] Constructing x0 and x1...")
    grasp_preds = pred_decode(end_points)  
    x0 = construct_pred_grasp_pose(grasp_preds)  # [B, 11, 1024]
    x1 = construct_gt_grasp_pose(end_points)    # [B, 11, 1024]
    cond = end_points['group_features'].detach()

    print(f"    x0 shape: {tuple(x0.shape)}, dtype: {x0.dtype}")
    print(f"    x1 shape: {tuple(x1.shape)}, dtype: {x1.dtype}")
    print(f"    cond shape: {tuple(cond.shape)}, dtype: {cond.dtype}")
    
    # Look at sample valid point vs invalid point
    if (valid_mask == 1).sum() > 0:
        first_valid = (valid_mask == 1).nonzero(as_tuple=True)[1][0]
        print(f"\n    DEBUG point data:")
        print(f"    -> [VALID point {first_valid}] x0_width: {x0[0,0,first_valid]:.4f}, x1_width (GT): {x1[0,0,first_valid]:.4f}")
    if (valid_mask == 0).sum() > 0:
        first_invalid = (valid_mask == 0).nonzero(as_tuple=True)[1][0]
        print(f"    -> [INVALID point {first_invalid}] x0_width: {x0[0,0,first_invalid]:.4f}, x1_width (GT_GARBAGE): {x1[0,0,first_invalid]:.4f}")

    # 6. Test Unmasked Loss vs Masked Loss directly
    print("\n[6] Computing Loss...")
    cfm_net.eval() # for deterministic forward temporarily
    with torch.no_grad():
        # Unmasked loss
        torch.manual_seed(0)
        loss_unmasked = cfm_net.compute_loss(x0, x1, cond, valid_mask=None)
        
        # Masked loss
        torch.manual_seed(0)
        loss_masked = cfm_net.compute_loss(x0, x1, cond, valid_mask=valid_mask)
        
        print(f"    Unmasked loss (with garbage): {loss_unmasked.item():.6f}")
        print(f"    Masked loss (valid only):     {loss_masked.item():.6f}")

    # 7. Training Step
    cfm_net.train()
    for step in range(2):
        optimizer.zero_grad()
        loss = cfm_net.compute_loss(x0.detach(), x1.detach(), cond, valid_mask=valid_mask)
        loss.backward()
        
        # Check gradients
        grad_norm = cfm_net.input_proj.weight.grad.norm().item()
        print(f"\n[7] Backpropagation (Step {step+1}):")
        print(f"    Loss output: {loss.item():.6f}")
        print(f"    Sample gradient norm (input_proj): {grad_norm:.6f}")
        
        optimizer.step()
        
        param_sample_after = cfm_net.output_proj[-1].weight
        print(f"    Output projection norm after step: {param_sample_after.norm().item():.6f}")
    
    print("\n"+ "="*60)
    print("SUCCESS")
    print("="*60)

if __name__ == '__main__':
    main()
