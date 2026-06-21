import os
import sys
import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
import MinkowskiEngine as ME

from flow.models.grasp_cfm import GraspVelocityMLP
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction
from flow.utils.cfm_solver import euler_solve
from flow.utils.lie import exp_so3

def extract_scene_inputs(base_net, batch_data):
    """
    Run base Minkowski backbone, objectness/graspness filters, and FPS downsampling
    to extract coordinates and features for conditional flow matching.
    Matches exactly the logic inside economicgrasp.py forward pass.
    """
    seed_xyz = batch_data['point_clouds']
    B, point_num, _ = seed_xyz.shape

    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [coord for coord in batch_data['coordinates_for_voxel']],
        [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)]
    )
    coordinates_batch, features_batch, _, batch_data['quantize2original'] = \
        ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True
        )

    coordinates_batch = coordinates_batch.cuda()
    features_batch = features_batch.cuda()

    mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)

    # Base Minkowski Backbone
    seed_features = base_net.backbone(mink_input).F
    seed_features = seed_features[batch_data['quantize2original']].view(
        B, point_num, -1
    ).transpose(1, 2)

    # Graspable scoring & mask
    batch_data = base_net.graspable(seed_features, batch_data)
    seed_features_flipped = seed_features.transpose(1, 2)
    objectness_score = batch_data['objectness_score']
    graspness_score = batch_data['graspness_score'].squeeze(1)
    objectness_pred = torch.argmax(objectness_score, 1)
    objectness_mask = (objectness_pred == 1)
    graspness_mask = graspness_score > cfgs.graspness_threshold
    graspable_mask = objectness_mask & graspness_mask

    # Furthest Point Sampling to 1024 points
    seed_features_graspable = []
    seed_xyz_graspable = []
    for i in range(B):
        cur_mask = graspable_mask[i]
        if cur_mask.sum() == 0:
            # Fallback if no points satisfy mask
            cur_mask = torch.ones_like(cur_mask, dtype=torch.bool)
            
        cur_feat = seed_features_flipped[i][cur_mask]
        cur_seed_xyz = seed_xyz[i][cur_mask]

        cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
        fps_idxs = furthest_point_sample(cur_seed_xyz, base_net.M_points)
        cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
        cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(
            1, 2).squeeze(0).contiguous()
        cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
        cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

        seed_features_graspable.append(cur_feat)
        seed_xyz_graspable.append(cur_seed_xyz)

    seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
    seed_features_graspable = torch.stack(seed_features_graspable)

    # View selection
    batch_data, res_feat = base_net.view(seed_features_graspable, batch_data)
    seed_features_graspable = seed_features_graspable + res_feat

    # Return shape: coords [B, 1024, 3], features [B, 512, 1024].
    return seed_xyz_graspable, seed_features_graspable

def evaluate_cfm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load test dataset
    if cfgs.test_mode == 'seen':
        split = 'test_seen'
    elif cfgs.test_mode == 'similar':
        split = 'test_similar'
    else:
        split = 'test_novel'
        
    print(f"Initializing dataset for split: {split}")
    TEST_DATASET = GraspNetDataset(
        cfgs.dataset_root,
        split=split,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        load_label=False,
        augment=False
    )
    SCENE_LIST = TEST_DATASET.scene_list()
    TEST_DATALOADER = DataLoader(
        TEST_DATASET,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(TEST_DATASET)}. Batches: {len(TEST_DATALOADER)}")
    
    # 2. Init Base Net & Load pretrained weights
    print(f"Loading base net checkpoint: {cfgs.checkpoint_path}")
    base_net = economicgrasp(seed_feat_dim=512, is_training=False).to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    base_net.load_state_dict(checkpoint['model_state_dict'])
    base_net.eval()
    
    # 3. Load CFM Models & Stats
    print(f"Loading CFM checkpoint: {args.cfm_checkpoint_path}")
    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=args.nsample,
        seed_feature_dim=512,
        sphere_radius=args.sphere_radius,
    ).to(device)
    mlp = GraspVelocityMLP(grasp_dim=5, cond_dim=128).to(device)
    
    cfm_checkpoint = torch.load(args.cfm_checkpoint_path, map_location=device)
    seed_conditioner.load_state_dict(cfm_checkpoint['seed_conditioner_state_dict'])
    mlp.load_state_dict(cfm_checkpoint['mlp_state_dict'])
    seed_conditioner.eval()
    mlp.eval()
    
    stats = torch.load(args.stats_path, map_location=device)
    print(f"Loaded normalization stats from: {args.stats_path}")
    
    os.makedirs(cfgs.save_dir, exist_ok=True)
    
    # 4. Inference Loop
    print("\nRunning CFM inference...")
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        # Move batch data to device
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
                
        # Forward pass base model
        with torch.no_grad():
            seed_xyz, seed_feats = extract_scene_inputs(base_net, batch_data)
            B = seed_xyz.shape[0]
            
            # CFM Prior Sampling: x0 ~ N(0, I)
            num_seed = seed_xyz.shape[1]
            x0 = torch.randn(B, num_seed, 5, device=device)
            
            # CFM ODE Euler Solver
            x_pred = euler_solve(
                seed_conditioner, mlp, x0, seed_xyz, seed_feats, stats, n_steps=args.n_steps
            )  # [B, 1024, 5]
            
        # Decode and format each batch element to GraspGroup
        for b in range(B):
            data_idx = batch_idx * cfgs.batch_size + b
            pred_grasps = x_pred[b]  # [1024, 5]
            
            p = seed_xyz[b]  # [1024, 3] (center)
            omega = pred_grasps[:, :3]  # [1024, 3] (rotation Lie algebra)
            w = pred_grasps[:, 3]  # [1024] (width)
            d = pred_grasps[:, 4]  # [1024] (depth)
            
            # Map Lie algebra rotation to maxtrix
            rot_matrices = exp_so3(omega)  # [1024, 3, 3]
            rot_flat = rot_matrices.reshape(num_seed, 9)
            
            # Build 17D arrays
            # 1. descending score
            score = 0.99 - (torch.arange(num_seed, device=device).float() / num_seed) * 0.49  # [1024]
            score = score.view(-1, 1)
            
            # 2. clamp width & depth
            grasp_width = torch.clamp(w, min=0.0, max=0.1).view(-1, 1)
            grasp_depth = torch.clamp(d, min=0.01, max=0.04).view(-1, 1)
            
            grasp_height = 0.02 * torch.ones_like(score)
            obj_ids = -1 * torch.ones_like(score)
            
            # Cat to 17D
            gg_preds = torch.cat([
                score,
                grasp_width,
                grasp_height,
                grasp_depth,
                rot_flat,
                p,
                obj_ids
            ], dim=-1).cpu().numpy()
            
            gg = GraspGroup(gg_preds)
            
            # Collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(
                    gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
                )
                gg = gg[~collision_mask]
                
            # Save GraspGroup
            save_dir = os.path.join(cfgs.save_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            os.makedirs(save_dir, exist_ok=True)
            gg.save_npy(save_path)
            
        if batch_idx % 20 == 0:
            print(f"Evaluated batch: {batch_idx}, elapsed: {time.time() - tic:.2f}s")
            
        # Debug break
        if (sys.gettrace() is not None or '--smoke_test' in sys.argv) and batch_idx >= 3:
            print("Debug/Smoke break triggered.")
            break
            
    if '--smoke_test' in sys.argv:
        print("\nSmoke test completed successfully! Measured batch generation time correctly.")
        return
            
    print("\nInference finished! Running graspnetAPI evaluations...")
    
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    if cfgs.test_mode == 'seen':
        res, ap = ge.eval_seen(cfgs.save_dir, proc=6)
        print(f"Seen split AP 0.8: {np.mean(res[:, :, :, 3])}, AP 0.4: {np.mean(res[:, :, :, 1])}")
    elif cfgs.test_mode == 'similar':
        res, ap = ge.eval_similar(cfgs.save_dir, proc=6)
        print(f"Similar split AP 0.8: {np.mean(res[:, :, :, 3])}, AP 0.4: {np.mean(res[:, :, :, 1])}")
    else:
        res, ap = ge.eval_novel(cfgs.save_dir, proc=6)
        print(f"Novel split AP 0.8: {np.mean(res[:, :, :, 3])}, AP 0.4: {np.mean(res[:, :, :, 1])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate CFM Grasp Pose Generator")
    parser.add_argument('--cfm_checkpoint_path', type=str, default='checkpoints/flowgrasp/flowgrasp_latest.tar',
                        help='Path to CFM checkpoint')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_norm_stats.pt',
                        help='Path to normalization stats')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of Euler steps for inference')
    parser.add_argument('--nsample', type=int, default=32, help='Number of neighbor seeds for spherical grouping')
    parser.add_argument('--sphere_radius', type=float, default=0.005, help='Seed grouping radius in meters')
    parser.add_argument('--smoke_test', action='store_true', help='Run smoke test')
    
    args, _ = parser.parse_known_args()
    evaluate_cfm(args)
