import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.label_generation import process_grasp_labels_without_seed_mapping
from utils.arguments import cfgs

def test_process_without_seed():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize dataset
    print("Initializing dataset...")
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

    # Get first batch
    print("Fetching a batch of data...")
    batch = next(iter(dataloader))
    
    # 2. Move batch to device
    for key in batch:
        if 'list' in key:
            for b_idx in range(len(batch[key])):
                for obj_idx in range(len(batch[key][b_idx])):
                    batch[key][b_idx][obj_idx] = batch[key][b_idx][obj_idx].to(device)
        else:
            batch[key] = batch[key].to(device)

    # 3. Test Function (Filtered down to 10 best per object, 50 best globally)
    print("\n=======================================================")
    print("Test: Running 10 Best per Object & 50 Best per Scene")
    print("=======================================================")
    
    end_points = batch.copy()
    
    # Call the processing function
    batch_grasp_views_rot, end_points_out = process_grasp_labels_without_seed_mapping(end_points)
    
    # Verify outputs are lists of tensors of varying sizes (one per batch item)
    batch_size = len(batch_grasp_views_rot)
    print(f"Batch Size: {batch_size}")
    
    for b in range(batch_size):
        num_points = end_points_out['batch_grasp_point'][b].shape[0]
        scores = end_points_out['batch_grasp_score'][b]
        
        # Calculate statistics
        num_non_zero = torch.sum(scores > 0.0).item()
        num_good = torch.sum(scores >= 0.1).item()
        pct_non_zero = (num_non_zero / num_points) * 100 if num_points > 0 else 0
        pct_good = (num_good / num_points) * 100 if num_points > 0 else 0
        min_score = scores.min().item() if num_points > 0 else 0
        max_score = scores.max().item() if num_points > 0 else 0
        mean_score = scores.mean().item() if num_points > 0 else 0
        
        print(f"\n  Scene {b+1}:")
        print(f"    Number of filtered grasp points: {num_points} (Expected: <= 50)")
        
        # Check shapes of each returned list item
        print(f"    grasp_points shape          : {end_points_out['batch_grasp_point'][b].shape} (Expected: [{num_points}, 3])")
        print(f"    grasp_views_rot shape       : {batch_grasp_views_rot[b].shape} (Expected: [{num_points}, 3, 3])")
        print(f"    grasp_rotations shape       : {end_points_out['batch_grasp_rotations'][b].shape} (Expected: [{num_points}])")
        print(f"    grasp_depth shape           : {end_points_out['batch_grasp_depth'][b].shape} (Expected: [{num_points}])")
        print(f"    grasp_score shape           : {scores.shape} (Expected: [{num_points}])")
        print(f"    grasp_width shape           : {end_points_out['batch_grasp_width'][b].shape} (Expected: [{num_points}])")
        
        print(f"\n    Top Filtered Grasp Quality Statistics:")
        print(f"      - Min Score               : {min_score:.4f}")
        print(f"      - Max Score               : {max_score:.4f}")
        print(f"      - Mean Score              : {mean_score:.4f}")
        print(f"      - Valid Grasps (Score > 0): {num_non_zero} / {num_points} ({pct_non_zero:.2f}%)")
        print(f"      - Good Grasps (Score >=0.1): {num_good} / {num_points} ({pct_good:.2f}%)")
        
        # Verify correctness of types, shapes, and constraints
        assert num_points <= 50
        assert end_points_out['batch_grasp_point'][b].shape == (num_points, 3)
        assert batch_grasp_views_rot[b].shape == (num_points, 3, 3)
        assert end_points_out['batch_grasp_rotations'][b].shape == (num_points,)
        assert end_points_out['batch_grasp_depth'][b].shape == (num_points,)
        assert end_points_out['batch_grasp_score'][b].shape == (num_points,)
        assert end_points_out['batch_grasp_width'][b].shape == (num_points,)

        # Assert scores are indeed sorted descending due to topk
        if num_points > 1:
            diff = scores[:-1] - scores[1:]
            assert (diff >= -1e-6).all(), "Scores must be in descending order!"

    print("\nAll checks passed successfully!")

if __name__ == '__main__':
    test_process_without_seed()
