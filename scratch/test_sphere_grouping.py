import os
import sys
import torch

# Thêm đường dẫn gốc của project vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs
from flow.models.grasp_cfm import economic_graspable
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Khởi tạo Dataset và DataLoader để lấy một batch thực tế
    print("Loading a test batch from GraspNetDataset...")
    dataset = GraspNetDataset(
        cfgs.dataset_root, 
        split='test_seen',
        camera=cfgs.camera, 
        num_points=cfgs.num_point, 
        remove_outlier=True, 
        load_label=False, 
        augment=False
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfgs.batch_size, 
        shuffle=False,
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    # Lấy 1 batch đầu tiên
    batch_data = next(iter(dataloader))
    
    # Chuyển batch_data lên GPU
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
            
    print("Batch loaded successfully.")

    # 2. Khởi tạo mô hình economic_graspable và Sphere_Grouping_Global_Interaction
    print("Initializing models...")
    base_net = economic_graspable(seed_feat_dim=512, is_training=False).to(device)
    
    checkpoint_path = "checkpoints/economicgrasp_realsense.tar"
    try:
        print(f"Loading base net checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        
    base_net.eval()
    
    # nsample=16, seed_feature_dim=512 (từ economic_graspable)
    sphere_grouping = Sphere_Grouping_Global_Interaction(
        nsample=16, 
        seed_feature_dim=512, 
        sphere_radius=0.05
    ).to(device)
    sphere_grouping.eval()

    # 3. Chạy inference qua economic_graspable
    print("\n--- Running economic_graspable ---")
    with torch.no_grad():
        end_points = base_net(batch_data)
        
    seed_xyz_graspable = end_points['xyz_graspable']
    seed_features_graspable = end_points['seed_features_graspable']
    
    print(f"xyz_graspable shape: {seed_xyz_graspable.shape}  # Expected: [B, 1024, 3]")
    print(f"seed_features_graspable shape: {seed_features_graspable.shape}  # Expected: [B, 512, 1024]")

    # 4. Chạy dữ liệu qua Sphere_Grouping_Global_Interaction
    print("\n--- Running Sphere_Grouping_Global_Interaction ---")
    try:
        with torch.no_grad():
            group_features = sphere_grouping(seed_xyz_graspable, seed_features_graspable)
            
        print(f"group_features shape: {group_features.shape}  # Expected: [B, 256, 1024]")
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")

if __name__ == "__main__":
    main()
