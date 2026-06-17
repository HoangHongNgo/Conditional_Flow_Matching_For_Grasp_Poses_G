import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from flow.utils.cfm_label_generation import process_grasp_labels

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    
    subset = torch.utils.data.Subset(dataset, [0, 1])
    dataloader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    ckpt = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()

    print("Running process_grasp_labels on a batch of 2...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            end_points = process_grasp_labels(end_points)
            
            print("Successfully processed.")
            print("batch_target_points shape:", end_points['batch_target_points'].shape)
            print("batch_target_views_rot shape:", end_points['batch_target_views_rot'].shape)
            print("batch_valid_mask shape:", end_points['batch_valid_mask'].shape)
            
            # Check how many valid configs are there on average
            valid_mask = end_points['batch_valid_mask']
            counts = valid_mask.sum(dim=-1) # [B, 1024]
            print(f"Mean valid configs per seed: {counts.float().mean().item():.2f}")
            print(f"Max valid configs per seed: {counts.max().item()}")
            
            break
            
if __name__ == '__main__':
    main()
