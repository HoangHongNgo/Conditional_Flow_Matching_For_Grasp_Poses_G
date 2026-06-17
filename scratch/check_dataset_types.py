import sys
import os
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs
from torch.utils.data import DataLoader

def check_dataset_types():
    print("Initializing dataset...")
    # Initialize same dataset as train.py
    dataset = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                              voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, 
                              remove_outlier=True, augment=True)
    
    print(f"Dataset length: {len(dataset)}")
    
    print("\n--- Getting one sample (__getitem__) ---")
    sample = dataset[0]
    
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: numpy array, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: torch Tensor, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list, length={len(value)}")
            if len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    print(f"  First element: numpy array, shape={value[0].shape}, dtype={value[0].dtype}")
                elif isinstance(value[0], torch.Tensor):
                    print(f"  First element: torch Tensor, shape={value[0].shape}, dtype={value[0].dtype}")
                else:
                    print(f"  First element type: {type(value[0])}")
        else:
            print(f"{key}: {type(value)}")

    print("\n--- Getting one batch (via DataLoader and collate_fn) ---")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: torch Tensor, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list, length={len(value)}")
            if len(value) > 0 and isinstance(value[0], list):
                print(f"  First element: list, length={len(value[0])}")
                if len(value[0]) > 0 and isinstance(value[0][0], torch.Tensor):
                    print(f"    Nested element: torch Tensor, shape={value[0][0].shape}, dtype={value[0][0].dtype}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == '__main__':
    check_dataset_types()
