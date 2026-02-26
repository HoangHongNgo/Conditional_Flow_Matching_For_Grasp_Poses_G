import torch
import numpy as np

from models.economicgrasp import economicgrasp, ecograsp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCENE_INDICES = [0, 1]  # List of scene indices to run inference on

# ---------------- LOAD DATASET ----------------
dataset = GraspNetDataset(
    cfgs.dataset_root,
    camera=cfgs.camera,
    split='train',
    voxel_size=cfgs.voxel_size,
    num_points=cfgs.num_point,
    remove_outlier=True,
    augment=False  # Disable augmentation for inference
)

# Load multiple scenes
data_list = [dataset[idx] for idx in SCENE_INDICES]

# Convert to batch format
batch_data = collate_fn(data_list)

# Move batch data to GPU/CPU
for key in batch_data:
    if 'list' in key:
        for i in range(len(batch_data[key])):
            for j in range(len(batch_data[key][i])):
                batch_data[key][i][j] = batch_data[key][i][j].to(DEVICE)
    else:
        batch_data[key] = batch_data[key].to(DEVICE)

# ---------------- LOAD MODEL ----------------
net = ecograsp(seed_feat_dim=512, is_training=False)
net.to(DEVICE)
net.eval()

# checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# net.load_state_dict(checkpoint['model_state_dict'])
# print(f"Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")

# ---------------- INFERENCE ----------------
with torch.no_grad():
    end_points = net(batch_data)

# ---------------- OUTPUT INSPECTION ----------------
for key, value in end_points.items():
    if torch.is_tensor(value):
        print(f"{key:25s}: {tuple(value.shape)}")
