"""
Visualize the probability distribution of TDUnet encoder features.
Uses a forward hook to capture backbone output, reduces dimension to 1,
and plots the distribution.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCENE_INDICES = [0, 1]

# ---------------- LOAD DATASET ----------------
dataset = GraspNetDataset(
    cfgs.dataset_root,
    camera=cfgs.camera,
    split='train',
    voxel_size=cfgs.voxel_size,
    num_points=cfgs.num_point,
    remove_outlier=True,
    augment=False
)

data_list = [dataset[idx] for idx in SCENE_INDICES]
batch_data = collate_fn(data_list)

for key in batch_data:
    if 'list' in key:
        for i in range(len(batch_data[key])):
            for j in range(len(batch_data[key][i])):
                batch_data[key][i][j] = batch_data[key][i][j].to(DEVICE)
    else:
        batch_data[key] = batch_data[key].to(DEVICE)

# ---------------- LOAD MODEL ----------------
net = economicgrasp(seed_feat_dim=512, is_training=False, attn_layer=False)
net.to(DEVICE)
net.eval()

# ---------------- HOOK: Capture backbone output ----------------
backbone_features = {}

def hook_fn(module, input, output):
    # output is ME.SparseTensor, .F gives dense features (N, C)
    backbone_features['encoder_output'] = output.F.detach().cpu()

hook_handle = net.backbone.register_forward_hook(hook_fn)

# ---------------- INFERENCE ----------------
with torch.no_grad():
    end_points = net(batch_data)

hook_handle.remove()

# ---------------- ANALYZE & PLOT ----------------
feats = backbone_features['encoder_output']  # (N_total, 512)
print(f"Backbone output shape: {feats.shape}")
print(f"  Mean: {feats.mean():.4f}, Std: {feats.std():.4f}")
print(f"  Min:  {feats.min():.4f}, Max: {feats.max():.4f}")

# Reduce dimension to 1: mean across feature channels per point
feat_reduced = feats.mean(dim=1).numpy()  # (N_total,)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram (Probability Distribution)
axes[0].hist(feat_reduced, bins=100, density=True, alpha=0.75, color='steelblue', edgecolor='white')
axes[0].set_title('Probability Distribution of Encoder Features\n(mean-reduced to 1D)', fontsize=13)
axes[0].set_xlabel('Feature Value (mean across 512 channels)')
axes[0].set_ylabel('Density')
axes[0].grid(True, alpha=0.3)

# Plot 2: Per-channel mean distribution
channel_means = feats.mean(dim=0).numpy()  # (512,)
axes[1].bar(range(len(channel_means)), channel_means, color='coral', alpha=0.7, width=1.0)
axes[1].set_title('Per-Channel Mean Activation', fontsize=13)
axes[1].set_xlabel('Channel Index')
axes[1].set_ylabel('Mean Activation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: feature_distribution.png")
plt.show()
