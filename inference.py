import torch
import numpy as np

from models.economicgrasp import economicgrasp, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.arguments import cfgs
from train_refine import construct_pred_grasp_pose, construct_gt_grasp_pose

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = 'results/economicgrasp_origin/economicgrasp_epoch10.tar'
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
net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
net.to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
net.load_state_dict(checkpoint['model_state_dict'])
print(f"Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")

# ---------------- INFERENCE ----------------
with torch.no_grad():
    end_points = net(batch_data)
    grasp_preds = pred_decode(end_points)

# ---------------- OUTPUT INSPECTION ----------------
for key, value in end_points.items():
    if torch.is_tensor(value):
        print(f"{key:25s}: {tuple(value.shape)}")


# =====================================================================
# TEST: construct_pred_grasp_pose & construct_gt_grasp_pose
# (imported from train_refine.py)
# =====================================================================
print("\n" + "=" * 60)
print("TESTING construct_pred_grasp_pose & construct_gt_grasp_pose")
print("(imported from train_refine.py)")
print("=" * 60)

# Check required keys exist in end_points
pred_keys = ['grasp_width_pred', 'grasp_depth_pred', 'grasp_angle_pred', 'grasp_top_view_xyz']
gt_keys = ['batch_grasp_width', 'batch_grasp_depth', 'batch_grasp_rotations', 'grasp_top_view_xyz']

print("\n--- Required keys for construct_pred_grasp_pose ---")
for k in pred_keys:
    if k in end_points:
        v = end_points[k]
        if torch.is_tensor(v):
            print(f"  ✓ {k:25s}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item():.4f}, max={v.max().item():.4f}")
    else:
        print(f"  ✗ {k:25s}: MISSING!")

print("\n--- Required keys for construct_gt_grasp_pose ---")
for k in gt_keys:
    if k in end_points:
        v = end_points[k]
        if torch.is_tensor(v):
            print(f"  ✓ {k:25s}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item():.4f}, max={v.max().item():.4f}")
    else:
        print(f"  ✗ {k:25s}: MISSING!")

# Construct x0 (predicted)
print("\n--- construct_pred_grasp_pose (x0) ---")
try:
    with torch.no_grad():
        x0 = construct_pred_grasp_pose(grasp_preds)
    print(f"  shape : {tuple(x0.shape)}")
    print(f"  dtype : {x0.dtype}")
    print(f"  min   : {x0.min().item():.6f}")
    print(f"  max   : {x0.max().item():.6f}")
    print(f"  mean  : {x0.mean().item():.6f}")
    print(f"  std   : {x0.std().item():.6f}")
    print(f"  NaN   : {torch.isnan(x0).any().item()}")
    print(f"  Inf   : {torch.isinf(x0).any().item()}")

    # Breakdown by component
    w0, d0, r0 = x0[:, 0:1, :], x0[:, 1:2, :], x0[:, 2:, :]
    print(f"  width  [{w0.min().item():.4f}, {w0.max().item():.4f}]  mean={w0.mean().item():.4f}")
    print(f"  depth  [{d0.min().item():.4f}, {d0.max().item():.4f}]  unique={torch.unique(d0).cpu().numpy()}")
    print(f"  rot    [{r0.min().item():.4f}, {r0.max().item():.4f}]  mean={r0.mean().item():.4f}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# Construct x1 (GT)
print("\n--- construct_gt_grasp_pose (x1) ---")
try:
    with torch.no_grad():
        x1 = construct_gt_grasp_pose(end_points)
    print(f"  shape : {tuple(x1.shape)}")
    print(f"  dtype : {x1.dtype}")
    print(f"  min   : {x1.min().item():.6f}")
    print(f"  max   : {x1.max().item():.6f}")
    print(f"  mean  : {x1.mean().item():.6f}")
    print(f"  std   : {x1.std().item():.6f}")
    print(f"  NaN   : {torch.isnan(x1).any().item()}")
    print(f"  Inf   : {torch.isinf(x1).any().item()}")

    # Breakdown by component
    w1, d1, r1 = x1[:, 0:1, :], x1[:, 1:2, :], x1[:, 2:, :]
    print(f"  width  [{w1.min().item():.4f}, {w1.max().item():.4f}]  mean={w1.mean().item():.4f}")
    print(f"  depth  [{d1.min().item():.4f}, {d1.max().item():.4f}]  unique={torch.unique(d1).cpu().numpy()}")
    print(f"  rot    [{r1.min().item():.4f}, {r1.max().item():.4f}]  mean={r1.mean().item():.4f}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# Compare x0 vs x1
print("\n--- Comparison: x0 (pred) vs x1 (GT) ---")
try:
    diff = (x0 - x1).abs()
    print(f"  |x0 - x1| mean  : {diff.mean().item():.6f}")
    print(f"  |x0 - x1| max   : {diff.max().item():.6f}")
    print(f"  width diff mean  : {(x0[:, 0:1, :] - x1[:, 0:1, :]).abs().mean().item():.6f}")
    print(f"  depth diff mean  : {(x0[:, 1:2, :] - x1[:, 1:2, :]).abs().mean().item():.6f}")
    print(f"  rot   diff mean  : {(x0[:, 2:, :] - x1[:, 2:, :]).abs().mean().item():.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
