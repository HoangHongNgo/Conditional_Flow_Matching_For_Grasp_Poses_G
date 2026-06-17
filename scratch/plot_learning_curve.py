"""Plot learning curve for the second CFM training run (log/refine)."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = 'log/refine/log_train_refine.txt'
# The last full run starts at line 33570 (EPOCH 0 START AT: 2026-05-02 17:45:28)
RUN_START_LINE = 33570

# Read lines from the second run onward
with open(LOG_FILE, 'r') as f:
    all_lines = f.readlines()
lines = all_lines[RUN_START_LINE - 1:]  # 0-indexed

# Parse per-epoch train loss (average of the batch-interval logs within each epoch)
# and eval loss
train_losses_per_epoch = []
eval_losses = []
current_epoch = None
current_epoch_losses = []

for line in lines:
    line = line.strip()
    # Epoch start
    m = re.match(r'\*+\s*EPOCH\s+(\d+)\s+START', line)
    if m:
        # Save previous epoch's average train loss
        if current_epoch is not None and current_epoch_losses:
            train_losses_per_epoch.append(np.mean(current_epoch_losses))
        current_epoch = int(m.group(1))
        current_epoch_losses = []
        continue
    # Train loss log line
    m = re.match(r'A: CFM Loss\s*:\s*([\d.]+)', line)
    if m:
        current_epoch_losses.append(float(m.group(1)))
        continue
    # Eval loss
    m = re.match(r'\[EVAL\]\s*CFM Loss:\s*([\d.]+)', line)
    if m:
        eval_losses.append(float(m.group(1)))
        continue

# Don't forget last epoch
if current_epoch is not None and current_epoch_losses:
    train_losses_per_epoch.append(np.mean(current_epoch_losses))

epochs_train = list(range(len(train_losses_per_epoch)))
epochs_eval = list(range(len(eval_losses)))

print(f"Train epochs: {len(train_losses_per_epoch)}")
print(f"Eval  epochs: {len(eval_losses)}")

# --- Plot ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(epochs_train, train_losses_per_epoch, 'o-', color='#FF6B35', linewidth=2,
        markersize=4, label='Train Loss', alpha=0.9)
ax.plot(epochs_eval, eval_losses, 's-', color='#004E89', linewidth=2,
        markersize=4, label='Eval Loss', alpha=0.9)

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('CFM Loss', fontsize=13)
ax.set_title('CFM Refinement Training — Run 2 (2026-05-02)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max(len(train_losses_per_epoch), len(eval_losses)) - 0.5)

# Also extract and plot eval MSE metrics
depth_base, depth_ref = [], []
rot_base, rot_ref = [], []
width_base, width_ref = [], []

for line in lines:
    line = line.strip()
    m = re.match(r'\[EVAL\]\s*Width\s+MSE:\s*base=([\d.]+)\s+ref=([\d.]+)', line)
    if m:
        width_base.append(float(m.group(1)))
        width_ref.append(float(m.group(2)))
    m = re.match(r'\[EVAL\]\s*Depth\s+MSE:\s*base=([\d.]+)\s+ref=([\d.]+)', line)
    if m:
        depth_base.append(float(m.group(1)))
        depth_ref.append(float(m.group(2)))
    m = re.match(r'\[EVAL\]\s*Rot\s+MSE:\s*base=([\d.]+)\s+ref=([\d.]+)', line)
    if m:
        rot_base.append(float(m.group(1)))
        rot_ref.append(float(m.group(2)))

if depth_base:
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epochs_mse = list(range(len(depth_base)))

    # Width MSE
    axes[0].plot(epochs_mse, width_base, '--', color='#888', linewidth=1.5, label='Base', alpha=0.7)
    axes[0].plot(epochs_mse, width_ref, '-', color='#2ECC71', linewidth=2, label='Refined')
    axes[0].set_title('Width MSE', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Depth MSE
    axes[1].plot(epochs_mse, depth_base, '--', color='#888', linewidth=1.5, label='Base', alpha=0.7)
    axes[1].plot(epochs_mse, depth_ref, '-', color='#3498DB', linewidth=2, label='Refined')
    axes[1].set_title('Depth MSE', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Rotation MSE
    axes[2].plot(epochs_mse, rot_base, '--', color='#888', linewidth=1.5, label='Base', alpha=0.7)
    axes[2].plot(epochs_mse, rot_ref, '-', color='#E74C3C', linewidth=2, label='Refined')
    axes[2].set_title('Rotation MSE', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig2.suptitle('Refinement Quality — Base vs. Refined (Eval)', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('log/refine/learning_curve_mse.png', dpi=150, bbox_inches='tight')
    print("Saved: log/refine/learning_curve_mse.png")

fig.tight_layout()
fig.savefig('log/refine/learning_curve_run2.png', dpi=150, bbox_inches='tight')
print("Saved: log/refine/learning_curve_run2.png")
