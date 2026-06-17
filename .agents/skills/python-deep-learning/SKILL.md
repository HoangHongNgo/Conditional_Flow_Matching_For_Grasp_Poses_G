---
name: python-deep-learning
description: Best practices for PyTorch deep learning code in EconomicGrasp
---

# PyTorch Deep Learning Patterns

## Model Development

### Adding a New Model Variant

1. Define the class in `models/economicgrasp.py`:
```python
class my_new_model(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, voxel_size=0.005):
        super().__init__()
        self.is_training = is_training
        # ... define layers
    
    def forward(self, end_points):
        # Process through pipeline, store results in end_points
        return end_points
```

2. Register in `train.py` and `test.py` model selection:
```python
if cfgs.model == 'my_new_model':
    net = my_new_model(seed_feat_dim=512, is_training=True)
```

### Adding a New Module

Add to `models/modules_economicgrasp.py`:
- Inherit from `nn.Module`
- Use `SharedMLP` for multi-layer conv blocks
- Maintain `[B, C, N]` tensor layout for conv layers
- Pass and update `end_points` dict

### Adding a New Loss Term

1. Add loss computation in `models/loss_economicgrasp.py`
2. Add weight argument in `utils/arguments.py`: `--my_loss_weight`
3. Sum weighted loss in `get_loss()` function

## GPU Memory Management

```python
# Check GPU memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache when needed
torch.cuda.empty_cache()

# Reduce batch size if OOM
# RTX 3090 (24GB): batch_size=4 for economicgrasp
# A100 (80GB): batch_size=8-16
```

## Checkpoint Handling

```python
# Save checkpoint (as done in train.py)
save_dict = {
    'epoch': epoch + 1,
    'optimizer_state_dict': optimizer.state_dict(),
    'model_state_dict': net.state_dict(),
}
torch.save(save_dict, os.path.join(cfgs.log_dir, f'model_epoch{epoch+1}.tar'))

# Load checkpoint
checkpoint = torch.load(path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
```

## Data Pipeline

### Custom Collate Function
The project uses `collate_fn` in `dataset/graspnet_dataset.py` to handle variable-length grasp labels:
- Keys with `_label` → standard batching
- Keys with `_list` → kept as list of lists (ragged tensors)

### DataLoader Configuration
```python
DataLoader(
    dataset,
    batch_size=cfgs.batch_size,
    shuffle=True,              # True for training
    num_workers=4,             # Adjust based on system
    worker_init_fn=my_worker_init_fn,  # For reproducibility
    collate_fn=collate_fn     # Custom collation
)
```
