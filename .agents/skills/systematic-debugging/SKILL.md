---
name: systematic-debugging
description: Debugging strategies for EconomicGrasp model, data pipeline, and CUDA issues
---

# Systematic Debugging Guide

## Common Issues & Solutions

### 1. Checkpoint Loading Errors

**`persistent_load function not specified`**
```python
# Wrong: file might be saved with different format
checkpoint = torch.load('data.pkl')

# Fix: Load with proper args
checkpoint = torch.load('model.tar', map_location=device, weights_only=False)
```

**`Missing key(s)` / `Unexpected key(s)`**
```python
# Diagnose
checkpoint = torch.load(path, map_location='cpu')
print(checkpoint.keys())
print(checkpoint['model_state_dict'].keys())

# Fix: Strict=False for partial loading
net.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### 2. Shape Mismatches

**Debugging tensor shapes through the pipeline:**
```python
def forward(self, end_points):
    for key, value in end_points.items():
        if torch.is_tensor(value):
            print(f"{key:30s}: {tuple(value.shape)}")
    # ... rest of forward
```

**Common shape issues:**
| Problem | Cause | Fix |
|---|---|---|
| `[B, N, C]` vs `[B, C, N]` | Wrong tensor layout | `x.permute(0, 2, 1)` |
| Wrong `M` dimension | `m_point` mismatch | Check `cfgs.m_point` |
| Batch dim missing | Single sample | `x.unsqueeze(0)` |

### 3. NaN/Inf Loss

```python
# Check for NaN in gradients
for name, param in net.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in: {name}")

# Check for NaN in end_points
for key, value in end_points.items():
    if torch.is_tensor(value) and torch.isnan(value).any():
        print(f"NaN in: {key}")
```

### 4. CUDA Errors

**`CUDA out of memory`**
- Reduce `--batch_size`
- Reduce `--num_point` (default 20000)
- Check for memory leaks (tensors not freed)

**`CUDA error: device-side assert`**
```bash
# Run with CUDA_LAUNCH_BLOCKING for better error messages
CUDA_LAUNCH_BLOCKING=1 python train.py ...
```

### 5. MinkowskiEngine Issues

**`Coordinate mismatch`**
- Ensure `voxel_size` is consistent (default: 0.005)
- Check that coordinates are properly quantized

### 6. PointNet2 CUDA Ops

**`RuntimeError: Not compiled with GPU support`**
```bash
# Recompile
cd libs/pointnet2
pip install -e .
```

## Debugging Workflow

1. **Isolate**: Run with `batch_size=1` and single sample
2. **Trace**: Add shape prints at each pipeline stage
3. **Validate**: Check `end_points` dict after each module
4. **Compare**: Use `scratch/` scripts for isolated testing
