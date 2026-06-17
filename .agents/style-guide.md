# EconomicGrasp — Style Guide

## Naming Conventions
| Element | Convention | Example |
|---|---|---|
| Class | PascalCase (descriptive) | `Grasp_Head_Local_Interaction`, `GraspableNet` |
| Module class | Underscored PascalCase | `Cylinder_Grouping_Global_Interaction` |
| Model class | lowercase | `economicgrasp`, `liteptgrasp` |
| Function | snake_case | `pred_decode`, `process_grasp_labels` |
| Variable | snake_case | `seed_features`, `vp_rot` |
| Constants | UPPER_CASE | `DEVICE`, `SCENE_LIST` |

## Data Flow Convention
All modules pass data through an `end_points` dictionary:
```python
def forward(self, end_points):
    # Read inputs
    point_cloud = end_points['point_clouds']
    # Process...
    # Write outputs
    end_points['objectness_score'] = objectness_score
    end_points['graspness_score'] = graspness_score
    return end_points
```

### Key `end_points` Fields
| Key | Shape | Description |
|---|---|---|
| `point_clouds` | `[B, N, 3]` | Input point cloud |
| `seed_xyz` | `[B, Ns, 3]` | Seed point positions |
| `seed_features` | `[B, 512, Ns]` | Backbone output features |
| `objectness_score` | `[B, 2, Ns]` | Object/non-object scores |
| `graspness_score` | `[B, 1, Ns]` | Point graspness |
| `grasp_view_score` | `[B, 300, M]` | View classification (M=1024) |
| `grasp_angle_cls_pred` | `[B, 12, M]` | In-plane angle classification |
| `grasp_depth_pred` | `[B, 5, M]` | Depth prediction (4 depths + 1) |
| `grasp_width_pred` | `[B, 1, M]` | Gripper width |
| `grasp_score_pred` | `[B, 6, M]` | Composite score (6 classes) |
| `*_label` / `*_list` | varies | Ground truth labels |

## Module Organization
1. New network modules → `models/modules_economicgrasp.py`
2. New model variants → `models/economicgrasp.py` (as new class)
3. New loss terms → `models/loss_economicgrasp.py`
4. New data utils → `utils/data_utils.py` or `utils/label_generation.py`
5. New CLI args → `utils/arguments.py` (add `parser.add_argument(...)`)

## PyTorch Conventions
- Use `nn.Conv1d` for point-wise feature transformations
- Use `SharedMLP` from `libs/pointnet2/pytorch_utils` for multi-layer conv blocks
- Tensor layout: `[B, C, N]` (batch, channels, points) for conv layers
- Use `furthest_point_sample` and `gather_operation` from PointNet2 for point sampling
- Model `is_training` flag controls label processing in `forward()`

## Import Order
```python
# 1. Standard library
import os, numpy as np, math, time

# 2. PyTorch
import torch, torch.nn as nn

# 3. External (MinkowskiEngine, graspnetAPI)
import MinkowskiEngine as ME

# 4. Local models/utils
from models.backbone import TDUnet
from utils.arguments import cfgs
from libs.pointnet2.pointnet2_utils import furthest_point_sample
```

## Comments & Documentation
- Use English for code comments
- Vietnamese is acceptable for research notes and TODOs
- Document tensor shapes in comments: `# [B, C, N]`
- Use `# DEBUG:` prefix for temporary debug code
