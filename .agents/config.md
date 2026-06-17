# EconomicGrasp — Project Configuration

## Overview
**EconomicGrasp** is a 6-DoF grasp detection framework (ECCV 2024) that uses economic supervision for efficient training while maintaining effective grasp performance on the GraspNet-1Billion dataset.

- **Paper**: [An Economic Framework for 6-DoF Grasp Detection](https://arxiv.org/abs/2407.08366)
- **Authors**: Xiao-Ming Wu, Jia-Feng Cai, Jian-Jian Jiang, Dian Zheng, Yi-Lin Wei, Wei-Shi Zheng

## Tech Stack
| Component | Version/Details |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 |
| CUDA | 12+ |
| MinkowskiEngine | Custom build for CUDA 12+ |
| PointNet2 | Custom CUDA ops (`libs/pointnet2/`) |
| KNN | Custom CUDA ops (`libs/knn/`) |
| LitePT | Lightweight Point Transformer (`libs/LitePT/`) |
| GraspNetAPI | Evaluation toolkit (`libs/graspnetAPI/`) |
| Open3D | ≥ 0.8, point cloud visualization |
| TensorBoard | 2.3, training monitoring |

## Python Environment (MANDATORY)

> **Before running ANY Python file**, you MUST activate the project virtualenv:
> ```bash
> source /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp/py310/bin/activate
> ```
> This applies to training, testing, inference, debugging scripts, and any one-off Python commands.
> Always use a persistent terminal with the venv activated to avoid re-activating on every command.

## Architecture

### Pipeline
```
Input Point Cloud (20000 pts)
    → MinkowskiEngine Backbone (TDUnet, voxel_size=0.005)
    → Seed Features (512-dim)
    → GraspableNet → objectness + graspness filtering (top 1024 pts)
    → ViewNet → 300-class grasp view prediction
    → Sphere Grouping + Global Interaction → view-aware features (256-dim)
    → Cylinder Grouping + Global Interaction → grasp-specific features
    → Grasp Head (Local Interaction) → angle, depth, width, score prediction
    → pred_decode → 6-DoF grasp poses
```

### Model Variants
| Model Class | Description |
|---|---|
| `economicgrasp` | Main model, uses TDUnet backbone |
| `liteptgrasp` | Variant using LitePT (Lightweight Point Transformer) backbone |

### Output Format
`pred_decode()` returns a list of `[Ni, 17]` tensors per batch item:
```
[score, width, height, depth, R(3x3), translation(3), obj_id]
```
where `R` is rotation matrix from 300-view discretization + in-plane angle.

## Directory Structure
```
EconomicGrasp/
├── models/
│   ├── economicgrasp.py        # Model definitions (economicgrasp, liteptgrasp, pred_decode)
│   ├── modules_economicgrasp.py # Modules: GraspableNet, ViewNet, Grouping, GraspHead, Attention
│   ├── backbone.py              # TDUnet (MinkowskiEngine sparse conv backbone)
│   └── loss_economicgrasp.py    # Multi-task loss function
├── dataset/
│   ├── graspnet_dataset.py      # GraspNetDataset, collate_fn
│   ├── generate_graspness.py    # Preprocessing: generate graspness labels
│   └── generate_economic.py     # Preprocessing: generate economic supervision labels
├── utils/
│   ├── arguments.py             # CLI args (cfgs = parser.parse_args())
│   ├── collision_detector.py    # ModelFreeCollisionDetector for testing
│   ├── data_utils.py            # Data preprocessing utilities
│   ├── label_generation.py      # process_grasp_labels, batch_viewpoint_params_to_matrix
│   └── loss_utils.py            # Loss helper functions
├── libs/
│   ├── pointnet2/               # PointNet++ CUDA ops (FPS, ball query, grouping)
│   ├── knn/                     # KNN CUDA operator
│   ├── LitePT/                  # Lightweight Point Transformer backbone
│   ├── graspnetAPI/              # GraspNet evaluation API
│   └── MinkowskiEngine/         # Sparse convolution engine
├── test_case/                   # Test scripts and visualization
├── train.py                     # Training entry point
├── test.py                      # Testing & evaluation entry point
├── inference.py                 # Quick inference script
└── checkpoints/                 # Saved model checkpoints
```

## Important Paths
| Path | Purpose |
|---|---|
| `/media/dsp520/Grasp_2T/graspnet` | GraspNet-1Billion dataset root |
| `results/` | Training logs and evaluation outputs |
| `checkpoints/` | Pre-trained and custom checkpoints |

## Key Patterns
- **Data flow**: All intermediate results pass through an `end_points` dict (key-value store)
- **Config**: All CLI arguments accessed via `from utils.arguments import cfgs`
- **Coordinate system**: Right-hand, Z-up; grasps discretized into 300 views × 12 angles × 4 depths
- **Batch handling**: Custom `collate_fn` for variable-length grasp labels (stored as `_list` keys)
