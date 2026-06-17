---
description: Train the EconomicGrasp model
---

# Training

## Prerequisites
- Dataset prepared (see `/data-prep`)
- Environment set up (see `/setup`)

## Standard Training Command

// turbo

```bash
source /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp/py310/bin/activate
```

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model economicgrasp \
    --camera kinect \
    --log_dir results/economicgrasp \
    --max_epoch 10 \
    --batch_size 4 \
    --dataset_root /media/dsp520/Grasp_2T/graspnet
```

## Key Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `graspness` | Model name: `economicgrasp` or `liteptgrasp` |
| `--camera` | `realsense` | Camera type: `kinect` or `realsense` |
| `--batch_size` | `4` | Batch size (RTX 3090: 4, A100: 8+) |
| `--max_epoch` | `20` | Total training epochs (paper uses 10) |
| `--learning_rate` | `0.001` | Initial LR with cosine annealing |
| `--voxel_size` | `0.005` | MinkowskiEngine voxel size (meters) |
| `--num_point` | `20000` | Input point cloud size |
| `--m_point` | `1024` | Points after graspness filtering |

## Training with LitePT Backbone

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model liteptgrasp \
    --camera kinect \
    --log_dir results/liteptgrasp \
    --load checkpoints/litept_S_plus_scannet/model_best.pth \
    --max_epoch 10 \
    --batch_size 28
```

## Resume Training

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model economicgrasp \
    --camera kinect \
    --log_dir results/economicgrasp \
    --checkpoint_path results/economicgrasp/economicgrasp_epoch5.tar \
    --resume \
    --max_epoch 10 \
    --batch_size 4 \
    --dataset_root /media/dsp520/Grasp_2T/graspnet
```

## Monitor Training

```bash
tensorboard --logdir results/economicgrasp
```

## Expected Resources (RTX 3090)
| Metric | Value |
|---|---|
| Training time | ~8.3 hours (10 epochs) |
| Main memory | ~4.2 GB |
| GPU memory | ~5.81 GB |

> [!TIP]
> The first ~60 batches are slow due to warmup. Subsequent runs will be faster.
