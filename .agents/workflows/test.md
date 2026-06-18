---
description: Test and evaluate the EconomicGrasp model on GraspNet-1Billion
---

# Testing & Evaluation

## Prerequisites
- Trained checkpoint available (e.g. `results/economicgrasp/economicgrasp_epoch10.tar`)
- Or download pre-trained: [kinect](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v1/economicgrasp_kinect.tar) | [realsense](https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v1/economicgrasp_realsense.tar)

## Python Environment

Use the project Python binary directly. Do not use plain `python`.

## Testing Commands

### Test on Seen Split
```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 ./py310/bin/python test.py \
    --model economicgrasp \
    --save_dir results/economicgrasp/test_ep10_seen \
    --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar \
    --camera kinect \
    --dataset_root /media/dsp520/Grasp_2T/graspnet \
    --test_mode seen \
    --inference
```

### Test on Similar Split
```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 ./py310/bin/python test.py \
    --model economicgrasp \
    --save_dir results/economicgrasp/test_ep10_similar \
    --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar \
    --camera kinect \
    --dataset_root /media/dsp520/Grasp_2T/graspnet \
    --test_mode similar \
    --inference
```

### Test on Novel Split
```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 ./py310/bin/python test.py \
    --model economicgrasp \
    --save_dir results/economicgrasp/test_ep10_novel \
    --checkpoint_path results/economicgrasp/economicgrasp_epoch10.tar \
    --camera kinect \
    --dataset_root /media/dsp520/Grasp_2T/graspnet \
    --test_mode novel \
    --inference
```

## Optional: Collision Detection
Add `--collision_thresh 0.01` to enable model-free collision detection during inference.

## Expected Results (Kinect)

| Split | AP |
|---|---|
| Seen | 62.59 |
| Similar | 51.73 |
| Novel | 19.54 |
