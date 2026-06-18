---
description: Run inference on specific scenes using the EconomicGrasp model
---

# Inference

## Quick Inference

Use the project Python binary directly. Do not use plain `python`.

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
CUDA_VISIBLE_DEVICES=0 ./py310/bin/python inference.py \
    --camera kinect \
    --dataset_root /media/dsp520/Grasp_2T/graspnet \
    --inference
```

## Custom Inference

Edit `inference.py` to customize:
- `SCENE_INDICES`: list of scene indices to process (default: `[0, 1]`)
- Model class: change `economicgrasp(...)` to `liteptgrasp(...)` if needed
- Checkpoint loading: uncomment `torch.load(...)` lines and set `CHECKPOINT_PATH`

## Output

The inference script prints tensor shapes for all output keys:
```
objectness_score     : (B, 2, N)
graspness_score      : (B, 1, N)
grasp_view_score     : (B, 300, M)
grasp_angle_cls_pred : (B, 12, M)
grasp_depth_pred     : (B, 5, M)
grasp_width_pred     : (B, 1, M)
grasp_score_pred     : (B, 6, M)
```

## Visualization

Use `test_case/visualize_grasp.py` for 3D grasp visualization with Open3D:
```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
./py310/bin/python test_case/test_visualize_grasp.py
```
