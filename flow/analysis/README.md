# Flow Analysis Scripts

This folder stores small analysis scripts used to inspect design choices for the
CFM data pipeline. These scripts should not generate the final CFM training
dataset directly unless their command-line interface explicitly says so.

## Seed Neighborhood Survey

Use `survey_seed_neighborhoods.py` to inspect how many ground-truth grasp
configurations fall within a radius around seed points selected by the frozen
`economic_graspable` model from `flow/models/grasp_cfm.py`.

Example:

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
./py310/bin/python flow/analysis/survey_seed_neighborhoods.py \
    --dataset_root /media/dsp520/Grasp_2T/graspnet \
    --checkpoint checkpoints/economicgrasp_realsense.tar \
    --camera realsense \
    --split train \
    --num_scenes 10 \
    --radius 0.005 \
    --output_dir flow/analysis/results/seed_neighborhoods
```

The script writes:

- `survey_summary.json`: aggregate count and score statistics.
- `per_scene_summary.csv`: one row per sampled scene/frame.
- `per_seed_summary.csv`: one row per selected seed point.

