#!/bin/bash
# Script to run evaluation of CFM model on GraspNet seen split using Realsense camera.

WORKSPACE_DIR="/media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp"
cd "$WORKSPACE_DIR" || exit 1

LOG_FILE="doc/test_cfm_realsense.log"
echo "Starting CFM Realsense Evaluation..." > "$LOG_FILE"
echo "Start Time: $(date)" >> "$LOG_FILE"

# Run the python command
./py310/bin/python flow/tests/test_cfm.py \
  --cfm_checkpoint_path checkpoints/flowgrasp/flowgrasp_latest.tar \
  --stats_path /media/dsp520/Grasp_2T/graspnet/cfm_norm_stats.pt \
  --checkpoint_path checkpoints/economicgrasp_realsense.tar \
  --camera realsense \
  --test_mode seen \
  --save_dir doc/test_cfm_outputs_seen_realsense \
  --batch_size 32 >> "$LOG_FILE" 2>&1

echo "Finished CFM Realsense Evaluation!" >> "$LOG_FILE"
echo "End Time: $(date)" >> "$LOG_FILE"
