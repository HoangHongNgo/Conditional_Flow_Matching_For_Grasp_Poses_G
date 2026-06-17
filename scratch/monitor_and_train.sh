#!/bin/bash
# monitor_and_train.sh
# Monitors generate_cfm_dataset.py --split eval in background.
# Once finished, it starts flow/train_cfm.py.

echo "================================================================="
echo "CFM Pipeline Monitor & Train Script Started"
echo "Start Time: $(date)"
echo "================================================================="

# Wait for generate_cfm_dataset.py --split eval to complete
echo "Waiting for generate_cfm_dataset.py --split eval to finish..."
while pgrep -f "generate_cfm_dataset.py --split eval" > /dev/null; do
    count=$(find /media/dsp520/Grasp_2T/graspnet/cfm_dataset_eval -name "*.pt" | wc -l)
    echo "[$(date '+%H:%M:%S')] Progress: $count / 7680 files generated."
    sleep 30
done

# Extra wait to ensure all files are fully flushed to disk
sleep 15

final_count=$(find /media/dsp520/Grasp_2T/graspnet/cfm_dataset_eval -name "*.pt" | wc -l)
echo "Dataset generation finished. Final file count: $final_count / 7680"
echo "VRAM is now freed. Initiating CFM training..."

# Start CFM training
# We use the py310 virtual environment python
./py310/bin/python flow/train_cfm.py --epochs 20 --batch_size 16

echo "================================================================="
echo "CFM Training Finished"
echo "End Time: $(date)"
echo "================================================================="
