import os
import numpy as np

def analyze_scene_grasp_scores(label_path):
    """
    Load a grasp label npz file and calculate statistics for grasp scores.
    
    Args:
        label_path (str): Path to the grasp label .npz file.
    """
    if not os.path.exists(label_path):
        print(f"File not found: {label_path}")
        return

    data = np.load(label_path)
    # The 'scores' array in the npz file has shape: [num_grasp_points, num_views]
    raw_scores = data['scores'].astype(np.float32)
    # Scaled scores as used in the model/dataset
    dataset_scores = raw_scores / 10.0

    print(f"File: {os.path.basename(label_path)}")
    print(f"Number of grasp points: {raw_scores.shape[0]}")
    print(f"Number of viewpoints: {raw_scores.shape[1]}")
    print(f"Raw Scores shape: {list(raw_scores.shape)}")
    
    # Calculate stats for all scores
    print("Statistics for ALL scores (including zero/negative/low scores):")
    print(f"  Raw Scores range: {raw_scores.min():.4f} to {raw_scores.max():.4f}")
    print(f"  Raw Scores mean: {raw_scores.mean():.4f}, std: {raw_scores.std():.4f}")
    print(f"  Dataset Scores range: {dataset_scores.min():.4f} to {dataset_scores.max():.4f}")
    print(f"  Dataset Scores mean: {dataset_scores.mean():.4f}, std: {dataset_scores.std():.4f}")

    # Calculate stats for non-zero scores (valid grasps usually have score > 0)
    valid_mask = raw_scores > 0
    valid_raw = raw_scores[valid_mask]
    valid_dataset = dataset_scores[valid_mask]
    
    if len(valid_raw) > 0:
        print(f"Statistics for VALID scores (scores > 0, count={len(valid_raw)}):")
        print(f"  Raw: min={valid_raw.min():.4f}, max={valid_raw.max():.4f}, mean={valid_raw.mean():.4f}, std={valid_raw.std():.4f}")
        print(f"  Dataset: min={valid_dataset.min():.4f}, max={valid_dataset.max():.4f}, mean={valid_dataset.mean():.4f}, std={valid_dataset.std():.4f}")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.percentile(valid_dataset, percentiles)
        print("  Percentiles of valid dataset scores:")
        for p, val in zip(percentiles, pct_vals):
            print(f"    {p}th: {val:.4f}")
    else:
        print("No scores > 0 found.")
    print("-" * 50)


if __name__ == "__main__":
    label_dir = "/media/dsp520/Grasp_2T/graspnet/economic_grasp_label_300views"
    if not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
    else:
        # Check a few scene files
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npz")])
        if len(files) == 0:
            print("No .npz files found in the directory.")
        else:
            # Let's inspect the first 3 files
            for f in files[:3]:
                analyze_scene_grasp_scores(os.path.join(label_dir, f))
