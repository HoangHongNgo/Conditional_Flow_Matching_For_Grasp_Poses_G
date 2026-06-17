"""
Khảo sát số lượng grasp pose hợp lệ trong 1 scene của GraspNet-1Billion.

Script này load trực tiếp từ các label files (không cần chạy model) để thống kê:
  1. Tổng số grasp point và grasp pose trong scene
  2. Phân loại theo từng object
  3. Phân loại theo score (friction coefficient)
  4. Kiểm tra collision label
  5. Thống kê grasp pose "thật sự dùng được" (score > 0 VÀ collision-free)

Usage:
    python count_valid_grasps.py --scene_id 0
    python count_valid_grasps.py --scene_id 0 --show_collision --score_thresh 0.4
    python count_valid_grasps.py --scene_id 0 --all_scenes
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# Thêm đường dẫn project vào sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def load_scene_meta(dataset_root, scene_id, camera='kinect', ann_id=0):
    """Load scene metadata để lấy danh sách object trong scene."""
    import scipy.io as scio
    scene_name = f'scene_{scene_id:04d}'
    meta_path = os.path.join(dataset_root, 'scenes', scene_name, camera, 'meta', f'{ann_id:04d}.mat')
    meta = scio.loadmat(meta_path)
    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    return obj_idxs


def load_economic_labels(dataset_root, scene_id):
    """Load economic grasp labels (pre-computed 300-view format)."""
    scene_name = f'scene_{scene_id:04d}'
    label_path = os.path.join(dataset_root, 'economic_grasp_label_300views', f'{scene_name}_labels.npz')
    if not os.path.exists(label_path):
        print(f"  [ERROR] Không tìm thấy label file: {label_path}")
        return None
    return np.load(label_path)


def load_raw_grasp_labels(dataset_root, obj_idx):
    """Load raw grasp labels cho 1 object (format gốc 300x12x4)."""
    label_path = os.path.join(dataset_root, 'grasp_label', f'{obj_idx:03d}_labels.npz')
    if not os.path.exists(label_path):
        return None
    return np.load(label_path)


def load_collision_labels(dataset_root, scene_id):
    """Load collision labels cho 1 scene."""
    scene_name = f'scene_{scene_id:04d}'
    collision_path = os.path.join(dataset_root, 'collision_label', scene_name, 'collision_labels.npz')
    if not os.path.exists(collision_path):
        return None
    return np.load(collision_path)


def analyze_scene(dataset_root, scene_id, camera='kinect', ann_id=0,
                  score_thresh=0.0, show_collision=False, show_raw=False):
    """Phân tích chi tiết grasp poses trong 1 scene."""
    scene_name = f'scene_{scene_id:04d}'
    print(f"\n{'='*70}")
    print(f"  SCENE {scene_id:04d} - Khảo sát Grasp Pose hợp lệ")
    print(f"{'='*70}")

    # 1. Load metadata
    obj_idxs = load_scene_meta(dataset_root, scene_id, camera, ann_id)
    print(f"\n  Camera: {camera} | Annotation: {ann_id}")
    print(f"  Số object trong scene: {len(obj_idxs)}")
    print(f"  Object IDs: {obj_idxs.tolist()}")

    # 2. Load economic labels (pre-computed)
    eco_labels = load_economic_labels(dataset_root, scene_id)
    if eco_labels is None:
        return None

    points = eco_labels['points']           # [N, 3]
    scores = eco_labels['scores']           # [N, 300] (uint8, scale /10 for friction)
    pointid = eco_labels['pointid']         # [N]
    rotations = eco_labels['rotations']     # [N, 300]
    depth_labels = eco_labels['depth']      # [N, 300]
    widths = eco_labels['widths']           # [N, 300]
    vgraspness = eco_labels['vgraspness']   # [N, 300]
    topview = eco_labels['topview']         # [N, 300]

    total_points = points.shape[0]
    num_views = scores.shape[1]             # 300
    total_poses = total_points * num_views

    # Economic score đã qua 3 bước lọc trong generate_economic.py:
    #   1. Loại grasp physically impossible (raw fric_coef = -1)
    #   2. Loại grasp bị collision trong scene
    #   3. Loại grasp có width > 0.1m
    # Sau đó invert: score = 1.1 - fric_coef → score cao = grasp tốt hơn
    # Score = 0 → invalid, Score > 0 → valid grasp
    scores_float = scores.astype(np.float32) / 10.0  # scale giống dataset
    valid_mask = scores_float > score_thresh
    total_valid = np.sum(valid_mask)

    print(f"\n  --- Tổng quan (Economic Labels - 300 views) ---")
    print(f"  Tổng grasp points (tất cả objects):     {total_points:>8,}")
    print(f"  Tổng grasp poses (points × views):      {total_poses:>8,}")
    print(f"  Score threshold:                         {score_thresh:.1f}")
    print(f"  Grasp poses hợp lệ (score > {score_thresh}):      {total_valid:>8,}  ({total_valid/total_poses*100:.2f}%)")
    print(f"  Grasp poses KHÔNG hợp lệ:               {total_poses - total_valid:>8,}  ({(total_poses - total_valid)/total_poses*100:.2f}%)")

    # 3. Phân tích theo từng object
    print(f"\n  --- Phân tích theo Object ---")
    print(f"  {'Object':>8} | {'Points':>8} | {'Valid Poses':>12} | {'Total Poses':>12} | {'Ratio':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    obj_stats = {}
    for obj_id in np.unique(pointid):
        mask_obj = pointid == obj_id
        n_pts = np.sum(mask_obj)
        obj_scores = scores_float[mask_obj]
        n_valid = np.sum(obj_scores > score_thresh)
        n_total = n_pts * num_views
        ratio = n_valid / n_total * 100 if n_total > 0 else 0
        obj_stats[int(obj_id)] = {
            'points': int(n_pts),
            'valid_poses': int(n_valid),
            'total_poses': int(n_total),
            'ratio': ratio
        }
        print(f"  {obj_id:>8} | {n_pts:>8,} | {n_valid:>12,} | {n_total:>12,} | {ratio:>7.2f}%")

    # 4. Phân loại theo mức score (friction coefficient bins)
    print(f"\n  --- Phân bố Score (đã invert: 1.1 - friction_coef) ---")
    print(f"  Score = raw_value / 10, range [0.0, 1.1]")
    print(f"  Score càng cao → grasp càng tốt (cần ít ma sát hơn để giữ vật)")
    print(f"  Score = 0: invalid (impossible / collision / width > 0.1m)")
    print()
    score_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"  {'Score Range':>15} | {'Count':>10} | {'Percentage':>10}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")

    for i in range(len(score_bins) - 1):
        lo, hi = score_bins[i], score_bins[i+1]
        if i == 0:
            count = np.sum(scores_float == 0)
            label = f"  = 0.0 (invalid)"
        else:
            count = np.sum((scores_float > lo) & (scores_float <= hi))
            label = f"  ({lo:.1f}, {hi:.1f}]"
        pct = count / total_poses * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:>17} | {count:>10,} | {pct:>8.2f}%  {bar}")

    # 5. Thống kê view graspness
    print(f"\n  --- View Graspness Stats ---")
    print(f"  Mean graspness:     {vgraspness.mean():.4f}")
    print(f"  Median graspness:   {np.median(vgraspness):.4f}")
    print(f"  Max graspness:      {vgraspness.max():.4f}")
    print(f"  % views with graspness > 0: {np.sum(vgraspness > 0) / vgraspness.size * 100:.2f}%")

    # 6. Phân tích points có ít nhất 1 valid grasp
    points_with_any_valid = np.sum(np.any(valid_mask, axis=1))
    points_all_invalid = total_points - points_with_any_valid
    print(f"\n  --- Phân tích theo Point ---")
    print(f"  Points có ít nhất 1 grasp hợp lệ: {points_with_any_valid:>6,} / {total_points:>6,}  ({points_with_any_valid/total_points*100:.2f}%)")
    print(f"  Points không có grasp nào hợp lệ:  {points_all_invalid:>6,} / {total_points:>6,}  ({points_all_invalid/total_points*100:.2f}%)")

    # Số valid views trung bình cho mỗi point
    valid_per_point = np.sum(valid_mask, axis=1)
    print(f"  Số views hợp lệ trung bình / point: {valid_per_point.mean():.1f} / {num_views}")
    print(f"  Min views hợp lệ / point:            {valid_per_point.min()}")
    print(f"  Max views hợp lệ / point:            {valid_per_point.max()}")

    # 7. Optional: Phân tích collision
    if show_collision:
        print(f"\n  --- Collision Analysis ---")
        collision_data = load_collision_labels(dataset_root, scene_id)
        if collision_data is None:
            print(f"  [WARN] Không tìm thấy collision labels cho scene {scene_id:04d}")
        else:
            total_collision_free = 0
            total_collision_entries = 0
            total_valid_and_collision_free = 0

            for obj_id in obj_idxs:
                key = f'arr_{obj_id}'
                if key not in collision_data:
                    continue
                col = collision_data[key]  # [obj_points, 300, 12, 4]
                n_pts_col = col.shape[0]
                collision_free = ~col  # True = collision free

                # Tổng collision-free poses (300 views × 12 rotations × 4 depths)
                total_cf = np.sum(collision_free)
                total_entries = col.size
                total_collision_free += total_cf
                total_collision_entries += total_entries

                print(f"  Object {obj_id:>3}: {n_pts_col:>5} points, "
                      f"collision-free = {total_cf:>10,} / {total_entries:>10,} "
                      f"({total_cf/total_entries*100:.2f}%)")

            if total_collision_entries > 0:
                print(f"  {'':>3} TỔNG: collision-free = {total_collision_free:>10,} / {total_collision_entries:>10,} "
                      f"({total_collision_free/total_collision_entries*100:.2f}%)")

    # 8. Optional: Raw label analysis (full 300×12×4 format)
    if show_raw:
        print(f"\n  --- Raw Grasp Label Analysis (300 views × 12 rotations × 4 depths) ---")
        for obj_id in obj_idxs:
            raw = load_raw_grasp_labels(dataset_root, obj_id)
            if raw is None:
                print(f"  Object {obj_id}: raw labels not found")
                continue
            raw_scores = raw['scores']  # [points, 300, 12, 4]
            raw_collision = raw['collision']  # [points, 300, 12, 4]
            n_pts_raw = raw_scores.shape[0]
            total_raw = raw_scores.size
            valid_raw = np.sum(raw_scores > score_thresh)
            cf_raw = np.sum(~raw_collision)
            both_raw = np.sum((raw_scores > score_thresh) & (~raw_collision))

            print(f"  Object {obj_id:>3}: {n_pts_raw:>5} pts | "
                  f"score > {score_thresh}: {valid_raw:>10,}/{total_raw:>10,} ({valid_raw/total_raw*100:.1f}%) | "
                  f"collision-free: {cf_raw:>10,} ({cf_raw/total_raw*100:.1f}%) | "
                  f"valid & CF: {both_raw:>10,} ({both_raw/total_raw*100:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TỔNG KẾT Scene {scene_id:04d}:")
    print(f"  → {len(obj_idxs)} objects, {total_points:,} grasp points, {total_poses:,} grasp poses")
    print(f"  → {total_valid:,} grasp poses hợp lệ (score > {score_thresh}) = {total_valid/total_poses*100:.2f}%")
    print(f"  → {points_with_any_valid:,}/{total_points:,} points có ít nhất 1 grasp hợp lệ")
    print(f"{'='*70}\n")

    return {
        'scene_id': scene_id,
        'num_objects': len(obj_idxs),
        'total_points': total_points,
        'total_poses': total_poses,
        'valid_poses': int(total_valid),
        'valid_ratio': total_valid / total_poses * 100,
        'points_with_valid': int(points_with_any_valid),
        'obj_stats': obj_stats
    }


def analyze_all_scenes(dataset_root, camera='kinect', score_thresh=0.0):
    """Khảo sát tất cả scenes và tạo bảng tổng hợp."""
    results = []
    scene_dirs = sorted([d for d in os.listdir(os.path.join(dataset_root, 'scenes'))
                         if d.startswith('scene_')])

    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.split('_')[1])
        # Chỉ load economic labels (nhanh, không cần meta)
        eco_labels = load_economic_labels(dataset_root, scene_id)
        if eco_labels is None:
            continue
        scores = eco_labels['scores'].astype(np.float32) / 10.0
        points = eco_labels['points']
        pointid = eco_labels['pointid']

        total_points = points.shape[0]
        total_poses = scores.size
        valid_poses = int(np.sum(scores > score_thresh))
        num_objects = len(np.unique(pointid))
        points_with_valid = int(np.sum(np.any(scores > score_thresh, axis=1)))

        results.append({
            'scene_id': scene_id,
            'num_objects': num_objects,
            'total_points': total_points,
            'total_poses': total_poses,
            'valid_poses': valid_poses,
            'valid_ratio': valid_poses / total_poses * 100 if total_poses > 0 else 0,
            'points_with_valid': points_with_valid,
        })

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  TỔNG HỢP TẤT CẢ SCENES (score > {score_thresh})")
    print(f"{'='*90}")
    print(f"  {'Scene':>6} | {'Objects':>7} | {'Points':>8} | {'Total Poses':>12} | "
          f"{'Valid Poses':>12} | {'Ratio':>7} | {'Pts w/ Valid':>12}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*12}")

    total_all_poses = 0
    total_all_valid = 0
    for r in results:
        total_all_poses += r['total_poses']
        total_all_valid += r['valid_poses']
        print(f"  {r['scene_id']:>6} | {r['num_objects']:>7} | {r['total_points']:>8,} | "
              f"{r['total_poses']:>12,} | {r['valid_poses']:>12,} | {r['valid_ratio']:>6.2f}% | "
              f"{r['points_with_valid']:>12,}")

    if results:
        avg_ratio = total_all_valid / total_all_poses * 100 if total_all_poses > 0 else 0
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*12}")
        print(f"  {'AVG':>6} | {'':>7} | {'':>8} | {total_all_poses:>12,} | "
              f"{total_all_valid:>12,} | {avg_ratio:>6.2f}% |")

        # Statistics across scenes
        ratios = [r['valid_ratio'] for r in results]
        print(f"\n  Thống kê valid ratio qua các scenes:")
        print(f"    Min:    {min(ratios):.2f}%  (scene {results[np.argmin(ratios)]['scene_id']:04d})")
        print(f"    Max:    {max(ratios):.2f}%  (scene {results[np.argmax(ratios)]['scene_id']:04d})")
        print(f"    Mean:   {np.mean(ratios):.2f}%")
        print(f"    Median: {np.median(ratios):.2f}%")
        print(f"    Std:    {np.std(ratios):.2f}%")

    print(f"{'='*90}\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Khảo sát số lượng grasp pose hợp lệ trong scene của GraspNet-1Billion"
    )
    parser.add_argument('--dataset_root', type=str,
                        default='/media/dsp520/Grasp_2T/graspnet',
                        help='Đường dẫn đến thư mục gốc của dataset GraspNet-1Billion')
    parser.add_argument('--scene_id', type=int, default=0,
                        help='ID của scene cần khảo sát (0-189)')
    parser.add_argument('--camera', type=str, default='kinect',
                        choices=['kinect', 'realsense'],
                        help='Loại camera')
    parser.add_argument('--ann_id', type=int, default=0,
                        help='ID của annotation (frame) để load metadata')
    parser.add_argument('--score_thresh', type=float, default=0.0,
                        help='Ngưỡng score để coi là hợp lệ (default: 0.0, tức score > 0)')
    parser.add_argument('--show_collision', action='store_true',
                        help='Hiển thị phân tích collision labels')
    parser.add_argument('--show_raw', action='store_true',
                        help='Hiển thị phân tích raw grasp labels (300×12×4 format)')
    parser.add_argument('--all_scenes', action='store_true',
                        help='Khảo sát tất cả scenes (bảng tổng hợp)')

    args = parser.parse_args()

    if args.all_scenes:
        analyze_all_scenes(args.dataset_root, args.camera, args.score_thresh)
    else:
        analyze_scene(
            args.dataset_root, args.scene_id, args.camera, args.ann_id,
            args.score_thresh, args.show_collision, args.show_raw
        )


if __name__ == '__main__':
    main()
