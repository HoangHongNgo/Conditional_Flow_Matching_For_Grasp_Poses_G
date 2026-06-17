import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
import math
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp, pred_decode
from models.flowgrasp import ConditionalFlowMatching
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from train_refine import construct_pred_grasp_pose, construct_gt_grasp_pose

# --- CONFIG ---
BASE_CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path else 'checkpoints/economicgrasp_kinect.tar'
CFM_CHECKPOINT_PATH = 'log/refine/cfm_latest.tar'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SCENES = 20
BATCH_SIZE = 4


# ============ METRIC UTILITIES ============

def rotation_geodesic_error(R_pred, R_gt):
    """
    Compute geodesic rotation error in degrees between predicted and GT
    rotation matrices.
    R_pred, R_gt: [B, 9, N] flattened rotation matrices.
    Returns: [B, N] angle error in degrees.
    """
    B, _, N = R_pred.shape
    # Reshape to [B*N, 3, 3]
    Rp = R_pred.permute(0, 2, 1).contiguous().reshape(-1, 3, 3)
    Rg = R_gt.permute(0, 2, 1).contiguous().reshape(-1, 3, 3)
    
    # R_diff = R_gt^T @ R_pred
    R_diff = torch.bmm(Rg.transpose(1, 2), Rp)
    
    # trace(R_diff) = 1 + 2*cos(theta)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * (180.0 / math.pi)
    
    return angle_deg.reshape(B, N)


def compute_stats(values_np):
    """Compute summary statistics for a numpy array."""
    if len(values_np) == 0:
        return {}
    return {
        'mean': float(np.mean(values_np)),
        'std': float(np.std(values_np)),
        'median': float(np.median(values_np)),
        'p25': float(np.percentile(values_np, 25)),
        'p75': float(np.percentile(values_np, 75)),
        'p90': float(np.percentile(values_np, 90)),
        'p95': float(np.percentile(values_np, 95)),
        'min': float(np.min(values_np)),
        'max': float(np.max(values_np)),
    }


def print_stats_block(name, stats, unit=""):
    """Print a formatted statistics block."""
    u = f" {unit}" if unit else ""
    print(f"    Mean ± Std : {stats['mean']:.6f} ± {stats['std']:.6f}{u}")
    print(f"    Median     : {stats['median']:.6f}{u}")
    print(f"    P25 / P75  : {stats['p25']:.6f} / {stats['p75']:.6f}{u}")
    print(f"    P90 / P95  : {stats['p90']:.6f} / {stats['p95']:.6f}{u}")
    print(f"    Min / Max  : {stats['min']:.6f} / {stats['max']:.6f}{u}")


# ============ MAIN ============

def main():
    print("=" * 90)
    print("  CONDITIONAL FLOW MATCHING (CFM) — COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 90)
    
    # ---- 1. Dataset ----
    print("\n[1] Initializing Dataset...")
    dataset = GraspNetDataset(
        cfgs.dataset_root, camera=cfgs.camera, split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    indices = list(range(min(NUM_SCENES, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    print(f"    Scenes: {len(subset)} | Batch size: {BATCH_SIZE}")

    # ---- 2. Base Model ----
    print("\n[2] Loading frozen base model...")
    base_net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
    base_net.to(DEVICE).eval()
    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # ---- 3. CFM Model ----
    print("[3] Loading trained CFM Model...")
    cfm_net = ConditionalFlowMatching(in_dim=11, out_dim=5, cond_dim=256, hidden_dim=256,
                                      time_dim=64, num_blocks=4)
    cfm_net.to(DEVICE).eval()
    ckpt_cfm = torch.load(CFM_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfm_net.load_state_dict(ckpt_cfm['model_state_dict'])
    trained_epoch = ckpt_cfm.get('epoch', 'N/A')
    print(f"    CFM checkpoint epoch: {trained_epoch}")
    total_params = sum(p.numel() for p in cfm_net.parameters())
    print(f"    CFM parameters: {total_params:,}")

    # ---- 4. Pre-cache base model outputs ----
    print("\n[4] Pre-caching base model outputs...")
    t0 = time.time()
    cached_batches = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(DEVICE)
                else:
                    batch_data[key] = batch_data[key].to(DEVICE)
            end_points = base_net(batch_data)
            grasp_preds = pred_decode(end_points)
            x0 = construct_pred_grasp_pose(grasp_preds)
            x1 = construct_gt_grasp_pose(end_points)
            cond = end_points['group_features'].detach()
            valid_mask = end_points['batch_valid_mask']
            cached_batches.append((x0, x1, cond, valid_mask))
    base_time = time.time() - t0
    print(f"    Pre-caching took {base_time:.2f}s")

    # ---- 5. Sweep over num_steps ----
    steps_to_test = [1, 5, 10, 20, 50]
    print(f"\n[5] Running num_steps sweep: {steps_to_test}")

    sweep_results = []

    for num_steps in steps_to_test:
        # Per-point collectors
        all_base_width_err = []
        all_ref_width_err = []
        all_base_depth_err = []
        all_ref_depth_err = []
        all_base_rot_angle = []
        all_ref_rot_angle = []
        all_base_overall = []
        all_ref_overall = []

        # Depth category accumulators
        depth_categories = [0.01, 0.02, 0.03, 0.04]
        depth_cat_base = {c: [] for c in depth_categories}
        depth_cat_ref = {c: [] for c in depth_categories}

        # Width category accumulators (every 0.02m bin)
        width_bins = [0.02, 0.04, 0.06, 0.08, 0.10]
        width_cat_base = {b: [] for b in width_bins}
        width_cat_ref = {b: [] for b in width_bins}

        cfm_time = 0.0

        with torch.no_grad():
            for x0, x1, cond, valid_mask in cached_batches:
                t_start = time.time()
                x_refined = cfm_net.sample(x0, cond, num_steps=num_steps)
                cfm_time += time.time() - t_start

                B, D, N = x0.shape
                vm = valid_mask.bool()  # [B, N]

                # --- Width errors [B, N] ---
                base_w_err = torch.abs(x0[:, 0, :] - x1[:, 0, :])
                ref_w_err = torch.abs(x_refined[:, 0, :] - x1[:, 0, :])
                all_base_width_err.append(base_w_err[vm].cpu().numpy())
                all_ref_width_err.append(ref_w_err[vm].cpu().numpy())

                # --- Depth errors [B, N] ---
                base_d_err = torch.abs(x0[:, 1, :] - x1[:, 1, :])
                ref_d_err = torch.abs(x_refined[:, 1, :] - x1[:, 1, :])
                all_base_depth_err.append(base_d_err[vm].cpu().numpy())
                all_ref_depth_err.append(ref_d_err[vm].cpu().numpy())

                # --- Rotation geodesic angle error [B, N] ---
                base_rot_deg = rotation_geodesic_error(x0[:, 2:, :], x1[:, 2:, :])
                ref_rot_deg = rotation_geodesic_error(x_refined[:, 2:, :], x1[:, 2:, :])
                all_base_rot_angle.append(base_rot_deg[vm].cpu().numpy())
                all_ref_rot_angle.append(ref_rot_deg[vm].cpu().numpy())

                # --- Overall L1 per point [B, N] ---
                base_overall = torch.sum(torch.abs(x0 - x1), dim=1)
                ref_overall = torch.sum(torch.abs(x_refined - x1), dim=1)
                all_base_overall.append(base_overall[vm].cpu().numpy())
                all_ref_overall.append(ref_overall[vm].cpu().numpy())

                # --- Depth category breakdown ---
                gt_depth = x1[:, 1, :]  # [B, N]
                for c in depth_categories:
                    cat = (torch.abs(gt_depth - c) < 0.005) & vm
                    if cat.any():
                        depth_cat_base[c].append(base_d_err[cat].cpu().numpy())
                        depth_cat_ref[c].append(ref_d_err[cat].cpu().numpy())

                # --- Width category breakdown ---
                gt_width = x1[:, 0, :]  # [B, N]
                for b in width_bins:
                    cat = (gt_width > b - 0.02) & (gt_width <= b) & vm
                    if cat.any():
                        width_cat_base[b].append(base_w_err[cat].cpu().numpy())
                        width_cat_ref[b].append(ref_w_err[cat].cpu().numpy())

        # Concatenate all per-point arrays
        base_w = np.concatenate(all_base_width_err)
        ref_w = np.concatenate(all_ref_width_err)
        base_d = np.concatenate(all_base_depth_err)
        ref_d = np.concatenate(all_ref_depth_err)
        base_r = np.concatenate(all_base_rot_angle)
        ref_r = np.concatenate(all_ref_rot_angle)
        base_o = np.concatenate(all_base_overall)
        ref_o = np.concatenate(all_ref_overall)

        total_pts = len(base_o)
        improved = int(np.sum(ref_o < base_o))
        worsened = int(np.sum(ref_o > base_o))

        sweep_results.append({
            'steps': num_steps,
            'total_pts': total_pts,
            'cfm_time': cfm_time,
            'base_w': base_w, 'ref_w': ref_w,
            'base_d': base_d, 'ref_d': ref_d,
            'base_r': base_r, 'ref_r': ref_r,
            'base_o': base_o, 'ref_o': ref_o,
            'improved': improved, 'worsened': worsened,
            'depth_cat_base': depth_cat_base, 'depth_cat_ref': depth_cat_ref,
            'width_cat_base': width_cat_base, 'width_cat_ref': width_cat_ref,
        })
        pct = improved / total_pts * 100
        print(f"    steps={num_steps:2d} | MAE Base={np.mean(base_o):.6f} Ref={np.mean(ref_o):.6f} | Improved={pct:.1f}% | CFM time={cfm_time:.3f}s")

    # ============ DETAILED REPORT ============
    best = min(sweep_results, key=lambda r: np.mean(r['ref_o']))

    print("\n" + "=" * 90)
    print("  SECTION A: NUM_STEPS SWEEP COMPARISON TABLE")
    print("=" * 90)
    header = f"  {'Steps':>5} | {'MAE Base':>10} | {'MAE Ref':>10} | {'Δ MAE':>10} | {'Width':>10} | {'Depth':>10} | {'Rot (°)':>10} | {'%Imp':>6} | {'Time(s)':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sweep_results:
        mae_b = np.mean(r['base_o'])
        mae_r = np.mean(r['ref_o'])
        delta = mae_r - mae_b
        sign = "+" if delta > 0 else ""
        w_mae = np.mean(r['ref_w'])
        d_mae = np.mean(r['ref_d'])
        r_mae = np.mean(r['ref_r'])
        pct = r['improved'] / r['total_pts'] * 100
        print(f"  {r['steps']:>5} | {mae_b:>10.6f} | {mae_r:>10.6f} | {sign}{delta:>9.6f} | {w_mae:>10.6f} | {d_mae:>10.6f} | {r_mae:>10.4f} | {pct:>5.1f}% | {r['cfm_time']:>7.3f}")

    print(f"\n  => Best num_steps = {best['steps']} (lowest overall MAE)")

    # ---- SECTION B: DETAILED COMPONENT ANALYSIS (best steps) ----
    print("\n" + "=" * 90)
    print(f"  SECTION B: COMPONENT-WISE ANALYSIS (num_steps = {best['steps']})")
    print("=" * 90)
    print(f"  Total valid grasp points: {best['total_pts']:,}")
    pts_imp = best['improved']
    pts_wor = best['worsened']
    pts_same = best['total_pts'] - pts_imp - pts_wor
    print(f"  Points improved : {pts_imp:>7,} ({pts_imp/best['total_pts']*100:.1f}%)")
    print(f"  Points worsened : {pts_wor:>7,} ({pts_wor/best['total_pts']*100:.1f}%)")
    print(f"  Points unchanged: {pts_same:>7,} ({pts_same/best['total_pts']*100:.1f}%)")

    # Width
    print(f"\n  --- WIDTH (meters) ---")
    print(f"  Base Model:")
    print_stats_block("Base Width", compute_stats(best['base_w']), "m")
    print(f"  Refined Model:")
    print_stats_block("Ref Width", compute_stats(best['ref_w']), "m")
    w_imp = int(np.sum(best['ref_w'] < best['base_w']))
    print(f"    Points improved: {w_imp}/{best['total_pts']} ({w_imp/best['total_pts']*100:.1f}%)")

    # Depth
    print(f"\n  --- DEPTH (meters) ---")
    print(f"  Base Model:")
    print_stats_block("Base Depth", compute_stats(best['base_d']), "m")
    print(f"  Refined Model:")
    print_stats_block("Ref Depth", compute_stats(best['ref_d']), "m")
    d_imp = int(np.sum(best['ref_d'] < best['base_d']))
    print(f"    Points improved: {d_imp}/{best['total_pts']} ({d_imp/best['total_pts']*100:.1f}%)")

    # Rotation (geodesic angle)
    print(f"\n  --- ROTATION (geodesic angle, degrees) ---")
    print(f"  Base Model:")
    print_stats_block("Base Rot", compute_stats(best['base_r']), "°")
    print(f"  Refined Model:")
    print_stats_block("Ref Rot", compute_stats(best['ref_r']), "°")
    r_imp = int(np.sum(best['ref_r'] < best['base_r']))
    print(f"    Points improved: {r_imp}/{best['total_pts']} ({r_imp/best['total_pts']*100:.1f}%)")

    # ---- SECTION C: DEPTH CATEGORY BREAKDOWN ----
    print("\n" + "=" * 90)
    print(f"  SECTION C: DEPTH CATEGORY BREAKDOWN (num_steps = {best['steps']})")
    print("=" * 90)
    print(f"  {'GT Depth':>10} | {'Count':>7} | {'Base MAE':>10} | {'Ref MAE':>10} | {'Δ':>10} | {'Result':>10}")
    print("  " + "-" * 65)
    for c in [0.01, 0.02, 0.03, 0.04]:
        b_list = best['depth_cat_base'][c]
        r_list = best['depth_cat_ref'][c]
        if len(b_list) > 0:
            b_arr = np.concatenate(b_list)
            r_arr = np.concatenate(r_list)
            cnt = len(b_arr)
            bm = np.mean(b_arr)
            rm = np.mean(r_arr)
            delta = rm - bm
            tag = "IMPROVED" if delta < 0 else "WORSENED"
            sign = "+" if delta > 0 else ""
            print(f"  {c:>10.2f}m | {cnt:>7,} | {bm:>10.6f} | {rm:>10.6f} | {sign}{delta:>9.6f} | {tag:>10}")

    # ---- SECTION D: WIDTH CATEGORY BREAKDOWN ----
    print("\n" + "=" * 90)
    print(f"  SECTION D: WIDTH CATEGORY BREAKDOWN (num_steps = {best['steps']})")
    print("=" * 90)
    print(f"  {'GT Width Bin':>12} | {'Count':>7} | {'Base MAE':>10} | {'Ref MAE':>10} | {'Δ':>10} | {'Result':>10}")
    print("  " + "-" * 67)
    for b in [0.02, 0.04, 0.06, 0.08, 0.10]:
        b_list = best['width_cat_base'][b]
        r_list = best['width_cat_ref'][b]
        if len(b_list) > 0:
            b_arr = np.concatenate(b_list)
            r_arr = np.concatenate(r_list)
            cnt = len(b_arr)
            bm = np.mean(b_arr)
            rm = np.mean(r_arr)
            delta = rm - bm
            tag = "IMPROVED" if delta < 0 else "WORSENED"
            sign = "+" if delta > 0 else ""
            lo = b - 0.02
            print(f"  ({lo:.2f}, {b:.2f}]m | {cnt:>7,} | {bm:>10.6f} | {rm:>10.6f} | {sign}{delta:>9.6f} | {tag:>10}")

    # ---- SECTION E: ERROR DISTRIBUTION (HISTOGRAM BUCKETS) ----
    print("\n" + "=" * 90)
    print(f"  SECTION E: ROTATION ERROR DISTRIBUTION (num_steps = {best['steps']})")
    print("=" * 90)
    angle_bins = [0, 5, 10, 15, 20, 30, 45, 60, 90, 180]
    print(f"  {'Angle Range':>15} | {'Base Count':>10} {'%':>6} | {'Ref Count':>10} {'%':>6}")
    print("  " + "-" * 55)
    for i in range(len(angle_bins) - 1):
        lo, hi = angle_bins[i], angle_bins[i + 1]
        b_cnt = int(np.sum((best['base_r'] >= lo) & (best['base_r'] < hi)))
        r_cnt = int(np.sum((best['ref_r'] >= lo) & (best['ref_r'] < hi)))
        b_pct = b_cnt / best['total_pts'] * 100
        r_pct = r_cnt / best['total_pts'] * 100
        print(f"  {lo:>5}° - {hi:<4}° | {b_cnt:>10,} {b_pct:>5.1f}% | {r_cnt:>10,} {r_pct:>5.1f}%")

    # ---- SECTION F: SUMMARY TABLE (paper-ready) ----
    print("\n" + "=" * 90)
    print("  SECTION F: SUMMARY TABLE (Paper-Ready)")
    print("=" * 90)
    b_stats = best
    print(f"  {'Metric':<30} | {'Base Model':>12} | {'CFM Refined':>12} | {'Δ':>12}")
    print("  " + "-" * 72)

    metrics = [
        ("Overall MAE", np.mean(b_stats['base_o']), np.mean(b_stats['ref_o'])),
        ("Width MAE (m)", np.mean(b_stats['base_w']), np.mean(b_stats['ref_w'])),
        ("Depth MAE (m)", np.mean(b_stats['base_d']), np.mean(b_stats['ref_d'])),
        ("Rotation Error (°)", np.mean(b_stats['base_r']), np.mean(b_stats['ref_r'])),
        ("Width Median (m)", np.median(b_stats['base_w']), np.median(b_stats['ref_w'])),
        ("Depth Median (m)", np.median(b_stats['base_d']), np.median(b_stats['ref_d'])),
        ("Rotation Median (°)", np.median(b_stats['base_r']), np.median(b_stats['ref_r'])),
    ]
    for name, bv, rv in metrics:
        delta = rv - bv
        sign = "+" if delta > 0 else ""
        print(f"  {name:<30} | {bv:>12.6f} | {rv:>12.6f} | {sign}{delta:>11.6f}")

    print(f"\n  Points improved over base: {b_stats['improved']:,}/{b_stats['total_pts']:,} ({b_stats['improved']/b_stats['total_pts']*100:.1f}%)")
    print(f"  CFM inference time: {b_stats['cfm_time']:.3f}s for {b_stats['total_pts']:,} points ({b_stats['total_pts']/b_stats['cfm_time']:.0f} pts/s)")
    print("=" * 90)


if __name__ == '__main__':
    main()
