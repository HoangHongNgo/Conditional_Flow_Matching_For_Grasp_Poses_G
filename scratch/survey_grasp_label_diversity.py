"""
Survey grasp label diversity around seed points.

For each seed point (after K-NN mapping to the nearest GT grasp point),
we explore the raw label landscape:
  - How many distinct (viewpoint, depth) pairs exist and are valid?
  - How spread out are the viewpoint directions (angular diversity)?
  - How does score vary across valid (viewpoint, depth) combos?
  - How diverse are rotation angles for the best viewpoints?

Based on this, we propose a practical 3-grasp selection strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.loss_utils import generate_grasp_views, transform_point_cloud, batch_viewpoint_params_to_matrix
from libs.knn.knn_modules import knn

NUM_FRAMES = 3       # number of training frames to survey
RADIUS_M  = 0.01    # neighborhood radius around each seed point (10 mm)
NUM_SEEDS_TO_PRINT = 3  # how many seed-point examples to print in detail

def load_model_and_data(device):
    """Load the model checkpoint and a small subset of the train dataset."""
    print("Loading dataset (realsense, train)...")
    dataset = GraspNetDataset(
        '/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='train',
        voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
        remove_outlier=True, augment=False
    )
    subset = torch.utils.data.Subset(dataset, list(range(NUM_FRAMES)))
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Loading model checkpoint...")
    ckpt_path = 'checkpoints/economicgrasp_realsense.tar'
    net = economicgrasp(seed_feat_dim=512, is_training=False)
    net.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net, dataloader


def get_transformed_labels(end_points, device):
    """
    For scene i=0, transform all per-object grasp labels to scene frame
    and return merged tensors. Mirrors the logic in process_grasp_labels.

    Returns:
        gt_points:      [M, 3]   - grasp point positions in scene frame
        gt_views_rot:   [M, 300, 3, 3] - view rotation matrices in scene frame
        gt_view_graspness: [M, 300] - graspness per view
        gt_scores:      [M, 60]  - quality score per top-view slot
        gt_rotations:   [M, 60]  - in-plane rotation index per top-view slot
        gt_depth:       [M, 60]  - depth index per top-view slot
        gt_widths:      [M, 60]  - width per top-view slot
        top_view_index: [M, 60]  - which of the 300 views corresponds to each slot
    """
    i = 0
    poses = end_points['object_poses_list'][i]

    merged_pts, merged_vrot, merged_vgsp = [], [], []
    merged_scores, merged_rots, merged_depths, merged_widths, merged_topv = [], [], [], [], []

    for obj_idx, pose in enumerate(poses):
        grasp_points   = end_points['grasp_points_list'][i][obj_idx]    # [P, 3]
        grasp_scores   = end_points['grasp_scores_list'][i][obj_idx]    # [P, 60]
        grasp_rots     = end_points['grasp_rotations_list'][i][obj_idx] # [P, 60]
        grasp_depth    = end_points['grasp_depth_list'][i][obj_idx]     # [P, 60]
        grasp_widths   = end_points['grasp_widths_list'][i][obj_idx]    # [P, 60]
        view_graspness = end_points['view_graspness_list'][i][obj_idx]  # [P, 300]
        top_view_index = end_points['top_view_index_list'][i][obj_idx]  # [P, 60]
        P = grasp_points.size(0)

        # Transform points and views to scene frame
        grasp_views  = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300, 3]
        pts_trans    = transform_point_cloud(grasp_points, pose, '3x4')      # [P, 3]
        views_trans  = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')  # [300, 3]

        # Rotation matrices for each template view, then rotate to scene frame
        angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
        views_rot      = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # [300, 3, 3]
        views_rot_trans = torch.matmul(pose[:3, :3], views_rot)                   # [300, 3, 3]

        # KNN: match transformed views back to canonical template indices
        gv_ = grasp_views.T.unsqueeze(0)        # [1, 3, 300]
        gvt_ = views_trans.T.unsqueeze(0)       # [1, 3, 300]
        view_inds = knn(gvt_, gv_, k=1).squeeze() - 1  # [300] — canonical idx for each scene view

        vgsp_trans     = torch.index_select(view_graspness, 1, view_inds)    # [P, 300]
        vrot_trans     = torch.index_select(views_rot_trans, 0, view_inds)   # [300, 3, 3]
        vrot_trans_exp = vrot_trans.unsqueeze(0).expand(P, -1, -1, -1)       # [P, 300, 3, 3]

        merged_pts.append(pts_trans)
        merged_vrot.append(vrot_trans_exp)
        merged_vgsp.append(vgsp_trans)
        merged_scores.append(grasp_scores)
        merged_rots.append(grasp_rots)
        merged_depths.append(grasp_depth)
        merged_widths.append(grasp_widths)
        merged_topv.append(top_view_index)

    gt_points      = torch.cat(merged_pts,    dim=0)  # [M, 3]
    gt_views_rot   = torch.cat(merged_vrot,   dim=0)  # [M, 300, 3, 3]
    gt_vgraspness  = torch.cat(merged_vgsp,   dim=0)  # [M, 300]
    gt_scores      = torch.cat(merged_scores, dim=0)  # [M, 60]
    gt_rotations   = torch.cat(merged_rots,   dim=0)  # [M, 60]
    gt_depth       = torch.cat(merged_depths, dim=0)  # [M, 60]
    gt_widths      = torch.cat(merged_widths, dim=0)  # [M, 60]
    top_view_index = torch.cat(merged_topv,   dim=0)  # [M, 60]

    return gt_points, gt_views_rot, gt_vgraspness, gt_scores, gt_rotations, gt_depth, gt_widths, top_view_index


def survey_one_scene(end_points, frame_idx, device):
    """
    For one forward-pass result, analyze the grasp label landscape
    around each of the 1024 seed points within RADIUS_M.
    Returns per-seed statistics dictionaries.
    """
    grasp_views = generate_grasp_views(cfgs.num_view).to(device)   # [300, 3]
    seed_xyz    = end_points['xyz_graspable'][0]                   # [1024, 3]

    (gt_points, gt_views_rot, gt_vgraspness,
     gt_scores, gt_rotations, gt_depth, gt_widths, top_view_index) = get_transformed_labels(end_points, device)

    M = gt_points.shape[0]

    # Pairwise distances between seeds and GT points: [1024, M]
    dists = torch.cdist(seed_xyz.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)

    # Per-seed aggregation
    results = {
        'num_gt_pts_in_radius':    [],
        'num_valid_viewdepth':     [],  # (viewpoint, depth) pairs with score > 0
        'num_distinct_views':      [],  # unique view indices with any valid grasp
        'mean_angular_spread_deg': [],  # mean pairwise angular distance between valid view directions
        'score_mean':              [],
        'score_max':               [],
        'rotation_diversity':      [],  # number of distinct in-plane rotation classes found
        'depth_diversity':         [],  # number of distinct depth levels found
        'example_details':         [],  # detailed per-seed info (only for first few seeds)
    }

    for s_idx in range(seed_xyz.shape[0]):
        neighbor_mask = dists[s_idx] <= RADIUS_M  # [M] bool
        num_nb = neighbor_mask.sum().item()
        results['num_gt_pts_in_radius'].append(num_nb)

        if num_nb == 0:
            results['num_valid_viewdepth'].append(0)
            results['num_distinct_views'].append(0)
            results['mean_angular_spread_deg'].append(0.0)
            results['score_mean'].append(0.0)
            results['score_max'].append(0.0)
            results['rotation_diversity'].append(0)
            results['depth_diversity'].append(0)
            continue

        # Gather neighbor labels
        nb_scores   = gt_scores[neighbor_mask]      # [nb, 60]
        nb_topv     = top_view_index[neighbor_mask]  # [nb, 60]
        nb_rots     = gt_rotations[neighbor_mask]    # [nb, 60]
        nb_depths   = gt_depth[neighbor_mask]        # [nb, 60]

        # Valid: score > 0 AND view index != -1 (i.e., could be mapped)
        valid_mask = (nb_scores > 0) & (nb_topv >= 0)  # [nb, 60]

        num_valid = valid_mask.sum().item()
        results['num_valid_viewdepth'].append(num_valid)

        if num_valid == 0:
            results['num_distinct_views'].append(0)
            results['mean_angular_spread_deg'].append(0.0)
            results['score_mean'].append(0.0)
            results['score_max'].append(0.0)
            results['rotation_diversity'].append(0)
            results['depth_diversity'].append(0)
            continue

        # Collect all valid view indices (into the 300 template views)
        valid_view_inds = nb_topv[valid_mask]   # [num_valid] — indices into 300 views
        valid_scores    = nb_scores[valid_mask].float() / 10.0  # scale to [0, 1]
        valid_rots      = nb_rots[valid_mask]
        valid_depths    = nb_depths[valid_mask]

        distinct_views = torch.unique(valid_view_inds)
        num_dv = distinct_views.shape[0]
        results['num_distinct_views'].append(num_dv)

        # Angular spread: mean pairwise angle between distinct view directions
        if num_dv >= 2:
            view_dirs = grasp_views[distinct_views]  # [num_dv, 3]
            view_dirs_n = torch.nn.functional.normalize(view_dirs, dim=-1)
            # Cosine similarity matrix
            cos_sim = torch.mm(view_dirs_n, view_dirs_n.t()).clamp(-1, 1)  # [num_dv, num_dv]
            angles_rad = torch.acos(cos_sim)  # [num_dv, num_dv]
            # Upper triangle only (excluding diagonal)
            iu = torch.triu_indices(num_dv, num_dv, offset=1)
            mean_angle_deg = angles_rad[iu[0], iu[1]].mean().item() * 180.0 / np.pi
        else:
            mean_angle_deg = 0.0
        results['mean_angular_spread_deg'].append(mean_angle_deg)

        results['score_mean'].append(valid_scores.mean().item())
        results['score_max'].append(valid_scores.max().item())

        # Distinct rotation classes (0-11)
        distinct_rots = torch.unique(valid_rots[valid_rots >= 0])
        results['rotation_diversity'].append(distinct_rots.shape[0])

        # Distinct depth levels (0-3)
        distinct_depths = torch.unique(valid_depths[valid_depths >= 0])
        results['depth_diversity'].append(distinct_depths.shape[0])

    return results


def main():
    """Main survey routine."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, dataloader = load_model_and_data(device)

    # Aggregate statistics across all frames
    agg = {
        'num_gt_pts_in_radius': [],
        'num_valid_viewdepth': [],
        'num_distinct_views': [],
        'mean_angular_spread_deg': [],
        'score_mean': [],
        'score_max': [],
        'rotation_diversity': [],
        'depth_diversity': [],
    }

    print(f"\nSurveying {NUM_FRAMES} frames, radius = {RADIUS_M*1000:.0f} mm ...\n")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Frames")):
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            end_points = net(batch_data)
            res = survey_one_scene(end_points, batch_idx, device)

            for k in agg:
                agg[k].extend(res[k])

    # ---- Print aggregated statistics ----
    print("\n" + "=" * 65)
    print(f"  GRASP LABEL DIVERSITY SURVEY (radius = {RADIUS_M*1000:.0f} mm, {NUM_FRAMES} frames)")
    print("=" * 65)

    def stat(arr, name, unit=""):
        a = np.array(arr)
        print(f"  {name}:")
        print(f"    Mean ± Std : {a.mean():.2f} ± {a.std():.2f} {unit}")
        print(f"    Median     : {np.median(a):.2f} {unit}")
        print(f"    10% / 90%  : {np.percentile(a, 10):.2f} / {np.percentile(a, 90):.2f} {unit}")
        print(f"    Min / Max  : {a.min():.2f} / {a.max():.2f} {unit}")
        # Fraction of seeds with > 0
        nonzero_frac = (a > 0).mean() * 100.0
        print(f"    Seeds with >0: {nonzero_frac:.1f}%")

    stat(agg['num_gt_pts_in_radius'],    "GT grasp POINTS in neighborhood")
    stat(agg['num_valid_viewdepth'],     "Valid (view, depth) configurations in neighborhood")
    stat(agg['num_distinct_views'],      "Distinct valid VIEW directions in neighborhood")
    stat(agg['mean_angular_spread_deg'], "Mean pairwise angular spread of views", unit="deg")
    stat(agg['score_mean'],              "Grasp Score mean (valid grasps only, 0-1)")
    stat(agg['score_max'],               "Grasp Score max (valid grasps only, 0-1)")
    stat(agg['rotation_diversity'],      "Distinct in-plane rotation classes (0-11)")
    stat(agg['depth_diversity'],         "Distinct depth levels (0-3)")

    # ---- Compute fraction of seeds where finding k diverse grasps is feasible ----
    print("\n--- Feasibility check: can we reliably pick k distinct grasps? ---")
    for k in [1, 2, 3, 5]:
        frac = (np.array(agg['num_valid_viewdepth']) >= k).mean() * 100.0
        frac_dv = (np.array(agg['num_distinct_views']) >= k).mean() * 100.0
        print(f"  k={k}: {frac:.1f}% seeds have >= {k} valid (view,depth) pairs  |  "
              f"{frac_dv:.1f}% have >= {k} distinct view directions")

    print("\n" + "=" * 65)
    print("  STRATEGY RECOMMENDATION FOR SELECTING 3 GRASPS PER SEED")
    print("=" * 65)
    print("""
Based on the survey results above, here is the recommended strategy:

STEP 1 — Collect the candidate pool
  For each seed point s:
    a. Find all GT grasp points within radius r (e.g. 10 mm).
    b. Enumerate all (gt_point_idx, view_slot_idx) pairs where
       score > 0  AND  top_view_index != -1.
    c. Decode each pair into a full grasp tuple:
       (position, view_rot_3x3, rotation_class, depth_class, width, score).

STEP 2 — Score-weighted view clustering (diversity via farthest-point in SO(3))
  a. Sort candidates by score descending.
  b. Greedy pick: select candidate with highest score as grasp #1.
  c. For grasp #2: from remaining candidates, pick the one whose view
     direction is most angularly distant from already-selected views
     (dot product with selected views is minimized), but score >= threshold.
  d. For grasp #3: same farthest-point logic, maximizing minimum angular
     distance to the two already-selected view directions.

  Key parameter: score threshold = 0.3 (= raw score >= 3), to ensure
  quality while still providing enough candidates for diversity.

STEP 3 — Fallback for seed points with too few candidates (< 3 valid pairs)
  - If only 2 valid pairs exist: use them + a zero-padded dummy.
  - If only 1 valid pair exists: duplicate it 3 times (valid_mask=1 only for 1).
  - If 0 valid pairs: mark all 3 as invalid (valid_mask=0).

WHY THIS WORKS:
  - Diversity is maximized by penalizing views similar to already-selected ones,
    matching the multi-modal nature of grasp distributions (SO(3) spread).
  - Score threshold ensures we do not waste a "slot" on a poor quality grasp.
  - The farthest-point approach is simple, differentiable in spirit, and matches
    how EconomicGrasp already uses the 300-view template sphere.
""")


if __name__ == '__main__':
    main()
