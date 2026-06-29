import argparse
import os
import sys

import numpy as np
import open3d as o3d

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from dataset.graspnet_dataset import GraspNetDataset  # noqa: E402
from graspnetAPI import GraspGroup  # noqa: E402
from graspnetAPI.utils.eval_utils import (  # noqa: E402
    compute_closest_points,
    parse_posevector,
    transform_points,
    voxel_sample_points,
)
from graspnetAPI.utils.xmlhandler import xmlReader  # noqa: E402
from utils.arguments import cfgs  # noqa: E402
sys.argv = _ORIGINAL_ARGV


def infer_split_from_scene(scene_name):
    """Return the GraspNet test split name that contains a scene."""
    scene_id = int(scene_name.split('_')[-1])
    if 100 <= scene_id < 130:
        return 'test_seen'
    if 130 <= scene_id < 160:
        return 'test_similar'
    if 160 <= scene_id < 190:
        return 'test_novel'
    raise ValueError(f"Scene {scene_name} is not in the GraspNet test scenes.")


def resolve_grasp_path(args):
    """Resolve a dump frame .npy path from either explicit path or dump components."""
    if args.grasp_path:
        return args.grasp_path
    if args.scene_id is None or args.frame_id is None:
        raise ValueError("Pass --grasp_path or provide both --scene_id and --frame_id.")

    scene_name = f"scene_{args.scene_id:04d}"
    frame_name = f"{args.frame_id:04d}.npy"
    return os.path.join(args.dump_dir, args.split, scene_name, args.camera, frame_name)


def infer_frame_metadata(grasp_path, args):
    """Infer scene name, frame id, camera, and dataset split for a dump frame path."""
    frame_id = args.frame_id
    scene_name = f"scene_{args.scene_id:04d}" if args.scene_id is not None else None
    camera = args.camera

    path_parts = os.path.normpath(grasp_path).split(os.sep)
    if frame_id is None:
        frame_id = int(os.path.splitext(path_parts[-1])[0])
    if len(path_parts) >= 3 and scene_name is None and path_parts[-3].startswith('scene_'):
        scene_name = path_parts[-3]
    if len(path_parts) >= 2 and camera is None:
        camera = path_parts[-2]

    if scene_name is None:
        raise ValueError("Could not infer scene_name. Pass --scene_id or use a path containing scene_XXXX.")
    if camera is None:
        camera = cfgs.camera

    dataset_split = args.dataset_split or infer_split_from_scene(scene_name)
    return scene_name, int(frame_id), camera, dataset_split


def resolve_dataset_index(dataset, scene_name, frame_id):
    """Find the dataset index for a scene/frame pair."""
    for idx, (candidate_scene, candidate_frame) in enumerate(zip(dataset.scenename, dataset.frameid)):
        if candidate_scene == scene_name and int(candidate_frame) == int(frame_id):
            return idx
    raise ValueError(f"Could not find {scene_name} frame {frame_id} in split {dataset.split}.")


def make_open3d_cloud(points, colors=None):
    """Build an Open3D point cloud from numpy point and color arrays."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def filter_grasps_by_score(grasp_group, top_k, score_min):
    """Return score-sorted grasps, useful for debugging non-eval selection."""
    scores = grasp_group.scores
    keep_mask = np.ones(scores.shape[0], dtype=bool)
    if score_min is not None:
        keep_mask &= scores >= score_min

    kept_indices = np.nonzero(keep_mask)[0]
    if kept_indices.size == 0:
        return GraspGroup(np.zeros((0, 17), dtype=np.float32))

    kept_indices = kept_indices[np.argsort(scores[kept_indices])[::-1]]
    if top_k is not None and top_k > 0:
        kept_indices = kept_indices[:top_k]
    return GraspGroup(grasp_group.grasp_group_array[kept_indices])


def load_eval_models_and_poses(dataset_root, camera, scene_name, frame_id):
    """Load object point models and poses needed by GraspNetEval pre-selection."""
    annotation_path = os.path.join(
        dataset_root,
        'scenes',
        scene_name,
        camera,
        'annotations',
        f'{frame_id:04d}.xml',
    )
    posevectors = xmlReader(annotation_path).getposevectorlist()

    model_list = []
    pose_list = []
    for posevector in posevectors:
        object_id, pose = parse_posevector(posevector)
        model_path = os.path.join(dataset_root, 'models', f'{object_id:03d}', 'nontextured.ply')
        model = o3d.io.read_point_cloud(model_path)
        model_list.append(voxel_sample_points(np.asarray(model.points), 0.008))
        pose_list.append(pose)
    return model_list, pose_list


def select_eval_grasps(grasp_group, dataset_root, camera, scene_name, frame_id, top_k, max_width):
    """Select grasps with the same pre-evaluation policy used by GraspNetEval."""
    grasp_group = GraspGroup(np.array(grasp_group.grasp_group_array, copy=True))

    gg_array = grasp_group.grasp_group_array
    gg_array[gg_array[:, 1] < 0, 1] = 0
    gg_array[gg_array[:, 1] > max_width, 1] = max_width
    grasp_group.grasp_group_array = gg_array

    grasp_group = grasp_group.nms(0.03, 30.0 / 180.0 * np.pi)
    if len(grasp_group) == 0:
        return GraspGroup(np.zeros((0, 17), dtype=np.float32)), []

    model_list, pose_list = load_eval_models_and_poses(dataset_root, camera, scene_name, frame_id)

    model_trans_list = []
    seg_mask = []
    for object_idx, model in enumerate(model_list):
        model_trans = transform_points(model, pose_list[object_idx])
        model_trans_list.append(model_trans)
        seg_mask.append(object_idx * np.ones(model_trans.shape[0], dtype=np.int32))

    scene = np.concatenate(model_trans_list, axis=0)
    seg_mask = np.concatenate(seg_mask, axis=0)
    closest_indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[closest_indices]

    per_object_grasps = []
    per_object_counts = []
    for object_idx in range(len(model_list)):
        object_grasps = grasp_group[model_to_grasp == object_idx]
        object_grasps.sort_by_score()
        object_top10 = object_grasps[:10].grasp_group_array
        per_object_grasps.append(object_top10)
        per_object_counts.append(len(object_top10))

    non_empty = [grasps for grasps in per_object_grasps if len(grasps) > 0]
    if not non_empty:
        return GraspGroup(np.zeros((0, 17), dtype=np.float32)), per_object_counts

    all_grasps = np.vstack(non_empty)
    sorted_indices = np.argsort(all_grasps[:, 0])[::-1]
    selected = all_grasps[sorted_indices[:top_k]]
    return GraspGroup(selected.astype(np.float32, copy=False)), per_object_counts


def paint_grasps_by_score(grasp_group):
    """Convert grasps to Open3D geometries and color each grasp by its score."""
    geoms = grasp_group.to_open3d_geometry_list()
    scores = grasp_group.scores
    if len(geoms) == 0:
        return geoms

    score_min = float(scores.min())
    score_max = float(scores.max())
    denom = max(score_max - score_min, 1e-8)
    for geom, score in zip(geoms, scores):
        t = float((score - score_min) / denom)
        color = [1.0 - t, 0.25 + 0.6 * t, 0.15]
        geom.paint_uniform_color(color)
    return geoms


def main():
    """Visualize one predicted dump frame with its raw GraspNet point cloud."""
    parser = argparse.ArgumentParser(description="Visualize one GraspGroup .npy frame from a dump directory.")
    parser.add_argument('--grasp_path', type=str, default='',
                        help='Direct path to a saved .npy grasp file.')
    parser.add_argument('--dump_dir', type=str, default='dump_cfm/top8_neighbor_scored',
                        help='Dump root containing split/scene/camera/frame.npy.')
    parser.add_argument('--split', type=str, default='seen', choices=['seen', 'similar', 'novel'],
                        help='Dump split folder used when --grasp_path is omitted.')
    parser.add_argument('--scene_id', type=int, default=None,
                        help='Scene id used when --grasp_path is omitted.')
    parser.add_argument('--frame_id', type=int, default=None,
                        help='Frame id used when --grasp_path is omitted.')
    parser.add_argument('--dataset_root', type=str, default=cfgs.dataset_root,
                        help='GraspNet dataset root.')
    parser.add_argument('--dataset_split', type=str, default='',
                        choices=['', 'test_seen', 'test_similar', 'test_novel'],
                        help='Optional GraspNet dataset split override.')
    parser.add_argument('--camera', type=str, default=cfgs.camera, choices=['realsense', 'kinect'],
                        help='Camera stream.')
    parser.add_argument('--selection', type=str, default='eval', choices=['eval', 'score', 'all'],
                        help='Which grasps to visualize: GraspNetEval-selected, score-sorted, or all.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of grasps to visualize for eval/score modes. Use 0 to show all in score mode.')
    parser.add_argument('--score_min', type=float, default=None,
                        help='Optional minimum score threshold for score mode.')
    parser.add_argument('--max_width', type=float, default=0.1,
                        help='Maximum gripper width used by GraspNetEval before selection.')
    parser.add_argument('--point_size', type=float, default=1.0,
                        help='Open3D point size.')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Load data and print stats without opening the Open3D window.')
    args = parser.parse_args()

    grasp_path = resolve_grasp_path(args)
    if not os.path.isfile(grasp_path):
        raise FileNotFoundError(f"Grasp dump file not found: {grasp_path}")

    scene_name, frame_id, camera, dataset_split = infer_frame_metadata(grasp_path, args)
    dataset = GraspNetDataset(
        args.dataset_root,
        camera=camera,
        split=dataset_split,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
        load_label=False,
    )
    data_idx = resolve_dataset_index(dataset, scene_name, frame_id)
    cloud_points, cloud_colors = dataset.get_data(data_idx, return_raw_cloud=True)
    cloud = make_open3d_cloud(cloud_points, cloud_colors)

    all_grasps = GraspGroup().from_npy(grasp_path)
    if args.selection == 'eval':
        shown_grasps, per_object_counts = select_eval_grasps(
            all_grasps,
            args.dataset_root,
            camera,
            scene_name,
            frame_id,
            args.top_k,
            args.max_width,
        )
    elif args.selection == 'score':
        shown_grasps = filter_grasps_by_score(all_grasps, args.top_k, args.score_min)
        per_object_counts = None
    else:
        shown_grasps = all_grasps
        per_object_counts = None
    grasp_geoms = paint_grasps_by_score(shown_grasps)

    print(
        f"Loaded {grasp_path}: all_grasps={len(all_grasps)}, shown_grasps={len(shown_grasps)}, "
        f"selection={args.selection}, scene={scene_name}, frame={frame_id}, split={dataset_split}, camera={camera}",
        flush=True,
    )
    if per_object_counts is not None:
        print(
            f"GraspNetEval pre-selection: objects={len(per_object_counts)}, "
            f"per_object_top10_counts={per_object_counts}",
            flush=True,
        )
    if len(shown_grasps) > 0:
        print(
            f"Score range all=[{float(all_grasps.scores.min()):.6f}, {float(all_grasps.scores.max()):.6f}], "
            f"shown=[{float(shown_grasps.scores.min()):.6f}, {float(shown_grasps.scores.max()):.6f}]",
            flush=True,
        )

    if args.smoke_test:
        print("Smoke test done. Geometry was prepared but Open3D window was skipped.", flush=True)
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{scene_name} frame {frame_id} ({len(shown_grasps)} grasps)")
    vis.add_geometry(cloud)
    for geom in grasp_geoms:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.point_size = args.point_size
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
