import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from graspnetAPI import GraspGroup, GraspNetEval
from torch.utils.data import DataLoader, Dataset

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ORIGINAL_ARGV = sys.argv[:]
if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
    sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in {'-h', '--help'}]

from flow.models.grasp_cfm import GraspVelocityMLP
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction
from flow.models.scoring_network import ScoringMLP, logits_to_expected_score
from flow.utils.cfm_solver import euler_solve
from flow.utils.lie import exp_so3
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from models.economicgrasp import economicgrasp
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetector
import MinkowskiEngine as ME

sys.argv = _ORIGINAL_ARGV


class CachedCFMSplitDataset(Dataset):
    """Load cached seed-conditioned CFM samples saved by generate_dataset.py."""

    def __init__(self, split_dir, split=None):
        """Initialize the cached split dataset from one directory of .pt files."""
        self.split_dir = split_dir
        files = sorted(
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if name.endswith('.pt')
        )
        if split is not None:
            files = [
                file_path for file_path in files
                if cached_file_belongs_to_split(file_path, split)
            ]
        self.files = files
        if not self.files:
            raise FileNotFoundError(f"No .pt files found under {split_dir}")

    def __len__(self):
        """Return the number of cached samples."""
        return len(self.files)

    def __getitem__(self, idx):
        """Load one cached CFM sample."""
        return torch.load(self.files[idx], map_location='cpu', weights_only=False)


def resolve_dataset_index(raw_dataset, scene_name, frame_id):
    """Map a scene/frame pair to the corresponding GraspNet dataset index."""
    return raw_dataset.scenename.index(scene_name) + int(frame_id)


def scene_name_belongs_to_split(scene_name, split):
    """Return whether a GraspNet test scene belongs to one evaluation split."""
    scene_id = int(scene_name.split('_')[-1])
    if split == 'test_seen':
        return 100 <= scene_id < 130
    if split == 'test_similar':
        return 130 <= scene_id < 160
    if split == 'test_novel':
        return 160 <= scene_id < 190
    return True


def cached_file_belongs_to_split(file_path, split):
    """Return whether a sequential sample_XXXXXX.pt cache file belongs to a test split."""
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    if not file_stem.startswith('sample_') or not file_stem[7:].isdigit():
        sample = torch.load(file_path, map_location='cpu', weights_only=False)
        return scene_name_belongs_to_split(sample['scene_name'], split)

    sample_idx = int(file_stem[7:])
    scene_id = 100 + (sample_idx // 256)
    if split == 'test_seen':
        return 100 <= scene_id < 130
    if split == 'test_similar':
        return 130 <= scene_id < 160
    if split == 'test_novel':
        return 160 <= scene_id < 190
    return True


def resolve_cached_split_dir(cached_dataset_root, split):
    """Resolve either root/test_seen-style or root/test-style cached datasets."""
    split_dir = os.path.join(cached_dataset_root, split)
    if os.path.isdir(split_dir):
        return split_dir, None

    test_dir = os.path.join(cached_dataset_root, 'test')
    if os.path.isdir(test_dir):
        return test_dir, split

    raise FileNotFoundError(
        f"Expected cached split directory {split_dir} or combined test directory {test_dir}."
    )


def resolve_cfm_checkpoint(checkpoint_path):
    """Resolve a CFM checkpoint path, preferring the newest seed-conditioner checkpoint."""
    if checkpoint_path:
        return checkpoint_path

    candidates = glob.glob("flow/results/*/flowgrasp_latest.tar")
    if not candidates:
        raise FileNotFoundError("No flowgrasp_latest.tar found under flow/results/*/.")

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def resolve_scoring_checkpoint(checkpoint_path):
    """Resolve a scoring-network checkpoint path, using the newest scoring_latest.tar when omitted."""
    if checkpoint_path:
        return checkpoint_path

    candidates = glob.glob("flow/results/**/scoring_latest.tar", recursive=True)
    if not candidates:
        raise FileNotFoundError("No scoring_latest.tar found under flow/results/**/.")

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def build_base_network(device):
    """Load the frozen EconomicGrasp backbone used to extract seed features."""
    if cfgs.checkpoint_path is None:
        cfgs.checkpoint_path = f"checkpoints/economicgrasp_{cfgs.camera}.tar"

    print(f"Loading base net checkpoint: {cfgs.checkpoint_path}")
    base_net = economicgrasp(seed_feat_dim=512, is_training=False).to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device, weights_only=False)
    base_net.load_state_dict(checkpoint['model_state_dict'])
    base_net.eval()
    return base_net


def build_cfm_models(args, device):
    """Load the trained seed-conditioned CFM pose generator."""
    cfm_checkpoint_path = resolve_cfm_checkpoint(args.cfm_checkpoint_path)
    print(f"Loading CFM checkpoint: {cfm_checkpoint_path}")
    cfm_checkpoint = torch.load(cfm_checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = cfm_checkpoint.get('args', {})

    nsample = args.nsample if args.nsample is not None else int(ckpt_args.get('nsample', 32))
    sphere_radius = (
        args.sphere_radius if args.sphere_radius is not None
        else float(ckpt_args.get('sphere_radius', 0.005))
    )

    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=nsample,
        seed_feature_dim=512,
        sphere_radius=sphere_radius,
    ).to(device)
    mlp = GraspVelocityMLP(grasp_dim=5, cond_dim=128).to(device)

    if 'seed_conditioner_state_dict' not in cfm_checkpoint:
        raise KeyError(
            f"{cfm_checkpoint_path} is not a seed-conditioned CFM checkpoint. "
            "Pass a checkpoint from flow/results/*/flowgrasp_latest.tar."
        )
    seed_conditioner.load_state_dict(cfm_checkpoint['seed_conditioner_state_dict'])
    mlp.load_state_dict(cfm_checkpoint['mlp_state_dict'])
    seed_conditioner.eval()
    mlp.eval()
    return seed_conditioner, mlp, cfm_checkpoint_path


def build_scoring_model(args, device):
    """Load the trained scoring network used to score each generated grasp pose."""
    scoring_checkpoint_path = resolve_scoring_checkpoint(args.scoring_checkpoint_path)
    print(f"Loading scoring checkpoint: {scoring_checkpoint_path}")
    checkpoint = torch.load(scoring_checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get('args', {})

    model = ScoringMLP(
        input_dim=int(ckpt_args.get('input_dim', args.scoring_input_dim)),
        num_classes=int(ckpt_args.get('num_classes', args.scoring_num_classes)),
        hidden_dim=int(ckpt_args.get('hidden_dim', args.scoring_hidden_dim)),
        dropout=float(ckpt_args.get('dropout', args.scoring_dropout)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, scoring_checkpoint_path


def extract_scene_inputs(base_net, batch_data):
    """Extract seed coordinates, features, graspness scores, and object ids from a raw batch."""
    seed_xyz = batch_data['point_clouds']  # [B, point_num, 3]
    B, point_num, _ = seed_xyz.shape

    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [coord for coord in batch_data['coordinates_for_voxel']],
        [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)],
    )
    coordinates_batch, features_batch, _, batch_data['quantize2original'] = ME.utils.sparse_quantize(
        coordinates_batch,
        features_batch,
        return_index=True,
        return_inverse=True,
    )

    device = seed_xyz.device
    coordinates_batch = coordinates_batch.to(device)
    features_batch = features_batch.to(device)
    mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)

    seed_features = base_net.backbone(mink_input).F
    seed_features = seed_features[batch_data['quantize2original']].view(
        B,
        point_num,
        -1,
    ).transpose(1, 2)  # [B, 512, point_num]

    batch_data = base_net.graspable(seed_features, batch_data)
    seed_features_flipped = seed_features.transpose(1, 2)
    objectness_score = batch_data['objectness_score']  # [B, 2, point_num]
    graspness_score = batch_data['graspness_score'].squeeze(1)  # [B, point_num]
    segmentation_label = batch_data.get('segmentation_label')
    if segmentation_label is not None:
        segmentation_label = segmentation_label.long()

    objectness_pred = torch.argmax(objectness_score, 1)
    graspable_mask = (objectness_pred == 1) & (graspness_score > cfgs.graspness_threshold)

    seed_xyz_graspable = []
    seed_features_graspable = []
    seed_graspness_graspable = []
    seed_object_ids_graspable = []
    for i in range(B):
        cur_mask = graspable_mask[i]
        if cur_mask.sum() == 0:
            cur_mask = torch.ones_like(cur_mask, dtype=torch.bool)

        cur_feat = seed_features_flipped[i][cur_mask]
        cur_seed_xyz = seed_xyz[i][cur_mask].unsqueeze(0)  # [1, M, 3]
        cur_graspness = graspness_score[i][cur_mask]
        if segmentation_label is None:
            cur_object_ids = torch.full_like(cur_graspness, fill_value=-1, dtype=torch.long)
        else:
            cur_object_ids = segmentation_label[i][cur_mask]
        fps_idxs = furthest_point_sample(cur_seed_xyz, base_net.M_points)  # [1, 1024]
        fps_idxs_flat = fps_idxs.squeeze(0).long()

        cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
        cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(
            1,
            2,
        ).squeeze(0).contiguous()

        cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
        cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()
        cur_graspness = cur_graspness[fps_idxs_flat].contiguous()
        cur_object_ids = cur_object_ids[fps_idxs_flat].contiguous()

        seed_xyz_graspable.append(cur_seed_xyz)
        seed_features_graspable.append(cur_feat)
        seed_graspness_graspable.append(cur_graspness)
        seed_object_ids_graspable.append(cur_object_ids)

    return (
        torch.stack(seed_xyz_graspable, 0),  # [B, 1024, 3]
        torch.stack(seed_features_graspable),  # [B, 512, 1024]
        torch.stack(seed_graspness_graspable, 0),  # [B, 1024]
        torch.stack(seed_object_ids_graspable, 0),  # [B, 1024]
    )


def generate_averaged_cfm_grasps(seed_conditioner, mlp, seed_xyz, seed_feats, norm_metadata, args, device):
    """Generate CFM grasps multiple times and average the 5D pose per seed."""
    num_samples = max(1, int(args.num_generation_samples))
    B, num_seed, _ = seed_xyz.shape  # [B, 1024, 3]
    x_pred_sum = None

    for _ in range(num_samples):
        x0 = torch.randn(B, num_seed, 5, device=device)  # [B, 1024, 5]
        x_pred_sample = euler_solve(
            seed_conditioner,
            mlp,
            x0,
            seed_xyz,
            seed_feats,
            norm_metadata,
            n_steps=args.n_steps,
        )  # [B, 1024, 5]
        if x_pred_sum is None:
            x_pred_sum = x_pred_sample
        else:
            x_pred_sum = x_pred_sum + x_pred_sample

    return x_pred_sum / float(num_samples)


@torch.no_grad()
def score_generated_grasps(scoring_model, seed_conditioner, x_pred, seed_xyz, seed_feats):
    """Score every generated grasp using [grasp_5d, seed_cond_128] inputs."""
    seed_cond = seed_conditioner(seed_xyz, seed_feats).transpose(1, 2).contiguous()  # [B, N, 128]
    scoring_input = torch.cat([x_pred, seed_cond], dim=-1).contiguous()  # [B, N, 133]
    logits = scoring_model(scoring_input)  # [B, N, 11]
    return logits_to_expected_score(logits)  # [B, N]


def graspgroup_from_5d(x_pred, seed_xyz, score_values):
    """Convert generated 5D grasp poses and scoring-network scores into a GraspGroup."""
    num_grasp = x_pred.shape[0]
    omega = x_pred[:, :3]  # [N, 3]
    width = x_pred[:, 3]  # [N]
    depth = x_pred[:, 4]  # [N]

    rot_flat = exp_so3(omega).reshape(num_grasp, 9)  # [N, 9]
    score = score_values.view(-1, 1)  # [N, 1]
    grasp_width = torch.clamp(width, min=0.0, max=0.1).view(-1, 1)
    grasp_depth = torch.clamp(depth, min=0.01, max=0.04).view(-1, 1)
    grasp_height = 0.02 * torch.ones_like(score)
    obj_ids = -1 * torch.ones_like(score)

    gg_preds = torch.cat(
        [
            score,
            grasp_width,
            grasp_height,
            grasp_depth,
            rot_flat,
            seed_xyz,
            obj_ids,
        ],
        dim=-1,
    )
    return GraspGroup(gg_preds.detach().cpu().numpy())


def move_batch_to_device(batch_data, device):
    """Move one collated GraspNet batch to the selected device."""
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def resolve_frame_metadata(args, raw_dataset, batch_idx, batch_size, batch_items, scene_list):
    """Return scene names, frame ids, and raw dataset indices for one output batch."""
    if args.cached_dataset_root:
        scene_names = [item['scene_name'] for item in batch_items]
        frame_ids = [int(item['frame_id']) for item in batch_items]
        raw_indices = [
            resolve_dataset_index(raw_dataset, scene_name, frame_id)
            for scene_name, frame_id in zip(scene_names, frame_ids)
        ]
        return scene_names, frame_ids, raw_indices

    raw_indices = [batch_idx * cfgs.batch_size + b for b in range(batch_size)]
    scene_names = [scene_list[data_idx] for data_idx in raw_indices]
    frame_ids = [data_idx % 256 for data_idx in raw_indices]
    return scene_names, frame_ids, raw_indices


def resolve_cached_dataset_source(cached_dataset_root, split):
    """Resolve cached test data from either a root directory or a direct test directory."""
    normalized_root = os.path.abspath(cached_dataset_root)

    if os.path.basename(normalized_root) == 'test' and os.path.isdir(normalized_root):
        return normalized_root, split

    return resolve_cached_split_dir(normalized_root, split)


def load_evaluation_dataset(args, split):
    """Build the evaluation dataset for one split, supporting raw and cached test inputs."""
    if args.cached_dataset_root:
        split_dir, filter_split = resolve_cached_dataset_source(args.cached_dataset_root, split)
        dataset = CachedCFMSplitDataset(split_dir, split=filter_split)
        collate = lambda batch: batch
        scene_list = None
        raw_dataset = economicgrasp_raw_dataset(split)
        return dataset, collate, scene_list, raw_dataset

    dataset = economicgrasp_raw_dataset(split)
    return dataset, flow_collate_fn, dataset.scene_list(), dataset


def economicgrasp_raw_dataset(split):
    """Build the raw GraspNet dataset used for metadata lookup and uncached evaluation."""
    from dataset.graspnet_dataset import GraspNetDataset

    return GraspNetDataset(
        cfgs.dataset_root,
        split=split,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        remove_outlier=True,
        load_label=False,
        augment=False,
    )


def flow_collate_fn(batch):
    """Import the project GraspNet collate function lazily to avoid side effects at module import."""
    from dataset.graspnet_dataset import collate_fn

    return collate_fn(batch)


def evaluate_split(args, split, save_dir):
    """Run CFM inference, score all generated grasps, save results, and evaluate one split."""
    device_name = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(f"Initializing dataset for split: {split}")

    dataset, collate, scene_list, raw_dataset = load_evaluation_dataset(args, split)
    dataloader = DataLoader(
        dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=0 if args.smoke_test else args.num_workers,
        collate_fn=collate,
    )
    print(f"Dataset size: {len(dataset)}. Batches: {len(dataloader)}")

    base_net = None if args.cached_dataset_root else build_base_network(device)
    seed_conditioner, mlp, _ = build_cfm_models(args, device)
    scoring_model, _ = build_scoring_model(args, device)
    norm_metadata = torch.load(args.stats_path, map_location=device, weights_only=True)
    print(f"Loaded normalization metadata from: {args.stats_path}")

    os.makedirs(save_dir, exist_ok=True)
    tic = time.time()
    print(f"\nRunning CFM inference and scoring with {args.num_generation_samples} generation sample(s)...")

    for batch_idx, batch_data in enumerate(dataloader):
        with torch.no_grad():
            if args.cached_dataset_root:
                batch_items = batch_data if isinstance(batch_data, list) else [batch_data]
                seed_xyz = torch.cat([item['xyz_graspable'].to(device) for item in batch_items], 0)
                seed_feats = torch.cat([item['seed_features_graspable'].to(device) for item in batch_items], 0)
            else:
                batch_data = move_batch_to_device(batch_data, device)
                batch_items = None
                seed_xyz, seed_feats, _, _ = extract_scene_inputs(base_net, batch_data)

            batch_size = seed_xyz.shape[0]
            scene_names, frame_ids, raw_indices = resolve_frame_metadata(
                args,
                raw_dataset,
                batch_idx,
                batch_size,
                batch_items,
                scene_list,
            )

            x_pred = generate_averaged_cfm_grasps(
                seed_conditioner,
                mlp,
                seed_xyz,
                seed_feats,
                norm_metadata,
                args,
                device,
            )  # [B, N, 5]
            scoring_scores = score_generated_grasps(
                scoring_model,
                seed_conditioner,
                x_pred,
                seed_xyz,
                seed_feats,
            )  # [B, N]

        for b in range(batch_size):
            gg_all = graspgroup_from_5d(x_pred[b], seed_xyz[b], scoring_scores[b])
            gg_to_save = gg_all

            if cfgs.collision_thresh > 0:
                cloud, _ = raw_dataset.get_data(raw_indices[b], return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(
                    gg_all,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                gg_to_save = gg_all[~collision_mask]

            save_scene_dir = os.path.join(save_dir, scene_names[b], cfgs.camera)
            save_path = os.path.join(save_scene_dir, str(frame_ids[b]).zfill(4) + '.npy')
            os.makedirs(save_scene_dir, exist_ok=True)
            gg_to_save.save_npy(save_path)

        if batch_idx % 20 == 0:
            print(f"Evaluated batch: {batch_idx}, elapsed: {time.time() - tic:.2f}s")

        if args.smoke_test and batch_idx >= 3:
            print("Smoke break triggered.")
            break


def evaluate_cfm(args):
    """Evaluate generated CFM grasps after rescoring them with the scoring network."""
    split_choices = ['seen', 'similar', 'novel']
    splits = split_choices if args.split == 'all' else [args.split]
    cfm_checkpoint_path = resolve_cfm_checkpoint(args.cfm_checkpoint_path)
    scoring_checkpoint_path = resolve_scoring_checkpoint(args.scoring_checkpoint_path)
    print(f"Evaluating CFM checkpoint: {cfm_checkpoint_path}")
    print(f"Scoring with checkpoint: {scoring_checkpoint_path}")
    print(f"Evaluating split(s): {', '.join(splits)}")

    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    default_save_name = "all_splits_scored" if args.split == 'all' else f"{args.split}_scored"
    if args.num_generation_samples > 1:
        default_save_name = f"{default_save_name}_avg{args.num_generation_samples}"
    base_save_dir = cfgs.save_dir or os.path.join("dump_cfm", default_save_name)
    os.makedirs(base_save_dir, exist_ok=True)

    for split in splits:
        split_save_dir = os.path.join(base_save_dir, split) if args.split == 'all' else base_save_dir
        evaluate_split(args, f"test_{split}", split_save_dir)
        if args.smoke_test:
            continue

        print(f"\nInference finished for {split}. Running graspnetAPI evaluation...")
        if split == 'seen':
            res, _ = ge.eval_seen(split_save_dir, proc=args.eval_proc)
        elif split == 'similar':
            res, _ = ge.eval_similar(split_save_dir, proc=args.eval_proc)
        else:
            res, _ = ge.eval_novel(split_save_dir, proc=args.eval_proc)
        np.save(os.path.join(split_save_dir, f"ap_{cfgs.camera}_{split}_scored.npy"), res)
        print(f"{split} split AP 0.8: {np.mean(res[:, :, :, 3])}, AP 0.4: {np.mean(res[:, :, :, 1])}")


def parse_args():
    """Parse command-line arguments for scored CFM evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate CFM generated grasps with ScoringMLP scores.")
    parser.add_argument('--cfm_checkpoint_path', type=str, default='',
                        help='Path to seed-conditioned CFM checkpoint. Empty uses latest under flow/results/*/.')
    parser.add_argument('--scoring_checkpoint_path', type=str, default='',
                        help='Path to scoring_latest.tar. Empty uses newest under flow/results/**/.')
    parser.add_argument('--cached_dataset_root', type=str, default='',
                        help='Optional root produced by flow/datasets/generate_dataset.py with split subdirs.')
    parser.add_argument('--split', type=str, default='all', choices=['seen', 'similar', 'novel', 'all'],
                        help='Evaluation split to run. Use all to evaluate seen, similar, and novel.')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_seed5d_norm_stats.pt',
                        help='Path to fixed normalization metadata.')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of Euler steps for CFM inference.')
    parser.add_argument('--nsample', type=int, default=None, help='Override CFM seed-conditioner neighbor count.')
    parser.add_argument('--sphere_radius', type=float, default=None, help='Override CFM seed-conditioner radius.')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers for full evaluation.')
    parser.add_argument('--num_generation_samples', type=int, default=1,
                        help='Number of CFM generations to average per seed before scoring.')
    parser.add_argument('--eval_proc', type=int, default=6, help='Number of graspnetAPI evaluation workers.')
    parser.add_argument('--device', type=str, default='', help='Device override, e.g. cuda:0 or cpu.')
    parser.add_argument('--scoring_input_dim', type=int, default=133, help='Fallback scoring-model input dimension.')
    parser.add_argument('--scoring_num_classes', type=int, default=11, help='Fallback scoring-model class count.')
    parser.add_argument('--scoring_hidden_dim', type=int, default=256, help='Fallback scoring-model hidden dimension.')
    parser.add_argument('--scoring_dropout', type=float, default=0.1, help='Fallback scoring-model dropout.')
    parser.add_argument('--smoke_test', action='store_true', help='Run a short inference-only smoke test.')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    evaluate_cfm(parse_args())
