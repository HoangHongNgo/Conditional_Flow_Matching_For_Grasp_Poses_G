import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from graspnetAPI import GraspGroup, GraspNetEval
from torch.utils.data import DataLoader

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ORIGINAL_ARGV = sys.argv[:]
if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
    sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in {'-h', '--help'}]

from flow.models.grasp_cfm import GraspVelocityMLP
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction
from flow.models.scoring_network import ScoringMLP, logits_to_expected_score
from flow.tests.test_cfm import (
    extract_scene_inputs,
    generate_averaged_cfm_grasps,
    load_evaluation_dataset,
    resolve_cfm_checkpoint,
    resolve_dataset_index,
    select_top_seed_indices,
)
from flow.utils.lie import exp_so3
from models.economicgrasp import economicgrasp
from utils.arguments import cfgs
from utils.collision_detector import ModelFreeCollisionDetector

sys.argv = _ORIGINAL_ARGV


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

    base_net = build_base_network(device)
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
                if 'seed_graspness_graspable' in batch_items[0] and 'seed_object_ids_graspable' in batch_items[0]:
                    seed_object_ids = torch.cat(
                        [item['seed_object_ids_graspable'].to(device) for item in batch_items],
                        0,
                    )
                else:
                    seed_object_ids = torch.full(
                        (seed_xyz.shape[0], seed_xyz.shape[1]),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
            else:
                batch_data = move_batch_to_device(batch_data, device)
                batch_items = None
                seed_xyz, seed_feats, _, seed_object_ids = extract_scene_inputs(base_net, batch_data)

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
            keep_mask = torch.ones(x_pred.shape[1], dtype=torch.bool, device=device)  # [N]

            if cfgs.collision_thresh > 0:
                cloud, _ = raw_dataset.get_data(raw_indices[b], return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(
                    gg_all,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                keep_mask = torch.from_numpy(~collision_mask).to(device=device, dtype=torch.bool)

            selected_indices = select_top_seed_indices(
                scoring_scores[b],
                seed_object_ids[b],
                keep_mask,
                max_per_object=args.max_per_object,
                total_top_k=args.total_top_k,
            )
            selected_np = selected_indices.detach().cpu().numpy()
            gg_selected = GraspGroup(gg_all.grasp_group_array[selected_np])

            save_scene_dir = os.path.join(save_dir, scene_names[b], cfgs.camera)
            save_path = os.path.join(save_scene_dir, str(frame_ids[b]).zfill(4) + '.npy')
            os.makedirs(save_scene_dir, exist_ok=True)
            gg_selected.save_npy(save_path)

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
    parser.add_argument('--max_per_object', type=int, default=10,
                        help='Maximum selected grasps per object before scene-level top-k.')
    parser.add_argument('--total_top_k', type=int, default=50,
                        help='Number of scored grasps saved per frame.')
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
