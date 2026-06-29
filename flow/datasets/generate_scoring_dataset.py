import argparse
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


KEEP_KEYS = {
    'xyz_graspable',
    'seed_group_features',
    'seed_grasp_rot_lie',
    'seed_valid_mask',
}


def pack_discrete_grasp_labels(end_points, batch_item_idx):
    """Pack score, width, and depth labels into lossless uint8 tensors for disk."""
    score = slice_batch_value(end_points['seed_grasp_score'], batch_item_idx)
    width = slice_batch_value(end_points['seed_grasp_width'], batch_item_idx)
    depth = slice_batch_value(end_points['seed_grasp_depth'], batch_item_idx)

    score_uint8 = torch.clamp(torch.round(score * 10.0), 0, 10).to(torch.uint8)
    width_uint8 = torch.clamp(torch.round(width * 1000.0), 0, 255).to(torch.uint8)
    depth_uint8 = torch.clamp(torch.round((depth - 0.01) / 0.01), 0, 3).to(torch.uint8)

    return {
        'seed_grasp_score_uint8': to_cpu_detached(score_uint8),
        'seed_grasp_width_uint8': to_cpu_detached(width_uint8),
        'seed_grasp_depth_uint8': to_cpu_detached(depth_uint8),
    }


def pack_half_precision_features(end_points, batch_item_idx):
    """Pack dense float feature tensors as float16 to reduce disk usage."""
    return {
        'seed_grasp_rot_lie': to_cpu_detached(
            slice_batch_value(end_points['seed_grasp_rot_lie'], batch_item_idx).to(torch.float16)
        ),
        'seed_group_features': to_cpu_detached(
            slice_batch_value(end_points['seed_group_features'], batch_item_idx).to(torch.float16)
        ),
    }


def move_batch_to_device(batch_data, device):
    """Move tensors and nested list tensors in a GraspNet batch to the selected device."""
    for key, value in batch_data.items():
        if 'list' in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    if isinstance(value[i][j], torch.Tensor):
                        value[i][j] = value[i][j].to(device)
        elif isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
    return batch_data


def to_cpu_detached(value):
    """Detach a generated value from autograd and move it to CPU recursively."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [to_cpu_detached(item) for item in value]
    return value


def slice_batch_value(value, batch_item_idx):
    """Extract one sample from a batched tensor or list value."""
    if isinstance(value, torch.Tensor):
        return value[batch_item_idx:batch_item_idx + 1]
    if isinstance(value, list):
        return [value[batch_item_idx]]
    return value


def build_output_sample(end_points, batch_item_idx, scene_name, frame_id):
    """Build one compact scoring-network sample from generated end_points."""
    output = {}
    for key in KEEP_KEYS:
        if key in {'seed_grasp_rot_lie', 'seed_group_features'}:
            continue
        output[key] = to_cpu_detached(slice_batch_value(end_points[key], batch_item_idx))
    output.update(pack_half_precision_features(end_points, batch_item_idx))
    output.update(pack_discrete_grasp_labels(end_points, batch_item_idx))

    output['scene_name'] = scene_name
    output['frame_id'] = frame_id
    return output


def get_contiguous_saved_count(save_dir):
    """Return the count of sequential sample files already present in save_dir."""
    prefix = 'sample_'
    suffix = '.pt'
    saved_indices = set()

    if not os.path.isdir(save_dir):
        return 0

    for entry in os.scandir(save_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        index_text = name[len(prefix):-len(suffix)]
        if index_text.isdigit():
            saved_indices.add(int(index_text))

    contiguous_count = 0
    while contiguous_count in saved_indices:
        contiguous_count += 1
    return contiguous_count


def save_output_sample(output_sample, save_path):
    """Atomically save one generated sample and report disk-space failures clearly."""
    save_dir = os.path.dirname(save_path)
    free_bytes = shutil.disk_usage(save_dir).free
    minimum_free_bytes = 256 * 1024 * 1024
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"Not enough free disk space to continue writing dataset files in {save_dir}. "
            f"Only {free_bytes / (1024 ** 3):.2f} GiB available."
        )

    tmp_path = f"{save_path}.tmp"
    try:
        torch.save(output_sample, tmp_path)
        os.replace(tmp_path, save_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        free_bytes = shutil.disk_usage(save_dir).free
        raise RuntimeError(
            f"Failed to save dataset sample to {save_path}. "
            f"Free space remaining in {save_dir}: {free_bytes / (1024 ** 3):.2f} GiB."
        ) from exc


def build_base_network(checkpoint_path, device):
    """Initialize economic_graspable and load frozen backbone/graspable weights."""
    from flow.models.grasp_cfm import economic_graspable

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Base checkpoint not found at {checkpoint_path}")

    net = economic_graspable(seed_feat_dim=512, is_training=True)
    net.to(device)
    net.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    mapped_dict = {}
    for key, value in state_dict.items():
        name = key[7:] if key.startswith('module.') else key
        if name.startswith('backbone.') or name.startswith('graspable.'):
            mapped_dict[name] = value

    net.load_state_dict(mapped_dict, strict=False)
    for param in net.parameters():
        param.requires_grad = False

    print(f"Loaded {len(mapped_dict)} base weight keys from {checkpoint_path}")
    return net


def build_seed_conditioner(checkpoint_path, device, nsample=None, sphere_radius=None):
    """Initialize the trained seed grouping module from a FlowGrasp checkpoint."""
    from flow.models.modules_flow import Sphere_Grouping_Global_Interaction

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"CFM checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get('args', {})
    resolved_nsample = nsample if nsample is not None else int(ckpt_args.get('nsample', 32))
    resolved_radius = sphere_radius if sphere_radius is not None else float(ckpt_args.get('sphere_radius', 0.005))

    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=resolved_nsample,
        seed_feature_dim=512,
        sphere_radius=resolved_radius,
    ).to(device)
    seed_conditioner.load_state_dict(checkpoint['seed_conditioner_state_dict'])
    seed_conditioner.eval()

    for param in seed_conditioner.parameters():
        param.requires_grad = False

    print(
        "Loaded seed conditioner from "
        f"{checkpoint_path} (nsample={resolved_nsample}, sphere_radius={resolved_radius})"
    )
    return seed_conditioner


def generate_split(base_net, seed_conditioner, device, args, split):
    """Generate one scoring dataset split from GraspNet frames."""
    from dataset.graspnet_dataset import GraspNetDataset, collate_fn
    from flow.utils.cfm_label_generation import process_scoring_label

    dataset = GraspNetDataset(
        args.dataset_root,
        camera=args.camera,
        split=split,
        voxel_size=args.voxel_size,
        num_points=args.num_point,
        remove_outlier=True,
        augment=False,
        load_label=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    save_dir = os.path.join(args.output_root, split)
    os.makedirs(save_dir, exist_ok=True)
    resume_count = get_contiguous_saved_count(save_dir)

    print(f"[{split}] Dataset size: {len(dataset)}")
    print(f"[{split}] Output directory: {save_dir}")
    if resume_count > 0:
        print(f"[{split}] Resuming from existing samples: {resume_count}")

    saved_count = resume_count
    valid_sample_count = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"[{split}] Generating scoring data")):
            if args.limit is not None and batch_idx >= args.limit:
                break

            batch_data = move_batch_to_device(batch_data, device)
            end_points = base_net(batch_data)

            group_features = seed_conditioner(
                end_points['xyz_graspable'].contiguous(),
                end_points['seed_features_graspable'].contiguous(),
            )
            end_points['seed_group_features'] = group_features.transpose(1, 2).contiguous()
            end_points = process_scoring_label(end_points)

            batch_size = end_points['xyz_graspable'].shape[0]
            for batch_item_idx in range(batch_size):
                if not end_points['seed_valid_mask'][batch_item_idx].any():
                    continue

                dataset_idx = batch_idx * args.batch_size + batch_item_idx
                scene_name = dataset.scenename[dataset_idx]
                frame_id = dataset.frameid[dataset_idx]
                if valid_sample_count < resume_count:
                    valid_sample_count += 1
                    continue

                output_sample = build_output_sample(
                    end_points,
                    batch_item_idx,
                    scene_name,
                    frame_id,
                )
                save_path = os.path.join(save_dir, f"sample_{saved_count:06d}.pt")
                save_output_sample(output_sample, save_path)
                saved_count += 1
                valid_sample_count += 1

    print(f"[{split}] Saved {saved_count - resume_count} new files ({saved_count} total).")
    return saved_count


def parse_args():
    """Parse command-line arguments for scoring dataset generation."""
    parser = argparse.ArgumentParser(description="Generate scoring-network seed feature and label datasets.")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet',
                        help='GraspNet dataset root.')
    parser.add_argument('--base_checkpoint_path', type=str, default='checkpoints/economicgrasp_realsense.tar',
                        help='Checkpoint containing economic_graspable backbone/graspable weights.')
    parser.add_argument('--cfm_checkpoint_path', type=str,
                        default='flow/results/top8_neighbor/flowgrasp_latest.tar',
                        help='FlowGrasp checkpoint containing the trained seed conditioner.')
    parser.add_argument('--camera', type=str, default='realsense', choices=['realsense', 'kinect'],
                        help='Camera split to generate.')
    parser.add_argument('--splits', nargs='+', default=['train', 'eval'],
                        help='Dataset splits to generate.')
    parser.add_argument('--output_root', type=str,
                        default='/media/dsp520/Grasp_2T/graspnet/scoring_dataset_top8_neighbor',
                        help='Output root for generated scoring .pt samples.')
    parser.add_argument('--batch_size', type=int, default=1, help='Generation batch size.')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers.')
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum batches per split.')
    parser.add_argument('--device', type=str, default=None, help='Device override, e.g. cuda:0 or cpu.')
    parser.add_argument('--num_point', type=int, default=20000,
                        help='Number of scene points sampled by GraspNetDataset.')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel size used by GraspNetDataset and sparse backbone.')
    parser.add_argument('--m_point', type=int, default=1024,
                        help='Number of graspable seed points produced by economic_graspable.')
    parser.add_argument('--num_view', type=int, default=300,
                        help='Number of grasp view slots kept per seed.')
    parser.add_argument('--nsample', type=int, default=None,
                        help='Override seed conditioner nsample. Defaults to checkpoint args.')
    parser.add_argument('--sphere_radius', type=float, default=None,
                        help='Override seed conditioner radius. Defaults to checkpoint args.')
    return parser.parse_args()


def main():
    """Generate scoring datasets for all requested splits."""
    args = parse_args()
    device_name = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    print(f"Using device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Base checkpoint: {args.base_checkpoint_path}")
    print(f"CFM checkpoint: {args.cfm_checkpoint_path}")
    print(f"Output root: {args.output_root}")
    print(f"Splits: {args.splits}")

    base_net = build_base_network(args.base_checkpoint_path, device)
    seed_conditioner = build_seed_conditioner(
        args.cfm_checkpoint_path,
        device,
        nsample=args.nsample,
        sphere_radius=args.sphere_radius,
    )

    total_count = 0
    for split in args.splits:
        total_count += generate_split(base_net, seed_conditioner, device, args, split)

    print(f"Done. Saved {total_count} scoring samples under {args.output_root}")


if __name__ == '__main__':
    main()
