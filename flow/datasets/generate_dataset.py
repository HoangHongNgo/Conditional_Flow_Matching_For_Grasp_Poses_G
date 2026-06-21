import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from flow.models.grasp_cfm import economic_graspable
from flow.utils.cfm_label_generation import process_grasp_labels
from utils.arguments import cfgs


KEEP_KEYS = {
    'xyz_graspable',
    'seed_features_graspable',
    'seed_grasp_rot_lie',
    'seed_grasp_width',
    'seed_grasp_depth',
    'seed_grasp_score',
    'seed_grasp_slot_mask',
    'seed_grasp_count',
    'seed_valid_mask',
}


def build_network(checkpoint_path, device):
    """Initialize frozen economic_graspable and load backbone/graspable weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    net = economic_graspable(seed_feat_dim=512, is_training=True)
    net.to(device)
    net.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    mapped_dict = {}
    for key, value in state_dict.items():
        name = key[7:] if key.startswith('module.') else key
        if name.startswith('backbone.') or name.startswith('graspable.'):
            mapped_dict[name] = value

    net.load_state_dict(mapped_dict, strict=False)
    for param in net.parameters():
        param.requires_grad = False

    print(f"Loaded {len(mapped_dict)} frozen weight keys from {checkpoint_path}")
    return net


def move_batch_to_device(batch_data, device):
    """Move tensors and nested list tensors in a GraspNet batch to device."""
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
    """Detach tensors from autograd and recursively move generated data to CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [to_cpu_detached(item) for item in value]
    return value


def slice_batch_value(value, batch_item_idx):
    """Extract one scene from a batched generated value."""
    if isinstance(value, torch.Tensor):
        return value[batch_item_idx:batch_item_idx + 1]
    if isinstance(value, list):
        return [value[batch_item_idx]]
    return value


def build_output_sample(end_points, batch_item_idx, scene_name, frame_id):
    """Build one compact frozen CFM sample from a batched end_points dict."""
    output = {}
    for key in KEEP_KEYS:
        output[key] = to_cpu_detached(slice_batch_value(end_points[key], batch_item_idx))

    output['scene_name'] = scene_name
    output['frame_id'] = frame_id
    return output


def generate_split(net, device, args, split):
    """Generate frozen seed-conditioned CFM labels for one GraspNet split."""
    dataset = GraspNetDataset(
        args.dataset_root,
        camera=args.camera,
        split=split,
        voxel_size=cfgs.voxel_size,
        num_points=cfgs.num_point,
        remove_outlier=True,
        augment=False,
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
    print(f"[{split}] Dataset size: {len(dataset)}")
    print(f"[{split}] Output directory: {save_dir}")

    saved_count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"[{split}] Generating")):
            if args.limit is not None and batch_idx >= args.limit:
                break

            batch_data = move_batch_to_device(batch_data, device)
            end_points = net(batch_data)
            end_points = process_grasp_labels(end_points)

            batch_size = end_points['seed_valid_mask'].shape[0]
            for batch_item_idx in range(batch_size):
                if not end_points['seed_valid_mask'][batch_item_idx].any():
                    continue

                dataset_idx = batch_idx * args.batch_size + batch_item_idx
                scene_name = dataset.scenename[dataset_idx]
                frame_id = dataset.frameid[dataset_idx]
                output_sample = build_output_sample(end_points, batch_item_idx, scene_name, frame_id)

                save_path = os.path.join(save_dir, f"sample_{saved_count:06d}.pt")
                torch.save(output_sample, save_path)
                saved_count += 1

    print(f"[{split}] Saved {saved_count} files.")
    return saved_count


def parse_args():
    """Parse command-line arguments for frozen CFM dataset generation."""
    parser = argparse.ArgumentParser(description="Generate frozen seed-conditioned 5D CFM dataset.")
    parser.add_argument('--dataset_root', type=str, default=cfgs.dataset_root, help='GraspNet dataset root.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/economicgrasp_realsense.tar',
                        help='Frozen economic_graspable checkpoint path.')
    parser.add_argument('--camera', type=str, default='realsense', choices=['realsense', 'kinect'],
                        help='Camera split to generate.')
    parser.add_argument('--splits', nargs='+', default=['train', 'eval', 'test'],
                        help='Dataset splits to generate.')
    parser.add_argument('--output_root', type=str,
                        default='/media/dsp520/Grasp_2T/graspnet/cfm_dataset_seed5d',
                        help='Output root for frozen .pt samples.')
    parser.add_argument('--batch_size', type=int, default=1, help='Generation batch size.')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers.')
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum batches per split.')
    parser.add_argument('--device', type=str, default=None, help='Device override, e.g. cuda:0 or cpu.')
    return parser.parse_args()


def main():
    """Generate all requested frozen CFM dataset splits."""
    args = parse_args()
    device_name = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    print(f"Using device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output root: {args.output_root}")
    print(f"Splits: {args.splits}")

    net = build_network(args.checkpoint_path, device)
    total_count = 0
    for split in args.splits:
        total_count += generate_split(net, device, args, split)

    print(f"Done. Saved {total_count} frozen CFM samples under {args.output_root}")


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
