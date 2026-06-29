import argparse
import os
import shutil
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

TEST_KEEP_KEYS = {
    'xyz_graspable',
    'seed_features_graspable',
    'seed_graspness_graspable',
    'seed_object_ids_graspable',
}


def build_network(checkpoint_path, device):
    """Initialize frozen economic_graspable and load backbone/graspable weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

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


def build_output_sample(end_points, batch_item_idx, scene_name, frame_id, keep_keys=KEEP_KEYS):
    """Build one compact frozen CFM sample from selected batched end_points keys."""
    output = {}
    for key in keep_keys:
        output[key] = to_cpu_detached(slice_batch_value(end_points[key], batch_item_idx))

    output['scene_name'] = scene_name
    output['frame_id'] = frame_id
    return output


def get_contiguous_saved_count(save_dir):
    """Return the number of sequentially saved `sample_XXXXXX.pt` files starting at zero."""
    prefix = 'sample_'
    suffix = '.pt'
    saved_indices = set()

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
    """Atomically save one output sample and raise a clearer error on disk write failure."""
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


def generate_split(net, device, args, split):
    """Generate frozen seed-conditioned CFM labels for one GraspNet split."""
    should_process_grasp_labels = split != 'test'
    keep_keys = KEEP_KEYS if should_process_grasp_labels else TEST_KEEP_KEYS

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
    resume_count = get_contiguous_saved_count(save_dir)
    print(f"[{split}] Dataset size: {len(dataset)}")
    print(f"[{split}] Output directory: {save_dir}")
    if resume_count > 0:
        print(f"[{split}] Resuming from existing samples: {resume_count}")

    saved_count = resume_count
    valid_sample_count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"[{split}] Generating")):
            if args.limit is not None and batch_idx >= args.limit:
                break

            batch_data = move_batch_to_device(batch_data, device)
            end_points = net(batch_data)
            if should_process_grasp_labels:
                end_points = process_grasp_labels(end_points)

            batch_size = end_points['xyz_graspable'].shape[0]
            for batch_item_idx in range(batch_size):
                if should_process_grasp_labels and not end_points['seed_valid_mask'][batch_item_idx].any():
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
                    keep_keys=keep_keys,
                )
                save_path = os.path.join(save_dir, f"sample_{saved_count:06d}.pt")
                save_output_sample(output_sample, save_path)
                saved_count += 1
                valid_sample_count += 1

    print(f"[{split}] Saved {saved_count - resume_count} new files ({saved_count} total).")
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
