import os
import sys
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from train_refine import construct_pred_grasp_pose, construct_gt_grasp_pose

# ----------- SPLITS TO EXTRACT ------------
# Each entry: (split_name, cache_folder_name)
# Add or remove splits as needed.
SPLITS_TO_EXTRACT = [
    ('train',        'dataset_cache_cfm'),
    ('test_similar', 'dataset_cache_cfm_test_similar'),
]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load base model once ---
    print("\n[1] Loading frozen base model...")
    base_net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
    base_net.to(device)
    base_net.eval()

    if not os.path.isfile(cfgs.checkpoint_path):
        print(f"Error: Base checkpoint not found at {cfgs.checkpoint_path}")
        return

    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device, weights_only=False)
    base_net.load_state_dict(checkpoint['model_state_dict'], strict=False)

    for param in base_net.parameters():
        param.requires_grad = False

    # --- Extract each split ---
    for split_name, cache_folder in SPLITS_TO_EXTRACT:
        cache_dir = os.path.join(cfgs.dataset_root, cache_folder)

        # Skip if cache already exists
        if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
            print(f"\n[SKIP] Cache already exists at {cache_dir} "
                  f"({len(os.listdir(cache_dir))} files). "
                  f"Delete the folder to re-extract.")
            continue

        os.makedirs(cache_dir, exist_ok=True)

        print(f"\n[*] Extracting split '{split_name}' -> {cache_dir}")

        dataset = GraspNetDataset(
            cfgs.dataset_root, camera=cfgs.camera, split=split_name,
            voxel_size=cfgs.voxel_size, num_points=cfgs.num_point,
            remove_outlier=True, augment=False
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False,
                                collate_fn=collate_fn, num_workers=2)

        global_idx = 0
        with torch.no_grad():
            for batch_data in tqdm(dataloader, total=len(dataloader),
                                   desc=f"  {split_name}"):
                for key in batch_data:
                    if 'list' in key:
                        for i in range(len(batch_data[key])):
                            for j in range(len(batch_data[key][i])):
                                batch_data[key][i][j] = batch_data[key][i][j].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                end_points = base_net(batch_data)
                grasp_preds = pred_decode(end_points)

                x0_batch = construct_pred_grasp_pose(grasp_preds).detach().cpu()
                x1_batch = construct_gt_grasp_pose(end_points).detach().cpu()
                cond_batch = end_points['group_features'].detach().cpu()
                valid_mask_batch = end_points['batch_valid_mask'].detach().cpu()

                # Slice the batch and save individual items
                B = x0_batch.shape[0]
                for i in range(B):
                    save_dict = {
                        'x0': x0_batch[i].clone(),                # [11, 1024]
                        'x1': x1_batch[i].clone(),                # [11, 1024]
                        'cond': cond_batch[i].clone(),            # [256, 1024]
                        'valid_mask': valid_mask_batch[i].clone()  # [1024]
                    }
                    torch.save(save_dict, os.path.join(cache_dir, f'sample_{global_idx:06d}.pt'))
                    global_idx += 1

        print(f"  => Saved {global_idx} samples for '{split_name}' to {cache_dir}")

    print("\nAll done!")


if __name__ == '__main__':
    # Fix EMFILE
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
