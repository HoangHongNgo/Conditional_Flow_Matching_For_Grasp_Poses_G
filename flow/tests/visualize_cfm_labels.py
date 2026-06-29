import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d

# Thêm đường dẫn project vào sys.path để import libs
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from graspnetAPI import GraspNet, GraspGroup
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from flow.utils.cfm_label_generation import process_grasp_labels
from models.economicgrasp import economicgrasp
from utils.arguments import cfgs


def build_graspgroup_from_seed_configs(seed_points, rot_matrices, widths, depths_raw, scores, valid_mask):
    """Build a GraspGroup from all valid grasp configs attached to seed points."""
    N, K = valid_mask.shape
    seed_points_expanded = seed_points[:, None, :].expand(N, K, 3)  # [1024, K, 3]

    valid_p = seed_points_expanded[valid_mask]  # [M, 3]
    valid_rot_matrices = rot_matrices[valid_mask]  # [M, 3, 3]
    valid_w = widths[valid_mask]  # [M]
    valid_d_raw = depths_raw[valid_mask]  # [M]
    valid_score = scores[valid_mask]  # [M]

    num_grasp = valid_p.shape[0]
    if num_grasp == 0:
        return GraspGroup(np.zeros((0, 17), dtype=np.float32))

    valid_d = torch.clamp(valid_d_raw.float() * 0.01, min=0.01, max=0.04)  # [M]
    rot_flat = valid_rot_matrices.reshape(num_grasp, 9)  # [M, 9]
    grasp_height = 0.02 * torch.ones_like(valid_score)  # [M]
    obj_ids = -1 * torch.ones_like(valid_score)  # [M]

    gg_preds = torch.cat([
        valid_score.view(-1, 1),
        valid_w.view(-1, 1),
        grasp_height.view(-1, 1),
        valid_d.view(-1, 1),
        rot_flat,
        valid_p,
        obj_ids.view(-1, 1),
    ], dim=-1).numpy()
    return GraspGroup(gg_preds)


def main():
    parser = argparse.ArgumentParser(description="Script hiển thị ground truth grasp labels cho CFM")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet', 
                        help='Đường dẫn đến thư mục gốc của dataset GraspNet-1Billion')
    parser.add_argument('--scene_id', type=int, default=0, help='ID của scene cần hiển thị')
    parser.add_argument('--camera', type=str, default='realsense', choices=['kinect', 'realsense'], help='Loại camera')
    parser.add_argument('--ann_id', type=int, default=0, help='ID của annotation (frame)')
    parser.add_argument('--num_seed', type=int, default=1024, help='Số lượng seed point muốn hiển thị')
    
    args = parser.parse_args()

    # 1. Khởi tạo đối tượng GraspNet API và load point cloud
    print(f"Đang tải dữ liệu từ {args.dataset_root}...")
    try:
        g = GraspNet(args.dataset_root, camera=args.camera, split='all')
    except Exception as e:
        print(f"Lỗi khởi tạo GraspNet: {e}")
        return

    print(f"Loading point cloud cho scene_{args.scene_id:04d}, frame {args.ann_id}...")
    cloud = g.loadScenePointCloud(args.scene_id, args.camera, args.ann_id)
    
    # 2. Tạo CFM Dataset Labels
    print("Loading CFM labels từ process_grasp_labels...")

    split = 'train' if args.scene_id < 100 else 'test'
    dataset = GraspNetDataset(args.dataset_root, camera=args.camera, split=split, augment=False, load_label=True, remove_outlier=True, num_points=cfgs.num_point)
    
    # Tìm index tương ứng trong dataset
    data_idx = -1
    scene_str = f'scene_{args.scene_id:04d}'
    for idx in range(len(dataset)):
        if dataset.scenename[idx] == scene_str and dataset.frameid[idx] == args.ann_id:
            data_idx = idx
            break
            
    if data_idx == -1:
        print(f"Không tìm thấy {scene_str} frame {args.ann_id} trong tập {split}.")
        return

    print("Đang chạy mạng cơ sở để lấy seed points...")
    raw_data = dataset[data_idx]
    batch_data = collate_fn([raw_data]) # Đưa vào batch size = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)

    # Load base_net để sinh seed points
    base_net = economicgrasp(seed_feat_dim=512, is_training=False).to(device)
    base_checkpoint = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device, weights_only=False)
    base_net.load_state_dict(base_checkpoint['model_state_dict'], strict=False)
    base_net.eval()
    
    with torch.no_grad():
        end_points = base_net(batch_data)
        
    print("Đang xử lý nhãn bằng process_grasp_labels (Padding 128 configs)...")
    end_points_cfm = process_grasp_labels(end_points)
    
    # Lấy thông tin từ batch đầu tiên
    seed_points = end_points_cfm['xyz_graspable'][0].cpu() # [1024, 3]
    target_rot_matrices = end_points_cfm['batch_target_views_rot'][0].cpu() # [1024, 128, 3, 3]
    target_w = end_points_cfm['batch_target_widths'][0].cpu() # [1024, 128]
    target_d_raw = end_points_cfm['batch_target_depths'][0].cpu() # [1024, 128]
    target_score = end_points_cfm['batch_target_scores'][0].cpu() # [1024, 128]
    valid_mask = end_points_cfm['batch_valid_mask'][0].cpu() # [1024, 128]

    print(f"Tổng số valid grasp configs trong pool: {valid_mask.sum().item()}")

    gg_all_seed_grasps = build_graspgroup_from_seed_configs(
        seed_points,
        target_rot_matrices,
        target_w,
        target_d_raw,
        target_score,
        valid_mask,
    )
    all_seed_grasp_geoms = gg_all_seed_grasps.to_open3d_geometry_list()
    for geom in all_seed_grasp_geoms:
        geom.paint_uniform_color([1.0, 0.45, 0.0]) # Orange

    # 3. Lấy mẫu 1 grasp cho mỗi seed point (giống hệt khi train)
    print("Sampling 1 grasp hợp lệ cho mỗi seed point...")
    N = seed_points.shape[0]
    K = valid_mask.shape[1]
    
    mask_float = valid_mask.float()
    weights = mask_float + 1e-6 
    
    # [1024, 1]
    sampled_idx = torch.multinomial(weights, num_samples=1)
    
    # Chọn tọa độ seed_points (giữ nguyên cho mọi config)
    target_p = seed_points # [1024, 3]
    
    # Squeeze the indices
    idx_squeeze = sampled_idx.squeeze(1) # [1024]
    
    # Gather attributes
    # target_rot_matrices: [1024, 128, 3, 3] -> [1024, 3, 3]
    final_rot_matrices = target_rot_matrices[torch.arange(N), idx_squeeze]
    final_w = target_w[torch.arange(N), idx_squeeze]
    final_d_raw = target_d_raw[torch.arange(N), idx_squeeze]
    final_score = target_score[torch.arange(N), idx_squeeze]
    final_valid = valid_mask[torch.arange(N), idx_squeeze]
    
    # Lọc ra chỉ những seed có chứa dữ liệu thực sự (vì những seed rỗng sẽ bị chọn đại 1 epsilon)
    valid_idx = torch.where(final_valid)[0]
    
    if len(valid_idx) > args.num_seed:
        # Nếu nhiều hơn số lượng muốn hiển thị thì lấy mẫu ngẫu nhiên
        perm = torch.randperm(len(valid_idx))[:args.num_seed]
        valid_idx = valid_idx[perm]
        
    final_p = target_p[valid_idx]
    final_rot_matrices = final_rot_matrices[valid_idx]
    final_w = final_w[valid_idx]
    final_d_raw = final_d_raw[valid_idx]
    final_score = final_score[valid_idx]
    
    # Chuyển depth về met
    final_d = torch.clamp(final_d_raw.float() * 0.01, min=0.01, max=0.04)
    
    n_target = final_p.shape[0]
    
    # 4. Map sang GraspGroup 17D
    target_rot_flat = final_rot_matrices.reshape(n_target, 9)
    target_h = 0.02 * torch.ones_like(final_score)
    target_obj_ids = -1 * torch.ones_like(final_score)
    
    gg_target_preds = torch.cat([
        final_score.view(-1, 1),
        final_w.view(-1, 1),
        target_h.view(-1, 1),
        final_d.view(-1, 1),
        target_rot_flat,
        final_p,
        target_obj_ids.view(-1, 1)
    ], dim=-1).numpy()
    
    gg_target = GraspGroup(gg_target_preds)
    target_geoms = gg_target.to_open3d_geometry_list()
    
    # Tô màu xanh dương (Target CFM)
    for geom in target_geoms:
        geom.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
        
    print("\n--- Mở cửa sổ Open3D hiển thị Seed Points (Green) ---")
    
    # Tạo các khối cầu (spheres) nhỏ 3D cho từng seed point để không bị chìm vào bề mặt
    seed_geoms = []
    for p in target_p.numpy():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003) # Bán kính 3mm
        sphere.translate(p)
        sphere.paint_uniform_color([0.0, 1.0, 0.0]) # Green
        seed_geoms.append(sphere)
    
    # Hiển thị cửa sổ thứ nhất: Chỉ có point cloud gốc và seed points (khối cầu xanh lá)
    o3d.visualization.draw_geometries([cloud] + seed_geoms, window_name=f"Seed Points (Green) - Scene {args.scene_id:04d}")

    print("\n--- Mở cửa sổ Open3D hiển thị Target Grasps (Blue) ---")
    
    # Hiển thị cửa sổ thứ hai: Có point cloud, seed points (xanh lá), và target grasps (xanh dương)
    o3d.visualization.draw_geometries([cloud] + seed_geoms + target_geoms, window_name=f"CFM Labels (Blue Grasps) - Scene {args.scene_id:04d}")

    print("\n--- Mở cửa sổ Open3D hiển thị toàn bộ grasp của 1024 seed points (Orange) ---")

    # Show every valid grasp config attached to the 1024 seed points.
    o3d.visualization.draw_geometries(
        [cloud] + seed_geoms + all_seed_grasp_geoms,
        window_name=f"All CFM Label Grasps for 1024 Seeds (Orange) - Scene {args.scene_id:04d}",
    )

if __name__ == '__main__':
    main()
