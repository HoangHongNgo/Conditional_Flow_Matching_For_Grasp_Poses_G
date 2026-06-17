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
from flow.utils.cfm_norm import denormalize_x
from utils.lie import exp_so3

def main():
    parser = argparse.ArgumentParser(description="Script hiển thị prior distribution và target grasp distribution của Flow Matching")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet', 
                        help='Đường dẫn đến thư mục gốc của dataset GraspNet-1Billion')
    parser.add_argument('--scene_id', type=int, default=0, help='ID của scene cần hiển thị')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'], help='Loại camera')
    parser.add_argument('--ann_id', type=int, default=0, help='ID của annotation (frame)')
    parser.add_argument('--num_grasp', type=int, default=50, help='Số lượng grasp muốn hiển thị mỗi loại')
    parser.add_argument('--fric_thresh', type=float, default=0.1, help='Ngưỡng hệ số ma sát cho target grasp (nhỏ hơn sẽ hiển thị mẫu grasp tốt hơn)')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_norm_stats.pt',
                        help='Đường dẫn tới file thống kê chuẩn hóa (stats)')
    
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
    cloud_points = np.asarray(cloud.points)
    
    geometries = [cloud]

    # Thêm import GraspNetDataset và hàm lấy label
    from dataset.graspnet_dataset import GraspNetDataset, collate_fn
    from utils.label_generation import process_grasp_labels_without_seed_mapping
    import torch.utils.data
    
    # 2. Tạo Target Grasp Distribution (Ground Truth Grasps) - Màu Xanh (Green)
    print("Loading target grasp distribution từ process_grasp_labels_without_seed_mapping...")

    # Load thông qua GraspNetDataset để khớp hoàn toàn với pipeline huấn luyện Flow Matching
    split = 'train' if args.scene_id < 100 else 'test'
    dataset = GraspNetDataset(args.dataset_root, camera=args.camera, split=split, augment=False, load_label=True, remove_outlier=True, num_points=20000)
    
    # Tìm index tương ứng trong dataset
    data_idx = -1
    scene_str = f'scene_{args.scene_id:04d}'
    for idx in range(len(dataset)):
        if dataset.scenename[idx] == scene_str and dataset.frameid[idx] == args.ann_id:
            data_idx = idx
            break
            
    if data_idx == -1:
        print(f"Không tìm thấy {scene_str} frame {args.ann_id} trong tập {split}.")
    else:
        print("Đang xử lý nhãn bằng process_grasp_labels_without_seed_mapping...")
        raw_data = dataset[data_idx]
        batch_data = collate_fn([raw_data]) # Đưa vào batch size = 1
        
        # Đưa toàn bộ tensor lên device phù hợp
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)

        # Gọi hàm tương tự như lúc tạo CFM dataset
        _, end_points_gt = process_grasp_labels_without_seed_mapping(batch_data)
        
        # Lấy ra các tham số grasp của mẫu duy nhất trong batch (index 0)
        target_p = end_points_gt['batch_grasp_point'][0].cpu() # [num_grasps, 3]
        target_rot_matrices = end_points_gt['batch_grasp_views_rot'][0].cpu() # [num_grasps, 3, 3]
        target_w = end_points_gt['batch_grasp_width'][0].cpu() # [num_grasps]
        target_d_raw = end_points_gt['batch_grasp_depth'][0].cpu() # [num_grasps]
        
        # Trong dataset gốc, depth lưu dưới dạng index (ví dụ: 1, 2, 3, 4 tương ứng 0.01m, 0.02m, 0.03m, 0.04m)
        # Hoặc 0,1,2,3. Chuyển đổi thành met và kẹp trong khoảng [0.01, 0.04]
        target_d = torch.clamp(target_d_raw.float() * 0.01, min=0.01, max=0.04)
        
        target_score = end_points_gt['batch_grasp_score'][0].cpu().view(-1, 1) # [num_grasps, 1]
        
        n_target = target_p.shape[0]
        
        # Map sang GraspGroup 17D
        target_rot_flat = target_rot_matrices.reshape(n_target, 9)
        target_h = 0.02 * torch.ones_like(target_score)
        target_obj_ids = -1 * torch.ones_like(target_score)
        
        gg_target_preds = torch.cat([
            target_score,
            target_w.view(-1, 1),
            target_h,
            target_d.view(-1, 1),
            target_rot_flat,
            target_p,
            target_obj_ids
        ], dim=-1).numpy()
        
        gg_target = GraspGroup(gg_target_preds)
        target_geoms = gg_target.to_open3d_geometry_list()
        
        # Tô màu xanh lá (Target)
        for geom in target_geoms:
            geom.paint_uniform_color([0.0, 1.0, 0.0])
            
        print(f"  -> Hiển thị {n_target} target grasps (Green) - khớp 100% logic train!")
        print("\n--- Mở cửa sổ Open3D hiển thị Target Grasps (Green) ---")
        o3d.visualization.draw_geometries([cloud] + target_geoms, window_name=f"Target (Green) - Scene {args.scene_id:04d}")

    # 3. Tạo Prior Distribution (Nhiễu Normal Distribution) - Màu Đỏ (Red)
    print("Generating prior grasp distribution (Normal Noise)...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load norm stats để denormalize
        stats = torch.load(args.stats_path, map_location=device)
        
        # Tính điểm seed_median từ point cloud hiện tại
        seed_median = torch.tensor(np.median(cloud_points, axis=0), dtype=torch.float32, device=device)
        
        # Tạo ngẫu nhiên x0 ~ N(0, I)
        n_samples = args.num_grasp
        x0 = torch.randn(n_samples, 8, device=device)
        
        # Denormalize x0 trở về không gian vật lý của scene hiện tại
        x_denorm = denormalize_x(x0, stats, seed_median) # [n_samples, 8]
        
        p = x_denorm[:, :3]      # [n_samples, 3] (center)
        omega = x_denorm[:, 3:6] # [n_samples, 3] (rotation Lie algebra)
        w = x_denorm[:, 6]       # [n_samples] (width)
        d = x_denorm[:, 7]       # [n_samples] (depth)
        
        # Convert Lie algebra sang ma trận quay 3x3
        rot_matrices = exp_so3(omega) # [n_samples, 3, 3]
        rot_flat = rot_matrices.reshape(n_samples, 9)
        
        # Ghép thành 17D array để tạo GraspGroup
        # Dummy score cho prior
        score = torch.ones(n_samples, 1, device=device) * 0.5 
        # Clamp giống khi inference để gripper không bị out-of-bounds
        grasp_width = torch.clamp(w, min=0.0, max=0.1).view(-1, 1)
        grasp_depth = torch.clamp(d, min=0.01, max=0.04).view(-1, 1)
        grasp_height = 0.02 * torch.ones_like(score)
        obj_ids = -1 * torch.ones_like(score)
        
        gg_prior_preds = torch.cat([
            score,
            grasp_width,
            grasp_height,
            grasp_depth,
            rot_flat,
            p,
            obj_ids
        ], dim=-1).cpu().numpy()
        
        gg_prior = GraspGroup(gg_prior_preds)
        prior_geoms = gg_prior.to_open3d_geometry_list()
        
        # Tô màu đỏ (Prior)
        for geom in prior_geoms:
            geom.paint_uniform_color([1.0, 0.0, 0.0])
            
        print(f"  -> Hiển thị {n_samples} prior grasps (Red)")
        print("\n--- Mở cửa sổ Open3D hiển thị Prior Grasps (Red) ---")
        o3d.visualization.draw_geometries([cloud] + prior_geoms, window_name=f"Prior (Red) - Scene {args.scene_id:04d}")
        
    except Exception as e:
        print(f"Lỗi khi generate prior grasps: {e}")

    # 4. Dự đoán từ mô hình CFM (Flow Matching) - Màu Xanh Dương (Blue)
    print("Predicting grasps from CFM Model...")
    try:
        from flow.models.grasp_cfm import SceneMinkEncoder, GraspVelocityMLP
        from flow.utils.cfm_solver import euler_solve
        from models.flowgrasp import economic_graspable
        from utils.arguments import cfgs

        # Load fresh batch_data on CPU for MinkowskiEngine, then move exactly what's needed to device
        batch_data_pred = collate_fn([raw_data])
        for key in batch_data_pred:
            if 'list' in key:
                for i in range(len(batch_data_pred[key])):
                    for j in range(len(batch_data_pred[key][i])):
                        batch_data_pred[key][i][j] = batch_data_pred[key][i][j].to(device)
            elif isinstance(batch_data_pred[key], torch.Tensor):
                batch_data_pred[key] = batch_data_pred[key].to(device)
        
        # Load base_net to extract seed features
        base_net = economic_graspable(seed_feat_dim=512, is_training=False).to(device)
        base_checkpoint = torch.load('checkpoints/economicgrasp_realsense.tar', map_location=device)
        base_net.load_state_dict(base_checkpoint['model_state_dict'], strict=False)
        base_net.eval()
        
        with torch.no_grad():
            end_points_pred = base_net(batch_data_pred)
            seed_xyz = end_points_pred['xyz_graspable']
            # transpose từ [B, 512, 1024] sang [B, 1024, 512] cho encoder
            seed_feats = end_points_pred['seed_features_graspable'].transpose(1, 2)
            
        # Load CFM Model
        cfm_checkpoint_path = 'flow/results/26-05-2026_13-30-09/flowgrasp_latest.tar'
        encoder = SceneMinkEncoder(in_channels=512, out_channels=256).to(device)
        mlp = GraspVelocityMLP(grasp_dim=8, cond_dim=256).to(device)
        cfm_checkpoint = torch.load(cfm_checkpoint_path, map_location=device)
        encoder.load_state_dict(cfm_checkpoint['encoder_state_dict'])
        mlp.load_state_dict(cfm_checkpoint['mlp_state_dict'])
        encoder.eval()
        mlp.eval()
        
        # Solve ODE (euler_solve đã tự động encode và denormalize bên trong)
        x0_cfm = torch.randn(1, args.num_grasp, 8, device=device)
        with torch.no_grad():
            x1_denorm = euler_solve(encoder, mlp, x0_cfm, seed_xyz, seed_feats, stats, n_steps=50)
            
        # Tách các thành phần vật lý
        pred_p = x1_denorm[0, :, :3].cpu()
        pred_rot = x1_denorm[0, :, 3:6].cpu()
        pred_w = x1_denorm[0, :, 6].cpu()
        pred_d = x1_denorm[0, :, 7].cpu()
        
        # Format as GraspGroup
        pred_rot_flat = exp_so3(pred_rot).reshape(args.num_grasp, 9).cpu()
        pred_score = torch.ones(args.num_grasp, 1).cpu()
        pred_w_clamped = torch.clamp(pred_w, min=0.0, max=0.1).view(-1, 1).cpu()
        pred_d_clamped = torch.clamp(pred_d, min=0.01, max=0.04).view(-1, 1).cpu()
        pred_h = 0.02 * torch.ones_like(pred_score).cpu()
        pred_obj_ids = -1 * torch.ones_like(pred_score).cpu()
        
        gg_pred_preds = torch.cat([
            pred_score, pred_w_clamped, pred_h, pred_d_clamped, pred_rot_flat, pred_p, pred_obj_ids
        ], dim=-1).numpy()
        
        gg_pred = GraspGroup(gg_pred_preds)
        pred_geoms = gg_pred.to_open3d_geometry_list()
        
        for geom in pred_geoms:
            geom.paint_uniform_color([0.0, 0.0, 1.0])
            
        print(f"  -> Hiển thị {args.num_grasp} predicted grasps (Blue)")
        print("\n--- Mở cửa sổ Open3D hiển thị Predicted Grasps (Blue) ---")
        o3d.visualization.draw_geometries([cloud] + pred_geoms, window_name=f"Predicted (Blue) - Scene {args.scene_id:04d}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Lỗi khi predict grasps: {e}")

    print("Hoàn tất!")

if __name__ == '__main__':
    main()
