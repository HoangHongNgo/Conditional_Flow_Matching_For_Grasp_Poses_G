import os
import sys
import open3d as o3d
import numpy as np
import torch

# Thêm đường dẫn project vào sys.path để import refs
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from graspnetAPI import GraspGroup
from models.economicgrasp import pred_decode

def showPredictedSceneGrasp(end_points, batch_idx=0, numGrasp=20, show_object=True, score_thresh=0.1):
    '''
    Visualize promised grasps outputted from EconomicGrasp model's end_points,
    similar to showSceneGrasp in graspnetAPI.

    **Input:**
    - end_points: dict của đầu ra mô hình EconomicGrasp sau khi chạy forward.
    - batch_idx: int, chỉ số của scene (mẫu) trong batch cần hiển thị.
    - numGrasp: int, số lượng grasp tối đa hiển thị (lấy ngẫu nhiên hoặc top từ kết quả).
    - show_object: bool, cho phép vẽ cả point cloud gốc thu được từ end_points.
    - score_thresh: float, ngưỡng điểm số grasp (chú ý: điểm predict từ model là càng cao càng tốt,
                    ngược lại với coef_fric trong dataset API gốc).
    '''
    geometries = []

    # 1. Sử dụng hàm pred_decode của model để giải mã tensor output thành định dạng list(Nx17) tensor
    grasp_preds_list = pred_decode(end_points)
    
    if batch_idx >= len(grasp_preds_list):
        print(f"Batch index {batch_idx} vượt quá giới hạn batch size ({len(grasp_preds_list)}).")
        return

    # Lấy tensor của dự đoán hiện tại và chuyển qua numpy
    grasp_preds = grasp_preds_list[batch_idx].detach().cpu().numpy()
    
    # 2. Tạo đối tượng GraspGroup của graspnetAPI từ mảng numpy dự đoán
    # GraspGroup tự động parse mảng (score, width, height, depth, rot matrix(9), center(3), obj_id)
    sceneGrasp = GraspGroup(grasp_preds)
    
    # 3. Lọc các grasp theo ngưỡng điểm số (score càng cao càng tốt)
    scores = sceneGrasp.scores
    mask = scores > score_thresh
    sceneGrasp = sceneGrasp[mask]
    
    if len(sceneGrasp) == 0:
        print(f"Cảnh báo: Không có predicted grasp nào thỏa mãn ngưỡng score_thresh > {score_thresh}.")
        return

    # 4. Giới hạn số lượng hiển thị (Lấy ra các grasp có điểm số cao nhất)
    if len(sceneGrasp) > numGrasp:
        sceneGrasp.sort_by_score()
        sceneGrasp = sceneGrasp[:numGrasp]
    
    print(f"Đang hiển thị {len(sceneGrasp)} predicted grasps...")
    
    # Lấy danh sách geometry hiển thị dạng hình vẽ kẹp chữ nhật
    geometries += sceneGrasp.to_open3d_geometry_list()

    # 5. Hiển thị đối tượng / scene point cloud nếu được bật
    if show_object and 'point_clouds' in end_points:
        point_cloud_tensor = end_points['point_clouds'][batch_idx]
        points = point_cloud_tensor.detach().cpu().numpy()
        
        scenePCD = o3d.geometry.PointCloud()
        scenePCD.points = o3d.utility.Vector3dVector(points)
        
        # Nếu end_points có lưu màu, tô màu Point Cloud
        if 'cloud_colors' in end_points:
            colors = end_points['cloud_colors'][batch_idx].detach().cpu().numpy()
            if colors.max() > 1.0:
                colors = colors / 255.0  # Normalize màu cho Open3D
            scenePCD.colors = o3d.utility.Vector3dVector(colors)
        else:
            scenePCD.paint_uniform_color([0.7, 0.7, 0.7])  # Màu xám cho dễ nhìn grasp
            
        geometries.append(scenePCD)
    elif show_object:
        print("Cảnh báo: Yêu cầu 'show_object=True' nhưng không tìm thấy trường 'point_clouds' trong end_points.")

    # 6. Vẽ tất cả kết hợp lại lên màn hình
    o3d.visualization.draw_geometries(geometries, window_name="EconomicGrasp - Predicted Grasps")

if __name__ == '__main__':
    import argparse
    from dataset.graspnet_dataset import GraspNetDataset, collate_fn
    from models.economicgrasp import economicgrasp
    
    parser = argparse.ArgumentParser(description="Inference và Visualizer trực tiếp cho EconomicGrasp model")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet', help='Thư mục gốc của GraspNet')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'])
    parser.add_argument('--scene_idx', type=int, default=0, help='Chỉ số thứ tự của dữ liệu trong tập dataset/dataloader')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(ROOT_DIR, 'checkpoints/economicgrasp_kinect.tar'), help='File checkpoint')
    parser.add_argument('--num_grasp', type=int, default=20, help='Giới hạn số lượng hiển thị trên màn hình')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='Ngưỡng để vứt bỏ các grasp rác (càng cao càng khắt khe)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"1. Đang tải Dataset GraspNet từ {args.dataset_root}...")
    try:
        dataset = GraspNetDataset(args.dataset_root, camera=args.camera, split='train', 
                                  voxel_size=0.005, num_points=20000, remove_outlier=True, augment=False)
        data = dataset[args.scene_idx]
        batch_data = collate_fn([data])
    except Exception as e:
        print(f"Lỗi khi đọc dataset: {e}")
        sys.exit(1)

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)

    print(f"2. Nạp mô hình EconomicGrasp sử dụng checkpoint tại {args.checkpoint}...")
    net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
    net.to(device)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback nếu checkpoint luu thang dict state
            net.load_state_dict(checkpoint) 
        print(f"  => Thành công!")
    except Exception as e:
        print(f"  => Lỗi do không thể mount Checkpoint: {e}")
        sys.exit(1)
        
    net.eval()

    print("3. Đang đưa Scene qua Inference Node của mô hình (Sẽ mất chút thời gian)...")
    with torch.no_grad():
        end_points = net(batch_data)

    print("4. Mở cửa sổ render 3D! Bạn có thể xoay / view / kéo các thành phần!")
    showPredictedSceneGrasp(end_points, batch_idx=0, numGrasp=args.num_grasp, show_object=True, score_thresh=args.score_thresh)

