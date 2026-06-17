import os
import sys
import argparse

# Thêm đường dẫn project vào sys.path để import libs hoặc models nếu cần thiết
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import GraspNet từ thư viện graspnetAPI 
# (Thư viện này có thể đã được cài bằng pip hoặc nằm trong thư mục libs)
from graspnetAPI import GraspNet

def main():
    parser = argparse.ArgumentParser(description="Script để hiển thị grasp của một scene bằng GraspNetAPI")
    parser.add_argument('--dataset_root', type=str, default='/media/dsp520/Grasp_2T/graspnet', 
                        help='Đường dẫn đến thư mục gốc của dataset GraspNet-1Billion')
    parser.add_argument('--scene_id', type=int, default=0, help='ID của scene cần hiển thị')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'], help='Loại camera')
    parser.add_argument('--ann_id', type=int, default=0, help='ID của annotation (frame)')
    parser.add_argument('--format', type=str, default='6d', choices=['6d', 'rect'], help='Định dạng grasp (6d hoặc rect)')
    parser.add_argument('--num_grasp', type=int, default=20, help='Số lượng grasp muốn hiển thị')
    parser.add_argument('--fric_thresh', type=float, default=0.1, help='Ngưỡng hệ số ma sát (nhỏ hơn sẽ hiển thị mẫu grasp tốt hơn)')
    
    args = parser.parse_args()

    # 1. Khởi tạo đối tượng GraspNet API
    print(f"Đang tải dữ liệu từ {args.dataset_root}...")
    try:
        g = GraspNet(args.dataset_root, camera=args.camera, split='all')
    except Exception as e:
        print(f"Lỗi khởi tạo GraspNet: {e}")
        return

    print(f"Đang nạp và hiển thị {args.num_grasp} grasp tốt nhất (fric <= {args.fric_thresh}) cho scene_{args.scene_id:04d}...")
    
    # 2. Gọi hàm showSceneGrasp có sẵn của API
    g.showSceneGrasp(
        sceneId=args.scene_id, 
        camera=args.camera, 
        annId=args.ann_id, 
        format=args.format, 
        numGrasp=args.num_grasp, 
        show_object=True, 
        coef_fric_thresh=args.fric_thresh
    )

if __name__ == '__main__':
    main()
