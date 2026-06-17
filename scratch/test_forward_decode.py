import os
import sys
import torch
from torch.utils.data import DataLoader

# Đưa thư mục gốc của project vào sys.path để có thể import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.arguments import cfgs
from models.economicgrasp import economicgrasp, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, collate_fn

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. KHỞI TẠO DATASET (Mô phỏng giống hệt train.py)
    print("\n1. Initializing dataset...")
    # train.py dùng augment=True, load_label=True (mặc định)
    dataset = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                              voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, 
                              remove_outlier=True, augment=True)
                              
    # Chỉ lấy 1 batch để test nhanh (batch_size=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # 2. KHỞI TẠO MÔ HÌNH
    print("\n2. Initializing economicgrasp model...")
    # train.py gốc dùng liteptgrasp, nhưng bạn yêu cầu dùng economicgrasp.
    # is_training=True để nó sinh ra batch_grasp_views_rot như khi train.
    net = economicgrasp(seed_feat_dim=512, is_training=True, is_refine=True)
    net.to(device)
    net.eval() # Set eval() để khi chạy forward không ảnh hưởng BatchNorm/Dropout

    # 3. LOAD DỮ LIỆU CỦA 1 SCENE
    print("\n3. Loading 1 scene...")
    batch_data_label = next(iter(dataloader))
    
    # Đẩy dữ liệu lên GPU (Giống hệt vòng lặp trong train.py)
    for key in batch_data_label:
        if 'list' in key:
            for i in range(len(batch_data_label[key])):
                for j in range(len(batch_data_label[key][i])):
                    batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data_label[key])):
                batch_data_label[key][i] = batch_data_label[key][i].to(device)
        else:
            batch_data_label[key] = batch_data_label[key].to(device)

    # 4. CHẠY FORWARD PASS
    print("\n4. Running model forward pass...")
    with torch.no_grad():
        end_points = net(batch_data_label)
        
    # 5. CHẠY HÀM PRED_DECODE
    print("\n5. Running pred_decode...")
    grasp_preds = pred_decode(end_points)
    
    # 6. IN KẾT QUẢ
    print(f"\n--- SUCCESS ---")
    print(f"Decode complete. Number of scenes in batch: {len(grasp_preds)}")
    print(f"Shape of predictions for the first scene: {grasp_preds[0].shape}")
    print(f" -> Điều này có nghĩa là có {grasp_preds[0].shape[0]} điểm gắp (seed points).")
    print(f" -> Mỗi điểm gắp được mô tả bằng {grasp_preds[0].shape[1]} thông số (Score, Width, Height, Depth, Rotation Matrix (9), Center XYZ (3), Obj_id).")
    
    # 7. LƯU SHAPE VÀO FILE TEXT
    print("\n7. Saving shapes to txt files...")
    
    # Lưu end_points
    endpoints_file = os.path.join(os.path.dirname(__file__), 'end_points_shapes.txt')
    with open(endpoints_file, 'w') as f:
        f.write("=== END_POINTS SHAPES ===\n\n")
        for k, v in end_points.items():
            if isinstance(v, torch.Tensor):
                f.write(f"Key: {k:<30} | Type: Tensor | Shape: {list(v.shape)}\n")
            elif isinstance(v, list):
                f.write(f"Key: {k:<30} | Type: List   | Length: {len(v)}\n")
                if len(v) > 0:
                    if isinstance(v[0], torch.Tensor):
                        f.write(f"    -> Item[0] is Tensor with Shape: {list(v[0].shape)}\n")
                    elif isinstance(v[0], list):
                        f.write(f"    -> Item[0] is List with Length: {len(v[0])}\n")
                        if len(v[0]) > 0 and isinstance(v[0][0], torch.Tensor):
                            f.write(f"        -> Item[0][0] is Tensor with Shape: {list(v[0][0].shape)}\n")
            elif isinstance(v, (int, float, str, bool)):
                f.write(f"Key: {k:<30} | Type: Scalar | Value: {v}\n")
            else:
                f.write(f"Key: {k:<30} | Type: {type(v)}\n")
                
    # Lưu grasp_preds
    grasppreds_file = os.path.join(os.path.dirname(__file__), 'grasp_preds_shapes.txt')
    with open(grasppreds_file, 'w') as f:
        f.write("=== GRASP_PREDS SHAPES ===\n\n")
        f.write(f"Type: List\n")
        f.write(f"Length (Batch Size): {len(grasp_preds)}\n")
        for i, pred in enumerate(grasp_preds):
            if isinstance(pred, torch.Tensor):
                f.write(f"Scene [{i}] | Type: Tensor | Shape: {list(pred.shape)}\n")
            else:
                f.write(f"Scene [{i}] | Type: {type(pred)}\n")
                
    print(f"Đã lưu thành công vào 2 file:")
    print(f" - {endpoints_file}")
    print(f" - {grasppreds_file}")

if __name__ == '__main__':
    main()
