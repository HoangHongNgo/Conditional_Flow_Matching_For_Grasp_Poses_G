# Hướng dẫn thiết lập môi trường (EconomicGrasp)

Dự án này sử dụng môi trường ảo Python 3.10 nằm trong thư mục `py310`.

## 1. Kích hoạt môi trường (Activation)

Để bắt đầu làm việc, bạn cần kích hoạt môi trường ảo bằng lệnh sau trong terminal:

```bash
# Di chuyển vào thư mục dự án nếu chưa ở đó
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp

# Kích hoạt môi trường
source py310/bin/activate
```

Sau khi kích hoạt, bạn sẽ thấy `(py310)` xuất hiện ở đầu dòng lệnh.

## 2. Kiểm tra cài đặt

Bạn có thể kiểm tra phiên bản Python và các thư viện quan trọng:

```bash
python --version
# Kết quả mong đợi: Python 3.10.15

python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Kết quả mong đợi: Torch: 2.5.1+cu121, CUDA: True
```

## 3. Chạy script Training

Để bắt đầu huấn luyện mô hình:

```bash
# Đảm bảo môi trường đã được kích hoạt
python train.py
```

*Lưu ý: Môi trường này đã bao gồm đầy đủ các dependencies cần thiết như PyTorch 2.5.1 với hỗ trợ CUDA.*
