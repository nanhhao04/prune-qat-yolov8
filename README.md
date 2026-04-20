# YOLO Optimization Pipeline cho Jetson Nano (Traffic Camera)

Pipeline này được thiết kế để tối ưu hóa mô hình YOLOv8 cho các bài toán nhận diện giao thông thực tế trên thiết bị Jetson Nano (Maxwell GPU). 
Pipeline kết hợp **Network Slimming (BN Pruning)** và **TensorRT (FP16)** để đạt được FPS cao nhất mà vẫn giữ được độ chính xác.

## 🚀 Luồng Pipeline (6 Bước)

### Bước 1: Sparsity Training
Huấn luyện mô hình với tham số `sr` (sparsity regularization) để ép các trọng số Batch Normalization (BN) về gần 0.
```bash
python scripts/01_sparsity_train.py
```

### Bước 2: Model Pruning
Cắt tỉa các channels có trọng số BN nhỏ. Bước này giúp giảm đáng kể GFLOPs, rất quan trọng cho Jetson Nano.
```bash
python scripts/02_prune.py
```
*Kết quả lưu tại: `weights/pruned_model.pt`*

### Bước 3: Finetuning
Huấn luyện lại mô hình đã cắt tỉa để phục hồi độ chính xác (mAP) cho bài toán giao thông.
```bash
python scripts/03_finetune.py
```

### Bước 4: QAT (Quantization-Aware Training) - *Tùy chọn*
Huấn luyện giả lập định dạng INT8. 
**Lưu ý:** Jetson Nano (Maxwell) không có Tensor Cores nên INT8 có thể không nhanh hơn FP16. Khuyến khích dùng FP16 ở bước export.
```bash
python scripts/04_qat.py
```

### Bước 5: TensorRT Export
Xuất mô hình sang định dạng `.engine` tối ưu cho Jetson Nano.
```bash
python scripts/05_export.py
```

### Bước 6: Inference & Demo
Chạy demo nhận diện giao thông thời gian thực trên Camera.
```bash
python scripts/06_inference_traffic.py --source 0 --weights runs/train-finetune/weights/best.engine
```

---

## 🛠 Cấu trúc thư mục
- `configs/`: Chứa file `data.yaml` (danh sách class giao thông) và `pipeline.yaml` (tham số huấn luyện).
- `core/`: Logic xử lý cắt tỉa mô hình (Pruning engine).
- `scripts/`: Các script thực thi từng bước.
- `weights/`: Lưu trữ checkpoints trung gian.

## ⚠️ Lưu ý cho Jetson Nano
1. **Model Size**: Nên bắt đầu với `yolov8n` (Nano version) để đạt FPS tốt nhất (>20 FPS).
2. **FP16 vs INT8**: Jetson Nano hoạt động tốt nhất ở chế độ **FP16**. Pipeline này mặc định cấu hình FP16 trong `configs/pipeline.yaml`.
3. **Swap Memory**: Hãy đảm bảo Jetson Nano đã được bật Swap memory (ít nhất 4GB) để quá trình build TensorRT engine không bị crash.

---
*Phát triển bởi Antigravity AI Assistant.*
