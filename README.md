# YOLOv8 Optimization & Deployment Pipeline (PC -> Jetson Nano)

Dự án này cung cấp quy trình 6 bước để tối ưu hóa YOLOv8, giúp đạt tốc độ cao trên thiết bị phần cứng hạn chế như Jetson Nano.

##  Quy trình thực hiện

### Giai đoạn 1: Huấn luyện & Tối ưu (PC / Colab)
Thực hiện trên máy có GPU mạnh để tạo ra mô hình nén và tối ưu.

1.  **Bước 1: Sparsity Training** (`scripts/01_sparsity_train.py`)
    *   **Mô tả:** Huấn luyện thưa hóa để chuẩn bị cho việc cắt tỉa.
    *   **Kết quả:** `runs/weights/best.pt`.

2.  **Bước 2: Model Pruning** (`scripts/02_prune.py`)
    *   **Mô tả:** Cắt bỏ các kênh thừa. Giảm kích thước mô hình và FLOPs.
    *   **Kết quả:** `weights/pruned_model.pt`.

3.  **Bước 3: Finetuning** (`scripts/03_finetune.py`)
    *   **Mô tả:** Huấn luyện lại để lấy lại độ chính xác (mAP) sau khi cắt.
    *   **Kết quả:** `runs/train-finetune/weights/best.pt`.

4.  **Bước 4: QAT (Quantization Aware Training)** (`scripts/04_qat.py`) - *Tùy chọn*
    *   **Mô tả:** Huấn luyện nhận biết lượng tử hóa, chuẩn bị cho việc tối ưu INT8.
    *   **Kết quả:** `weights/qat_model.pt`.

---

### Giai đoạn 2: Triển khai & Chạy thực tế (Jetson Nano)
Copy file tối ưu nhất (`best.pt` hoặc `qat_model.pt`) sang Jetson.

5.  **Bước 5: Export TensorRT** (`scripts/05_export.py`)
    *   **Mô tả:** Chuyển đổi sang định dạng `.engine` sử dụng TensorRT. Ưu tiên FP16 cho Jetson Nano.
    *   **Kết quả:** File `.engine`.

6.  **Bước 6: Chạy Inference (Demo)** (`scripts/06_inference_traffic.py`)
    *   **Mô tả:** Chạy nhận diện giao thông từ camera/video thực tế.

---

## 🛠 Cấu hình
Mọi thông số đều được quản lý tại:
*   `configs/pipeline.yaml`: Toggles cho từng bước và thông số hyperparameter.
*   `configs/data.yaml`: Cấu hình đường dẫn dữ liệu ảnh.

---
*Dự án được tối ưu hóa bởi Antigravity AI.*
