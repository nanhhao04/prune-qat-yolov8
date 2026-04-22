# YOLOv8 Optimization Pipeline cho Jetson Nano (Traffic Camera)

Dự án này cung cấp một quy trình hoàn chỉnh để tối ưu hóa mô hình YOLOv8 cho thiết bị phần cứng hạn chế như **Jetson Nano (Maxwell GPU)**. Chúng tôi sử dụng kết hợp **Structured Pruning (Network Slimming)** và **TensorRT (FP16)** để tối đa hóa FPS mà vẫn giữ được độ chính xác (mAP).

##  Luồng tối ưu hóa đề xuất (PC -> Jetson)

Để đạt hiệu quả cao nhất, quy trình được chia làm 2 giai đoạn:

### Giai đoạn 1: Tối ưu trên Máy tính mạnh (PC có GPU)
Mục tiêu là huấn luyện và cắt tỉa mô hình để có file `.pt` nhẹ nhất nhưng vẫn chính xác.

1.  **Chuẩn bị dữ liệu:** `python scripts/prepare_data.py`
2.  **Huấn luyện tối ưu:** Chạy script tự động để thực hiện Sparsity -> Pruning -> Finetuning:
    ```bash
    python scripts/pc_train_pipeline.py
    ```
    *Kết quả: File model tối ưu sẽ nằm tại `weights/for_jetson/optimized_yolo_jetson.pt`*

### Giai đoạn 2: Triển khai & Benchmark trên Jetson Nano
Copy file `.pt` đã tối ưu sang Jetson và thực hiện:

1.  **Export TensorRT:**
    ```bash
    python scripts/05_export.py --weights optimized_yolo_jetson.pt
    ```
2.  **So sánh hiệu năng (Benchmark):**
    So sánh trực tiếp tốc độ và độ chính xác giữa bản gốc (.pt) và bản TensorRT (.engine):
    ```bash
    python scripts/compare_models.py --pt optimized_yolo_jetson.pt --engine optimized_yolo_jetson.engine
    ```
    *Báo cáo chi tiết sẽ được lưu tại `runs/logs/comparison_report.txt`*

---

## 🛠 Hướng dẫn chi tiết

### 1. Cài đặt môi trường
Đảm bảo bạn đã bật **Swap Memory** (ít nhất 4GB) trên Jetson Nano:
```bash
sudo apt-get install zram-config
# Hoặc tạo swapfile thủ công (khuyên dùng 4GB-8GB)
```

Cài đặt dependencies:
```bash
pip install -r requirements.txt
# Lưu ý: Với Jetson, nên cài PyTorch từ file .whl chính thức của NVIDIA
```

### 2. Triển khai Inference (Demo)
Sau khi đã có file `.engine`, chạy nhận diện thực tế:
```bash
# Chạy trên camera local
python scripts/06_inference_traffic.py --source 0 --weights optimized_yolo_jetson.engine

# Chạy trên folder ảnh để test
python scripts/06_inference_traffic.py --source datasets/traffic/images/val --weights optimized_yolo_jetson.engine
```

---

##  Mẹo tối ưu FPS trên Jetson Nano

1.  **Power Mode:** Đưa Jetson Nano về chế độ hiệu năng cao nhất (10W):
    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```
2.  **FP16 Precision:** Luôn sử dụng `--half` khi export TensorRT. Jetson Nano (Maxwell) hoạt động hiệu quả nhất với FP16.
3.  **Input Size:** Sử dụng `imgsz=320` hoặc `416` nếu bạn cần FPS cực cao (>30 FPS). Mặc định pipeline để `640`.

---
*Dự án được phát triển và tối ưu bởi Antigravity AI Assistant.*
