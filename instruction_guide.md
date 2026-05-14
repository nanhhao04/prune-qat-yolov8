# Instruction Guide — YOLOv8 Traffic Violation Detection

## Tổng quan hệ thống

```
Video / Camera
    ↓
YOLOv8 (Vehicle Detection + Tracking)
    ↓
Violation Zone Check (Polygon + Traffic Light State)
    ↓
YOLOv8 (Plate Detection) → OCR
    ↓
Log CSV + Ảnh bằng chứng
```

**Hai chế độ chạy:**
- `main.py` — Chạy trên PC với file `.pt`
- `inference_on_edge.py` — Chạy trên Jetson Nano với file `.engine` (TensorRT FP16)

---

## Cấu trúc thư mục

```
yolo_pipeline_hao/
├── main.py                  # Pipeline PC
├── inference_on_edge.py     # Pipeline Jetson Nano
├── benchmark.py             # So sánh metric các model
├── calibrate_roi.py         # Công cụ lấy tọa độ ROI
├── configs/
│   ├── data.yaml            # Dataset xe
│   ├── data-plate.yaml      # Dataset biển số
│   └── pipeline.yaml        # Tham số train/export
├── scripts/
│   ├── 01_sparsity_train.py
│   ├── 02_prune.py
│   ├── 03_finetune.py
│   ├── 04_qat.py
│   └── 05_export.py
└── weights/
    ├── 01-sparsity/
    ├── 02-pruning/
    ├── 03-finetuning/       # ← Model quan trọng nhất
    │   ├── finetune/weights/best.pt        (xe)
    │   └── finetune-data-plate/weights/best.pt  (biển số)
    └── 04-qat/
```

---

## PHẦN 1 — Chạy trên PC

### Cài đặt môi trường

```bash
pip install ultralytics easyocr opencv-python psutil
```

> ⚠️ Nếu gặp lỗi `cv2.imshow not implemented`: gỡ `opencv-python-headless` và cài lại `opencv-python`
> ```bash
> pip uninstall opencv-python-headless -y
> pip install opencv-python
> ```


### Chạy hệ thống

```bash
python main.py
```

**Controls:**
| Phím | Tác dụng |
|------|----------|
| `q`  | Thoát |
| `p`  | Pause / Resume |
| `c`  | Lưu frame để calibrate ROI |

---

## PHẦN 2 — Deploy lên Jetson Nano

### Bước 1: Copy project sang Jetson

```bash
# Trên PC — tạo file zip (bỏ dataset nặng)
cd C:\Users\Admin\Desktop
Compress-Archive -Path yolo_pipeline_hao -DestinationPath deploy.zip

# Copy sang Jetson (thay <jetson_ip>)
scp deploy.zip nano@<jetson_ip>:/home/nano/
```

### Bước 2: Giải nén và cài đặt trên Jetson

```bash
ssh nano@<jetson_ip>

cd /home/nano
unzip deploy.zip
cd yolo_pipeline_hao

# Cài thư viện (Jetson đã có PyTorch + CUDA từ JetPack)
pip3 install ultralytics easyocr psutil
```

**Kiểm tra CUDA:**
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Phải ra: CUDA: True
```

### Bước 3: Build TensorRT Engine trên Jetson

> ⚠️ Engine **phải build trực tiếp trên Jetson** — không thể dùng engine từ PC

```bash
# Export model xe
python3 scripts/05_export.py --weights weights/03-finetuning/finetune/weights/best.pt

# Export model biển số
python3 scripts/05_export.py --weights weights/03-finetuning/finetune-data-plate/weights/best.pt
```

> Build lần đầu mất **5-10 phút**. Kết quả: `best.engine` xuất hiện cạnh `best.pt`.

### Bước 4: Chạy trên Jetson

```bash
python3 inference_on_edge.py
```

**Tối ưu hiệu năng:**

```bash
# Bật max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Tắt màn hình để tăng FPS (headless mode)
# Sửa trong inference_on_edge.py:
# cv2.imshow(...)  →  comment dòng này
```

---

## PHẦN 3 — Benchmark & So sánh Metric

### Chạy benchmark

```bash
pip install psutil  # nếu chưa có

# Chạy trên PC hoặc Jetson
python benchmark.py --source violetetraffic.mp4 --n_frames 200
```

Kết quả được in ra terminal và lưu vào `benchmark_results.csv`.

### Metric so sánh mong đợi

| Model                  | FPS (Jetson) | Avg Latency | RAM     | Ghi chú              |
|------------------------|:------------:|:-----------:|:-------:|----------------------|
| `01_sparsity.pt`       | ~3–5         | ~200 ms     | ~900 MB | Baseline có L1 mask  |
| `02_pruned.pt`         | ~5–7         | ~150 ms     | ~700 MB | Cắt 30% channel      |
| `03_finetune.pt`       | ~5–8         | ~130 ms     | ~700 MB | Finetune sau pruning |
| `03_finetune.engine`   | **~20–30**   | **~35 ms**  | ~400 MB | **TensorRT FP16 ✓**  |

> Engine FP16 nhanh hơn ~**4–5×** so với PyTorch trên Jetson Nano Maxwell.

---

## PHẦN 4 — Output & Log

### File log vi phạm

`violation_logs.csv` (PC) hoặc `edge_violation_logs.csv` (Jetson):

```csv
Time,                    ID,  Class,  Type
2025-09-25 08:42:31,     7,   Car,    RED_LIGHT
2025-09-25 08:43:15,     12,  Moto,   RED_LIGHT
```

### Ảnh bằng chứng

Tự động lưu tại thời điểm phát hiện vi phạm:
```
violation_7.jpg
violation_12.jpg
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|-----|------------|-----|
| `cv2.error: inRange Bad argument` | Dùng list thay vì np.array | Wrap với `np.array([...])` |
| `cv2.imshow not implemented` | Dùng opencv-headless | `pip install opencv-python` |
| `CUDA: False` trên Jetson | JetPack chưa cài đủ | Kiểm tra `torch` version |
| Engine không tìm thấy | Chưa build trên Jetson | Chạy `05_export.py` trên Jetson |
| FPS thấp trên Jetson | Chế độ power save | `sudo nvpmodel -m 0 && sudo jetson_clocks` |
