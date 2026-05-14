"""
benchmark.py — So sánh latency, FPS, mAP, RAM giữa các model:
  - best.pt  (PyTorch FP32 - baseline)
  - best.pt  (sau Pruning)
  - best.pt  (sau Finetune)
  - best.engine (TensorRT FP16)

Usage:
    python benchmark.py --source violetetraffic.mp4 --n_frames 200
"""

import cv2
import time
import argparse
import torch
import numpy as np
import os
import csv
import sys
from ultralytics import YOLO

# Tự động nạp DLL từ thư mục lib của TensorRT và bin của cuDNN
trt_lib = r"C:\Program Files\TensorRT-8.6.1.6\lib"
cudnn_bin = r"C:\Program Files\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin"

for path in [trt_lib, cudnn_bin]:
    if os.path.exists(path):
        if sys.platform == 'win32':
            try:
                os.add_dll_directory(path)
            except Exception:
                pass
        if path not in os.environ['PATH']:
            os.environ['PATH'] += os.path.pathsep + path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed — RAM metrics disabled. Run: pip install psutil")

MODELS = {
    "1_sparsity_pt":   "weights/01-sparsity/sparsity/weights/best.pt",
    "2_pruned_pt":     "weights/02-pruning/pruned_model.pt",
    "3_finetune_pt":   "weights/03-finetuning/finetune/weights/best.pt",
    "4_finetune_onnx": "weights/03-finetuning/finetune/weights/best.onnx",
    "5_engine_fp16":   "weights/05-export/best.engine",
}

def ram_mb():
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return -1

def benchmark_model(name, model_path, source, n_frames, imgsz=640):
    print(f"\n{'='*55}")
    print(f"  Model: {name}")
    print(f"  Path : {model_path}")
    print(f"{'='*55}")

    if not os.path.exists(model_path):
        print(f"  [SKIP] File not found: {model_path}")
        return None

    try:
        model = YOLO(model_path, task='detect')

        cap = cv2.VideoCapture(source) if isinstance(source, str) else cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [SKIP] Cannot open video: {source}")
            return None

        latencies = []
        total_detections = 0
        
        # 1. Warm-up phase (Bỏ qua 10 frame đầu để ONNX/TensorRT ổn định)
        print(f"  [INFO] Warming up 10 frames...")
        for _ in range(10):
            ret, frame = cap.read()
            if not ret: break
            _ = model(frame, verbose=False, imgsz=imgsz, device=0 if torch.cuda.is_available() else "cpu", conf=0.25, iou=0.45)
        
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ram_before = ram_mb()

        # 2. Real benchmark
        print(f"  [INFO] Benchmarking {n_frames} frames...")
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret: break

            t0 = time.perf_counter()
            # Ép dùng chung conf và iou để so sánh độ chính xác (total dets)
            results = model(frame, verbose=False, imgsz=imgsz, device=0 if torch.cuda.is_available() else "cpu", conf=0.25, iou=0.45)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            total_detections += len(results[0].boxes)

        cap.release()
        del model  # Free memory

        lat = np.array(latencies)
        ram_after = ram_mb()

        result = {
            "model":          name,
            "path":           model_path,
            "n_frames":       len(latencies),
            "avg_latency_ms": round(float(np.mean(lat)), 2),
            "p50_ms":         round(float(np.percentile(lat, 50)), 2),
            "p95_ms":         round(float(np.percentile(lat, 95)), 2),
            "min_ms":         round(float(np.min(lat)), 2),
            "max_ms":         round(float(np.max(lat)), 2),
            "fps":            round(1000 / float(np.mean(lat)), 2),
            "ram_delta_mb":   round(ram_after - ram_before, 1),
            "total_dets":     total_detections,
        }

        print(f"  Avg Latency : {result['avg_latency_ms']} ms")
        print(f"  FPS         : {result['fps']}")
        print(f"  P50 / P95   : {result['p50_ms']} / {result['p95_ms']} ms")
        print(f"  Min / Max   : {result['min_ms']} / {result['max_ms']} ms")
        print(f"  RAM Δ       : {result['ram_delta_mb']} MB")
        print(f"  Total dets  : {total_detections} over {len(latencies)} frames")
        return result
    except Exception as e:
        print(f"  [ERROR] Could not benchmark {name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    default="violetetraffic.mp4")
    parser.add_argument("--n_frames",  type=int, default=200)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--out",       default="benchmark_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[BENCH] Device: {device} | frames: {args.n_frames} | imgsz: {args.imgsz}")

    results = []
    for name, path in MODELS.items():
        # Set device=0 for GPU in the benchmark function
        r = benchmark_model(name, path, args.source, args.n_frames, args.imgsz)
        if r: results.append(r)

    # Summary table
    print(f"\n{'='*75}")
    print(f"  {'Model':<25} {'FPS':>6} {'Avg(ms)':>9} {'P95(ms)':>9} {'RAM Δ MB':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*10}")
    for r in results:
        print(f"  {r['model']:<25} {r['fps']:>6} {r['avg_latency_ms']:>9} {r['p95_ms']:>9} {r['ram_delta_mb']:>10}")
    print(f"{'='*75}")

    # Save CSV
    if results:
        keys = results[0].keys()
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[DONE] Results saved to: {args.out}")

if __name__ == "__main__":
    main()
