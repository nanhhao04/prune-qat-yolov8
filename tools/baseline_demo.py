"""
baseline_demo.py
================
Chạy YOLOv8n gốc (COCO pretrained, chưa fine-tune) trên video
để so sánh với bản đã fine-tune trong main.py.

Usage:
    python baseline_demo.py
    python baseline_demo.py --video violete2.mp4 --conf 0.25
"""

import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO

VEHICLE_CLS = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="violete2.mp4")
    parser.add_argument("--model",  default="yolov8n.pt",  help="COCO baseline model")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--imgsz",  type=int,   default=640)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[BASELINE] Model : {args.model}  (no fine-tune)")
    print(f"[BASELINE] Video : {args.video}")
    print(f"[BASELINE] Device: {'CUDA' if device == 0 else 'CPU'}")

    model = YOLO(args.model)  # auto-download yolov8n.pt nếu chưa có
    cap   = cv2.VideoCapture(args.video)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[BASELINE] Video : {W}×{H}  |  q=quit\n")

    # Stop line đơn giản để tham chiếu
    stop_y = int(H * 0.42)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False,
                        conf=args.conf,
                        imgsz=args.imgsz,
                        classes=list(VEHICLE_CLS.keys()),
                        device=device)

        boxes = results[0].boxes
        n_det = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            name = VEHICLE_CLS.get(cls, "?")

            color = (180, 180, 0)  # Cyan-ish — phân biệt với bản fine-tune (xanh/đỏ)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Stop line tham chiếu
        cv2.line(frame, (0, stop_y), (W, stop_y), (255, 255, 0), 1)

        # Header
        cv2.putText(frame, f"BASELINE yolov8n | dets: {n_det}",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        cv2.imshow("Baseline (no fine-tune)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")

if __name__ == "__main__":
    main()
