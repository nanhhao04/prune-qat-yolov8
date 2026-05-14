"""
Traffic Violation Detection — Jetson Edge (TensorRT .engine)
Run: python inference_on_edge.py
Keys: q=quit

Requires:
  - YOLOv8 .engine built on same Jetson (tensorrt version must match)
  - pip install ultralytics easyocr opencv-python
  - GStreamer pipeline cho CSI camera (thay video_source nếu dùng cam)
"""

import csv, datetime, os
import cv2, numpy as np
import torch
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "vehicle_engine": "weights/best_vehicle.engine",
    "plate_engine":   "weights/best_plate.engine",

    # Camera: 0 = USB cam, dùng GStreamer string cho CSI cam Jetson
    # "video_source": "nvarguscamerasrc ! ... ! appsink"
    "video_source":   "violetetraffic.mp4",
    "log_file":       "edge_violation_logs.csv",
    "evidence_dir":   "evidence",

    # Inference — giữ nhỏ để đủ FPS trên Jetson
    "conf": 0.25, "iou": 0.45, "imgsz": 640,

    # Violation zone — tỷ lệ W/H
    "zone": [(0.18, 0.38), (0.80, 0.38), (0.88, 0.56), (0.10, 0.56)],

    # Traffic light ROI [x1,y1,x2,y2]
    "light_roi": [917, 111, 947, 155],
    "light_update_hz": 5,
    "red_thr": 800, "green_thr": 800,

    # Jetson options
    "show_display": True,   # False để max FPS khi không có màn hình
    "save_evidence": True,  # lưu ảnh vi phạm
}

VEHICLE_CLS = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

# ── HELPERS (copy từ main.py — giữ 2 file độc lập) ───────────────────────────
def make_zone(W, H, pts):
    return np.array([[int(x*W), int(y*H)] for x,y in pts], np.int32)

def light_score(frame, roi):
    x1,y1,x2,y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return 0, 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r = cv2.inRange(hsv,np.array([0,120,120]),np.array([10,255,255])) + cv2.inRange(hsv,np.array([160,120,120]),np.array([180,255,255]))
    g = cv2.inRange(hsv,np.array([40,120,120]),np.array([90,255,255]))
    return int(np.sum(r)), int(np.sum(g))

def detect_light(frame, roi, red_thr, green_thr, prev):
    r, g = light_score(frame, roi)
    if r > g and r > red_thr:  return "RED"
    if g > green_thr:          return "GREEN"
    return prev

def draw_minimal(frame, zone, stop_y, light_state, n_viol):
    """Overlay tối giản — ít tốn GPU hơn trên Jetson."""
    cv2.polylines(frame,[zone],True,(0,220,220),2)
    cv2.line(frame,(zone[0][0],stop_y),(zone[1][0],stop_y),(0,220,220),2)
    col = (0,230,0) if light_state=="GREEN" else (0,0,230)
    cv2.putText(frame,f"LIGHT:{light_state} | VIO:{n_viol}",
                (16,36),cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2)

def init_log(path):
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            csv.writer(f).writerow(["Timestamp","VehicleID","Class","Plate","Type"])

def write_log(path, obj_id, cls_name, plate="PENDING"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path,'a',newline='') as f:
        csv.writer(f).writerow([ts, obj_id, cls_name, plate, "RED_LIGHT"])
    print(f"[VIOLATION] {ts}  id={obj_id}  cls={cls_name}  plate={plate}")

def ocr_plate(plate_model, vehicle_crop):
    """Chạy plate detector rồi trả về text. Bỏ qua nếu crop rỗng."""
    if vehicle_crop.size == 0: return "UNKNOWN"
    res = plate_model(vehicle_crop, verbose=False, imgsz=320, device=0 if torch.cuda.is_available() else "cpu")
    for r in res:
        for box in r.boxes:
            px1,py1,px2,py2 = map(int,box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0: continue
            # EasyOCR — import lazy để không crash nếu chưa cài
            try:
                import easyocr
                if not hasattr(ocr_plate,'_reader'):
                    ocr_plate._reader = easyocr.Reader(['en'], gpu=True)
                out = ocr_plate._reader.readtext(plate_crop)
                if out: return out[0][1].upper()
            except Exception as e:
                print(f"[OCR] {e}")
    return "UNKNOWN"

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CFG
    os.makedirs(cfg["evidence_dir"], exist_ok=True)

    # Kiểm tra CUDA — bắt buộc trên Jetson
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — check JetPack installation")
        DEVICE = "cpu"
    else:
        DEVICE = 0  # GPU index 0
        print(f"[INIT] CUDA OK — {torch.cuda.get_device_name(0)}")

    print("[INIT] Loading TensorRT engines...")
    vehicle_model = YOLO(cfg["vehicle_engine"], task="detect")
    plate_model   = YOLO(cfg["plate_engine"],   task="detect")
    print("[INIT] Engines loaded")

    cap = cv2.VideoCapture(cfg["video_source"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {cfg['video_source']}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Video: {W}×{H}")

    zone    = make_zone(W, H, cfg["zone"])
    stop_y  = zone[0][1]
    init_log(cfg["log_file"])

    light_state  = "GREEN"
    violated_ids = set()
    frame_idx    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # ── Traffic light ────────────────────────────────────────────────
        if frame_idx % cfg["light_update_hz"] == 0:
            light_state = detect_light(
                frame, cfg["light_roi"],
                cfg["red_thr"], cfg["green_thr"], light_state)

        # ── Inference (TensorRT engine trên GPU) ────────────────────────
        results = vehicle_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False,
            conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
            agnostic_nms=True, classes=list(VEHICLE_CLS.keys()),
            device=DEVICE)

        # ── Violation logic ──────────────────────────────────────────────
        if results[0].boxes.id is not None:
            for box, oid, cls, conf in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.id.cpu().numpy().astype(int),
                    results[0].boxes.cls.cpu().numpy().astype(int),
                    results[0].boxes.conf.cpu().numpy()):

                x1,y1,x2,y2 = map(int,box)
                cx = (x1+x2)//2
                in_zone = cv2.pointPolygonTest(
                    zone,(float(cx),float(y2)),False) >= 0
                violated = light_state=="RED" and in_zone

                if violated and oid not in violated_ids:
                    violated_ids.add(oid)
                    # OCR biển số từ crop xe
                    crop  = frame[max(0,y1):y2, max(0,x1):x2]
                    plate = ocr_plate(plate_model, crop)
                    write_log(cfg["log_file"], oid,
                              VEHICLE_CLS.get(cls,"Vehicle"), plate)
                    if cfg["save_evidence"]:
                        path = os.path.join(cfg["evidence_dir"],
                                            f"vio_{oid}.jpg")
                        cv2.imwrite(path, frame)

                if cfg["show_display"]:
                    col = (0,0,220) if violated else (0,200,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                    cv2.putText(frame,
                                f"{VEHICLE_CLS.get(cls,'?')} {oid} {conf:.2f}",
                                (x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,col,1)

        # ── Display ──────────────────────────────────────────────────────
        if cfg["show_display"]:
            draw_minimal(frame,zone,stop_y,light_state,len(violated_ids))
            cv2.imshow("Edge Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            # Headless: print FPS mỗi 100 frame
            if frame_idx % 100 == 0:
                print(f"[PROGRESS] frame={frame_idx}"
                      f"  light={light_state}"
                      f"  violations={len(violated_ids)}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Violations: {len(violated_ids)} — log: {cfg['log_file']}")

if __name__ == "__main__":
    main()