"""
Traffic Violation Detection — Desktop/Dev
Run: python main.py
Keys: q=quit  p=pause  c=save calibration frame
"""

import csv, datetime, functools, os
import cv2, numpy as np, torch
from ultralytics import YOLO

torch.load = functools.partial(torch.load, weights_only=False)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "vehicle_model": "weights/03-finetuning/finetune/weights/best.pt",
    "video_source":  "violete2.mp4",
    "log_file":      "violation_logs.csv",

    # Inference
    "conf": 0.15, "iou": 0.40, "imgsz": 1280, "augment": True,

    # Violation zone — tỷ lệ W/H, tự scale theo resolution
    #   TL ─────── TR   ← stop line
    #  /             \
    # BL ─────────── BR
    "zone": [(0.18, 0.38), (0.80, 0.38), (0.88, 0.56), (0.10, 0.56)],

    # Traffic light ROI [x1,y1,x2,y2] — chỉnh bằng calibrate_roi.py
    "light_roi": [917, 111, 947, 155],
    "light_update_hz": 5,       # check đèn mỗi N frame
    "red_thr": 800, "green_thr": 800,

    "calibrate": False,
    # headless: tự detect — Windows luôn show, Linux check $DISPLAY
    "headless": False,  # override thủ công nếu cần
}

VEHICLE_CLS = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

# ── HELPERS ───────────────────────────────────────────────────────────────────
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

def draw_zone(frame, zone, stop_y, state):
    col = (0,60,180) if state=="RED" else (0,140,0)
    ov = frame.copy(); cv2.fillPoly(ov,[zone],col)
    cv2.addWeighted(ov,0.15,frame,0.85,0,frame)
    cv2.polylines(frame,[zone],True,(0,220,220),2)
    cv2.line(frame,(zone[0][0],stop_y),(zone[1][0],stop_y),(0,220,220),3)
    cv2.putText(frame,"STOP LINE",(zone[0][0]+8,stop_y-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,220),2)

def draw_box(frame, x1,y1,x2,y2, label, color):
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
    cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
    cv2.putText(frame,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

def init_log(path):
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            csv.writer(f).writerow(["Timestamp","VehicleID","Class","Plate","Type"])

def write_log(path, obj_id, cls_name):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path,'a',newline='') as f:
        csv.writer(f).writerow([ts, obj_id, cls_name, "PENDING_OCR", "RED_LIGHT"])
    print(f"[VIOLATION] {ts}  id={obj_id}  cls={cls_name}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CFG
    device = 0 if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        print(f"[INIT] CUDA OK — {torch.cuda.get_device_name(device)}")
    else:
        print("[INIT] Device: CPU")
        print("[WARN] No GPU — inference will be slow")

    # Auto-detect headless: Windows luôn có display, Linux check $DISPLAY
    import platform, os as _os
    headless = cfg["headless"]
    if platform.system() == "Linux" and not _os.environ.get("DISPLAY"):
        headless = True
        print("[INIT] No $DISPLAY detected — headless mode")
    elif platform.system() == "Windows":
        headless = False

    model = YOLO(cfg["vehicle_model"])
    cap   = cv2.VideoCapture(cfg["video_source"])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Video: {W}×{H}")

    zone    = make_zone(W, H, cfg["zone"])
    stop_y  = zone[0][1]
    init_log(cfg["log_file"])

    light_state  = "GREEN"
    violated_ids = set()
    frame_idx    = 0

    print("[RUN] q=quit  p=pause  c=save frame\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Traffic light
        if frame_idx % cfg["light_update_hz"] == 0:
            light_state = detect_light(
                frame, cfg["light_roi"],
                cfg["red_thr"], cfg["green_thr"], light_state)

        # Inference
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False,
            conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
            augment=cfg["augment"], agnostic_nms=True,
            classes=list(VEHICLE_CLS.keys()),
            device=device)

        # Draw
        draw_zone(frame, zone, stop_y, light_state)

        s_col = (0,230,0) if light_state=="GREEN" else (0,0,230)
        cv2.putText(frame,f"LIGHT: {light_state}",(20,42),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,s_col,3)
        cv2.putText(frame,f"Violations: {len(violated_ids)}",(20,78),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(240,240,240),2)

        if cfg["calibrate"]:
            x1,y1,x2,y2 = cfg["light_roi"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
            r,g = light_score(frame, cfg["light_roi"])
            cv2.putText(frame,f"R:{r//100} G:{g//100}",(x1,y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1)

        # Detections
        if results[0].boxes.id is not None:
            for box,oid,cls,conf in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.id.cpu().numpy().astype(int),
                    results[0].boxes.cls.cpu().numpy().astype(int),
                    results[0].boxes.conf.cpu().numpy()):

                x1,y1,x2,y2 = map(int,box)
                name = VEHICLE_CLS.get(cls,"Vehicle")
                cx   = (x1+x2)//2
                in_zone = cv2.pointPolygonTest(zone,(float(cx),float(y2)),False)>=0
                violated = light_state=="RED" and in_zone

                if violated and oid not in violated_ids:
                    violated_ids.add(oid)
                    write_log(cfg["log_file"], oid, name)
                    cv2.imwrite(f"violation_{oid}.jpg", frame)

                color = (0,0,220) if violated else (0,210,0)
                draw_box(frame,x1,y1,x2,y2,f"{name} {oid} {conf:.2f}",color)
                cv2.circle(frame,(cx,y2),4,(0,220,220),-1)

        if not cfg["headless"]:
            cv2.imshow("Traffic Monitoring", frame)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('p'): cv2.waitKey(0)
            elif key == ord('c'):
                cv2.imwrite("calibration_frame.png", frame)
                print("[CAL] Saved calibration_frame.png")
        else:
            if frame_idx % 100 == 0:
                print(f"[PROGRESS] frame={frame_idx}  light={light_state}  violations={len(violated_ids)}")

    cap.release(); cv2.destroyAllWindows()
    print(f"\n[DONE] Violations: {len(violated_ids)} — log: {cfg['log_file']}")

if __name__ == "__main__":
    main()