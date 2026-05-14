import cv2
import time
import os
from ultralytics import YOLO

def run_inference(engine_path, source="violetetraffic.mp4"):
    """
    Runs inference on Windows using a TensorRT engine.
    """
    if not os.path.exists(engine_path):
        print(f"[ERROR] Engine file not found: {engine_path}")
        print("Please run 'python export_to_trt.py' first.")
        return

    # Load the TensorRT engine
    print(f"[INFO] Loading TensorRT engine: {engine_path}")
    model = YOLO(engine_path, task="detect")

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print(f"[INFO] Starting inference on {source}...")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # Inference
        # device=0 ensures we use the NVIDIA GPU
        results = model.predict(frame, conf=0.25, device=0, verbose=False)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Draw results
        annotated_frame = results[0].plot()
        
        # Overlay FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f} (TensorRT)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 TensorRT Windows Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference finished.")

if __name__ == "__main__":
    # Path to the generated engine file
    # If you exported best.pt, the engine will be best.engine in the same folder
    engine_file = "weights/03-finetuning/finetune/weights/best.engine"
    
    # Fallback to demo engine if best.engine doesn't exist
    if not os.path.exists(engine_file):
        engine_file = "yolov8n.engine"

    video_source = "violetetraffic.mp4" # Or 0 for webcam
    
    run_inference(engine_file, video_source)
