import os
from ultralytics import YOLO
import time

def test_onnx_inference(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        return

    print(f"[INFO] Loading ONNX model: {model_path}")
    # Load the model - Ultralytics will automatically use ONNX Runtime if .onnx is passed
    # task='detect' is usually inferred but good to be explicit
    model = YOLO(model_path, task='detect')

    print(f"[INFO] Running inference on GPU (CUDA/TensorRT)...")
    
    # Warmup
    _ = model.predict(image_path, device=0, verbose=False)

    # Benchmark
    start_time = time.time()
    results = model.predict(image_path, device=0, imgsz=640)
    end_time = time.time()

    print(f"\n[RESULTS]")
    print(f"- Inference time: {(end_time - start_time)*1000:.2f} ms")
    
    # Show detections
    for result in results:
        print(f"- Found {len(result.boxes)} objects")
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            print(f"  * {name}: {conf:.2f}")

    # Save output image
    out_dir = "runs/onnx_test"
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "result.jpg")
    results[0].save(filename=res_path)
    print(f"\n[DONE] Result saved to: {res_path}")

if __name__ == "__main__":
    MODEL = "weights/03-finetuning/finetune/weights/best.onnx"
    IMAGE = "data/traffic/images/train/00002754.jpg"
    
    test_onnx_inference(MODEL, IMAGE)
