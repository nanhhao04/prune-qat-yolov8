import os
import shutil
import torch
import sys
from ultralytics import YOLO

# Tự động nạp DLL từ thư mục lib của TensorRT và bin của cuDNN
trt_lib = r"C:\Program Files\TensorRT-8.6.1.6\lib"
cudnn_bin = r"C:\Program Files\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin"

for path in [trt_lib, cudnn_bin]:
    if os.path.exists(path):
        if path not in os.environ['PATH']:
            os.environ['PATH'] += os.path.pathsep + path
        if sys.platform == 'win32':
            os.add_dll_directory(path)
        print(f"[INFO] Activated binaries from: {path}")

def export_model(pt_path, imgsz=640, half=True):
    """
    Exports a YOLOv8 .pt model to .onnx and .engine (TensorRT) on Windows.
    Moves the result to weights/05-export/.
    """
    if not os.path.exists(pt_path):
        print(f"[ERROR] Weights not found at: {pt_path}")
        return

    # Create target directory
    export_dir = os.path.abspath("weights/05-export")
    os.makedirs(export_dir, exist_ok=True)

    print(f"[INFO] Loading model from: {pt_path}")
    model = YOLO(pt_path)

    print(f"[INFO] Exporting to TensorRT (imgsz={imgsz}, half={half})...")
    try:
        # format='engine' automatically exports to ONNX first, then builds the engine.
        # It usually saves them in the same directory as the .pt file
        exported_path = model.export(format='engine', imgsz=imgsz, half=half, device=0)
        
        # model.export returns the path to the .engine file
        engine_filename = os.path.basename(exported_path)
        dest_engine_path = os.path.join(export_dir, engine_filename)
        
        # Move the engine file
        shutil.move(exported_path, dest_engine_path)
        print(f"[SUCCESS] Engine model saved at: {dest_engine_path}")

        # Also try to move the ONNX file (it usually sits next to the .pt file)
        onnx_source = exported_path.replace(".engine", ".onnx")
        if os.path.exists(onnx_source):
            dest_onnx_path = os.path.join(export_dir, os.path.basename(onnx_source))
            shutil.move(onnx_source, dest_onnx_path)
            print(f"[INFO] ONNX model also moved to: {dest_onnx_path}")

    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        print("\nPossible solutions for Windows:")
        print("1. Ensure CUDA and cuDNN are installed.")
        print("2. Ensure TensorRT SDK is downloaded and the 'bin' folder is in your System PATH.")
        print("3. Check if your GPU supports the requested TensorRT version.")

if __name__ == "__main__":
    # Path to your finetuned weights
    weights_to_export = "weights/03-finetuning/finetune/weights/best.pt"
    
    # Check if file exists, if not, try the base model
    if not os.path.exists(weights_to_export):
        print(f"[WARN] Finetuned weights not found, using base yolov8n.pt for demonstration.")
        weights_to_export = "yolov8n.pt"

    export_model(weights_to_export)
