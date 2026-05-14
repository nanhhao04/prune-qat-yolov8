import torch
import functools
import os
import subprocess
import argparse
import yaml
from ultralytics import YOLO

torch.load = functools.partial(torch.load, weights_only=False)


def export_via_ultralytics(model, onnx_path, engine_path, imgsz, half):
    """Dùng Ultralytics export trực tiếp (hoạt động với ultralytics <= 8.2.x)."""
    try:
        # Ultralytics saves in the same dir as the model by default
        result = model.export(format="engine", imgsz=imgsz, half=half, device=0)
        
        # Move the results to the target paths if they differ
        if os.path.abspath(result) != os.path.abspath(engine_path):
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            import shutil
            shutil.move(result, engine_path)
            
            onnx_result = result.replace(".engine", ".onnx")
            if os.path.exists(onnx_result):
                shutil.move(onnx_result, onnx_path)
                
        print(f"[DONE] Engine saved: {engine_path}")
        return True
    except TypeError as e:
        if "dynamo" in str(e):
            print(f"[WARN] Ultralytics version too new (dynamo kwarg error)")
            print(f"       Fix: pip install ultralytics==8.2.103")
        else:
            print(f"[ERROR] Ultralytics export failed: {e}")
        return False


def export_via_torch_onnx(model, onnx_path, imgsz):
    """Bypass Ultralytics: dùng torch.onnx.export trực tiếp."""
    torch_model = model.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      device  : {device}")
    torch_model = torch_model.to(device)
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)

    torch.onnx.export(
        torch_model, dummy, onnx_path,
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )
    print(f"[1/2] ONNX saved: {onnx_path}")
    return True


def find_trtexec():
    candidates = [
        "trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
        r = subprocess.run(f"which {c}", shell=True, capture_output=True)
        if r.returncode == 0:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT engine")
    parser.add_argument("--weights", type=str,
                        default="weights/03-finetuning/finetune/weights/best.pt")
    parser.add_argument("--pipeline_cfg", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--onnx_only", action="store_true")
    args = parser.parse_args()

    weights_path = os.path.abspath(args.weights)
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    with open(args.pipeline_cfg, "r") as f:
        cfg = yaml.safe_load(f)["export"]

    imgsz       = cfg["imgsz"]
    half        = cfg.get("half", True)
    
    # Create target directory
    export_dir = os.path.abspath("weights/05-export")
    os.makedirs(export_dir, exist_ok=True)
    
    filename = os.path.basename(weights_path).replace(".pt", "")
    onnx_path   = os.path.join(export_dir, f"{filename}.onnx")
    engine_path = os.path.join(export_dir, f"{filename}.engine")

    print(f"\n[EXPORT] {weights_path}")
    print(f"         Output Dir: {export_dir}")
    print(f"         imgsz={imgsz}  half={half}")

    model = YOLO(weights_path)

    # ── Thử 1: Ultralytics native (nhanh nhất, hoạt động trên ultralytics <= 8.2.x)
    print("\n[TRY 1] Ultralytics native export (engine)...")
    if export_via_ultralytics(model, onnx_path, engine_path, imgsz, half):
        return

    # ── Thử 2: torch.onnx.export → trtexec
    print("\n[TRY 2] torch.onnx.export + trtexec...")
    if not export_via_torch_onnx(model, onnx_path, imgsz):
        return

    if args.onnx_only:
        print("[DONE] ONNX-only mode.")
        return

    trtexec_bin = find_trtexec()
    if trtexec_bin:
        print(f"[2/2] Found trtexec: {trtexec_bin}")
        half_flag = "--fp16" if half else ""
        cmd = (f"{trtexec_bin} --onnx={onnx_path} --saveEngine={engine_path} "
               f"{half_flag} "
               f"--minShapes=images:1x3x{imgsz}x{imgsz} "
               f"--optShapes=images:1x3x{imgsz}x{imgsz} "
               f"--maxShapes=images:1x3x{imgsz}x{imgsz} "
               f"--workspace=4096")
        print(f"[2/2] Building engine...\n      {cmd}\n")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode == 0:
            print(f"\n[DONE] Engine saved: {engine_path}")
            return
        print(f"[ERROR] trtexec failed (code {ret.returncode})")

    # ── Hướng dẫn thủ công
    half_str = "--fp16" if half else ""
    print("\n" + "="*60)
    print("  ONNX đã sẵn sàng. Để build engine, chạy một trong các lệnh sau:")
    print("  1. Downgrade ultralytics (khuyến nghị):")
    print("     pip install ultralytics==8.2.103")
    print("     python3 scripts/05_export.py --weights", args.weights)
    print()
    print("  2. Cài trtexec rồi chạy thủ công:")
    print("     sudo apt-get install -y tensorrt")
    print(f"     trtexec --onnx={onnx_path} --saveEngine={engine_path} {half_str}")
    print("="*60)


if __name__ == "__main__":
    main()
