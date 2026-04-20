from ultralytics import YOLO
import yaml

def main():
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["export"]
    
    # Path to the best trained model (either from QAT or Finetune)
    # For Jetson Nano, we might use the finetuned FP16 model instead of QAT INT8
    # Let's assume user wants to export the finetuned one by default
    model_path = "runs/train-finetune/weights/best.pt"
    
    model = YOLO(model_path)
    
    # Export to TensorRT engine
    model.export(
        format="engine",
        imgsz=cfg["imgsz"],
        half=cfg["half"],        # Required for FP16
        int8=cfg["int8"],        # Usually False for Jetson Nano
        workspace=cfg["workspace"],
        device=0
    )
    
    print(f"Model exported to TensorRT engine format.")

if __name__ == "__main__":
    main()
