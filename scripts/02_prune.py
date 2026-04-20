import sys
import os
import yaml
import torch
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.pruner import YOLOPruner

def main():
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["prune"]
    
    weights = cfg["weights"]
    ratio = cfg["ratio"]
    model_size = cfg["model_size"]
    
    # Path to base YOLOv8 config (from ultralytics repo)
    # Assuming the current working directory relative to project root
    base_cfg = "../ultralytics/cfg/models/v8/yolov8.yaml"
    
    pruner = YOLOPruner(weights, base_cfg, model_size)
    pruned_model, maskbndict = pruner.prune(ratio=ratio)
    
    # Save pruned model
    save_path = "weights/pruned_model.pt"
    os.makedirs("weights", exist_ok=True)
    torch.save({
        "model": pruned_model,
        "maskbndict": maskbndict
    }, save_path)
    
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    main()
