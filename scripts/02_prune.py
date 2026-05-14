import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

import sys
import os
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.pruner import YOLOPruner

import argparse

def main():
    parser = argparse.ArgumentParser(description="Model Pruning")
    parser.add_argument("--pipeline_cfg", type=str, default="configs/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    with open(args.pipeline_cfg, "r") as f:
        cfg = yaml.safe_load(f)["prune"]
    
    # Generate the expected weight path based on data_cfg
    data_name = os.path.splitext(os.path.basename(args.data_cfg))[0]
    suffix = f"-{data_name}" if data_name != "data" else ""
    
    # If weights path in config is the default 'runs/weights/best.pt' 
    # or if we want to auto-locate from Step 1:
    weights = cfg["weights"]
    if "01-sparsity" in weights:
        # Construct path: weights/01-sparsity/sparsity[-suffix]/weights/best.pt
        # We use absolute path to be safe on Colab
        base_dir = os.path.abspath("weights/01-sparsity")
        weights = os.path.join(base_dir, f"sparsity{suffix}", "weights", "best.pt")
        
    print(f"Pruning model using weights from: {weights}")
    
    if not os.path.exists(weights):
        # Fallback: check if it's in runs/detect/weights/...
        alt_weights = os.path.join(os.getcwd(), "runs/detect", "weights/01-sparsity", f"sparsity{suffix}", "weights", "best.pt")
        if os.path.exists(alt_weights):
            weights = alt_weights
            print(f"Found weights in alternative path: {weights}")
        else:
            print(f"Error: Weights not found at {weights}")
            return

    ratio = cfg["ratio"]
    model_size = cfg["model_size"]
    
    # Path to base YOLOv8 config (from installed ultralytics package)
    import ultralytics
    ultralytics_path = Path(ultralytics.__file__).parent
    base_cfg = ultralytics_path / "cfg/models/v8/yolov8.yaml"
    
    if not base_cfg.exists():
        # Fallback for older versions of ultralytics
        base_cfg = ultralytics_path / "models/v8/yolov8.yaml"
    
    pruner = YOLOPruner(weights, str(base_cfg), model_size)
    pruned_model, maskbndict = pruner.prune(ratio=ratio)
    
    # Save pruned model in a format compatible with YOLO()
    save_path = os.path.abspath("weights/02-pruning/pruned_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Minimal checkpoint structure for YOLO() loading
    ckpt = {
        "model": pruned_model,
        "maskbndict": maskbndict,
        "train_args": {}, # Placeholder
    }
    torch.save(ckpt, save_path)
    
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    main()
