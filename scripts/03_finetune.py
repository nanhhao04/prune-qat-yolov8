import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

from ultralytics import YOLO
import yaml

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Finetuning")
    parser.add_argument("--pipeline_cfg", type=str, default="configs/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    with open(args.pipeline_cfg, "r") as f:
        cfg = yaml.safe_load(f)["finetune"]
    
    # Load the pruned model we just saved
    # Note: If running for plate, you might want to specify which weights to load.
    # By default it loads weights/02-pruning/pruned_model.pt
    pruned_checkpoint = os.path.abspath("weights/02-pruning/pruned_model.pt")
    if not os.path.exists(pruned_checkpoint):
        print(f"Error: Pruned model not found at {pruned_checkpoint}")
        return
        
    model = YOLO(pruned_checkpoint)
    
    # Generate a run name based on data config
    data_name = os.path.splitext(os.path.basename(args.data_cfg))[0]
    suffix = f"-{data_name}" if data_name != "data" else ""
    train_name = "finetune" + suffix

    model.train(
        data=args.data_cfg,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        lr0=cfg["lr0"],
        device=cfg.get("device", 0),
        project=os.path.abspath("weights/03-finetuning"),
        name=train_name,
        exist_ok=True
    )

if __name__ == "__main__":
    main()
