import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

import torch.nn as nn
from ultralytics import YOLO
import yaml
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description="Sparsity Training")
    parser.add_argument("--pipeline_cfg", type=str, default="configs/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    # Load pipeline config
    with open(args.pipeline_cfg, "r") as f:
        cfg = yaml.safe_load(f)["sparsity"]
    
    # Initialize model
    model = YOLO(cfg["base_weights"])
    
    # Determine if we use sparsity or normal training
    is_pipeline = cfg.get("enabled", True)
    sr_val = cfg["sr"] if is_pipeline else None
    
    # Generate a run name based on data config
    data_name = os.path.splitext(os.path.basename(args.data_cfg))[0]
    suffix = f"-{data_name}" if data_name != "data" else ""
    train_name = ("sparsity" if is_pipeline else "normal") + suffix
    
    print(f"Mode: {'Pipeline (Sparsity)' if is_pipeline else 'Normal Train'}")
    print(f"Data: {args.data_cfg}")
    
    # Add callback for sparsity penalty (Network Slimming)
    if is_pipeline and sr_val:
        def on_after_backward(trainer):
            for m in trainer.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # L1 regularization on BN weights (gamma)
                    m.weight.grad.data.add_(sr_val * torch.sign(m.weight.data))
        
        model.add_callback('on_after_backward', on_after_backward)
        print(f"Sparsity penalty applied with sr={sr_val}")

    # Start training
    model.train(
        data=args.data_cfg,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        lr0=cfg["lr0"],
        device=cfg.get("device", 0),
        project=os.path.abspath("weights/01-sparsity"),
        name=train_name,
        exist_ok=True
    )

if __name__ == "__main__":
    main()
