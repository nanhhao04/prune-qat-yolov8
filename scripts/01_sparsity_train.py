import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml
import os

def main():
    # Load pipeline config
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["sparsity"]
    
    # Initialize model
    model = YOLO(cfg["base_weights"])
    
    # Determine if we use sparsity or normal training
    is_pipeline = cfg.get("enabled", True)
    sr_val = cfg["sr"] if is_pipeline else None
    train_name = "train-sparsity" if is_pipeline else "train-normal"
    
    print(f"Mode: {'Pipeline (Sparsity)' if is_pipeline else 'Normal Train'}")
    
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
        data="configs/data.yaml",
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        lr0=cfg["lr0"],
        device=cfg.get("device", 0),
        project="runs",
        name=train_name,
        exist_ok=True
    )

if __name__ == "__main__":
    main()
