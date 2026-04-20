from ultralytics import YOLO
import yaml
import os

def main():
    # Load pipeline config
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["sparsity"]
    
    # Initialize model
    model = YOLO(cfg["base_weights"])
    
    # Start sparsity training
    model.train(
        data="configs/data.yaml",
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        sr=cfg["sr"],
        lr0=cfg["lr0"],
        project="runs",
        name="train-sparsity",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
