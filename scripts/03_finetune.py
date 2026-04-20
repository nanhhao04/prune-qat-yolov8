from ultralytics import YOLO
import yaml

def main():
    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["finetune"]
    
    # Load the pruned model we just saved
    # Note: pruned models use custom logic so they might need special loading
    # if not using the YOLO() wrapper cleanly. However, original project uses 
    # model = YOLO(pruned_checkpoint)
    
    model = YOLO("weights/pruned_model.pt")
    
    model.train(
        data="configs/data.yaml",
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        lr0=cfg["lr0"],
        project="runs",
        name="train-finetune",
        exist_ok=True,
        finetune=True  # Ensure finetune mode is ON
    )

if __name__ == "__main__":
    main()
