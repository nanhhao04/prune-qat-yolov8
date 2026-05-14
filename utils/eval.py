from ultralytics import YOLO
import argparse
import os
import time
import json
from pathlib import Path

def evaluate(weights, data="configs/data.yaml", imgsz=640, device=""):
    """
    Evaluate a YOLO model and report comprehensive metrics.
    """
    print(f"Evaluating model: {weights}")
    
    # Load model
    model = YOLO(weights)
    
    # Run validation
    results = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
    
    # Extract metrics
    metrics = {
        "mAP50": round(results.results_dict["metrics/mAP50(B)"], 4),
        "mAP50-95": round(results.results_dict["metrics/mAP50-95(B)"], 4),
        "Precision": round(results.results_dict["metrics/precision(B)"], 4),
        "Recall": round(results.results_dict["metrics/recall(B)"], 4),
        "Fitness": round(results.fitness, 4),
    }
    
    # Latency / Speed metrics (ms per image)
    speed = results.speed
    latency_preprocess = speed["preprocess"]
    latency_inference = speed["inference"]
    latency_postprocess = speed["postprocess"]
    total_latency = latency_preprocess + latency_inference + latency_postprocess
    
    metrics["Latency_Preprocess (ms)"] = round(latency_preprocess, 2)
    metrics["Latency_Inference (ms)"] = round(latency_inference, 2)
    metrics["Latency_Postprocess (ms)"] = round(latency_postprocess, 2)
    metrics["Total_Latency (ms)"] = round(total_latency, 2)
    metrics["FPS"] = round(1000 / total_latency, 2) if total_latency > 0 else 0
    
    # Print results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:25}: {v}")
    print("="*40)
    
    # Save results
    save_dir = Path("runs/eval")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(weights).stem
    save_path = save_dir / f"eval_{model_name}.json"
    
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to: {save_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Evaluation Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt or .engine)")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (e.g. 0 or cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
    else:
        evaluate(args.weights, args.data, args.imgsz, args.device)
