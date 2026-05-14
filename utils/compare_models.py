import argparse
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

def evaluate_model(weights, data="configs/data.yaml", imgsz=640):
    """
    Evaluates a model and returns key metrics.
    """
    print(f"\n--- Evaluating: {weights} ---")
    if not os.path.exists(weights):
        print(f"Error: Weights file not found: {weights}")
        return None
        
    model = YOLO(weights)
    # Run validation
    results = model.val(data=data, imgsz=imgsz, verbose=False)
    
    # Extract metrics
    metrics = {
        "Model": Path(weights).name,
        "mAP50": round(results.results_dict["metrics/mAP50(B)"], 4),
        "mAP50-95": round(results.results_dict["metrics/mAP50-95(B)"], 4),
        "Inference (ms)": round(results.speed["inference"], 2),
        "FPS": round(1000 / results.speed["inference"], 2) if results.speed["inference"] > 0 else 0
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8 .pt and .engine models")
    parser.add_argument("--pt", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine model")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    results = []
    # Test .pt file
    pt_metrics = evaluate_model(args.pt, args.data, args.imgsz)
    if pt_metrics:
        results.append(pt_metrics)
    
    # Test .engine file
    engine_metrics = evaluate_model(args.engine, args.data, args.imgsz)
    if engine_metrics:
        results.append(engine_metrics)

    if len(results) < 2:
        print("\nError: Could not evaluate both models. Comparison aborted.")
        return

    # Display comparison table
    comparison_table = df.to_string(index=False)
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON ON JETSON")
    print("="*70)
    print(comparison_table)
    
    # Calculate improvement
    pt_fps = results[0]['FPS']
    engine_fps = results[1]['FPS']
    
    summary_lines = []
    if pt_fps > 0:
        fps_boost = engine_fps / pt_fps
        boost_msg = f"🚀 TensorRT Speedup: {fps_boost:.2f}x faster than PyTorch"
        print("-" * 70)
        print(boost_msg)
        summary_lines.append(boost_msg)
    
    # Calculate mAP drop (if any)
    map_diff = results[1]['mAP50'] - results[0]['mAP50']
    map_msg = f"📊 mAP50 Change: {map_diff:+.4f}"
    print(map_msg)
    summary_lines.append(map_msg)
    print("="*70)

    # Save to log
    log_dir = Path("runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "comparison_report.txt"
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("YOLOv8 PERFORMANCE COMPARISON REPORT\n")
        f.write("="*70 + "\n")
        f.write(comparison_table + "\n")
        f.write("-" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("="*70 + "\n")
    
    print(f"\nReport saved to: {log_path}")

if __name__ == "__main__":
    main()
