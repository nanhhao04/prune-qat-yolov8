import yaml
import subprocess
import os
import sys
from pathlib import Path
import shutil

def run_step(script_path, desc):
    print(f"\n{'='*20}")
    print(f"STEP: {desc}")
    print(f"{'='*20}")
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Error in {desc}: {e}")
        return False

def main():
    # 1. Load config
    config_path = "configs/pipeline.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(" Starting PC Optimization Pipeline...")
    
    # 2. Run Sparsity Training
    if not run_step("scripts/01_sparsity_train.py", "Sparsity Training (Step 1/3)"):
        return

    # 3. Run Pruning
    if not run_step("scripts/02_prune.py", "Model Pruning (Step 2/3)"):
        return

    # 4. Run Finetuning
    if not run_step("scripts/03_finetune.py", "Finetuning (Step 3/3)"):
        return

    # 5. Collect final weights
    print("\n Optimization Pipeline Completed!")
    
    # Path to the final finetuned weights (usually runs/train-finetune/weights/best.pt)
    # Note: Check your 03_finetune.py 'name' parameter if different
    final_weights_src = Path("runs/train-finetune/weights/best.pt")
    dist_dir = Path("weights/for_jetson")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    if final_weights_src.exists():
        dest_path = dist_dir / "optimized_yolo_jetson.pt"
        shutil.copy(final_weights_src, dest_path)
        print(f"\n FINAL MODEL READY: {dest_path}")
        print("Copy this file to your Jetson Nano to export to TensorRT.")
    else:
        print(f"\n Warning: Could not find final weights at {final_weights_src}")

if __name__ == "__main__":
    main()
