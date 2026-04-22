import yaml
import subprocess
import os
import sys
from pathlib import Path

def run_script(script_path, desc):
    print(f"\n>>>> Starting: {desc} ({script_path})")
    try:
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f">>>> Completed: {desc}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        return False

def main():
    config_path = "configs/pipeline.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Step 1: Sparsity Training
    if cfg.get("sparsity", {}).get("enabled", False):
        if not run_script("scripts/01_sparsity_train.py", "Sparsity Training"):
            return
    else:
        print("Skipping Sparsity Training (disabled in config)")

    # Step 2: Pruning
    if cfg.get("prune", {}).get("enabled", False):
        if not run_script("scripts/02_prune.py", "Model Pruning"):
            return
    else:
        print("Skipping Model Pruning (disabled in config)")

    # Step 3: Finetuning
    if cfg.get("finetune", {}).get("enabled", False):
        if not run_script("scripts/03_finetune.py", "Finetuning"):
            return
    else:
        print("Skipping Finetuning (disabled in config)")

    # Step 4: QAT
    if cfg.get("qat", {}).get("enabled", False):
        if not run_script("scripts/04_qat.py", "Quantization Aware Training"):
            return
    else:
        print("Skipping QAT (disabled in config)")

    # Step 5: Export
    if cfg.get("export", {}).get("enabled", False):
        if not run_script("scripts/05_export.py", "TensorRT Export"):
            return
    else:
        print("Skipping Export (disabled in config)")

    print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    main()
