import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

from ultralytics.qat.nvidia_tensorrt.qat_pruned_trainer import QuantizationPrunedTrainer
from ultralytics.utils import DEFAULT_CFG_DICT
import yaml
import argparse

import os

def main():
    parser = argparse.ArgumentParser(description="QAT Training")
    parser.add_argument("--pipeline_cfg", type=str, default="configs/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--data_cfg", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="runs/train-finetune/weights/best.pt", help="Path to finetuned weights")
    args = parser.parse_args()

    with open(args.pipeline_cfg, "r") as f:
        pipe_cfg = yaml.safe_load(f)["qat"]
    
    # Generate a run name based on data config
    data_name = os.path.splitext(os.path.basename(args.data_cfg))[0]
    suffix = f"-{data_name}" if data_name != "data" else ""
    project_name = os.path.abspath("weights/04-qat")
    run_name = f"qat{suffix}"

    overrides = {
        'model': 'yolov8n.yaml',
        'data': args.data_cfg,
        'epochs': pipe_cfg['epochs'],
        'imgsz': pipe_cfg['imgsz'],
        'batch': pipe_cfg['batch'],
        'lr0': pipe_cfg['lr0'],
        'project': project_name,
        'name': run_name,
        'pruned_checkpoint': os.path.abspath('weights/02-pruning/pruned_model.pt'),
        'exist_ok': True
    }

    trainer = QuantizationPrunedTrainer(cfg=DEFAULT_CFG_DICT.copy(), overrides=overrides)
    
    # Load weight from finetune
    trainer.model = trainer.get_model(weights=args.weights)

    trainer.train()

if __name__ == "__main__":
    main()
