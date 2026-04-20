from ultralytics.qat.nvidia_tensorrt.qat_pruned_trainer import QuantizationPrunedTrainer
from ultralytics.utils import DEFAULT_CFG_DICT
import yaml
import argparse

def main():
    with open("configs/pipeline.yaml", "r") as f:
        pipe_cfg = yaml.safe_load(f)["qat"]
    
    overrides = {
        'model': 'yolov8n.yaml',
        'data': 'configs/data.yaml',
        'epochs': pipe_cfg['epochs'],
        'imgsz': pipe_cfg['imgsz'],
        'batch': pipe_cfg['batch'],
        'lr0': pipe_cfg['lr0'],
        'project': 'runs/qat-pruned',
        'name': 'train',
        'pruned_checkpoint': 'weights/pruned_model.pt',
        'exist_ok': True
    }

    trainer = QuantizationPrunedTrainer(cfg=DEFAULT_CFG_DICT.copy(), overrides=overrides)
    
    # Load best weight from finetune
    weight_to_load = "runs/train-finetune/weights/best.pt"
    trainer.model = trainer.get_model(weights=weight_to_load)

    trainer.train()

if __name__ == "__main__":
    main()
