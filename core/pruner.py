import re
import os
import yaml
import torch
import torch.nn as nn
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.block_pruned import C2fPruned, SPPFPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import colorstr, LOGGER

class YOLOPruner:
    def __init__(self, weights, cfg_path, model_size='s'):
        self.weights = weights
        self.cfg_path = cfg_path
        self.model_size = model_size
        self.model = AutoBackend(weights, fuse=False)
        self.model.eval()
        
    def prune(self, ratio=0.3):
        LOGGER.info(f"Starting pruning with ratio {ratio}...")
        
        bn_dict = {}
        ignore_bn_list = []
        chunk_bn_list = []
        
        # Step 1: Identify BN layers and ignore lists (residual blocks)
        for name, module in self.model.model.named_modules():
            if isinstance(module, Bottleneck):
                if module.add:
                    ignore_bn_list.append(f"{name[:-4]}.cv1.bn")
                    ignore_bn_list.append(f"{name}.cv2.bn")
                else:
                    chunk_bn_list.append(f"{name[:-4]}.cv1.bn")
            if isinstance(module, nn.BatchNorm2d):
                bn_dict[name] = module
        
        # Step 2: Filter BN layers
        filtered_bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_list}
        
        # Step 3: Gather gamma weights to calculate threshold
        bn_weights = []
        for name, module in filtered_bn_dict.items():
            bn_weights.extend(module.weight.data.abs().clone().cpu().tolist())
        
        sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
        
        # Step 4: Calculate threshold
        highest_thre = min([m.weight.data.abs().max() for m in filtered_bn_dict.values()])
        percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
        
        if ratio > percent_limit:
            LOGGER.warning(f"Requested ratio {ratio} exceeds safe limit {percent_limit:.3f}. Clamping.")
            ratio = percent_limit
            
        thre = sorted_bn[int(len(sorted_bn) * ratio)]
        LOGGER.info(f"Pruning threshold: {thre:.4f}")

        # Step 5: Build pruned YAML (Custom architecture definition)
        pruned_yaml = self._generate_pruned_yaml()
        
        # Step 6: Generate masks and update original BN weights
        maskbndict = {}
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                origin_channels = module.weight.data.size()[0]
                mask = torch.ones(origin_channels)
                if name not in ignore_bn_list:
                    mask = module.weight.data.abs().gt(thre).float()
                    # Keep even channels for C2f structure split
                    if name in chunk_bn_list and mask.sum() % 2 == 1:
                        mask = self._adjust_to_even_channels(module, mask, thre)
                
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                maskbndict[name] = mask
        
        # Step 7: Reconstruct model
        pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
        
        # Step 8: Map weights
        self._map_weights(self.model.model, pruned_model, maskbndict)
        
        return pruned_model, maskbndict

    def _adjust_to_even_channels(self, module, mask, thre):
        flattened_sorted_weight = torch.sort(module.weight.data.abs().view(-1))[0]
        idx = torch.min(torch.nonzero(flattened_sorted_weight.gt(thre))).item()
        thre_ = flattened_sorted_weight[idx - 1] - 1e-6
        return module.weight.data.abs().gt(thre_).float()

    def _generate_pruned_yaml(self):
        with open(self.cfg_path, 'r') as f:
            base_yaml = yaml.safe_load(f)
        
        pruned = {
            "nc": self.model.model.nc,
            "scales": base_yaml["scales"],
            "scale": self.model_size,
            "backbone": [
                [-1, 1, 'Conv', [64, 3, 2]],
                [-1, 1, 'Conv', [128, 3, 2]],
                [-1, 3, 'C2fPruned', [128, True]],
                [-1, 1, 'Conv', [256, 3, 2]],
                [-1, 6, 'C2fPruned', [256, True]],
                [-1, 1, 'Conv', [512, 3, 2]],
                [-1, 6, 'C2fPruned', [512, True]],
                [-1, 1, 'Conv', [1024, 3, 2]],
                [-1, 3, 'C2fPruned', [1024, True]],
                [-1, 1, 'SPPFPruned', [1024, 5]],
            ],
            "head": [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 3, 'C2fPruned', [512]],
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 3, 'C2fPruned', [256]],
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],
                [-1, 3, 'C2fPruned', [512]],
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 9], 1, 'Concat', [1]],
                [-1, 3, 'C2fPruned', [1024]],
                [[15, 18, 21], 1, 'DetectPruned', [self.model.model.nc]],
            ]
        }
        return pruned

    def _map_weights(self, model_org, model_pruned, maskbndict):
        # Implementation of step 9 from original prune.py
        # Logic is complex, mapping weights from old to new modules based on masks
        current_to_prev = model_pruned.current_to_prev
        pattern_c2f = re.compile(r"model.\d+.m.0.cv1.bn")
        pattern_detect = re.compile(r"model.\d+.cv\d.\d.2")
        
        for (name_org, module_org), (name_pruned, module_pruned) in \
            zip(model_org.named_modules(), model_pruned.named_modules()):
            
            if 'dfl' in name_org: break
            
            if pattern_detect.fullmatch(name_org):
                prev_bn = current_to_prev[name_org]
                mask = maskbndict[prev_bn].to(torch.bool)
                module_pruned.weight.data = module_org.weight.data[:, mask, :, :]
                if module_org.bias is not None:
                    module_pruned.bias.data = module_org.bias.data
                continue

            if isinstance(module_org, nn.Conv2d):
                curr_bn = name_org[:-4] + 'bn'
                mask_out = maskbndict[curr_bn].to(torch.bool)
                prev_bn = current_to_prev.get(curr_bn, None)
                
                if isinstance(prev_bn, list):
                    mask_in = torch.cat([maskbndict[ni] for ni in prev_bn], dim=0).to(torch.bool)
                elif prev_bn is not None:
                    mask_in = maskbndict[prev_bn].to(torch.bool)
                    if pattern_c2f.fullmatch(curr_bn):
                        mask_in = mask_in.chunk(2, 0)[1]
                    if name_org == "model.9.cv2.conv":
                        mask_in = torch.cat([mask_in for _ in range(4)], dim=0)
                else:
                    mask_in = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)
                
                weight = module_org.weight.data[mask_out, :, :, :]
                module_pruned.weight.data = weight[:, mask_in, :, :]
                if module_org.bias is not None:
                    module_pruned.bias.data = module_org.bias.data[mask_out]

            if isinstance(module_org, nn.BatchNorm2d):
                mask = maskbndict[name_org].to(torch.bool)
                module_pruned.weight.data = module_org.weight.data[mask]
                module_pruned.bias.data = module_org.bias.data[mask]
                module_pruned.running_mean = module_org.running_mean[mask]
                module_pruned.running_var = module_org.running_var[mask]
