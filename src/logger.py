import torch
import torch.distributed as dist
from typing import Dict


# Epoch-wise logger for distributed setting (seems enough for me)
class Logger:
    def __init__(self, wandb_logger=None):
        # wandb logger
        self.wandb_logger = wandb_logger

        # distributed?
        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1

        # memory
        self.bucket = {}
        self.count = {}
                
    def update(self, metrics: Dict[str, torch.Tensor]):
        count = 1
        if 'count' in metrics.keys():
            count = metrics['count']
            del metrics['count']
        
        for key in metrics.keys():
            if key in self.bucket.keys():
                self.bucket[key].append(metrics[key].item())
                self.count[key].append(count)
            else:
                self.bucket[key] = [metrics[key].item(),]
                self.count[key] = [count,]
    
    def log(self, metrics: Dict = None):
        metrics = dict() if metrics is None else metrics
        for key in self.bucket.keys():
            if self.bucket[key]: # if not empty
                count = torch.cuda.FloatTensor(self.count[key])
                value = torch.cuda.FloatTensor(self.bucket[key])
                
                # sum-value and count
                value = (value * count).sum()
                count = count.sum()
                if self.distributed:
                    dist.reduce(value, dst=0)
                    dist.reduce(count, dst=0)
                    value = value / count
                metrics[key] = value
                
                # reset
                self.bucket[key] = []
                self.count[key] = []
        
        if self.wandb_logger:
            self.wandb_logger.log(metrics, commit=True)


