import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class StepLinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            iters_per_epoch: int,
            warm_up_epochs: int,
            max_epochs: int,
            last_epoch: int = -1,
            min_lr: float = 0.,
            warm_up_start_lr: float = 0.,
    ) -> None:
        self.iters_per_epoch = iters_per_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_iters = iters_per_epoch * warm_up_epochs
        self.max_epochs = max_epochs
        self.warm_up_start_lr = warm_up_start_lr
        self.min_lr = min_lr

        super(StepLinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Linear warm-up
        if self._step_count < self.warm_up_iters + 1:
            ratio = self._step_count / (self.warm_up_epochs * self.iters_per_epoch + 1)
            return [self.warm_up_start_lr + ratio * (base_lr - self.warm_up_start_lr) for base_lr in self.base_lrs]

        # Cosine annealing
        epoch = (self._step_count - 1) / self.iters_per_epoch
        _ratio = (epoch - self.warm_up_epochs) / (self.max_epochs - self.warm_up_epochs)
        ratio = 0.5 * (1. + math.cos(math.pi * _ratio))
        lrs = [self.base_lrs[i] if ('fix_lr' in param_group and param_group['fix_lr']) 
               else self.min_lr + (self.base_lrs[i] - self.min_lr) * ratio
               for i, param_group in enumerate(self.optimizer.param_groups)]
        return lrs

    def _get_closed_form_lr(self) -> List[float]:
        return self.get_lr()
