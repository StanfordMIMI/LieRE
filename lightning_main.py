from lightning.pytorch.cli import LightningCLI
from lightning_modules.lightning_liere_image_classification import (
    LiereImageClassification,
    FLOPsAnalysisCallback,
)
from lightning_modules.lightning_data_image import Cifar100, Imagenet

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.cosine_annealing_scheduler = CosineAnnealingLR(
            optimizer,
            self.max_epochs - self.warmup_epochs,
            eta_min=0,
            last_epoch=last_epoch - warmup_epochs if last_epoch != -1 else -1,
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        return self.cosine_annealing_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, lr):
                param_group["lr"] = lr
        else:
            self.cosine_annealing_scheduler.step(epoch - self.warmup_epochs)


cli = LightningCLI(save_config_kwargs={"overwrite": True}, save_config_callback=None)
