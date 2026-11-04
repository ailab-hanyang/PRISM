from typing import Dict, List, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, OneCycleLR, LinearLR, SequentialLR,
)


def build_optimizer(param_groups: List[Dict], config) -> Optimizer:
    if config.optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    return torch.optim.Adam(param_groups)


def build_scheduler(optimizer: Optimizer, config) -> Union[Dict, None]:
    """Returns a Lightning lr_scheduler dict, or None if no scheduler."""
    scheduler_type = getattr(config, "scheduler_type", "cosine")

    if scheduler_type == "warmup_cosine":
        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                          total_iters=config.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=config.cosine_T_max - config.warmup_epochs,
                                   eta_min=config.cosine_eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[config.warmup_epochs])
        return {"scheduler": scheduler, "monitor": "val/loss"}

    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.cosine_T_max,
                                      eta_min=config.cosine_eta_min)
        return {"scheduler": scheduler, "monitor": "val/loss"}

    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma)
        return {"scheduler": scheduler}

    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=config.onecycle_max_lr,
                               epochs=config.epochs,
                               steps_per_epoch=config.steps_per_epoch,
                               pct_start=config.onecycle_pct_start)
        return {"scheduler": scheduler, "interval": "step"}

    return None
