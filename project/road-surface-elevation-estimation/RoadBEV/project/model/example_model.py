from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, StepLR

from project.template.model import AbstractLightningModule
from project.config.model_config import RoadBEVModelConfig

class BaseModel(nn.Module):
    def __init__(self, config: RoadBEVModelConfig):
        super().__init__()
        self.config = config
        # Define model layers here
        self.layer = nn.Linear(10, 10)  # Example layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ModelManager(AbstractLightningModule):
    def __init__(self, model: nn.Module, config: RoadBEVModelConfig):
        super().__init__(model)
        self.config = config
        self.model = model

    def _step(self, batch: Dict[str, torch.Tensor], logging_prefix: str) -> torch.Tensor:
        features, targets = batch
        predictions = self.model(features['trajectory'])  # Example forward pass
        loss = self.compute_loss(features, targets, predictions)
        self.log(f"{logging_prefix}_loss", loss)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        criterion = nn.MSELoss()  # Example loss function
        loss = criterion(predictions, targets['gt_trajectories'])
        return loss

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Returns the optimizers that are used by the pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }