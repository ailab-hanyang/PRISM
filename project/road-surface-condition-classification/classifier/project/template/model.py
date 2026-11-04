from typing import Dict, Union, Dict, List
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl

from project.template.builder import AbstractFeatureBuilder, AbstractTargetBuilder
from project.dataset.dataclasses import *


class AbstractLightningModule(ABC, pl.LightningModule):

    def __init__(self):
        super().__init__()
 
    
    def _step(self, batch: Dict[str, torch.Tensor], logging_prefix: str) -> torch.Tensor:
        pass
    
    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        pass


    @abstractmethod
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Model does not support training.")
    
    
    @abstractmethod
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        pass
    

    def get_training_callbacks(self) -> List[pl.Callback]:
        return []