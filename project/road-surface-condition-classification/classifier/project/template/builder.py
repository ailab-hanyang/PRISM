from abc import abstractmethod
from typing import Dict

from torch import Tensor

from project.dataset.dataclasses import ModelInput, Anno


class AbstractFeatureBuilder:
    """
    - Abstract class of feature builder for model training.
    - Do:
        1) Implement feature computation from ModelInput.
        2) Executes data augmentation or preprocessing if necessary.
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_features(self, model_input: ModelInput) -> Dict[str, Tensor]:
        """
        args:
            model_input: ModelInput object containing input data for feature computation.
        return: 
            A dictionary where each item has a unique identifier and maps to a single feature tensor.
        """
        pass


class AbstractTargetBuilder:
    def __init__(self):
        pass

    @abstractmethod
    def compute_targets(self, anno: Anno) -> Dict[str, Tensor]:
        """
        args:
            anno: Anno object containing annotation data for target computation.
        return:
            A dictionary where each item has a unique identifier and maps to a single target tensor.
        """
        pass
