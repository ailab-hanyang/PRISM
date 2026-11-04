from typing import List, Dict, Tuple, Any

import torch
from torch import Tensor


def model_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str,Tensor]:
    collated_batch = {}
    
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
        # Need to pad if tensors have different shapes!
    return collated_batch
    