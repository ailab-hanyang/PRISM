from typing import List, Dict, Any

import torch
from torch import Tensor


def _collate(items: list) -> Any:
    if isinstance(items[0], dict):
        return {k: _collate([item[k] for item in items]) for k in items[0]}
    if isinstance(items[0], Tensor):
        return torch.stack(items, dim=0)
    return items


def model_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _collate(batch)
