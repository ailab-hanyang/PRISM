from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

_EFFICIENTNET_BUILDERS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
}


def build_efficientnet(backbone_type: str, num_input_channels: int) -> Tuple[nn.Module, int]:
    """Returns (feature_extractor, out_channels). Output: (B, C, H, W) spatial feature map."""
    model = _EFFICIENTNET_BUILDERS[backbone_type](weights=None)

    if num_input_channels != 3:
        orig_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_input_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

    out_channels = model.classifier[1].in_features
    return model.features, out_channels
