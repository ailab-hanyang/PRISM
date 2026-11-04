from typing import Tuple
import torch.nn as nn
from torchvision import models

_MOBILENET_BUILDERS = {
    "mobilenet_v2":       models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
}


def build_mobilenet(backbone_type: str, num_input_channels: int) -> Tuple[nn.Module, int]:
    """Returns (feature_extractor, out_channels). Output: (B, C, H, W) spatial feature map."""
    model = _MOBILENET_BUILDERS[backbone_type](weights=None)

    if num_input_channels != 3:
        orig_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            num_input_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

    if backbone_type == "mobilenet_v2":
        out_channels = model.classifier[1].in_features
    else:  # v3 small / large: classifier is [Linear, Hardswish, Dropout, Linear]
        out_channels = model.classifier[0].in_features

    return model.features, out_channels
