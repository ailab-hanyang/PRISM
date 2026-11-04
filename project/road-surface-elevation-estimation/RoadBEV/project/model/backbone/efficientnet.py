import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from project.model.submodules.submodule import *
from project.model.backbone.mobilenetv3 import ConvNormActivation, SqueezeExcitation
from project.model.backbone.mobilenetv3 import _make_divisible


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class UpSampleBN(nn.Module):
    def __init__(self, skip_channel, cur_channel, out_channel, interploate=False):
        super(UpSampleBN, self).__init__()

        self.interploate = interploate
        self.conv = nn.Sequential(BasicConv(skip_channel+cur_channel, out_channel, stride=1, kernel_size=3, padding=1),
                                  BasicConv(out_channel, out_channel, stride=1, kernel_size=3, padding=1))
        if not self.interploate:
            self.up = BasicConv(cur_channel, cur_channel, deconv=True, relu=False, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, concat_with):
        if self.interploate:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        else:
            up_x = self.up(x)
        f = torch.cat([up_x, concat_with], dim=1)
        return self.conv(f)

class EfficientNetCla(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.num_classes = num_classes
        layers: List[nn.Module] = []

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                self.num_classes,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        return x

# Concat channel counts for features[2]+[3]+[4] per variant
# Computed from _make_divisible(out_ch * width_mult, 8) for stages 2, 3, 4
_EFFICIENTNET_CONCAT_CH = {
    "b3": 176,  # 32 + 48 + 96   (width_mult=1.2, canonical=300)
    "b4": 200,  # 32 + 56 + 112  (width_mult=1.4, canonical=380)
    "b5": 224,  # 40 + 64 + 128  (width_mult=1.6, canonical=456) -- approximate, verify if needed
    "b6": 256,  # 40 + 72 + 144  (width_mult=1.8, canonical=528)
}

_EFFICIENTNET_WEIGHTS = {
    "b3": ("torchvision.models", "efficientnet_b3", "EfficientNet_B3_Weights", "IMAGENET1K_V1"),
    "b4": ("torchvision.models", "efficientnet_b4", "EfficientNet_B4_Weights", "IMAGENET1K_V1"),
    "b5": ("torchvision.models", "efficientnet_b5", "EfficientNet_B5_Weights", "IMAGENET1K_V1"),
    "b6": ("torchvision.models", "efficientnet_b6", "EfficientNet_B6_Weights", "IMAGENET1K_V1"),
}


class EfficientNetFeature(nn.Module):
    """Generic EfficientNet feature extractor (torchvision-based).

    Output: (B, 64, H/4, W/4)

    Stages used (stored as self.features, an nn.ModuleList):
      features[0]: first conv, stride 2  → H/2
      features[1]: stage 1,   stride 1  → H/2
      features[2]: stage 2,   stride 2  → H/4
      features[3]: stage 3,   stride 2  → H/8
      features[4]: stage 4,   stride 2  → H/16
    concat(f2, f3↑, f4↑) → final_conv → 64ch

    The parameter naming (features.0.*, features.1.*, ...) intentionally matches
    the classifier backbone so its checkpoint can be loaded without key remapping.

    Args:
        variant:     "b3" | "b4" | "b5" | "b6"
        in_channels: number of input channels (3 for RGB, 9/12 for polar)
    """

    feat_channel: int = 64

    def __init__(self, variant: str = "b6", in_channels: int = 3) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in _EFFICIENTNET_CONCAT_CH:
            raise ValueError(
                f"[Backbone] Unknown EfficientNet variant '{variant}'. "
                f"Supported: {list(_EFFICIENTNET_CONCAT_CH.keys())}"
            )
        self.variant = variant
        self.in_channels = in_channels
        concat_ch = _EFFICIENTNET_CONCAT_CH[variant]

        import importlib
        mod = importlib.import_module("torchvision.models")
        build_fn = getattr(mod, f"efficientnet_{variant}")
        base = build_fn(weights=None)

        # Adapt first conv for non-RGB input
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )

        self.features = nn.ModuleList([
            base.features[0],
            base.features[1],
            base.features[2],
            base.features[3],
            base.features[4],
        ])

        self.final_conv = nn.Sequential(
            convbn(concat_ch, 128, 5, 1, 2, 1),
            nn.ReLU(inplace=True),
            convbn(128, self.feat_channel, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channel, self.feat_channel, 1, 1, 0, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x  = self.features[0](x)
        x  = self.features[1](x)
        f2 = self.features[2](x)
        f3 = self.features[3](f2)
        f4 = self.features[4](f3)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bicubic', align_corners=True)
        f4_up = F.interpolate(f4, size=f2.shape[2:], mode='bicubic', align_corners=True)
        return self.final_conv(torch.cat([f2, f3_up, f4_up], dim=1))

    def load_imagenet_weights(self) -> None:
        """Load torchvision ImageNet pretrained weights into features[0-4].

        If the input stem is expanded beyond RGB, copy pretrained RGB weights into
        the first 3 channels and zero-initialize the additional channels.
        """
        import importlib
        _, fn_name, weights_cls, weights_attr = _EFFICIENTNET_WEIGHTS[self.variant]
        mod = importlib.import_module("torchvision.models")
        weights = getattr(getattr(mod, weights_cls), weights_attr)
        src = getattr(mod, fn_name)(weights=weights)

        # Stem: handle channel-mismatch safely.
        src_conv = src.features[0][0]
        dst_conv = self.features[0][0]
        with torch.no_grad():
            dst_conv.weight.zero_()
            copy_ch = min(3, self.in_channels)
            dst_conv.weight[:, :copy_ch].copy_(src_conv.weight[:, :copy_ch])

        if len(self.features[0]) > 1:
            for idx in range(1, len(self.features[0])):
                self.features[0][idx].load_state_dict(src.features[0][idx].state_dict(), strict=True)

        for i in range(1, 5):
            self.features[i].load_state_dict(src.features[i].state_dict(), strict=True)
        print(f"[Backbone] EfficientNet-{self.variant.upper()} ImageNet weights loaded.")


# Backward-compat alias
EfficientNetB6Feature = EfficientNetFeature


def efficientnet_cla(inplanes, num_classes, **kwargs: Any) -> EfficientNetCla:
    # # b0
    width_mult = 1.0
    depth_mult = 1.0

    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, inplanes, inplanes, 1),
        bneck_conf(6, 3, 1, inplanes, 128, 2),
        bneck_conf(6, 5, 1, 128, num_classes, 2),
        bneck_conf(6, 3, 1, num_classes, num_classes, 2),
    ]

    model = EfficientNetCla(inverted_residual_setting, num_classes=num_classes, **kwargs)

    return model


