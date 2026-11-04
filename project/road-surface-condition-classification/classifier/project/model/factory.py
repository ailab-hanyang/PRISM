from project.model.backbone.efficientnet import build_efficientnet
from project.model.backbone.mobilenet import build_mobilenet
from project.model.model import Classifier, ClassifierManager
from project.dataset.dataclasses import RoadConditionClassMapper

_BACKBONE_REGISTRY = {
    **{f"efficientnet_b{i}": build_efficientnet for i in range(8)},
    **{k: build_mobilenet for k in ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]},
}


_INPUT_MODE_CHANNELS = {
    "rgb":       3,   # single-stream: RGB-camera RGB(3)
    "polar_rgb": 3,   # single-stream: polar-camera RGB(3)
    "mono_polar": 3,  # single-stream: DoLP(1) + AoLP(2)
    "tri_polar":  9,  # single-stream: DoLP(3) + AoLP(6)
    # two-stream mid-fusion: RGB enc(3) + polar enc
    "rgb_mono":   6,  # polar enc: DoLP(1)+AoLP(2) = 3ch
    "rgb_tri":   12,  # polar enc: DoLP(3)+AoLP(6) = 9ch
}

_TWO_STREAM_POLAR_CHANNELS = {
    "rgb_mono": 3,
    "rgb_tri":  9,
}


def get_num_input_channels(config) -> int:
    mode = config.input_mode
    if mode not in _INPUT_MODE_CHANNELS:
        raise ValueError(f"Unknown input_mode '{mode}'. Choose from: {list(_INPUT_MODE_CHANNELS)}")
    return _INPUT_MODE_CHANNELS[mode]


def build_model(config) -> ClassifierManager:
    backbone_type = config.backbone_type
    if backbone_type not in _BACKBONE_REGISTRY:
        supported = sorted(_BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone_type '{backbone_type}'. Supported: {supported}")

    input_mode = config.input_mode
    build_fn = _BACKBONE_REGISTRY[backbone_type]
    head_dropout = getattr(config, "head_dropout", 0.2)

    if input_mode in _TWO_STREAM_POLAR_CHANNELS:
        polar_in_ch = _TWO_STREAM_POLAR_CHANNELS[input_mode]
        backbone, out_channels = build_fn(backbone_type, 3)
        polar_backbone, _ = build_fn(backbone_type, polar_in_ch)
    else:
        num_input_channels = get_num_input_channels(config)
        backbone, out_channels = build_fn(backbone_type, num_input_channels)
        polar_backbone = None

    model = Classifier(
        backbone=backbone,
        backbone_out_channels=out_channels,
        num_material_classes=RoadConditionClassMapper.N_SURFACE_MATERIAL,
        num_condition_classes=RoadConditionClassMapper.N_SURFACE_STATE,
        head_dropout=head_dropout,
        input_mode=input_mode,
        polar_backbone=polar_backbone,
    )
    return ClassifierManager(model=model, config=config)
