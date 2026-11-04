from typing import Dict
import torch


def _get_polar_rgb_input(features: Dict) -> torch.Tensor:
    p = features.get("polar", {})
    rgb = p.get("rgb_image")
    if rgb is None:
        raise KeyError("Polar RGB input is missing. Expected features['polar']['rgb_image'].")
    return rgb


def extract_input(features: Dict, input_mode: str) -> torch.Tensor:
    """Assemble the model input tensor from the features dict.

    input_mode    channels
    ----------    --------
    rgb           RGB-camera RGB(3)               =  3 ch
    polar_rgb     polar-camera RGB(3)             =  3 ch
    mono_polar    DoLP(1) + AoLP(2)               =  3 ch
    tri_polar     DoLP(3) + AoLP(6)               =  9 ch
    rgb_mono      polar RGB(3) + DoLP(1) + AoLP(2)=  6 ch
    rgb_tri       polar RGB(3) + DoLP(3) + AoLP(6)= 12 ch
    """
    p = features.get("polar", {})

    if input_mode == "rgb":
        return features["image"]
    elif input_mode == "polar_rgb":
        return _get_polar_rgb_input(features)
    elif input_mode in ("mono_polar", "tri_polar"):
        return torch.cat([p["dolp"], p["aolp"]], dim=1)
    elif input_mode == "rgb_mono":
        polar_rgb = _get_polar_rgb_input(features)
        return torch.cat([polar_rgb, p["dolp"], p["aolp"]], dim=1)
    elif input_mode == "rgb_tri":
        polar_rgb = _get_polar_rgb_input(features)
        return torch.cat([polar_rgb, p["dolp"], p["aolp"]], dim=1)
    else:
        raise ValueError(
            f"Unknown input_mode '{input_mode}'. "
            f"Choose from: rgb, polar_rgb, mono_polar, tri_polar, rgb_mono, rgb_tri"
        )
