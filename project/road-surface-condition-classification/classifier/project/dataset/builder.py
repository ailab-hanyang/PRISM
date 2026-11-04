from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import copy
import pickle

import numpy as np
import torch
import cv2
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from project.template.builder import AbstractFeatureBuilder, AbstractTargetBuilder
from project.dataset.dataclasses import *
from project.config.model_config import ClassifierConfig

class ModelFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

        input_h = getattr(config, "input_h", 540)
        input_w = getattr(config, "input_w", 960)
        self.transform_image = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((input_h, input_w), interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        self.transform_polar_image = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.transform_polar_mono_image = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5], std=[0.5])  
            ])

        self._rgb_undistort_map: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._polar_undistort_map: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def preprocess_image(self, image: ImageData, calibration: Calibration) -> Tuple[ImageData, Calibration]:
        if self._rgb_undistort_map is not None:
            undistorted_image = cv2.remap(image.data, self._rgb_undistort_map[0], self._rgb_undistort_map[1], cv2.INTER_LINEAR)
        else:
            undistorted_image = cv2.undistort(image.data, calibration.rgb_intrinsic, calibration.rgb_distortion)
        # Crop to 1920x1080 from bottom-left corner
        # Original size: 2448 x 2048
        # Target size: 1920 x 1080
        # Bottom-left crop: x=0, y=2048-1080=968
        H, W = undistorted_image.shape[:2]
        crop_h = getattr(self.config, "rgb_crop_h", 1080)
        crop_w = getattr(self.config, "rgb_crop_w", 1920)
        input_h = getattr(self.config, "input_h", 540)
        input_w = getattr(self.config, "input_w", 960)
        y_start = H - crop_h
        x_start = 0
        cropped_image = undistorted_image[y_start:y_start+crop_h, x_start:x_start+crop_w]
        # Convert BGR to RGB for ToTensor (ToTensor accepts numpy array)
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        transformed_image = self.transform_image(cropped_image_rgb)

        image_data = ImageData(
            data_path=image.data_path,
            data=transformed_image
        )

        # TEMP: calibration intrinsic update skipped (result unused by model)
        return (image_data, calibration)
    
    def preprocess_polar_image(self, polar_image: PolarImageData, calibration: Calibration) -> Tuple[PolarImageData, Calibration]:
        """
        Preprocess polarimetric camera images.

        Two modes controlled by use_trichromatic flag:
        1. TRICHROMATIC (True): Demosaic → per-channel Stokes → RGB DoLP/AoLP → 12ch input
        2. MONOCHROMATIC (False): Raw Bayer → averaged Stokes → single DoLP/AoLP → 5ch or 6ch input
        """
        # === Step 1: Undistort and Crop ===
        if self._polar_undistort_map is not None:
            m1, m2 = self._polar_undistort_map
            undistorted_0d   = cv2.remap(polar_image.data_0d,   m1, m2, cv2.INTER_LINEAR)
            undistorted_45d  = cv2.remap(polar_image.data_45d,  m1, m2, cv2.INTER_LINEAR)
            undistorted_90d  = cv2.remap(polar_image.data_90d,  m1, m2, cv2.INTER_LINEAR)
            undistorted_135d = cv2.remap(polar_image.data_135d, m1, m2, cv2.INTER_LINEAR)
        else:
            undistorted_0d   = cv2.undistort(polar_image.data_0d,   calibration.polar_intrinsic, calibration.polar_distortion)
            undistorted_45d  = cv2.undistort(polar_image.data_45d,  calibration.polar_intrinsic, calibration.polar_distortion)
            undistorted_90d  = cv2.undistort(polar_image.data_90d,  calibration.polar_intrinsic, calibration.polar_distortion)
            undistorted_135d = cv2.undistort(polar_image.data_135d, calibration.polar_intrinsic, calibration.polar_distortion)

        H, W = undistorted_0d.shape[:2]
        crop_h = getattr(self.config, "input_h", 540)
        crop_w = getattr(self.config, "input_w", 960)
        y_start, x_start = H - crop_h, 0

        crop_0d = undistorted_0d[y_start:y_start+crop_h, x_start:x_start+crop_w]
        crop_45d = undistorted_45d[y_start:y_start+crop_h, x_start:x_start+crop_w]
        crop_90d = undistorted_90d[y_start:y_start+crop_h, x_start:x_start+crop_w]
        crop_135d = undistorted_135d[y_start:y_start+crop_h, x_start:x_start+crop_w]

        # === Step 2: Mode Selection ===
        # Derive trichromatic flag from input_mode (tri/rgb_tri → True, else → False)
        input_mode = getattr(self.config, "input_mode", None)
        if input_mode is not None:
            use_trichromatic = input_mode in ("tri_polar", "rgb_tri")
        else:
            use_trichromatic = bool(getattr(self.config, "use_trichromatic", True))
        if use_trichromatic:
            # ============================================================
            # TRICHROMATIC MODE: Demosaic → Per-channel Stokes
            # ============================================================
            # Demosaic each polarization channel to RGB (H, W, 3)
            demosaic_0d = cv2.cvtColor(crop_0d, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 255.0
            demosaic_45d = cv2.cvtColor(crop_45d, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 255.0
            demosaic_90d = cv2.cvtColor(crop_90d, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 255.0
            demosaic_135d = cv2.cvtColor(crop_135d, cv2.COLOR_BayerRG2RGB).astype(np.float32) / 255.0

            # RGB reconstruction: average of all 4 demosaiced channels
            avg = 0.25 * (demosaic_0d + demosaic_45d + demosaic_90d + demosaic_135d)
            polar_rgb_image = np.clip(avg * 255, 0, 255).astype(np.uint8)

            # Per-channel Stokes: (H, W, 3) for R, G, B separately
            s0_per_ch = 0.5 * (demosaic_0d + demosaic_90d + demosaic_45d + demosaic_135d)  # (H, W, 3)
            s1_per_ch = demosaic_0d - demosaic_90d
            s2_per_ch = demosaic_45d - demosaic_135d

            # Per-channel DoLP/AoLP (H, W, 3)
            dolp = np.sqrt(s1_per_ch**2 + s2_per_ch**2) / (s0_per_ch + 1e-7)
            aolp = 0.5 * np.arctan2(s2_per_ch, s1_per_ch)  # Range: [-π/2, π/2]

            # Transpose to (C, H, W) format
            dolp_transposed = dolp.transpose(2, 0, 1)  # (3, H, W)
            aolp_transposed = aolp.transpose(2, 0, 1)  # (3, H, W)

            # AoLP sincos encoding: [R_sin, R_cos, G_sin, G_cos, B_sin, B_cos]
            aolp_sin = np.sin(2.0 * aolp_transposed).astype(np.float32)
            aolp_cos = np.cos(2.0 * aolp_transposed).astype(np.float32)

            aolp_channels = np.empty((6, crop_h, crop_w), dtype=np.float32)
            aolp_channels[0::2] = aolp_sin  # R_sin, G_sin, B_sin at 0, 2, 4
            aolp_channels[1::2] = aolp_cos  # R_cos, G_cos, B_cos at 1, 3, 5

            dolp_tensor = torch.from_numpy(dolp_transposed)  # (3, H, W)
            aolp_tensor = torch.from_numpy(aolp_channels)    # (6, H, W)

            # Mono images for 0d_45d_90d_135d mode (grayscale)
            mono_0d = crop_0d.astype(np.float32) / 255.0
            mono_45d = crop_45d.astype(np.float32) / 255.0
            mono_90d = crop_90d.astype(np.float32) / 255.0
            mono_135d = crop_135d.astype(np.float32) / 255.0

            # Stokes vector placeholder (averaged)
            s0 = s0_per_ch.mean(axis=(0, 1))
            s1 = s1_per_ch.mean(axis=(0, 1))
            s2 = s2_per_ch.mean(axis=(0, 1))

        else:
            # ============================================================
            # MONOCHROMATIC MODE: Raw Bayer → Averaged Stokes
            # ============================================================
            # Convert to float and normalize
            mono_0d = crop_0d.astype(np.float32) / 255.0
            mono_45d = crop_45d.astype(np.float32) / 255.0
            mono_90d = crop_90d.astype(np.float32) / 255.0
            mono_135d = crop_135d.astype(np.float32) / 255.0

            # Compute averaged Stokes parameters (H, W)
            s0, s1, s2, dolp, aolp = polar_image.compute_polar_info(
                mono_0d, mono_45d, mono_90d, mono_135d
            )

            # RGB reconstruction: average of all 4 raw channels, then demosaic
            # Cast to float32 before summing to avoid uint8 overflow
            polar_rgb_src = (crop_0d.astype(np.float32) + crop_45d.astype(np.float32)
                             + crop_90d.astype(np.float32) + crop_135d.astype(np.float32)) * 0.25
            polar_rgb_src = np.clip(polar_rgb_src, 0, 255).astype(np.uint8)
            polar_rgb_image = cv2.cvtColor(polar_rgb_src, cv2.COLOR_BayerRG2RGB)

            # AoLP encoding
            aolp_sin = np.sin(2.0 * aolp).astype(np.float32)
            aolp_cos = np.cos(2.0 * aolp).astype(np.float32)
            
            aolp_tensor = torch.from_numpy(np.stack([aolp_sin, aolp_cos], axis=0))  # (2, H, W)
            dolp_tensor = torch.from_numpy(dolp.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        # === Step 3: Build Output Dataclass ===
        polar_data = PolarImageData(
            data_path_0d=polar_image.data_path_0d,
            data_path_45d=polar_image.data_path_45d,
            data_path_90d=polar_image.data_path_90d,
            data_path_135d=polar_image.data_path_135d,

            data_0d=self.transform_polar_mono_image(mono_0d),
            data_45d=self.transform_polar_mono_image(mono_45d),
            data_90d=self.transform_polar_mono_image(mono_90d),
            data_135d=self.transform_polar_mono_image(mono_135d),

            rgb_image=self.transform_polar_image(polar_rgb_image),
            dolp=dolp_tensor,
            aolp=aolp_tensor,
            stokes=torch.from_numpy(np.stack([s0, s1, s2], axis=0).astype(np.float32))
        )

        # TEMP: calibration intrinsic update skipped (result unused by model)
        return (polar_data, calibration)


    def build_undistort_cache(self, calibration: Any) -> None:
        """Precompute undistort maps for RGB and polar cameras.

        intrinsic/distortion are shared across all datasets, so one calibration suffices.
        Replaces per-sample cv2.undistort() with a single initUndistortRectifyMap + remap.
        """
        rgb_h = getattr(self.config, "rgb_sensor_h", 2048)
        rgb_w = getattr(self.config, "rgb_sensor_w", 2448)
        polar_h = getattr(self.config, "polar_sensor_h", 1024)
        polar_w = getattr(self.config, "polar_sensor_w", 1224)

        map1, map2 = cv2.initUndistortRectifyMap(
            calibration.rgb_intrinsic, calibration.rgb_distortion,
            None, calibration.rgb_intrinsic, (rgb_w, rgb_h), cv2.CV_32FC1
        )
        self._rgb_undistort_map = (map1, map2)

        map1, map2 = cv2.initUndistortRectifyMap(
            calibration.polar_intrinsic, calibration.polar_distortion,
            None, calibration.polar_intrinsic, (polar_w, polar_h), cv2.CV_32FC1
        )
        self._polar_undistort_map = (map1, map2)


    def compute_features(self, model_input: ModelInput) -> Dict[str, torch.Tensor]:
        image_data, _ = self.preprocess_image(image=model_input.image, calibration=model_input.calibration)
        polar_image_data, _ = self.preprocess_polar_image(
            polar_image=model_input.polar_image,
            calibration=model_input.calibration,
        )

        features = {
            "image": image_data.data.float(),
            "polar": {
                "mono_image_0d": polar_image_data.data_0d.float(),
                "mono_image_45d": polar_image_data.data_45d.float(),
                "mono_image_90d": polar_image_data.data_90d.float(),
                "mono_image_135d": polar_image_data.data_135d.float(),
                "rgb_image": polar_image_data.rgb_image.float(),
                "dolp": polar_image_data.dolp.float(),
                "aolp": polar_image_data.aolp.float(),
                "stokes": polar_image_data.stokes.float()
            }
        }
        
        return features


class ModelTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

    def compute_targets(self, anno: Anno) -> Dict[str, torch.Tensor]:
        joint_id = RoadConditionClassMapper.encode_joint(anno.material_label, anno.condition_label)
        return {
            "material": torch.tensor(anno.material_label, dtype=torch.long),
            "condition": torch.tensor(anno.condition_label, dtype=torch.long),
            "joint": torch.tensor(joint_id, dtype=torch.long),
        }
