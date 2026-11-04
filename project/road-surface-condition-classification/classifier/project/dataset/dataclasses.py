from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2


@dataclass
class Calibration:
    lidar2ego: np.ndarray # shape: (4, 4)
    rgb2ego: np.ndarray   # shape: (4, 4)
    polar2ego: np.ndarray # shape: (4, 4)
    
    rgb_intrinsic: np.ndarray  # shape: (3, 3)
    rgb_distortion: np.ndarray  # shape: (1, 4)
    polar_intrinsic: np.ndarray  # shape: (3, 3)
    polar_distortion: np.ndarray  # shape: (1, 4)
    
    
    def get_rgb_intrinsic_after_crop(self, x_offset: int, y_offset: int) -> np.ndarray:

        intrinsic = deepcopy(self.rgb_intrinsic)
        intrinsic[0, 2] -= x_offset
        intrinsic[1, 2] -= y_offset

        return intrinsic

    def get_polar_intrinsic_after_crop(self, x_offset: int, y_offset: int) -> np.ndarray:

        intrinsic = deepcopy(self.polar_intrinsic)
        intrinsic[0, 2] -= x_offset
        intrinsic[1, 2] -= y_offset

        return intrinsic
    
    def get_rgb_intrinsic_after_resizing(self, intrinsic: np.ndarray, ori_size: Tuple[int, int], target_size: Tuple[int, int]) -> np.ndarray:
        
        ori_w, ori_h = ori_size
        target_w, target_h = target_size
        
        scale_x = target_w / ori_w
        scale_y = target_h / ori_h
        
        resized_intrinsic = deepcopy(intrinsic)
        resized_intrinsic[0, 0] *= scale_x
        resized_intrinsic[1, 1] *= scale_y
        resized_intrinsic[0, 2] *= scale_x
        resized_intrinsic[1, 2] *= scale_y
        
        return resized_intrinsic
    


@dataclass
class ImageData:
    data_path: Path
    data: np.ndarray  # shape: (H, W, 3)
    
    def open_image(self) -> np.ndarray:
        img = cv2.imread(str(self.data_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

@dataclass
class PolarImageData:
    data_path_0d: Path
    data_path_45d: Path
    data_path_90d: Path
    data_path_135d: Path

    data_0d: np.ndarray
    data_45d: np.ndarray
    data_90d: np.ndarray
    data_135d: np.ndarray

    rgb_image: Optional[np.ndarray] = None
    dolp: Optional[np.ndarray] = None
    aolp: Optional[np.ndarray] = None
    stokes: Optional[np.ndarray] = None

    def open_polar_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_0d = cv2.imread(str(self.data_path_0d), cv2.IMREAD_GRAYSCALE)
        img_45d = cv2.imread(str(self.data_path_45d), cv2.IMREAD_GRAYSCALE)
        img_90d = cv2.imread(str(self.data_path_90d), cv2.IMREAD_GRAYSCALE)
        img_135d = cv2.imread(str(self.data_path_135d), cv2.IMREAD_GRAYSCALE)

        return img_0d, img_45d, img_90d, img_135d
    
    def compute_polar_info(self, img_0d: np.ndarray, img_45d: np.ndarray, img_90d: np.ndarray, img_135d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s0, s1, s2 = self._compute_stokes(img_0d, img_45d, img_90d, img_135d)
        dolp = np.sqrt(s1**2 + s2**2) / (s0 + 1e-7)
        aolp = 0.5 * np.arctan2(s2, s1)
        return s0, s1, s2, dolp, aolp

    def _compute_stokes(self, img_0d: np.ndarray, img_45d: np.ndarray, img_90d: np.ndarray, img_135d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s0 = 0.5 * (img_0d + img_90d + img_45d + img_135d)
        s1 = img_0d - img_90d
        s2 = img_45d - img_135d
        return s0, s1, s2

@dataclass
class RoadSurfaceCondition:
    weather: str
    road_type: str
    surface_material: str
    surface_state: str


class RoadConditionClassMapper:
    SURFACE_MATERIAL = ["asphalt", "concrete", "belgian_block", "gravel"]
    SURFACE_STATE    = ["dry", "damp", "wet", "slush", "snow_covered"]

    N_SURFACE_MATERIAL = len(SURFACE_MATERIAL)
    N_SURFACE_STATE    = len(SURFACE_STATE)

    NUM_CLASSES = N_SURFACE_MATERIAL * N_SURFACE_STATE

    @classmethod
    def encode(cls, condition: "RoadSurfaceCondition") -> int:
        m = cls.SURFACE_MATERIAL.index(condition.surface_material)
        s = cls.SURFACE_STATE.index(condition.surface_state)
        return cls.encode_joint(m, s)

    @classmethod
    def encode_joint(cls, material_label: int, condition_label: int) -> int:
        return int(material_label) * cls.N_SURFACE_STATE + int(condition_label)

    @classmethod
    def decode(cls, class_id: int) -> "RoadSurfaceCondition":
        m, s = divmod(class_id, cls.N_SURFACE_STATE)
        return RoadSurfaceCondition(
            weather="",
            road_type="",
            surface_material=cls.SURFACE_MATERIAL[m],
            surface_state=cls.SURFACE_STATE[s],
        )


@dataclass
class ModelInput:
    calibration: Calibration
    image: ImageData
    polar_image: PolarImageData
    dataset_folder: str = ""

@dataclass
class Anno:
    material_label: int
    condition_label: int
