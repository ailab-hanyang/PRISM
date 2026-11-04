from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union, Any
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
    
    def generate_reference_calibration(self, reference_height: int, align_to_camera: str = "rgb"):
        """
        Generate reference frame calibration.

        Args:
            reference_height: Z-axis offset from LiDAR (typically 1.8m)
            align_to_camera: Which camera's X position to align reference frame to.
                           Options: "rgb" or "polar". Default "rgb" to match GT generation.

        The reference frame is defined as:
            - Origin X: Aligned to specified camera's X position in LiDAR frame
            - Origin Y: Same as LiDAR (0)
            - Origin Z: LiDAR Z + reference_height

        This ensures model's reference frame matches GT generation coordinate system.
        """
        ego2rgb = np.linalg.inv(self.rgb2ego)
        ego2polar = np.linalg.inv(self.polar2ego)

        # Build reference frame transform with dynamic camera offset
        lidar2reference = np.eye(4, dtype=np.float32)
        lidar2reference[0, 3] = 0.0
        lidar2reference[2, 3] = reference_height  # Lift to reference height
        reference2lidar = np.linalg.inv(lidar2reference)

        self.reference2rgb = (ego2rgb @ self.lidar2ego @ reference2lidar).astype(np.float32)
        self.reference2polar = (ego2polar @ self.lidar2ego @ reference2lidar).astype(np.float32)


@dataclass
class VoxelCenters:
    roi_x: List[float]
    roi_y: List[float]
    roi_z: List[float]
    
    grid_resolution_x: float
    grid_resolution_y: float
    grid_resolution_z: float

    
    data: np.ndarray
    
    def generate_voxel_centers(self,):

        self.num_grid_x: int = int((self.roi_x[1] - self.roi_x[0]) / self.grid_resolution_x)
        self.num_grid_y: int = int((self.roi_y[1] - self.roi_y[0]) / self.grid_resolution_y)
        self.num_grid_z: int = int((self.roi_z[1] - self.roi_z[0]) / self.grid_resolution_z)

        voxel_centers = np.zeros((self.num_grid_x, self.num_grid_y, self.num_grid_z, 3), dtype=np.float32)

        # X coordinate (front direction in reference frame)
        voxel_centers[:, :, :, 0] = (
            np.arange(self.num_grid_x) * self.grid_resolution_x
            + self.roi_x[0]
            + self.grid_resolution_x / 2
        )[:, None, None].repeat(self.num_grid_y, axis=1).repeat(self.num_grid_z, axis=2)

        # Y coordinate (left direction in reference frame)
        voxel_centers[:, :, :, 1] = (
            np.arange(self.num_grid_y) * self.grid_resolution_y
            + self.roi_y[0]
            + self.grid_resolution_y / 2
        )[None, :, None].repeat(self.num_grid_x, axis=0).repeat(self.num_grid_z, axis=2)

        # Z coordinate (up direction in reference frame, elevation)
        voxel_centers[:, :, :, 2] = (
            np.arange(self.num_grid_z) * self.grid_resolution_z
            + self.roi_z[0]
            + self.grid_resolution_z / 2
        )[None, None, :].repeat(self.num_grid_x, axis=0).repeat(self.num_grid_y, axis=1)

        # Flatten: (X, Y, Z, 3) -> (X*Y*Z, 3) -> (3, X*Y*Z)
        voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)

        self.data = voxel_centers

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
class ElevationMap:
    data_path: Path
    data: np.ndarray  # shape: (H, W)
    mask: np.ndarray  # shape: (H, W)
    

@dataclass
class ModelInput:
    calibration: Calibration
    image: ImageData
    polar_image: PolarImageData
    voxel_centers: VoxelCenters
    dataset_folder: str = ""

@dataclass
class Anno:
    elevation_map: ElevationMap
