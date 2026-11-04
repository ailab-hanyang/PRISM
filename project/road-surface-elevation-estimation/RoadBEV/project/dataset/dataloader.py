import re
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import yaml
import numpy as np
import pickle
import copy

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from project.dataset.dataclasses import *
from project.dataset.builder import ModelFeatureBuilder, ModelTargetBuilder
from project.config.model_config import RoadBEVModelConfig
from project.dataset.collate import model_collate_fn


class SceneLoader:
    def __init__(self, config: RoadBEVModelConfig):
        self.config = config

    def load_data(self, mode: str):
        if mode == "train":
            pkl_path = self.config.data_root / self.config.train_pkl
        elif mode in ("val", "test"):
            pkl_path = self.config.data_root / self.config.val_pkl
        else:
            raise ValueError("Invalid mode. Choose 'train', 'val' or 'test'.")

        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)
            self.meta_info = raw['meta_info']
            all_data = raw['data_list']

        n_total = len(all_data)
        self.data = [s for s in all_data
                     if not s.get('ignore_elevation_label', False)
                     and s.get('elevation_map', {}).get('elevation_map_path')]
        n_filtered = n_total - len(self.data)

        print(f"pkl path: {pkl_path}")
        print(f"data_list length: {len(self.data)} "
              f"(total={n_total}, ignored={n_filtered})")

        self.data_root = Path(self.meta_info.get('dataset_root') or self.config.data_root)

        self.get_calib_infos()

        self.voxel_centers = self.generate_voxel_centers(self.config)
    
    def get_calib_infos(self):
        self.calibrations = {}
        lidar2ego = np.array(self.meta_info['sensor_calibration']['lidar']['lidar2ego'], dtype=np.float32)

        calib_infos = {}
        for (dataset_folder, value) in self.meta_info['calibration_paths'].items():
            calib_yaml_path = self.config.devkit_root / value
            with open(calib_yaml_path, 'r') as f:
                calib_info = yaml.safe_load(f)
                for (calib_key, calib_value) in calib_info.items():
                    if isinstance(calib_value, list):
                        calib_info[calib_key] = np.array(calib_value, dtype=np.float32)
            calib_infos[dataset_folder] = calib_info

        # Intrinsics: try meta_info first; fall back to first calibration yaml if empty
        _first_calib = next(iter(calib_infos.values())) if calib_infos else {}
        rgb_meta = self.meta_info["sensor_calibration"]["cameras"]["rgb_camera"]["cam_intrinsic"]
        polar_meta = self.meta_info["sensor_calibration"]["cameras"]["polar_camera"]["cam_intrinsic"]

        if rgb_meta.get("fx") is not None:
            rgb_src = rgb_meta
        else:
            rgb_src = _first_calib.get("rgb_cam_intrinsic", {})

        if polar_meta.get("fx") is not None:
            polar_src = polar_meta
        else:
            polar_src = _first_calib.get("polar_cam_intrinsic", {})

        rgb_intrinsic = np.zeros((3, 3), dtype=np.float32)
        rgb_intrinsic[0, 0] = rgb_src["fx"]
        rgb_intrinsic[1, 1] = rgb_src["fy"]
        rgb_intrinsic[0, 2] = rgb_src["cx"]
        rgb_intrinsic[1, 2] = rgb_src["cy"]
        rgb_intrinsic[2, 2] = 1.0
        rgb_distortion = np.array(self.meta_info["sensor_calibration"]["cameras"]["rgb_camera"]["cam_distortion"], dtype=np.float32)

        # Polar intrinsic is calibrated at full sensor resolution (2448x2048),
        # but each polarization channel image is at half resolution (1224x1024)
        # due to the 2x2 micro-polarizer array. Scale by 0.5 to match actual image size.
        polar_intrinsic = np.zeros((3, 3), dtype=np.float32)
        polar_intrinsic[0, 0] = polar_src["fx"] * 0.5
        polar_intrinsic[1, 1] = polar_src["fy"] * 0.5
        polar_intrinsic[0, 2] = polar_src["cx"] * 0.5
        polar_intrinsic[1, 2] = polar_src["cy"] * 0.5
        polar_intrinsic[2, 2] = 1.0
        polar_distortion = np.array(self.meta_info["sensor_calibration"]["cameras"]["polar_camera"]["cam_distortion"], dtype=np.float32)

        for dataset_folder, calib_info in calib_infos.items():
            rgb2ego = calib_info['rgb_cam_to_ego']
            polar2ego = calib_info['polar_cam_to_ego']

            calibration = Calibration(
                lidar2ego=lidar2ego,
                rgb2ego=rgb2ego,
                polar2ego=polar2ego,
                rgb_intrinsic=rgb_intrinsic,
                rgb_distortion=rgb_distortion,
                polar_intrinsic=polar_intrinsic,
                polar_distortion=polar_distortion
            )
            calibration.generate_reference_calibration(reference_height=1.8, align_to_camera="rgb")
            self.calibrations[dataset_folder] = calibration
        

    def generate_voxel_centers(self, config: RoadBEVModelConfig) -> VoxelCenters:
        voxel_centers = VoxelCenters(
            roi_x=[config.roi_x[0], config.roi_x[1]],
            roi_y=[config.roi_y[0], config.roi_y[1]],
            roi_z=[config.roi_z[0], config.roi_z[1]],
            grid_resolution_x=config.grid_resolution_x,
            grid_resolution_y=config.grid_resolution_y,
            grid_resolution_z=config.grid_resolution_z,
            data=None
        )
        voxel_centers.generate_voxel_centers()
        
        return voxel_centers
        
    def _resolve_path(self, path: Path) -> Path:
        if self.data_root is not None and not path.is_absolute():
            return self.data_root / path
        return path

    def get_image(self, idx: int) -> ImageData:
        data_path = self._resolve_path(Path(self.data[idx]['images']['rgb_camera']['img_path']))
        data = cv2.imread(str(data_path))
        
        return ImageData(
            data_path=data_path,
            data=data
        )

    def get_polar_image(self, idx: int) -> PolarImageData:
        data_path_0d = self._resolve_path(Path(self.data[idx]['images']['polar_camera_0d']['img_path']))
        data_path_45d = self._resolve_path(Path(self.data[idx]['images']['polar_camera_45d']['img_path']))
        data_path_90d = self._resolve_path(Path(self.data[idx]['images']['polar_camera_90d']['img_path']))
        data_path_135d = self._resolve_path(Path(self.data[idx]['images']['polar_camera_135d']['img_path']))

        # Load images
        data_0d = cv2.imread(str(data_path_0d), cv2.IMREAD_GRAYSCALE)
        data_45d = cv2.imread(str(data_path_45d), cv2.IMREAD_GRAYSCALE)
        data_90d = cv2.imread(str(data_path_90d), cv2.IMREAD_GRAYSCALE)
        data_135d = cv2.imread(str(data_path_135d), cv2.IMREAD_GRAYSCALE)

        return PolarImageData(
            data_path_0d = data_path_0d,
            data_path_45d = data_path_45d,
            data_path_90d = data_path_90d,
            data_path_135d = data_path_135d,
            data_0d = data_0d,
            data_45d = data_45d,
            data_90d = data_90d,
            data_135d = data_135d
        )
        
    
    def get_elevation_map(self, idx: int) -> ElevationMap:
        data_path = self._resolve_path(Path(self.data[idx]['elevation_map']['elevation_map_path']))
        with open(data_path, 'rb') as f:
            elevation_data = pickle.load(f)
        
        data = elevation_data[0].numpy()
        mask = elevation_data[1].numpy()
        
        return ElevationMap(
            data_path=data_path,
            data=data,
            mask=mask
        )
    

    def get_model_input(self, idx: int) -> ModelInput:
        image = self.get_image(idx)
        polar_image = self.get_polar_image(idx)
        dataset_folder = self.data[idx]['dataset_folder']
        calibration = self.calibrations[dataset_folder]
        voxel_centers = self.voxel_centers

        return ModelInput(image=image,
                          polar_image=polar_image,
                          calibration=calibration,
                          voxel_centers=voxel_centers,
                          dataset_folder=dataset_folder)

    def get_annotation(self, idx: int) -> Anno:
        elevation_map = self.get_elevation_map(idx)
        return Anno(elevation_map=elevation_map)


class ModelDataset(Dataset):
    def __init__(self, config: RoadBEVModelConfig, mode: str):
        self.loader = SceneLoader(config)
        self.loader.load_data(mode)

        self.feature_builder = ModelFeatureBuilder(config=config)
        sample_calibration = next(iter(self.loader.calibrations.values()))
        self.feature_builder.build_undistort_cache(sample_calibration)
        self.feature_builder.build_voxel_uv_cache(
            calibrations=self.loader.calibrations,
            voxel_centers=self.loader.voxel_centers
        )
        self.target_builder = ModelTargetBuilder(config=config)

    def __len__(self):
        return len(self.loader.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        model_input = self.loader.get_model_input(idx)
        anno = self.loader.get_annotation(idx)

        features = self.feature_builder.compute_features(model_input)
        targets = self.target_builder.compute_targets(anno)

        return {
            "features": features,
            "targets": targets,
        }


class ModelDataModule(pl.LightningDataModule):
    def __init__(self, config: RoadBEVModelConfig):
        super().__init__()
        self.config = config
        
        print(f"batch size: {config.batch_size}")

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ModelDataset(self.config, mode="train")
        self.val_dataset = ModelDataset(self.config, mode="val")
        self.test_dataset = ModelDataset(self.config, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.config.batch_size, 
                          shuffle=True,
                          num_workers=self.config.num_workers,
                          pin_memory=True,
                          persistent_workers=self.config.num_workers > 0,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size, 
                          num_workers=self.config.num_workers,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.config.batch_size, 
                          num_workers=self.config.num_workers,
                          )
