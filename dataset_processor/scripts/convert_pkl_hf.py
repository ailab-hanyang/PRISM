"""
PKL Converter for Pre-Split PRISM Dataset (HuggingFace Download)

For datasets downloaded from HuggingFace where train/val splits are already separated into folders.
Simply converts all data in train/ → train.pkl and val/ → val.pkl.

Usage:
  python convert_pkl_hf.py --config config/pkl_config_hf.yaml
"""

import os
import json
import pickle
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Classification class definitions ──────────────────────────────────────────
MATERIAL_CLASSES = ["asphalt", "concrete", "belgian_block", "gravel"]
CONDITION_CLASSES = ["dry", "damp", "wet", "slush", "snow_covered"]
IGNORE_MATERIALS = {"other"}


def resolve_folder_labels(folder_info: Dict[str, Any],
                           timestamps: List[str]) -> Dict[str, Dict]:
    """Resolve ignore_label / material_label / condition_label / weather for every timestamp."""
    ignored_ranges: List[Dict] = folder_info.get("ignored_frames", [])
    segments: List[Dict]       = folder_info.get("ordered_segments", [])

    result: Dict[str, Dict] = {}
    for ts in timestamps:
        # Step 1: ignored_frames
        if any(r["ts_start"] <= ts <= r["ts_end"] for r in ignored_ranges):
            result[ts] = {"ignore_classification_label": True, "material_label": None, "condition_label": None, "weather": None}
            continue

        # Step 2: find matching segment
        matched_label = None
        for seg in segments:
            if seg["ts_start"] <= ts <= seg["ts_end"]:
                matched_label = seg["label"]
                break

        if matched_label is None:
            result[ts] = {"ignore_classification_label": True, "material_label": None, "condition_label": None, "weather": None}
            continue

        # Step 3: decode
        material  = matched_label.get("surface_material")
        condition = matched_label.get("surface_state")
        weather   = matched_label.get("weather")

        mat_idx  = MATERIAL_CLASSES.index(material)   if material  in MATERIAL_CLASSES  else None
        cond_idx = CONDITION_CLASSES.index(condition)  if condition in CONDITION_CLASSES else None

        if material in IGNORE_MATERIALS:
            result[ts] = {"ignore_classification_label": True, "material_label": None, "condition_label": cond_idx, "weather": weather}
        else:
            ignore = (mat_idx is None or cond_idx is None)
            result[ts] = {"ignore_classification_label": ignore, "material_label": mat_idx, "condition_label": cond_idx, "weather": weather}

    return result


def load_classification_labels(label_json_path: str) -> Dict[str, Any]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config is None:
        raise RuntimeError(f"Config loaded as None from {config_path}")
    return config


def get_timestamp_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0]


def get_lidar2ego(calibration: Dict[str, Any]) -> np.ndarray:
    if 'lidar_to_ego' in calibration:
        return np.array(calibration['lidar_to_ego'], dtype=np.float32)
    return np.eye(4, dtype=np.float32)


def get_cam_intrinsic(camera_type: str, calibration: Dict[str, Any]) -> Dict[str, float]:
    if camera_type == 'rgb' and 'rgb_cam_intrinsic' in calibration:
        return calibration['rgb_cam_intrinsic']
    elif camera_type.startswith('polar') and 'polar_cam_intrinsic' in calibration:
        return calibration['polar_cam_intrinsic']
    return {}


def get_cam_distortion(camera_type: str, calibration: Dict[str, Any]) -> np.ndarray:
    if camera_type == 'rgb' and 'rgb_cam_distortion' in calibration:
        return np.array(calibration['rgb_cam_distortion'], dtype=np.float32)
    elif camera_type.startswith('polar') and 'polar_cam_distortion' in calibration:
        return np.array(calibration['polar_cam_distortion'], dtype=np.float32)
    return np.array([], dtype=np.float32)


def discover_sequences(dataset_base: Path) -> List[Path]:
    if not dataset_base.exists():
        return []
    return sorted([d for d in dataset_base.iterdir()
                   if d.is_dir() and d.name.startswith('sequence_')])


def get_timestamps_for_sequence(sequence_dir: Path,
                                 elevation_map_dir: str = "elevation_map_v2") -> List[str]:
    elevation_dir = sequence_dir / elevation_map_dir
    if elevation_dir.exists():
        pkl_files = sorted(elevation_dir.glob("*.pkl"))
        if pkl_files:
            return [get_timestamp_from_filename(f.name) for f in pkl_files]

    lidar_dir = sequence_dir / "lidar_accum_scan"
    if lidar_dir.exists():
        pcd_files = sorted(lidar_dir.glob("*.pcd"))
        return [get_timestamp_from_filename(f.name) for f in pcd_files]

    return []


def check_valid_ratio(pkl_path: Path) -> float:
    """Returns valid cell ratio in [0,1], or -1.0 on error/missing."""
    if not pkl_path.exists():
        return -1.0
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        ele_mask = data[1]
        if isinstance(ele_mask, torch.Tensor):
            return ele_mask.float().mean().item()
        return float(np.mean(ele_mask.astype(float)))
    except Exception:
        return -1.0


def _relpath(path: Path, root: Optional[Path]) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def process_sample(dataset_path: Path, dataset_name: str, sequence: str, timestamp: str,
                   topics: Dict[str, List[str]], calibration: Dict[str, Any],
                   elevation_map_dir: str = "elevation_map_filtered",
                   ignore_classification_label: bool = True,
                   ignore_elevation_label: bool = False,
                   material_label: Optional[int] = None,
                   condition_label: Optional[int] = None,
                   weather: Optional[str] = None,
                   dataset_root: Optional[Path] = None) -> Dict[str, Any]:
    sample = {
        'dataset_name':    dataset_name,
        'dataset_folder':  dataset_path.name,
        'sample_sequence': sequence,
        'ignore_classification_label': ignore_classification_label,
        'ignore_elevation_label':      ignore_elevation_label,
        'material_label':  material_label,
        'condition_label': condition_label,
        'weather':         weather,
        'elevation_map':   {},
        'images':          {},
    }

    # Elevation Map PKL
    elevation_path = dataset_path / sequence / elevation_map_dir
    if elevation_path.exists():
        pkl_file = elevation_path / f"{timestamp}.pkl"
        if pkl_file.exists():
            sample['elevation_map'] = {
                'elevation_map_path': _relpath(pkl_file, dataset_root),
                'timestamp': timestamp,
            }

    # RGB Camera
    rgb_path = dataset_path / sequence / "rgb"
    if rgb_path.exists():
        for ext in ['.png', '.jpg']:
            img_file = rgb_path / f"{timestamp}{ext}"
            if img_file.exists():
                sample['images']['rgb_camera'] = {
                    'img_path':  _relpath(img_file, dataset_root),
                    'timestamp': timestamp,
                }
                break

    # Polar Camera
    polar_path = dataset_path / sequence / "polar"
    if polar_path.exists():
        for angle in topics.get('polar_angles', ['0d', '45d', '90d', '135d']):
            angle_path = polar_path / angle
            if angle_path.exists():
                for ext in ['.png', '.jpg']:
                    img_file = angle_path / f"{timestamp}{ext}"
                    if img_file.exists():
                        sample['images'][f'polar_camera_{angle}'] = {
                            'img_path':  _relpath(img_file, dataset_root),
                            'timestamp': timestamp,
                        }
                        break

    return sample


def resolve_dataset_calib_path(dataset_base: Path, dataset_calib_dir: Path,
                               project_root: Path) -> str:
    calib_path = dataset_calib_dir / f'{dataset_base.name}.yaml'
    try:
        return str(calib_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(calib_path)


def process_split_folder(split_name: str, split_dir: Path, config: Dict[str, Any],
                         dataset_root: Path, devkit_root: Path, project_root: Path,
                         calibration: Dict[str, Any], dataset_calib_dir: Optional[Path],
                         folders_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process all datasets in a split folder (train/ or val/)."""

    elevation_map_dir: str = config.get('elevation_map_dir', 'elevation_map')
    topics: Dict           = config.get('topics', {'polar_angles': ['0d', '45d', '90d', '135d']})

    filter_cfg             = config.get('filter', {})
    min_valid_ratio: float = filter_cfg.get('min_valid_ratio', 0.0)

    data_list: List[Dict] = []
    total_scanned = 0
    total_filtered = 0

    # Discover all dataset folders in split directory
    dataset_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*70}")
    print(f"Split directory: {split_dir}")
    print(f"Datasets found:  {len(dataset_folders)}")
    print(f"Elevation dir:   {elevation_map_dir}")
    print(f"Min valid ratio: {min_valid_ratio:.0%}")
    print(f"{'='*70}")

    calibration_paths: Dict[str, str] = {}

    for dataset_path in dataset_folders:
        folder_name = dataset_path.name

        if dataset_calib_dir is not None:
            calib_path = resolve_dataset_calib_path(dataset_path, dataset_calib_dir, project_root)
            calibration_paths[folder_name] = calib_path

        print(f"\n  Dataset: {folder_name}")
        print(f"  Path:    {dataset_path}")

        sequences = discover_sequences(dataset_path)
        if not sequences:
            print(f"  [WARNING] No sequences found")
            continue
        print(f"  Sequences: {len(sequences)}")

        # Resolve per-timestamp labels for the whole folder
        folder_labels: Dict[str, Dict] = {}
        if folder_name in folders_info:
            all_ts: List[str] = []
            for seq_path in sequences:
                all_ts.extend(get_timestamps_for_sequence(seq_path, elevation_map_dir))
            all_ts.sort()
            folder_labels = resolve_folder_labels(folders_info[folder_name], all_ts)
        else:
            print(f"  [LABEL] WARNING: '{folder_name}' not in labels.json"
                  " — all samples ignore_label=True")

        for seq_path in sequences:
            seq_name   = seq_path.name
            timestamps = get_timestamps_for_sequence(seq_path, elevation_map_dir)

            if not timestamps:
                print(f"    {seq_name}: 0 samples (skipped)")
                continue

            n_elev_ignored = 0
            n_valid = 0
            total_scanned += len(timestamps)

            for ts in timestamps:
                pkl_path = seq_path / elevation_map_dir / f"{ts}.pkl"
                ratio    = check_valid_ratio(pkl_path)

                ignore_elev = ratio < min_valid_ratio
                if ignore_elev:
                    n_elev_ignored += 1
                    total_filtered += 1
                else:
                    n_valid += 1

                lbl = folder_labels.get(ts, {
                    "ignore_classification_label": True, "material_label": None,
                    "condition_label": None, "weather": None
                })
                sample = process_sample(
                    dataset_path, folder_name, seq_name, ts,
                    topics, calibration, elevation_map_dir,
                    ignore_classification_label = lbl["ignore_classification_label"],
                    ignore_elevation_label      = ignore_elev,
                    material_label              = lbl["material_label"],
                    condition_label             = lbl["condition_label"],
                    weather                     = lbl["weather"],
                    dataset_root                = dataset_root,
                )
                data_list.append(sample)

            elev_info = f" | ignore_elev={n_elev_ignored}" if n_elev_ignored else ""
            print(f"    [{split_name:5s}] {seq_name}: {len(timestamps)} samples, {n_valid} valid{elev_info}")

    print(f"\n{split_name.upper()} Summary:")
    print(f"  Total scanned:  {total_scanned}")
    print(f"  Total filtered: {total_filtered}")
    print(f"  Total valid:    {len([s for s in data_list if not s['ignore_elevation_label']])}")

    # Return calibration_paths for meta_info
    return data_list, calibration_paths


def create_pkl(config: Dict[str, Any]):
    """Create train.pkl and val.pkl from pre-split HuggingFace dataset structure."""

    # --- Environment variables ---
    dataset_root_env = os.environ.get('PRISM_DATASET_ROOT')
    if not dataset_root_env:
        raise EnvironmentError("PRISM_DATASET_ROOT is not set.")
    dataset_root = Path(dataset_root_env)

    devkit_root_env = os.environ.get('PRISM_DEVKIT_ROOT')
    if not devkit_root_env:
        raise EnvironmentError("PRISM_DEVKIT_ROOT is not set.")
    devkit_root = Path(devkit_root_env)

    def _resolve_data(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else dataset_root / q

    def _resolve_devkit(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else devkit_root / q

    # --- Output directory ---
    output_dir = _resolve_data(config.get('output_dir', 'pkl'))
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent.parent

    # --- Labels ---
    label_path_cfg = config.get('classification_label_path')
    folders_info: Dict[str, Any] = {}
    if label_path_cfg:
        label_path = str(_resolve_data(label_path_cfg))
        all_labels = load_classification_labels(label_path)
        folders_info = all_labels.get("folders", {})
        print(f"[LABEL] Loaded: {label_path}  ({len(folders_info)} folders)")
    else:
        print("[LABEL] WARNING: classification_label_path not set")

    # --- Calibration ---
    calibration = {}
    dataset_calib_dir = None
    if 'calibration_path' in config:
        common_calib = _resolve_devkit(config['calibration_path'])
        dataset_calib_dir = common_calib.parent / 'extrinsics'
        print(f"[CAL] {common_calib}")
        if common_calib.exists():
            cal_data = load_config(str(common_calib))
            calibration = cal_data.get('calibration', cal_data)
        else:
            print(f"[WARNING] Calibration not found: {common_calib}")

    # --- Process train/ and val/ folders ---
    train_dir = dataset_root / "train"
    val_dir   = dataset_root / "val"

    train_data_list = []
    val_data_list = []
    all_calibration_paths = {}

    if train_dir.exists():
        train_data_list, train_calib_paths = process_split_folder(
            "train", train_dir, config, dataset_root, devkit_root, project_root,
            calibration, dataset_calib_dir, folders_info
        )
        all_calibration_paths.update(train_calib_paths)
    else:
        print(f"\n[WARNING] Train directory not found: {train_dir}")

    if val_dir.exists():
        val_data_list, val_calib_paths = process_split_folder(
            "val", val_dir, config, dataset_root, devkit_root, project_root,
            calibration, dataset_calib_dir, folders_info
        )
        all_calibration_paths.update(val_calib_paths)
    else:
        print(f"\n[WARNING] Val directory not found: {val_dir}")

    # --- Meta info ---
    train_datasets = [d.name for d in train_dir.iterdir() if d.is_dir()] if train_dir.exists() else []
    val_datasets   = [d.name for d in val_dir.iterdir() if d.is_dir()] if val_dir.exists() else []
    all_datasets   = sorted(set(train_datasets + val_datasets))

    meta_info = {
        'dataset_version': config.get('dataset_version', 'unknown'),
        'dataset_root':    str(dataset_root),
        'devkit_root':     str(devkit_root),
        'datasets':        all_datasets,
        'split': {
            'method': 'pre_split_hf',
            'description': 'Data downloaded from HuggingFace with pre-separated train/val folders',
        },
        'label': {
            'material_classes':  MATERIAL_CLASSES,
            'condition_classes': CONDITION_CLASSES,
            'ignore_materials':  sorted(IGNORE_MATERIALS),
        },
        'calibration_paths': all_calibration_paths,
        'sensor_calibration': {
            'lidar': {'lidar2ego': get_lidar2ego(calibration).tolist()},
            'cameras': {
                'rgb_camera': {
                    'cam_intrinsic':  get_cam_intrinsic('rgb', calibration),
                    'cam_distortion': get_cam_distortion('rgb', calibration).tolist(),
                },
                'polar_camera': {
                    'cam_intrinsic':  get_cam_intrinsic('polar', calibration),
                    'cam_distortion': get_cam_distortion('polar', calibration).tolist(),
                },
            },
        },
    }

    # --- Save ---
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if train_data_list:
        train_path = output_dir / 'train.pkl'
        with open(train_path, 'wb') as f:
            pickle.dump({'meta_info': meta_info, 'data_list': train_data_list}, f)
        print(f"Train: {len(train_data_list)} samples -> {train_path}")
    else:
        print("Train: 0 samples (not saved)")

    if val_data_list:
        val_path = output_dir / 'val.pkl'
        with open(val_path, 'wb') as f:
            pickle.dump({'meta_info': meta_info, 'data_list': val_data_list}, f)
        print(f"Val:   {len(val_data_list)} samples -> {val_path}")
    else:
        print("Val:   0 samples (not saved)")

    train_cls_ign  = sum(1 for s in train_data_list if s['ignore_classification_label'])
    val_cls_ign    = sum(1 for s in val_data_list   if s['ignore_classification_label'])
    train_elev_ign = sum(1 for s in train_data_list if s['ignore_elevation_label'])
    val_elev_ign   = sum(1 for s in val_data_list   if s['ignore_elevation_label'])
    train_cls      = len(train_data_list) - train_cls_ign
    val_cls        = len(val_data_list)   - val_cls_ign
    train_elev     = len(train_data_list) - train_elev_ign
    val_elev       = len(val_data_list)   - val_elev_ign
    total_pkl      = len(train_data_list) + len(val_data_list)
    total_cls      = train_cls + val_cls
    total_elev     = train_elev + val_elev

    print(f"\n--- Statistics ---")
    print(f"In PKL  train / val            : {len(train_data_list)} / {len(val_data_list)}")
    print(f"  ignore_classification_label  : {train_cls_ign} / {val_cls_ign}")
    print(f"  classifiable                 : {train_cls} / {val_cls}")
    print(f"  ignore_elevation_label       : {train_elev_ign} / {val_elev_ign}")
    print(f"  elevation-trainable          : {train_elev} / {val_elev}")
    if total_pkl > 0:
        r = len(train_data_list) / total_pkl
        print(f"Overall split                  : {r:.1%} train / {1-r:.1%} val")
    if total_cls > 0:
        r = train_cls / total_cls
        print(f"Classifiable split             : {r:.1%} train / {1-r:.1%} val")
    if total_elev > 0:
        r = train_elev / total_elev
        print(f"Elevation-trainable split      : {r:.1%} train / {1-r:.1%} val")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='PKL Converter for Pre-Split PRISM Dataset (HuggingFace)')
    parser.add_argument('--config', default='dataset_processor/config/pkl_config_hf.yaml')
    args = parser.parse_args()
    create_pkl(load_config(args.config))


if __name__ == '__main__':
    main()
