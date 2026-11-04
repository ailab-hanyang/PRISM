"""
PKL Converter for PRISM Dataset

Timestamp-range based label resolution with explicit scenario-level train/val split.

Usage:
  python convert_pkl.py --config config/pkl_config.yaml
"""

import os
import json
import pickle
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ── Classification class definitions ──────────────────────────────────────────
# material_label index: asphalt=0, concrete=1, belgian_block=2, gravel=3
MATERIAL_CLASSES = ["asphalt", "concrete", "belgian_block", "gravel"]
# condition_label index: dry=0, damp=1, wet=2, slush=3, snow_covered=4
CONDITION_CLASSES = ["dry", "damp", "wet", "slush", "snow_covered"]
# Materials stored in PKL but excluded from classification (ignore_label=True)
IGNORE_MATERIALS = {"other"}


def resolve_folder_labels(folder_info: Dict[str, Any],
                           timestamps: List[str]) -> Dict[str, Dict]:
    """Resolve ignore_label / material_label / condition_label / weather for every timestamp.

    Label resolution order per timestamp:
      1. Falls in any ignored_frames range  → ignore_label=True, labels=None
      2. Matches an ordered_segment range   → decode material/condition/weather
         - material in IGNORE_MATERIALS     → ignore_label=True
         - otherwise                        → ignore_label=False
      3. Not covered by any segment         → ignore_label=True, labels=None

    Timestamp range comparison uses direct string comparison.
    Format 'XXXXXXXXXX_YYY' is fixed-width, so lexicographic order == numeric order.

    Returns:
        Dict mapping ts_str -> {ignore_label, material_label, condition_label, weather}
    """
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


def resolve_dataset_calib_path(dataset_base: Path, dataset_calib_dir: Path,
                               project_root: Path) -> str:
    calib_path = dataset_calib_dir / f'{dataset_base.name}.yaml'
    try:
        return str(calib_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(calib_path)


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
        print(f"  [WARNING] Dataset not found: {dataset_base}")
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


def check_terrain_flatness(pkl_path: Path) -> Tuple[float, float, float]:
    """Returns (x_slope_mean, z_range, status). status=-1.0 on error."""
    if not pkl_path.exists():
        return -1.0, -1.0, -1.0
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        ele_gt, ele_mask = data[0], data[1]
        if isinstance(ele_gt,   torch.Tensor): ele_gt   = ele_gt.numpy()
        if isinstance(ele_mask, torch.Tensor): ele_mask = ele_mask.numpy()
        valid_z = ele_gt[ele_mask]
        if len(valid_z) == 0:
            return -1.0, -1.0, -1.0
        z_range = float(valid_z.max() - valid_z.min())
        H = ele_gt.shape[0]
        slopes = [abs(float(ele_gt[i+1, j] - ele_gt[i, j]))
                  for i in range(H - 1)
                  for j in range(ele_gt.shape[1])
                  if ele_mask[i, j] and ele_mask[i+1, j]]
        x_slope_mean = float(np.mean(slopes)) if slopes else 0.0
        return x_slope_mean, z_range, 1.0
    except Exception:
        return -1.0, -1.0, -1.0


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


def create_pkl(config: Dict[str, Any]):
    """Create train.pkl and val.pkl with explicit scenario-level split."""

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

    # --- Config ---
    dataset_bases: List[str] = config.get('dataset_bases', [])
    if not dataset_bases:
        print("[ERROR] No dataset_bases in config"); return

    val_entire_folders: set            = set(config.get('val_entire_folders', []))
    val_sequences: Dict[str, List]     = config.get('val_sequences', {})
    # {folder: {_seq_key_ignored: ratio}} — merges all folder timestamps, tail ratio% → val (seq key ignored)
    val_intra_sequences: Dict[str, Dict] = config.get('val_intra_sequences', {})
    exclude_sequences: Dict[str, List] = config.get('exclude_sequences', {})

    elevation_map_dir: str = config.get('elevation_map_dir', 'elevation_map')
    topics: Dict           = config.get('topics', {'polar_angles': ['0d', '45d', '90d', '135d']})

    filter_cfg             = config.get('filter', {})
    min_valid_ratio: float = filter_cfg.get('min_valid_ratio', 0.0)
    enable_flatness: bool  = filter_cfg.get('enable_flatness_filter', False)
    max_x_slope: float     = filter_cfg.get('max_x_slope_mean', 5.0)
    max_z_range: float     = filter_cfg.get('max_z_range', 80.0)

    output_dir = _resolve_data(config.get('output_dir', 'prism_pkl'))
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

    calibration_paths: Dict[str, str] = {}

    meta_info = {
        'dataset_version': config.get('dataset_version', 'unknown'),
        'dataset_root':    str(dataset_root),
        'devkit_root':     str(devkit_root),
        'datasets':        [Path(ds).name for ds in dataset_bases],
        'split': {
            'method':             'explicit_scenario',
            'val_entire_folders': sorted(val_entire_folders),
            'val_sequences':      {k: sorted(v) for k, v in val_sequences.items()},
        },
        'label': {
            'material_classes':  MATERIAL_CLASSES,
            'condition_classes': CONDITION_CLASSES,
            'ignore_materials':  sorted(IGNORE_MATERIALS),
        },
        'calibration_paths': calibration_paths,
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

    train_data_list: List[Dict] = []
    val_data_list:   List[Dict] = []
    total_scanned = total_filtered = total_no_elev = total_flat = 0

    print(f"\n{'='*70}")
    print("PKL Converter  (Explicit Scenario-Level Split)")
    print(f"{'='*70}")
    print(f"Elevation dir  : {elevation_map_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Val folders    : {sorted(val_entire_folders)}")
    print(f"Val sequences  : {dict(val_sequences)}")
    intra_display = {f: next(iter(s.values())) for f, s in val_intra_sequences.items()}
    print(f"Val intra-split: {intra_display}  (folder-level tail split; seq key in config is ignored)")
    print(f"Min valid ratio: {min_valid_ratio:.0%}")
    if enable_flatness:
        print(f"Flatness filter: max_slope={max_x_slope}, max_z={max_z_range}")
    print(f"Datasets       : {len(dataset_bases)}")
    print(f"Dataset root   : {dataset_root}")
    print(f"{'='*70}")

    for ds_str in dataset_bases:
        dataset_path = _resolve_data(ds_str)
        folder_name  = dataset_path.name

        if dataset_calib_dir is not None:
            calib_path = resolve_dataset_calib_path(dataset_path, dataset_calib_dir, project_root)
            calibration_paths[folder_name] = calib_path

        print(f"\n  Dataset: {folder_name}")
        print(f"  Path   : {dataset_path}")

        sequences = discover_sequences(dataset_path)
        if not sequences:
            print(f"  [WARNING] No sequences found"); continue
        print(f"  Sequences: {len(sequences)}")

        folder_is_val        = folder_name in val_entire_folders
        folder_val_seqs      = set(val_sequences.get(folder_name, []))
        folder_intra_seqs    = val_intra_sequences.get(folder_name, {})  # {seq_name: ratio}
        folder_excl_seqs     = set(exclude_sequences.get(folder_name, []))
        folder_train_valid   = 0
        folder_val_valid     = 0

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

        # Folder-level intra split: merge all eligible sequence timestamps and split at once
        # so that tail N% of the global timeline becomes val regardless of per-sequence data density
        folder_intra_split: Dict[str, str] = {}
        folder_intra_ratio: Optional[float] = None
        if folder_intra_seqs and not folder_is_val:
            folder_intra_ratio = next(iter(folder_intra_seqs.values()))
            eligible_ts: List[str] = []
            for sp in sequences:
                if sp.name not in folder_excl_seqs and sp.name not in folder_val_seqs:
                    eligible_ts.extend(get_timestamps_for_sequence(sp, elevation_map_dir))
            eligible_ts.sort()
            n_total   = len(eligible_ts)
            val_start = max(1, int(n_total * (1.0 - folder_intra_ratio)))
            for ts in eligible_ts[:val_start]:
                folder_intra_split[ts] = "train"
            for ts in eligible_ts[val_start:]:
                folder_intra_split[ts] = "val"
            n_intra_val = n_total - val_start
            print(f"  Intra-split (tail {folder_intra_ratio:.0%}): {n_total} scanned → {val_start} train / {n_intra_val} val  [before valid-ratio filter]")

        for seq_path in sequences:
            seq_name   = seq_path.name
            timestamps = get_timestamps_for_sequence(seq_path, elevation_map_dir)

            if seq_name in folder_excl_seqs:
                print(f"    {seq_name}: EXCLUDED"); continue
            if not timestamps:
                print(f"    {seq_name}: 0 samples (skipped)"); continue

            is_val = folder_is_val or (seq_name in folder_val_seqs)

            n_elev_ignored     = 0
            n_seq_train_valid  = 0
            n_seq_val_valid    = 0
            total_scanned += len(timestamps)

            # Determine per-timestamp split assignment
            if folder_intra_split and not is_val:
                ts_split  = [(folder_intra_split.get(ts, "train"), ts) for ts in timestamps]
                split_tag = f"intra(val={folder_intra_ratio:.0%})"
            else:
                split_tag = "val" if is_val else "train"
                ts_split  = [(split_tag, ts) for ts in timestamps]

            for sp, ts in ts_split:
                pkl_path = seq_path / elevation_map_dir / f"{ts}.pkl"
                ratio    = check_valid_ratio(pkl_path)

                ignore_elev = ratio < min_valid_ratio  # also covers ratio < 0 (missing file)
                if enable_flatness and not ignore_elev:
                    slope, zr, ok = check_terrain_flatness(pkl_path)
                    if ok < 0 or slope > max_x_slope or zr > max_z_range:
                        ignore_elev = True

                if ignore_elev:
                    n_elev_ignored  += 1
                    total_filtered  += 1
                elif sp == "train":
                    n_seq_train_valid += 1
                else:
                    n_seq_val_valid   += 1

                lbl = folder_labels.get(ts, {
                    "ignore_classification_label": True, "material_label": None, "condition_label": None
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
                (val_data_list if sp == "val" else train_data_list).append(sample)

            elev_info = f" | ignore_elev={n_elev_ignored}" if n_elev_ignored else ""
            if folder_intra_split and not is_val:
                print(f"    [{split_tag}] {seq_name}: {len(timestamps)} scanned → valid train={n_seq_train_valid} / val={n_seq_val_valid}{elev_info}")
                folder_train_valid += n_seq_train_valid
                folder_val_valid   += n_seq_val_valid
            else:
                print(f"    [{split_tag:5s}] {seq_name}: {len(timestamps)} samples{elev_info}")

        if folder_intra_split:
            print(f"  → intra valid total: {folder_train_valid} train / {folder_val_valid} val")

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
        print(f"Val  : {len(val_data_list)} samples -> {val_path}")
    else:
        print("Val  : 0 samples (not saved)")

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
    print(f"Scanned                        : {total_scanned}")
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
    parser = argparse.ArgumentParser(description='PKL Converter for PRISM Dataset')
    parser.add_argument('--config', default='config/pkl_config.yaml')
    args = parser.parse_args()
    create_pkl(load_config(args.config))


if __name__ == '__main__':
    main()
