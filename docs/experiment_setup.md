### 2. Prepare Ground Truth & Create .pkl files for train / test

Before training or evaluation, you need to generate ground truth elevation maps from raw LiDAR point clouds, then package them into `.pkl` index files for the dataloader.

#### Prerequisites

Make sure the following environment variables are set (see [Environment Setup](./environment_setup.md)):

```bash
export PRISM_DATASET_ROOT=/path/to/PRISM-Dataset   # root directory of the downloaded dataset
export PRISM_DEVKIT_ROOT=/path/to/PRISM            # repo root
```

---

#### Step 1 — Generate ground truth elevation maps

```bash
cd dataset_processor
export PYTHONPATH=$(pwd)

python scripts/create_gt.py
```

This script reads `dataset_processor/config/gt_config.yaml` and processes every `.pcd` file under each scene's `lidar_accum_scan/` directory in parallel (80% of available CPU cores by default).

**Output** — for each sequence:
```
{PRISM_DATASET_ROOT}/{scene}/sequence_{id}/elevation_map_10cmx10cm/{timestamp}.pkl
```

Each `.pkl` file contains:
- `ele_gt` — elevation map tensor `(H, W)`, `torch.float32`, unit: cm
- `ele_mask` — valid pixel mask `(H, W)`, `torch.bool`

> **Note**: Processing the full dataset takes several hours. Progress is shown per sequence.

Configuration options in `gt_config.yaml`:
| Key | Description |
|-----|-------------|
| `input.dataset_bases` | List of scene directory names to process |
| `roi` | Region of interest in camera coordinates (cm) |
| `grid.resolution_x/y` | BEV grid resolution (default: 10 cm × 10 cm) |
| `include_normal_gt` | Also compute surface normal GT (default: `True`) |

---

#### Step 2 — Create train / val index files

Choose one of the two options below depending on how your dataset is structured.

---

##### Option 1 — HuggingFace download (Recommended)

Use this if you downloaded the dataset from HuggingFace, where the train/val split is **already provided** as separate top-level folders:

```
{PRISM_DATASET_ROOT}/
├── train/
│   ├── 0106_dataset/
│   ├── 0112_dataset/
│   └── ...
└── val/
    ├── 0128_dataset/
    └── ...
```

```bash
python scripts/convert_pkl_hf.py --config config/pkl_config_hf.yaml
```

The script automatically discovers all scene folders under `train/` and `val/` — no split configuration needed.

Configuration options in `pkl_config_hf.yaml`:
| Key | Description |
|-----|-------------|
| `elevation_map_dir` | GT subdirectory name (default: `elevation_map_10cmx10cm`) |
| `filter.min_valid_ratio` | Minimum valid pixel ratio to include a frame (default: `0.6`) |
| `classification_label_path` | Path to the surface label JSON file (default: `labels.json`) |

---

##### Option 2 — Custom split

Use this if your dataset has a flat structure (no `train/`/`val/` folders) and you want to define the train/val split yourself via `pkl_config.yaml`:

```
{PRISM_DATASET_ROOT}/
├── 0106_dataset/
├── 0112_dataset/
└── ...
```

```bash
python scripts/convert_pkl.py --config config/pkl_config.yaml
```

Split assignment is controlled by these keys in `pkl_config.yaml`:
| Key | Description |
|-----|-------------|
| `val_entire_folders` | Scenes assigned entirely to val |
| `val_sequences` | Per-scene sequence-level val assignment |
| `val_intra_sequences` | Tail-ratio intra-scene val split (e.g. last 20% → val) |
| `exclude_sequences` | Sequences to skip entirely |
| `filter.min_valid_ratio` | Minimum valid pixel ratio to include a frame |
| `classification_label_path` | Path to the surface label JSON file |

---

**Output** (both options):
```
{PRISM_DATASET_ROOT}/pkl/
├── train.pkl
└── val.pkl
```

Each `.pkl` index file contains:
- `meta_info` — dataset version, sensor calibration, label class definitions
- `data_list` — list of samples, each with:
  - Paths to RGB and polarimetric images (0°, 45°, 90°, 135°)
  - Path to the pre-generated elevation map `.pkl`
  - `material_label` / `condition_label` for surface classification
  - `ignore_elevation_label` / `ignore_classification_label` flags
