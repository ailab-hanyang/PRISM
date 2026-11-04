import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict


@dataclass
class RoadBEVModelConfig:
    # >>> Data Path Configurations <<<
    train_pkl: Path = Path("pkl/train.pkl")   # relative to data_root
    val_pkl: Path = Path("pkl/val.pkl")      # relative to data_root
    data_root: Optional[Path] = None          # default: PRISM_DATASET_ROOT
    devkit_root: Optional[Path] = None        # default: PRISM_DEVKIT_ROOT

    def __post_init__(self):
        if self.data_root is None:
            root = os.environ.get('PRISM_DATASET_ROOT')
            if not root:
                raise EnvironmentError("PRISM_DATASET_ROOT is not set.\n  export PRISM_DATASET_ROOT=/path/to/dataset")
            self.data_root = Path(root)
        if self.devkit_root is None:
            root = os.environ.get('PRISM_DEVKIT_ROOT')
            if not root:
                raise EnvironmentError("PRISM_DEVKIT_ROOT is not set.\n  export PRISM_DEVKIT_ROOT=/path/to/PRISM")
            self.devkit_root = Path(root)
    
    # >>> Model Run Path Configurations <<<
    run_path: Path = Path("./runs/debug_run")
    checkpoint_save_path: Path = run_path / Path("checkpoints")
    logs_save_path: Path = run_path / Path("logs")
    
    # >>> Calibration Mode <<<

    # >>> Image Crop / Resize Configurations <<<
    # RGB: original 2448x2048 → crop (rgb_crop_w x rgb_crop_h) from bottom-left → resize to (input_w x input_h)
    # Polar: original 1224x1024 → crop (input_w x input_h) from bottom-left (no resize)
    rgb_crop_h: int = 1080
    rgb_crop_w: int = 1920
    input_h: int = 540
    input_w: int = 960

    # >>> Input Mode <<<
    # Controls which sensor data is fed into the model.
    # All modes use two-stream mid-fusion (separate RGB / polar encoders, 1×1 Conv fusion).
    # ┌─────────────┬──────────────────────────────────────────────────┬────────┐
    # │ input_mode  │ streams                                          │ total  │
    # ├─────────────┼──────────────────────────────────────────────────┼────────┤
    # │ rgb         │ single-stream: RGB-camera RGB(3)                 │  3 ch  │
    # │ polar_rgb   │ single-stream: polar-camera RGB(3)               │  3 ch  │
    # │ mono_polar  │ single-stream: DoLP(1) + AoLP(2)                 │  3 ch  │
    # │ tri_polar   │ single-stream: DoLP(3) + AoLP(6)                 │  9 ch  │
    # │ rgb_mono    │ two-stream: RGB enc(3) + polar enc(DoLP1+AoLP2)  │  6 ch  │
    # │ rgb_tri     │ two-stream: RGB enc(3) + polar enc(DoLP3+AoLP6)  │ 12 ch  │
    # └─────────────┴──────────────────────────────────────────────────┴────────┘
    input_mode = "rgb_tri"

    # >>> Backbone Pretrained Weight Configurations <<<
    use_backbone_pretrained = False
    backbone_pretrained_path = None  # Set to None if not using pretrained weights

    # >>> Backbone Fine-tuning Strategy <<<
    # Differential LR: backbone gets (backbone_lr_scale * learning_rate), head gets learning_rate
    # Set < 1.0 to preserve pretrained features. e.g., 0.1 = backbone LR 10x smaller than head
    backbone_lr_scale: float = 1.0

    roi_x = [7.0, 12.0]
    roi_y = [-2.0, 2.0]
    roi_z = [-0.2, 0.2]

    grid_resolution_x = 0.1
    grid_resolution_y = 0.1
    grid_resolution_z = 0.01
    num_grids_x = int((roi_x[1] - roi_x[0]) / grid_resolution_x)
    num_grids_y = int((roi_y[1] - roi_y[0]) / grid_resolution_y)
    num_grids_z = int((roi_z[1] - roi_z[0]) / grid_resolution_z)

    ele_range = (roi_z[1] - roi_z[0]) / 2.0
    cla_res = 1.0  # in cm
    # >>> Model Configurations <<<
    backbone_type = "efficientnet_b6"
    use_temporal = True
    
    # >>> Training Configurations <<<
    epochs = 50
    batch_size = 16
    accumulate_grad_batches = 1
    num_workers = 8

    devices = "auto"

    # Optimizer settings
    optimizer_type = "adamw"  # "adam" or "adamw"
    learning_rate = 8e-4 
    weight_decay = 1e-4

    # Scheduler settings
    scheduler_type = "warmup_cosine"
    # For CosineAnnealingLR
    warmup_epochs = 10  # number of linear warmup epochs
    cosine_T_max = 50  # Usually same as epochs
    cosine_eta_min = 1e-6
    # For StepLR
    step_size = 10
    step_gamma = 0.1
    # For OneCycleLR
    onecycle_max_lr = 1e-4
    onecycle_pct_start = 0.02
    
    precision = "32"
    strategy = "ddp"
    seed = 42

    # >>> Logging Configurations <<<
    wandb_entity_name = "your_wandb_entity"  # Set to your actual Weights & Biases entity name
    wandb_project_name = "your_wandb_project"  # Set to your actual Weights & Biases project name
    wandb_run_name = "your_run_name"  # Set to your desired run name or use None for auto-naming


