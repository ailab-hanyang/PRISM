### 3. Train the model

Both benchmarks share the same training entry point and config structure.  
Configuration is managed via a **Python dataclass** (`project/config/model_config.py`).  
Edit the config file directly before launching training — there are no CLI overrides for hyperparameters.

---

## Track 1 — Road Surface Condition Classification

```bash
cd project/road-surface-condition-classification/classifier
export PYTHONPATH=$(pwd)

# Train from scratch
python scripts/train.py --config project/config/model_config.py

# Resume from checkpoint
python scripts/train.py --config project/config/model_config.py \
    --ckpt_path /path/to/checkpoint.ckpt
```

### Configuration (`classifier/project/config/model_config.py`)

#### Model selection — `backbone_type`

```python
backbone_type = "efficientnet_b6"   # default
```

| Value | Model |
|-------|-------|
| `efficientnet_b0` ~ `efficientnet_b7` | EfficientNet B0–B7 |
| `mobilenet_v2` | MobileNet V2 |
| `mobilenet_v3_small` | MobileNet V3 Small |
| `mobilenet_v3_large` | MobileNet V3 Large |

#### Input modality — `input_mode`

```python
input_mode = "rgb_tri"   # default
```

| Value | Input streams | Channels |
|-------|--------------|:--------:|
| `rgb` | RGB-camera RGB | 3 |
| `polar_rgb` | Polar-camera RGB (demosaiced) | 3 |
| `mono_polar` | DoLP (1) + AoLP sin/cos (2) | 3 |
| `tri_polar` | per-channel DoLP (3) + per-channel AoLP (6) | 9 |
| `rgb_mono` | RGB encoder (3) + polar encoder (DoLP+AoLP, 3) | 6 |
| `rgb_tri` | RGB encoder (3) + polar encoder (per-ch DoLP+AoLP, 9) | 12 |

Two-stream modes (`rgb_mono`, `rgb_tri`) use mid-fusion: separate backbones for each stream merged via 1×1 Conv.

#### Image resolution

```python
# RGB: 2448×2048 → crop rgb_crop_w×rgb_crop_h from bottom-left → resize to input_w×input_h
# Polar: 1224×1024 → crop input_w×input_h from bottom-left (no resize)
rgb_crop_h: int = 1080
rgb_crop_w: int = 1920
input_h:    int = 540
input_w:    int = 960
```

#### Training hyperparameters

```python
epochs                  = 50
batch_size              = 16
accumulate_grad_batches = 1
num_workers             = 8
devices                 = "auto"   # "auto" | [0] | [0,1] | ...

optimizer_type   = "adamw"          # "adam" | "adamw"
learning_rate    = 8e-4
weight_decay     = 1e-4

scheduler_type   = "warmup_cosine"  # "warmup_cosine" | "step" | "onecycle"
warmup_epochs    = 10               # (warmup_cosine) linear warm-up epochs
cosine_T_max     = 50               # (warmup_cosine) usually == epochs
cosine_eta_min   = 1e-6
step_size        = 10               # (step) decay every N epochs
step_gamma       = 0.1              # (step) multiplicative factor
onecycle_max_lr  = 1e-4            # (onecycle)
onecycle_pct_start = 0.02          # (onecycle)

precision = "32"    # "32" | "16-mixed"
strategy  = "ddp"
seed      = 42
```

#### Logging (Weights & Biases)

```python
wandb_entity_name  = "your_entity"
wandb_project_name = "your_project"
wandb_run_name     = "efficientnet_b6_rgb_tri"
```

---

## Track 2 — Road Surface Elevation Estimation (RoadBEV)

```bash
cd project/road-surface-elevation-estimation/RoadBEV
export PYTHONPATH=$(pwd)

# Train from scratch
python scripts/train.py --config project/config/model_config.py

# Resume from checkpoint
python scripts/train.py --config project/config/model_config.py \
    --ckpt_path /path/to/checkpoint.ckpt
```

### Configuration (`RoadBEV/project/config/model_config.py`)

#### Input modality — `input_mode`

Same options as Track 1. See the table above.

#### Backbone

```python
backbone_type = "efficientnet_b6"   # default (same registry as Track 1)
```

#### Pretrained backbone

```python
use_backbone_pretrained   = False
backbone_pretrained_path  = None    # path to a .ckpt or .pth file
```

#### Training hyperparameters

Same structure as Track 1. See the table above.

#### Logging (Weights & Biases)

```python
wandb_entity_name  = "your_entity"
wandb_project_name = "your_project"
wandb_run_name     = "roadbev_rgb_tri"
```

---

## Multi-GPU training

Set `devices` in the config to specify which GPUs to use:

```python
devices = "auto"    # use all available GPUs
devices = [0, 1]    # use GPU 0 and 1
devices = [0]       # single GPU
```

`strategy = "ddp"` is set by default and is recommended for multi-GPU runs.  
`batch_size` is **per GPU** — effective batch size = `batch_size × num_gpus × accumulate_grad_batches`.
