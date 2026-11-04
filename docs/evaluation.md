### 4. Evaluation

#### 1. Road Surface Condition Classification
```bash
cd project/road-surface-condition-classification/classifier
export PYTHONPATH=$(pwd)

python scripts/test.py \
    --config project/config/model_config.py \
    --ckpt /path/to/checkpoint.ckpt \
    --output_dir outputs/test
```

Outputs saved to `outputs/test/`:
- `per_material.csv` / `per_material.png` — per-class metrics for surface material
- `per_condition.csv` / `per_condition.png` — per-class metrics for surface condition
- `per_joint.csv` / `per_joint.png` — per-class metrics for joint (material × condition)

#### 2. Road Surface Elevation Estimation (RoadBEV)
```bash
cd project/road-surface-elevation-estimation/RoadBEV
export PYTHONPATH=$(pwd)

python -m scripts.test \
    --config project/config/model_config.py \
    --ckpt /path/to/checkpoint.ckpt \
    --output_dir outputs/test
```

Outputs saved to `outputs/test/`:
- `metrics.json` — full breakdown (overall, per material, per condition)
- `metrics.csv` — same results in CSV format

Reported metrics: MAE (cm), RMSE (cm), % frames with error > 0.5 / 1.0 / 2.0 cm
