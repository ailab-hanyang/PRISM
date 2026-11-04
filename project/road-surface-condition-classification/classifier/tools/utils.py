import csv
from pathlib import Path
import shutil
import importlib.util
import sys
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def _serialize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(v) for v in value]
    return repr(value)


def config_to_dict(config: Any) -> Dict[str, Any]:
    config_dict: Dict[str, Any] = {}

    for key in dir(config):
        if key.startswith("_"):
            continue

        value = getattr(config, key)
        if callable(value):
            continue

        config_dict[key] = _serialize_config_value(value)

    return config_dict


def save_config_files(save_dir: Path, timestamp: str, config_file_path: str):

    save_dir.mkdir(parents=True, exist_ok=True)

    if Path(config_file_path).exists():
        dest_path = save_dir / f"{timestamp}_{Path(config_file_path).name}"
        shutil.copy2(config_file_path, dest_path)
        print(f"Config saved: {dest_path}")
    else:
        print(f"Config file not found: {config_file_path}")

    print(f"All config files saved to: {save_dir}")

def import_module_from_file(module_name: str, file_path: str):

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def save_per_class_csv(per_class: Dict[str, Dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "f1", "precision", "recall"])
        writer.writeheader()
        for cls_label, cls_m in per_class.items():
            writer.writerow({"class": cls_label, **cls_m})


def save_per_class_figure(per_class: Dict[str, Dict[str, float]], save_path: Path) -> None:
    labels     = list(per_class.keys())
    f1_vals    = [per_class[l]["f1"]        for l in labels]
    prec_vals  = [per_class[l]["precision"] for l in labels]
    rec_vals   = [per_class[l]["recall"]    for l in labels]

    n = len(labels)
    y = np.arange(n)
    bar_h = 0.25

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    ax.barh(y + bar_h,  prec_vals, bar_h, label="Precision", color="#4C72B0")
    ax.barh(y,          rec_vals,  bar_h, label="Recall",    color="#55A868")
    ax.barh(y - bar_h,  f1_vals,   bar_h, label="F1",        color="#C44E52")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
