"""
RoadBEV Track 2 Evaluation Script

Computes elevation estimation metrics on the val set:
  - Overall (all non-ignored elevation frames)
  - Per surface material  (only frames with valid classification labels)
  - Per surface condition/state  (only frames with valid classification labels)

Ignore rules (from convert_pkl.py):
  - ignore_elevation_label=True      → excluded from ALL evaluation
  - ignore_classification_label=True → excluded from material / condition breakdown only

Run from RoadBEV project root:
    python -m scripts.test \\
        --config project/config/model_config.py \\
        --ckpt   exp/RoadBEV/RoadBEV_rgb/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from project.dataset.dataloader import ModelDataset
from project.model.model import RoadBEV
from tools.utils import import_module_from_file


# ---------------------------------------------------------------------------
# Class definitions (must match convert_pkl.py)
# ---------------------------------------------------------------------------

MATERIAL_CLASSES   = ["asphalt", "concrete", "belgian_block", "gravel"]
CONDITION_CLASSES  = ["dry", "damp", "wet", "slush", "snow_covered"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str):
    module = import_module_from_file("RoadBEVModelConfig", str(Path(config_path).resolve()))
    return module.RoadBEVModelConfig()


def load_checkpoint(model: RoadBEV, ckpt_path: Path) -> None:
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(state_dict)}")
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded : {ckpt_path}")
    print(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[ckpt] missing sample:", missing[:10])
    if unexpected:
        print("[ckpt] unexpected sample:", unexpected[:10])


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_idx: int):
        return self.indices[local_idx], self.dataset[self.indices[local_idx]]


def _stack_nested(items):
    if torch.is_tensor(items[0]):
        return torch.stack(items, dim=0)
    if isinstance(items[0], dict):
        return {k: _stack_nested([item[k] for item in items]) for k in items[0]}
    return items


def _indexed_collate(batch):
    return [item[0] for item in batch], _stack_nested([item[1] for item in batch])


# ---------------------------------------------------------------------------
# Metric accumulator (frame-mean — identical to test_per_scenario.py)
# ---------------------------------------------------------------------------

def _new_acc() -> Dict[str, float]:
    return {
        "samples":            0.0,
        "valid_ratio_sum":    0.0,
        "valid_pixels":       0.0,
        "valid_batches":      0.0,
        "mae_batch_sum":      0.0,
        "rmse_batch_sum":     0.0,
        "over_0_5_batch_sum": 0.0,
        "over_1_0_batch_sum": 0.0,
        "over_2_0_batch_sum": 0.0,
    }


def _update_acc(
    acc: Dict[str, float],
    pred_cm: torch.Tensor,
    gt_cm: torch.Tensor,
    mask_raw: torch.Tensor,
    ele_range_cm: float,
) -> None:
    valid = (gt_cm > -ele_range_cm) & (gt_cm < ele_range_cm) & mask_raw.bool()
    total_pixels = int(valid.numel())
    valid_pixels = int(valid.sum().item())

    acc["samples"]         += 1.0
    acc["valid_ratio_sum"] += float(valid_pixels / max(total_pixels, 1))
    if valid_pixels <= 0:
        return

    diff    = (pred_cm - gt_cm)[valid].to(torch.float32)
    abs_err = diff.abs()

    acc["valid_pixels"]       += float(valid_pixels)
    acc["valid_batches"]      += 1.0
    acc["mae_batch_sum"]      += float(abs_err.mean().item())
    acc["rmse_batch_sum"]     += float(torch.sqrt((diff * diff).mean()).item())
    acc["over_0_5_batch_sum"] += float((abs_err > 0.5).float().mean().item())
    acc["over_1_0_batch_sum"] += float((abs_err > 1.0).float().mean().item())
    acc["over_2_0_batch_sum"] += float((abs_err > 2.0).float().mean().item())


def _finalize_acc(acc: Dict[str, float]) -> Dict[str, Any]:
    samples       = int(acc["samples"])
    valid_batches = int(acc["valid_batches"])
    out: Dict[str, Any] = {
        "samples":          samples,
        "valid_ratio_mean": float(acc["valid_ratio_sum"] / max(samples, 1)),
        "valid_pixels":     int(acc["valid_pixels"]),
        "mae_cm":           None,
        "rmse_cm":          None,
        "over_0_5cm_pct":   None,
        "over_1_0cm_pct":   None,
        "over_2_0cm_pct":   None,
    }
    if valid_batches > 0:
        out["mae_cm"]       = float(acc["mae_batch_sum"]      / valid_batches)
        out["rmse_cm"]      = float(acc["rmse_batch_sum"]     / valid_batches)
        out["over_0_5cm_pct"] = float(acc["over_0_5_batch_sum"] / valid_batches * 100.0)
        out["over_1_0cm_pct"] = float(acc["over_1_0_batch_sum"] / valid_batches * 100.0)
        out["over_2_0cm_pct"] = float(acc["over_2_0_batch_sum"] / valid_batches * 100.0)
    return out


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_HDR = f"  {'Group':<24} {'N':>6} {'MAE':>8} {'RMSE':>8} {'>0.5%':>7} {'>1.0%':>7} {'>2.0%':>7}"
_SEP = "  " + "-" * (len(_HDR) - 2)


def _fmt_val(v: Optional[float], w: int, decimals: int) -> str:
    if v is None:
        return "-".rjust(w)
    return f"{v:.{decimals}f}".rjust(w)


def _print_row(label: str, m: Dict[str, Any]) -> str:
    return (
        f"  {label:<24} {m['samples']:>6}"
        + _fmt_val(m["mae_cm"],       8, 4)
        + _fmt_val(m["rmse_cm"],      8, 4)
        + _fmt_val(m["over_0_5cm_pct"], 7, 2)
        + _fmt_val(m["over_1_0cm_pct"], 7, 2)
        + _fmt_val(m["over_2_0cm_pct"], 7, 2)
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoadBEV Track 2 evaluation")
    parser.add_argument("--config",      type=str, default="project/config/model_config.py")
    parser.add_argument("--ckpt",        type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="outputs/test")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    dataset = ModelDataset(config=config, mode="test")
    model   = RoadBEV(config=config).to(device)
    load_checkpoint(model, Path(args.ckpt).resolve())
    model.eval()

    ele_range_cm = float(config.ele_range * 100.0)

    # Build index lists and per-frame metadata needed for breakdowns.
    # ignore_elevation_label=True  → skip entirely
    # ignore_classification_label  → skip for material/condition breakdowns only
    elev_indices: List[int] = []
    for idx, meta in enumerate(dataset.loader.data):
        if not meta.get("ignore_elevation_label", False):
            elev_indices.append(idx)

    print(f"Total frames for evaluation: {len(elev_indices)} "
          f"(ignored elevation: {len(dataset.loader.data) - len(elev_indices)})")

    # Accumulators
    overall_acc: Dict[str, float] = _new_acc()
    mat_accs:  Dict[int, Dict[str, float]] = {i: _new_acc() for i in range(len(MATERIAL_CLASSES))}
    cond_accs: Dict[int, Dict[str, float]] = {i: _new_acc() for i in range(len(CONDITION_CLASSES))}

    loader = DataLoader(
        IndexedSubset(dataset, elev_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_indexed_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    with torch.no_grad():
        for batch_indices, batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            preds = model(_to_device(batch["features"], device))["elevation_map_pred"].detach().cpu()
            gts   = batch["targets"]["elevation_map"]
            masks = batch["targets"]["elevation_mask"]

            for i, idx in enumerate(batch_indices):
                pred = preds[i]
                gt   = gts[i]
                mask = masks[i]
                meta = dataset.loader.data[idx]

                # Overall (elevation label is valid by construction of elev_indices)
                _update_acc(overall_acc, pred, gt, mask, ele_range_cm)

                # Material / condition breakdown
                if meta.get("ignore_classification_label", True):
                    continue

                mat_idx  = meta.get("material_label")
                cond_idx = meta.get("condition_label")

                if mat_idx is not None and mat_idx in mat_accs:
                    _update_acc(mat_accs[mat_idx], pred, gt, mask, ele_range_cm)

                if cond_idx is not None and cond_idx in cond_accs:
                    _update_acc(cond_accs[cond_idx], pred, gt, mask, ele_range_cm)

    # Finalize
    overall_m = _finalize_acc(overall_acc)
    mat_m     = {MATERIAL_CLASSES[i]:  _finalize_acc(acc) for i, acc in mat_accs.items()}
    cond_m    = {CONDITION_CLASSES[i]: _finalize_acc(acc) for i, acc in cond_accs.items()}

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"Config : {args.config}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"{'=' * 72}")

    print(f"\n[Overall]  (N={overall_m['samples']})")
    print(f"  MAE  (cm)   : {overall_m['mae_cm']}")
    print(f"  RMSE (cm)   : {overall_m['rmse_cm']}")
    print(f"  >0.5cm (%)  : {overall_m['over_0_5cm_pct']}")
    print(f"  >1.0cm (%)  : {overall_m['over_1_0cm_pct']}")
    print(f"  >2.0cm (%)  : {overall_m['over_2_0cm_pct']}")

    print(f"\n[Per Surface Material]  (ignore_classification_label=False only)")
    print(_HDR)
    print(_SEP)
    for name in MATERIAL_CLASSES:
        print(_print_row(name, mat_m[name]))

    print(f"\n[Per Surface Condition]  (ignore_classification_label=False only)")
    print(_HDR)
    print(_SEP)
    for name in CONDITION_CLASSES:
        print(_print_row(name, cond_m[name]))

    print(f"\n{'=' * 72}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    report = {
        "config_path": str(Path(args.config).resolve()),
        "ckpt_path":   str(Path(args.ckpt).resolve()),
        "device":      str(device),
        "overall":     overall_m,
        "per_material":  mat_m,
        "per_condition": cond_m,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV: one row per group
    fieldnames = ["group", "category", "samples", "valid_pixels",
                  "mae_cm", "rmse_cm",
                  "over_0_5cm_pct", "over_1_0cm_pct", "over_2_0cm_pct"]

    def _csv_row(group: str, category: str, m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "group":           group,
            "category":        category,
            "samples":         m["samples"],
            "valid_pixels":    m["valid_pixels"],
            "mae_cm":          m["mae_cm"],
            "rmse_cm":         m["rmse_cm"],
            "over_0_5cm_pct":  m["over_0_5cm_pct"],
            "over_1_0cm_pct":  m["over_1_0cm_pct"],
            "over_2_0cm_pct":  m["over_2_0cm_pct"],
        }

    csv_rows = [_csv_row("overall", "overall", overall_m)]
    for name in MATERIAL_CLASSES:
        csv_rows.append(_csv_row("material", name, mat_m[name]))
    for name in CONDITION_CLASSES:
        csv_rows.append(_csv_row("condition", name, cond_m[name]))

    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[Output]")
    print(f"  {output_dir / 'metrics.json'}")
    print(f"  {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
