import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.dataset.collate import model_collate_fn
from project.dataset.dataloader import ModelDataset
from project.dataset.dataclasses import RoadConditionClassMapper
from project.model.factory import build_model
from project.utils.metric import ClassificationMetrics, MATERIAL_LABELS, CONDITION_LABELS, JOINT_LABELS
from tools.utils import import_module_from_file, save_per_class_csv, save_per_class_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Classifier test: overall metrics")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--ckpt",        type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="outputs/test")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def load_checkpoint(manager, ckpt_path: Path):
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    # ckpt saved with full mobilenet model (backbone.features.*) but current code
    # returns model.features directly (backbone.*), so remap keys if needed
    if any("backbone.features." in k for k in state_dict) and not any("backbone.features." in k for k in manager.state_dict()):
        state_dict = {k.replace("backbone.features.", "backbone."): v for k, v in state_dict.items()}
        print("[ckpt] remapped backbone.features.* → backbone.*")
    missing, unexpected = manager.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[ckpt] missing sample:", missing[:5])
    if unexpected:
        print("[ckpt] unexpected sample:", unexpected[:5])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module = import_module_from_file("ClassifierConfig", str(Path(args.config).resolve()))
    config = module.ClassifierConfig()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ModelDataset(config=config, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=model_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    manager = build_model(config).to(device)
    load_checkpoint(manager, Path(args.ckpt).resolve())
    manager.eval()

    metrics = {
        "material": ClassificationMetrics(
            num_classes=RoadConditionClassMapper.N_SURFACE_MATERIAL,
            labels=MATERIAL_LABELS,
            track_cm=False,
            track_per_class=True,
        ).to(device),
        "condition": ClassificationMetrics(
            num_classes=RoadConditionClassMapper.N_SURFACE_STATE,
            labels=CONDITION_LABELS,
            track_cm=False,
            track_per_class=True,
        ).to(device),
        "joint": ClassificationMetrics(
            num_classes=RoadConditionClassMapper.NUM_CLASSES,
            labels=JOINT_LABELS,
            track_cm=False,
            track_per_class=True,
        ).to(device),
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="test", dynamic_ncols=True):
            features = _to_device(batch["features"], device)
            targets = _to_device(batch["targets"], device)
            outputs = manager.model(features)

            material_preds = outputs["material"].argmax(dim=1)
            condition_preds = outputs["condition"].argmax(dim=1)
            joint_preds = material_preds * RoadConditionClassMapper.N_SURFACE_STATE + condition_preds

            metrics["material"].update(material_preds, targets["material"])
            metrics["condition"].update(condition_preds, targets["condition"])
            metrics["joint"].update(joint_preds, targets["joint"])

    results = {task: metric.compute() for task, metric in metrics.items()}

    print("\n=== Overall ===")
    for task in ("condition", "material", "joint"):
        result = results[task]
        print(f"\n[{task}]")
        print(f"  Accuracy  : {float(result['acc']):.4f}")
        print(f"  F1 Macro  : {float(result['f1_macro']):.4f}")
        print(f"  Precision : {float(result['precision_macro']):.4f}")
        print(f"  Recall    : {float(result['recall_macro']):.4f}")

    for task, result in results.items():
        save_per_class_csv(result["per_class"], output_dir / f"per_{task}.csv")
        save_per_class_figure(result["per_class"], output_dir / f"per_{task}.png")
    print(f"\nOutput: {output_dir}")

if __name__ == "__main__":
    main()
