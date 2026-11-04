from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

from project.dataset.dataclasses import RoadConditionClassMapper


MATERIAL_LABELS = RoadConditionClassMapper.SURFACE_MATERIAL
CONDITION_LABELS = RoadConditionClassMapper.SURFACE_STATE
JOINT_LABELS = [
    f"{material}/{condition}"
    for material in MATERIAL_LABELS
    for condition in CONDITION_LABELS
]


class ClassificationMetrics(nn.Module):
    def __init__(
        self,
        num_classes: int,
        labels: Optional[List[str]] = None,
        track_cm: bool = False,
        track_per_class: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.labels = labels or [str(i) for i in range(num_classes)]
        self.track_cm = track_cm
        self.track_per_class = track_per_class

        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        if track_per_class:
            self.f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average="none")
            self.precision_per_class = Precision(task="multiclass", num_classes=num_classes, average="none")
            self.recall_per_class = Recall(task="multiclass", num_classes=num_classes, average="none")
        if track_cm:
            self.cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def update(self, preds, labels):
        self.acc(preds, labels)
        self.f1(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        if self.track_per_class:
            self.f1_per_class(preds, labels)
            self.precision_per_class(preds, labels)
            self.recall_per_class(preds, labels)
        if self.track_cm:
            self.cm(preds, labels)

    def compute(self) -> dict:
        result = {
            "acc": self.acc.compute(),
            "f1_macro": self.f1.compute(),
            "precision_macro": self.precision.compute(),
            "recall_macro": self.recall.compute(),
        }
        if self.track_per_class:
            f1_pc = self.f1_per_class.compute().cpu()
            precision_pc = self.precision_per_class.compute().cpu()
            recall_pc = self.recall_per_class.compute().cpu()
            result["per_class"] = {
                self.labels[i]: {
                    "f1": float(f1_pc[i]),
                    "precision": float(precision_pc[i]),
                    "recall": float(recall_pc[i]),
                }
                for i in range(self.num_classes)
            }
        if self.track_cm:
            result["cm_figure"] = self._build_cm_figure()
        return result

    def _build_cm_figure(self) -> plt.Figure:
        cm = self.cm.compute().cpu().numpy()
        fig_size = max(6, min(18, self.num_classes * 0.8))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.labels,
            yticklabels=self.labels,
            xlabel="Predicted",
            ylabel="True",
        )
        tick_fontsize = 6 if self.num_classes > 12 else 8
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)
        fig.tight_layout()
        return fig

    def reset(self):
        self.acc.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        if self.track_per_class:
            self.f1_per_class.reset()
            self.precision_per_class.reset()
            self.recall_per_class.reset()
        if self.track_cm:
            self.cm.reset()


class MarginalMetrics(nn.Module):
    """Backward-compatible projection metrics for joint class IDs."""

    def __init__(self, axis: str):
        super().__init__()
        assert axis in ("state", "material"), f"axis must be 'state' or 'material', got {axis!r}"
        self.axis = axis
        if axis == "state":
            self.metrics = ClassificationMetrics(
                num_classes=RoadConditionClassMapper.N_SURFACE_STATE,
                labels=CONDITION_LABELS,
                track_per_class=True,
            )
        else:
            self.metrics = ClassificationMetrics(
                num_classes=RoadConditionClassMapper.N_SURFACE_MATERIAL,
                labels=MATERIAL_LABELS,
                track_per_class=True,
            )

    def _project(self, x):
        if self.axis == "state":
            return x % RoadConditionClassMapper.N_SURFACE_STATE
        return x // RoadConditionClassMapper.N_SURFACE_STATE

    def update(self, preds, labels):
        self.metrics.update(self._project(preds), self._project(labels))

    def compute(self) -> dict:
        return self.metrics.compute()["per_class"]

    def reset(self):
        self.metrics.reset()
