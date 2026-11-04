from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from project.template.model import AbstractLightningModule
from project.dataset.dataclasses import RoadConditionClassMapper
from project.model.utils.input import extract_input
from project.utils.optim import build_optimizer, build_scheduler
from project.utils.metric import ClassificationMetrics, MATERIAL_LABELS, CONDITION_LABELS, JOINT_LABELS


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out_channels: int,
                 num_material_classes: int, num_condition_classes: int,
                 head_dropout: float = 0.2,
                 input_mode: str = "rgb",
                 polar_backbone: Optional[nn.Module] = None):
        super().__init__()
        self._input_mode = input_mode
        self._two_stream = polar_backbone is not None
        self.backbone = backbone
        if self._two_stream:
            self.polar_backbone = polar_backbone
            self.fusion_proj = nn.Conv2d(2 * backbone_out_channels, backbone_out_channels, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.material_head = ClassificationHead(backbone_out_channels, num_material_classes, head_dropout)
        self.condition_head = ClassificationHead(backbone_out_channels, num_condition_classes, head_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features: Dict) -> Dict[str, torch.Tensor]:
        if self._two_stream:
            p = features.get("polar", {})
            x_rgb = p["rgb_image"]
            x_polar = torch.cat([p["dolp"], p["aolp"]], dim=1)
            f_rgb = self.backbone(x_rgb)         # (B, C, H, W)
            f_polar = self.polar_backbone(x_polar)  # (B, C, H, W)
            fused = self.fusion_proj(torch.cat([f_rgb, f_polar], dim=1))  # (B, C, H, W)
        else:
            x = extract_input(features, self._input_mode)
            fused = self.backbone(x)  # (B, C, H, W)
        feat = self.avgpool(fused).flatten(1)  # (B, C)
        return {
            "material": self.material_head(feat),
            "condition": self.condition_head(feat),
        }


class ClassifierManager(AbstractLightningModule):
    def __init__(self, model: Classifier, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.eval_metrics = nn.ModuleDict({
            "material": ClassificationMetrics(
                num_classes=RoadConditionClassMapper.N_SURFACE_MATERIAL,
                labels=MATERIAL_LABELS,
                track_cm=True,
            ),
            "condition": ClassificationMetrics(
                num_classes=RoadConditionClassMapper.N_SURFACE_STATE,
                labels=CONDITION_LABELS,
                track_cm=True,
            ),
            "joint": ClassificationMetrics(
                num_classes=RoadConditionClassMapper.NUM_CLASSES,
                labels=JOINT_LABELS,
                track_cm=True,
            ),
        })

    def _joint_labels(self, material: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return material * RoadConditionClassMapper.N_SURFACE_STATE + condition

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        material_loss = self.criterion(outputs["material"], targets["material"])
        condition_loss = self.criterion(outputs["condition"], targets["condition"])
        return {
            "material": material_loss,
            "condition": condition_loss,
            "total": material_loss + condition_loss,
        }

    def _eval_step(self, batch: Dict) -> None:
        features, targets = batch["features"], batch["targets"]
        outputs = self.model(features)

        material_preds = outputs["material"].argmax(dim=1)
        condition_preds = outputs["condition"].argmax(dim=1)
        joint_preds = self._joint_labels(material_preds, condition_preds)

        self.eval_metrics["material"].update(material_preds, targets["material"])
        self.eval_metrics["condition"].update(condition_preds, targets["condition"])
        self.eval_metrics["joint"].update(joint_preds, targets["joint"])

    def _log_eval_metrics(self, prefix: str):
        for task, metrics in self.eval_metrics.items():
            result = metrics.compute()
            self.log(f"{prefix}/{task}/acc", result["acc"], prog_bar=True, sync_dist=True)
            self.log(f"{prefix}/{task}/f1_score", result["f1_macro"], prog_bar=False, sync_dist=True)
            self.log(f"{prefix}/{task}/precision", result["precision_macro"], prog_bar=False, sync_dist=True)
            self.log(f"{prefix}/{task}/recall", result["recall_macro"], prog_bar=False, sync_dist=True)
            if self.logger:
                self.logger.experiment.log(
                    {f"{prefix}/{task}/confusion_matrix": result["cm_figure"]},
                    step=self.trainer.global_step,
                )
                plt.close(result["cm_figure"])
            metrics.reset()

    def training_step(self, batch, batch_idx):
        features, targets = batch["features"], batch["targets"]
        outputs = self.model(features)
        losses = self._compute_losses(outputs, targets)
        self.log("train/loss", losses["total"], prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/material/loss", losses["material"], prog_bar=False, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/condition/loss", losses["condition"], prog_bar=False, sync_dist=True, on_step=True, on_epoch=True)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch)

    def on_validation_epoch_end(self):
        self._log_eval_metrics("val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch)

    def on_test_epoch_end(self):
        self._log_eval_metrics("test")

    def compute_loss(self, features, targets, predictions):
        return self._compute_losses(predictions, targets)["total"]

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model.parameters(), self.config)
        scheduler_dict = build_scheduler(optimizer, self.config)

        if scheduler_dict is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
