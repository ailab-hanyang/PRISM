from __future__ import print_function
from re import M
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, StepLR, CosineAnnealingLR, LinearLR, SequentialLR

import math
from typing import Dict, Union

from project.model.head.ele_head import *
from project.model.backbone.efficientnet import EfficientNetFeature
from project.template.model import AbstractLightningModule
from project.config.model_config import RoadBEVModelConfig
from project.model.loss_functions.loss import MyLoss
from project.model.loss_functions.metric import Metric
from project.model.utils.input import extract_input, _INPUT_MODE_CHANNELS


# ---------------------------------------------------------------------------
# Pretrained weight helpers
# ---------------------------------------------------------------------------

def _load_classifier_backbone_weights(feature_extraction: nn.Module, ckpt_path: str,
                                      label: str = "", prefix: str = "model.backbone.") -> None:
    """Load EfficientNet backbone weights from a classifier Lightning checkpoint.

    Supports two Track 1 Classifier conventions:
        self.backbone = model.features
        → keys: model.backbone.N.*

    The two are detected by inspecting the first stripped key and aligned to the
    RoadBEV target's ModuleList naming (`features.N.*`) before loading.

    Pass prefix="model.polar_backbone." to load the polar encoder weights.
    """
    import os
    tag = f"[Backbone{':' + label if label else ''}]"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{tag} Checkpoint not found: {ckpt_path}")

    print(f"{tag} Loading classifier pretrained weights from:\n  {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state = ckpt.get("state_dict", ckpt)

    PREFIX = prefix
    if not any(k.startswith(PREFIX) for k in full_state):
        available = sorted({".".join(k.split(".")[:3]) for k in full_state})
        raise RuntimeError(
            f"{tag} No keys with prefix '{PREFIX}' found.\n"
            f"  Available top-level keys: {available}"
        )

    # Strip "model.backbone." prefix; let strict=False handle unused keys (stages 5-8, head)
    backbone_state = {k[len(PREFIX):]: v for k, v in full_state.items() if k.startswith(PREFIX)}

    # Align NEW Track 1 naming ('N.*') to target's 'features.N.*' by prepending 'features.'.
    sample_key = next(iter(backbone_state.keys()), "")
    if sample_key and not sample_key.startswith("features."):
        backbone_state = {f"features.{k}": v for k, v in backbone_state.items()}
        print(f"{tag} Detected NEW Track 1 (spatial-fusion) naming; prepended 'features.' to {len(backbone_state)} keys.")

    # Handle first-conv channel mismatch (e.g. classifier trained on RGB=3ch, RoadBEV needs 12ch)
    stem_key = "features.0.0.weight"
    if stem_key in backbone_state:
        src_w   = backbone_state[stem_key]
        dst_conv = feature_extraction.features[0][0]
        if src_w.shape != dst_conv.weight.shape:
            src_ch, dst_ch = src_w.shape[1], dst_conv.weight.shape[1]
            print(f"{tag} Stem channel mismatch: classifier={src_ch}ch, RoadBEV={dst_ch}ch. "
                  f"Copying first {min(src_ch, dst_ch)} channels; rest zero-initialized.")
            new_w = torch.zeros_like(dst_conv.weight)
            new_w[:, :min(src_ch, dst_ch)] = src_w[:, :min(src_ch, dst_ch)]
            backbone_state[stem_key] = new_w

    missing, unexpected = feature_extraction.load_state_dict(backbone_state, strict=False)
    backbone_missing = [k for k in missing if not k.startswith("final_conv")]
    if backbone_missing:
        print(f"{tag} Missing backbone keys ({len(backbone_missing)}): {backbone_missing[:5]}")
    loaded = len(backbone_state) - len(unexpected)
    print(f"{tag} Loaded {loaded}/{len(backbone_state)} keys from classifier checkpoint.")



def _build_backbone(backbone_type: str, in_channels: int) -> nn.Module:
    """Instantiate a feature extractor by backbone_type.

    Supported values:
        "efficientnet_b3" | "efficientnet_b4" | "efficientnet_b5" | "efficientnet_b6"
        "resnet50"
    """
    if backbone_type.startswith("efficientnet_"):
        variant = backbone_type.split("_", 1)[1]  # "b3", "b4", ...
        return EfficientNetFeature(variant=variant, in_channels=in_channels)
    elif backbone_type == "resnet50":
        return ResNet50Feature(in_channels=in_channels)
    else:
        raise ValueError(
            f"[Backbone] Unknown backbone_type: '{backbone_type}'. "
            "Use 'efficientnet_b3/b4/b5/b6' or 'resnet50'."
        )


class RoadBEV(nn.Module):
    def __init__(self, config: RoadBEVModelConfig):
        super(RoadBEV, self).__init__()
        self.config = config
        self.num_grids_x, self.num_grids_y, self.num_grids_z = config.num_grids_x, config.num_grids_y, config.num_grids_z
        self.num_grids = [self.num_grids_x, self.num_grids_y, self.num_grids_z]
        self.ele_range = config.ele_range   # in meter
        self.cla_res = config.cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32)*self.cla_res + self.ele_range*100 - self.cla_res/2
        self.register_buffer('ele_values', ele_values.reshape(1, self.num_classes, 1, 1))
        
        input_mode = str(getattr(self.config, "input_mode", "rgb_tri"))
        if input_mode not in _INPUT_MODE_CHANNELS:
            raise ValueError(
                f"Unsupported input_mode: '{input_mode}'. "
                f"Choose from: {list(_INPUT_MODE_CHANNELS)}"
            )
        in_channels = _INPUT_MODE_CHANNELS[input_mode]

        backbone_type = str(getattr(self.config, "backbone_type", "efficientnet_b6"))
        self._two_stream = input_mode in ("rgb_tri", "rgb_mono")

        if self._two_stream:
            polar_in_ch = 9 if input_mode == "rgb_tri" else 3
            self.feature_extraction_rgb = _build_backbone(backbone_type, 3)
            self.feature_extraction_polar = _build_backbone(backbone_type, polar_in_ch)
            feat_ch = self.feature_extraction_rgb.feat_channel
            self.fusion_proj = nn.Conv2d(2 * feat_ch, feat_ch, kernel_size=1)
        else:
            self.feature_extraction = _build_backbone(backbone_type, in_channels)
            feat_ch = self.feature_extraction.feat_channel

        # Mono only - regressor for monocular camera
        self.ele_head = EleCla2D(feat_ch, self.num_grids, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # Load pretrained backbone weights (after random init)
        if getattr(self.config, "use_backbone_pretrained", False):
            ckpt_path = getattr(self.config, "backbone_pretrained_path", None)
            if self._two_stream:
                if ckpt_path:
                    _load_classifier_backbone_weights(self.feature_extraction_rgb, str(ckpt_path), label="RGB",
                                                      prefix="model.backbone.")
                    _load_classifier_backbone_weights(self.feature_extraction_polar, str(ckpt_path), label="Polar",
                                                      prefix="model.polar_backbone.")
                else:
                    self.feature_extraction_rgb.load_imagenet_weights()
                    print("[Backbone:Polar] No pretrained path set; polar encoder stays randomly initialized.")
            else:
                if ckpt_path:
                    _load_classifier_backbone_weights(self.feature_extraction, str(ckpt_path))
                elif input_mode == "rgb":
                    self.feature_extraction.load_imagenet_weights()
                else:
                    raise ValueError(
                        "[Backbone] use_backbone_pretrained=True but "
                        "backbone_pretrained_path is not set. "
                        "Set backbone_pretrained_path in config or "
                        "set use_backbone_pretrained=False."
                    )

    def forward(self, input_features: Dict[str, torch.Tensor]):
        proj_index = input_features["voxel_uv"]
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]

        if self._two_stream:
            p = input_features.get("polar", {})
            x_rgb = p["rgb_image"]                               # (B, 3, H, W)
            x_polar = torch.cat([p["dolp"], p["aolp"]], dim=1)  # (B, 9, H, W)
            f_rgb = self.feature_extraction_rgb(x_rgb)
            f_polar = self.feature_extraction_polar(x_polar)
            features = self.fusion_proj(torch.cat([f_rgb, f_polar], dim=1))
            spatial_h, spatial_w = x_rgb.shape[2], x_rgb.shape[3]
        else:
            feature_input = extract_input(input_features, self.config.input_mode)
            features = self.feature_extraction(feature_input)
            spatial_h, spatial_w = feature_input.shape[2], feature_input.shape[3]

        B, C, H, W = features.shape
        features = features.reshape(B, C, -1)

        # Legacy-style projection indexing:
        # proj_index is image-space (crop+resize) and is scaled here to feature-map space.
        scale_h = H / spatial_h
        scale_w = W / spatial_w
        scaled_idx_x = (proj_index[:, 0, :].float() * scale_w).long().clamp(0, W - 1)
        scaled_idx_y = (proj_index[:, 1, :].float() * scale_h).long().clamp(0, H - 1)
        linear_indices = scaled_idx_y * W + scaled_idx_x

        voxel_feat = features.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))

        # voxel_centers are flattened in (X, Y, Z) order, so recover the same layout.
        voxel_feat = voxel_feat.reshape(B, C, self.num_grids_x, self.num_grids_y, self.num_grids_z)

        # Standard LiDAR coords: X=front, Y=left, Z=up

        ele_pred = self.ele_head(voxel_feat) 

        if not self.training:
            ele_pred = F.softmax(ele_pred, dim=1)
            ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

            # pred_class = torch.max(ele_pred.data, 1)[1]
            # ele_pred = self.ele_values[pred_class.type(torch.long)]
        
        predictions = {
            "elevation_map_pred": ele_pred
        }

        return predictions


class RoadBEVManager(AbstractLightningModule):
    def __init__(self, model: nn.Module, config: RoadBEVModelConfig):
        super().__init__()
        self.config = config
        self.model = model
        
        self.loss_fn = MyLoss(
            ele_range=self.config.ele_range,
            voxel_ele_res=self.config.grid_resolution_z,
            cla_res=self.config.cla_res
        )
        
        self.metric_fn = Metric(
            ele_range=self.config.ele_range,
            num_grids_z=self.config.num_grids_z,
            distance_wise=False
        )

    def _step(self, batch: Dict[str, torch.Tensor], logging_prefix: str) -> torch.Tensor:
        features, targets = batch["features"], batch["targets"]
        
        predictions = self.model(features)  # Example forward pass
        
        if logging_prefix == "train":
            loss = self.compute_loss(targets, predictions)
            self.log(f"{logging_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        elif logging_prefix in {"val", "test"}:
            metric = self.compute_metric(targets, predictions)
            if logging_prefix == "val":
                self._last_val_predictions = {k: v.detach().cpu() for k, v in predictions.items()}
                self._last_val_targets = {k: v.detach().cpu() for k, v in targets.items()}
            return metric

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.model.train()
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.model.eval()
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.model.eval()
        return self._step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self.metric_fn.clear()

    def on_validation_epoch_end(self) -> None:
        if self.metric_fn.count_all == 0:
            return

        metric_all, _ = self.metric_fn.get_metric()
        mae = float(metric_all[0])
        rmse = float(metric_all[1])
        over_0_5cm_pct = float(metric_all[2] * 100.0)
        over_1_0cm_pct = float(metric_all[3] * 100.0)
        over_2_0cm_pct = float(metric_all[4] * 100.0)

        # Legacy-style validation metrics:
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_over_0_5cm_pct", over_0_5cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_over_1_0cm_pct", over_1_0cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_over_2_0cm_pct", over_2_0cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metric_fn.clear()

    def on_test_epoch_start(self) -> None:
        self.metric_fn.clear()

    def on_test_epoch_end(self) -> None:
        if self.metric_fn.count_all == 0:
            return

        metric_all, _ = self.metric_fn.get_metric()
        mae = float(metric_all[0])
        rmse = float(metric_all[1])
        over_0_5cm_pct = float(metric_all[2] * 100.0)
        over_1_0cm_pct = float(metric_all[3] * 100.0)
        over_2_0cm_pct = float(metric_all[4] * 100.0)

        self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_over_0_5cm_pct", over_0_5cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_over_1_0cm_pct", over_1_0cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_over_2_0cm_pct", over_2_0cm_pct, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metric_fn.clear()

    def compute_loss(
        self,
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        
        loss = self.loss_fn(predictions, targets)
        return loss
    
    def compute_metric(
        self,
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the metric used for validation based on the features, targets and model predictions.
        """
        self.metric_fn.compute(predictions, targets)
        metric_values = self.metric_fn.get_metric()
        return metric_values[0]

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Returns the optimizers that are used by the pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        # Build parameter groups with optional differential LR for backbone vs head.
        # Differential LR only applies when use_backbone_pretrained=True,
        # since backbone_lr_scale is meant to protect pretrained features.
        base_lr = self.config.learning_rate
        backbone_lr_scale = float(getattr(self.config, "backbone_lr_scale", 1.0))
        use_pretrained = bool(getattr(self.config, "use_backbone_pretrained", False))
        if use_pretrained and backbone_lr_scale != 1.0:
            if self.model._two_stream:
                backbone_params = list(self.model.feature_extraction_rgb.parameters()) + \
                                  list(self.model.feature_extraction_polar.parameters())
            else:
                backbone_params = list(self.model.feature_extraction.parameters())
            param_groups = [
                {"params": backbone_params,
                 "lr": base_lr * backbone_lr_scale,
                 "name": "backbone"},
                {"params": self.model.ele_head.parameters(),
                 "lr": base_lr,
                 "name": "head"},
            ]
        else:
            param_groups = self.model.parameters()

        # Create optimizer based on config
        if self.config.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")

        # Create scheduler based on config
        if self.config.scheduler_type.lower() == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.config.step_size,
                gamma=self.config.step_gamma
            )
        elif self.config.scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.cosine_T_max,
                eta_min=self.config.cosine_eta_min
            )
        elif self.config.scheduler_type.lower() == "onecycle":
            # OneCycleLR requires total_steps
            # FIXED: Use trainer.estimated_stepping_batches for correct total_steps calculation
            # This matches the original RoadBEV: steps_per_epoch * epochs
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.onecycle_max_lr,
                total_steps=total_steps,
                pct_start=self.config.onecycle_pct_start,
                div_factor=20,  # matches original RoadBEV
                three_phase=False,  # matches original RoadBEV
                anneal_strategy='linear'  # matches original RoadBEV
            )
        elif self.config.scheduler_type.lower() == "warmup_cosine":
            # LinearWarmup + CosineAnnealingLR (step-based)
            total_steps = self.trainer.estimated_stepping_batches
            warmup_epochs = int(getattr(self.config, "warmup_epochs", 10))
            steps_per_epoch = max(1, total_steps // self.config.epochs)
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = max(1, total_steps - warmup_steps)

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1.0 / 100,   # start: base_lr / 100
                end_factor=1.0,             # end: base_lr
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.config.cosine_eta_min,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.config.scheduler_type}")

        # warmup_cosine and onecycle must be updated every step
        step_based = self.config.scheduler_type.lower() in ("onecycle", "warmup_cosine")
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step" if step_based else "epoch",
            "frequency": 1,
        }

        # For schedulers that need monitoring (e.g., ReduceLROnPlateau)
        if self.config.scheduler_type.lower() not in ["onecycle", "step"]:
            lr_scheduler_config["monitor"] = "val_metric_epoch"

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

