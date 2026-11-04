import random
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WandbVisualizationCallback(Callback):
    """Log polar-RGB input (once, with ROI cuboid) + BEV GT/pred comparison (each epoch).

    RGB is written to wandb.run.summary (not time-series) so it appears only once and
    does not accumulate with each epoch, avoiding the infinite-copy panel issue.
    BEV comparisons are logged once per epoch via on_validation_epoch_end with a single
    log() call (no explicit step), which prevents duplicate entries from per-batch calls.
    Samples are chosen by reservoir sampling (Knuth's Algorithm R) for uniform randomness.
    """

    # Cuboid edge connectivity. Corner index k = xi*4 + yi*2 + zi, xi/yi/zi in {0=min,1=max}.
    _CUBOID_EDGES = [
        (0, 4), (1, 5), (2, 6), (3, 7),   # x-edges
        (0, 2), (1, 3), (4, 6), (5, 7),   # y-edges
        (0, 1), (2, 3), (4, 5), (6, 7),   # z-edges (connect z_min↔z_max faces)
    ]

    def __init__(self, config, num_samples: int = 4, dpi: int = 100):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.dpi = dpi
        self.num_grids_x = int(config.num_grids_x)
        self.num_grids_y = int(config.num_grids_y)
        self.roi_x = list(config.roi_x)
        self.roi_y = list(config.roi_y)
        self.roi_z = list(config.roi_z)
        self.input_mode = str(getattr(config, "input_mode", "rgb_tri"))
        self._img_h = getattr(config, "input_h", 540)
        self._img_w = getattr(config, "input_w", 960)

        self._reservoir: List[Dict] = []       # reservoir-sampled validation data
        self._samples_seen = 0
        self._roi_corners_uv: Optional[np.ndarray] = None   # (8, 2) projected corners

    # ------------------------------------------------------------------
    # ROI cuboid helpers
    # ------------------------------------------------------------------

    def _try_init_roi_corners(self, trainer: pl.Trainer) -> None:
        """Lazy-init ROI corner projections from the first available calibration."""
        if self._roi_corners_uv is not None:
            return
        try:
            dm = trainer.datamodule
            if dm is None or not hasattr(dm, "train_dataset"):
                return
            cal = next(iter(dm.train_dataset.loader.calibrations.values()))
            self._roi_corners_uv = self._compute_roi_corners_uv(cal)
        except Exception:
            pass

    def _compute_roi_corners_uv(self, calibration) -> np.ndarray:
        """Project 8 ROI cuboid corners from reference frame to image pixel coords.

        Corner ordering: k = xi*4 + yi*2 + zi  (xi,yi,zi in {0,1} → min,max).
        Returns float32 (8, 2); corners behind the camera are set to nan.
        """
        if self.input_mode == "rgb":
            R = calibration.reference2rgb[:3, :]         # (3, 4)
            K = calibration.rgb_intrinsic.copy()
            crop_h = getattr(self.config, "rgb_crop_h", 1080)
            crop_w = getattr(self.config, "rgb_crop_w", 1920)
            sensor_h = getattr(self.config, "rgb_sensor_h", 2048)
            K[1, 2] -= sensor_h - crop_h                 # bottom-left crop: adjust cy
            K[0] *= self._img_w / crop_w                 # resize x scale
            K[1] *= self._img_h / crop_h                 # resize y scale
        else:
            R = calibration.reference2polar[:3, :]        # (3, 4)
            K = calibration.polar_intrinsic.copy()
            sensor_h = getattr(self.config, "polar_sensor_h", 1024)
            K[1, 2] -= sensor_h - self._img_h            # bottom-left crop: adjust cy

        corners_ref = np.array(
            [[x, y, z, 1.0] for x in self.roi_x for y in self.roi_y for z in self.roi_z],
            dtype=np.float32,
        ).T                                               # (4, 8)

        pts_cam = R @ corners_ref                         # (3, 8)
        uvz = K @ pts_cam                                 # (3, 8)
        uv = (uvz[:2] / uvz[2:]).T.astype(np.float32)    # (8, 2)
        uv[pts_cam[2] < 0.1] = np.nan                    # behind camera → skip
        return uv

    def _draw_roi_cuboid(self, image: np.ndarray) -> np.ndarray:
        """Draw 12 ROI cuboid edges on a uint8 HxWx3 RGB image.

        Edge colours by z-level:
          green  – z_min face edges
          cyan   – z_max face edges
          orange – vertical (z-connector) edges
        """
        H, W = image.shape[:2]
        img = image.copy()
        uv = self._roi_corners_uv
        for a, b in self._CUBOID_EDGES:
            ua, va, ub, vb = uv[a, 0], uv[a, 1], uv[b, 0], uv[b, 1]
            if any(np.isnan(c) for c in (ua, va, ub, vb)):
                continue
            zi_a, zi_b = a % 2, b % 2
            if zi_a != zi_b:
                color = (255, 165, 0)   # orange – z-connector
            elif zi_a == 0:
                color = (0, 255, 0)     # green  – z_min face
            else:
                color = (0, 220, 255)   # cyan   – z_max face
            cv2.line(
                img,
                (int(np.clip(ua, 0, W - 1)), int(np.clip(va, 0, H - 1))),
                (int(np.clip(ub, 0, W - 1)), int(np.clip(vb, 0, H - 1))),
                color, 2, cv2.LINE_AA,
            )
        return img

    # ------------------------------------------------------------------
    # PL callbacks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._reservoir = []
        self._samples_seen = 0
        self._try_init_roi_corners(trainer)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank != 0:
            return
        if not hasattr(pl_module, "_last_val_predictions") or \
           not hasattr(pl_module, "_last_val_targets"):
            return

        # _last_val_{predictions,targets} are already on CPU (set in RoadBEVManager._step)
        ele_pred_b = pl_module._last_val_predictions["elevation_map_pred"].numpy()
        ele_gt_b   = pl_module._last_val_targets["elevation_map"].numpy()
        ele_mask_b = pl_module._last_val_targets["elevation_mask"].numpy()
        rgb_b = (
            batch["features"]["image"] if self.input_mode == "rgb"
            else batch["features"]["polar"]["rgb_image"]
        ).detach().cpu()

        for i in range(ele_pred_b.shape[0]):
            self._samples_seen += 1
            sample = {
                "rgb":      rgb_b[i].clone(),
                "ele_pred": ele_pred_b[i].copy(),
                "ele_gt":   ele_gt_b[i].copy(),
                "ele_mask": ele_mask_b[i].copy(),
                "idx":      batch_idx * ele_pred_b.shape[0] + i,
            }
            # Reservoir sampling – Knuth's Algorithm R
            if len(self._reservoir) < self.num_samples:
                self._reservoir.append(sample)
            else:
                j = random.randint(0, self._samples_seen - 1)
                if j < self.num_samples:
                    self._reservoir[j] = sample

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.global_rank != 0 or not self._reservoir:
            return

        for slot, s in enumerate(self._reservoir):
            bev_img = self._create_bev_comparison(
                s["ele_gt"], s["ele_pred"], s["ele_mask"], s["idx"]
            )
            rgb = self._tensor_to_rgb(s["rgb"])
            if self._roi_corners_uv is not None:
                rgb = self._draw_roi_cuboid(rgb)

            caption = f"Epoch {trainer.current_epoch} | idx {s['idx']}"
            trainer.logger.experiment.log(
                {
                    f"val_vis/bev_{slot}":       wandb.Image(bev_img, caption=caption),
                    f"val_vis/input_rgb_{slot}": wandb.Image(rgb,     caption=caption),
                },
                step=trainer.global_step,
            )

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
        arr = image.permute(1, 2, 0).numpy()
        return np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    def _create_bev_comparison(
        self,
        ele_gt: np.ndarray,
        ele_pred: np.ndarray,
        ele_mask: np.ndarray,
        sample_idx: int,
    ) -> np.ndarray:
        mask = ele_mask.astype(bool)
        gt, pred = ele_gt.copy(), ele_pred.copy()
        gt[~mask] = np.nan
        gt_vis  = np.flip(gt,   axis=1)
        pred_vis = np.flip(pred, axis=1)

        finite = np.concatenate([
            gt_vis[np.isfinite(gt_vis)], pred_vis[np.isfinite(pred_vis)],
        ])
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if np.isclose(vmin, vmax):
                vmin -= 1.0; vmax += 1.0

        fig, axes = plt.subplots(1, 2, figsize=(11, 8), dpi=self.dpi)
        extent = [self.roi_y[0], self.roi_y[1], self.roi_x[0], self.roi_x[1]]
        kw = dict(extent=extent, origin="lower", aspect="equal",
                  cmap="coolwarm", vmin=vmin, vmax=vmax)

        axes[0].imshow(gt_vis,   **kw); axes[0].set_title("Ground Truth")
        im = axes[1].imshow(pred_vis, **kw); axes[1].set_title("Prediction")
        for ax in axes:
            ax.set_xlabel("lateral (m)")
            ax.set_ylabel("longitudinal (m)")

        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.16)
        cax = fig.add_axes([0.90, 0.22, 0.018, 0.56])
        fig.colorbar(im, cax=cax).set_label("elevation (cm)")
        fig.suptitle(f"BEV Elevation – Sample {sample_idx}")

        arr = self._fig_to_numpy(fig)
        plt.close(fig)
        return arr

    @staticmethod
    def _fig_to_numpy(fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        return buf.reshape(nrows, ncols, 4)[..., :3].copy()
