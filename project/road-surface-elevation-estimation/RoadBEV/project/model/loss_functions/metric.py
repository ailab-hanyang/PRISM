import numpy as np
import matplotlib.pyplot as plt
from project.utils.experiment import make_nograd_func
from typing import Dict

import torch


class Metric():
    def __init__(self, ele_range, num_grids_z, distance_wise=False):
        self.ele_range = ele_range * 100  # in cm

        self.metric_all = np.zeros(5,)
        self.count_all = 0

        # if compute the distance-wise metric in the ROI grid
        self.distance_wise = distance_wise
        self.intervals = 11  # number of grids for every segment
        self.num_intervals = int(num_grids_z / self.intervals) + 1

        self.metric_wise = np.zeros((self.num_intervals, 5))
        self.count_wise = np.zeros(self.num_intervals)

    @make_nograd_func
    def clear(self):
        self.count_all = 0
        self.metric_all *= 0
        self.count_wise *= 0
        self.metric_wise *= 0

    @make_nograd_func
    def plot_depthwise(self, metric_depthwise):
        plt.figure()
        plt.subplot(121)
        plt.plot(np.flip(metric_depthwise[:, 0]), marker='*')
        plt.title('MAE')
        plt.subplot(122)
        plt.plot(np.flip(metric_depthwise[:, 1]), marker='*')
        plt.title('RMSE')
        plt.show()

    @make_nograd_func
    def get_metric(self):
        if self.count_all > 0:
            metric_all = self.metric_all / self.count_all
        else:
            metric_all = np.zeros_like(self.metric_all)

        if self.distance_wise:
            metric_wise = np.zeros_like(self.metric_wise)
            np.divide(
                self.metric_wise,
                self.count_wise.reshape(-1, 1),
                out=metric_wise,
                where=self.count_wise.reshape(-1, 1) > 0,
            )
            return [metric_all, metric_wise]
        else:
            return [metric_all, None]

    @make_nograd_func
    def compute_values(self, ele_gt: torch.Tensor, ele_pred: torch.Tensor):
        mae = torch.abs(ele_gt - ele_pred)
        rmse = torch.sqrt(((ele_gt - ele_pred) ** 2).mean())
        ratio_thresh_0_5 = torch.mean((mae > 0.5).float())
        ratio_thresh_1_0 = torch.mean((mae > 1.0).float())
        ratio_thresh_2_0 = torch.mean((mae > 2.0).float())

        return np.array([
            float(torch.mean(mae).item()),
            float(rmse.item()),
            float(ratio_thresh_0_5.item()),
            float(ratio_thresh_1_0.item()),
            float(ratio_thresh_2_0.item())
        ])

    @make_nograd_func
    def compute(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # ele_pred: [B, H, W]
        ele_gt = targets["elevation_map"]
        ele_mask = targets["elevation_mask"]
        ele_pred = preds["elevation_map_pred"]

        mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(mask_roi, ele_mask)

        gt_valid = ele_gt[ele_mask].to(torch.float32)
        if gt_valid.numel() == 0:
            return
        pred_valid = ele_pred[ele_mask].to(torch.float32)

        self.count_all += 1
        self.metric_all += self.compute_values(gt_valid, pred_valid)

        if self.distance_wise:
            for i in range(self.num_intervals):
                try:
                    ele_gt_ = ele_gt[:, i * self.intervals:(i + 1) * self.intervals, :]
                    ele_pred_ = ele_pred[:, i * self.intervals:(i + 1) * self.intervals, :]
                    ele_mask_ = ele_mask[:, i * self.intervals:(i + 1) * self.intervals, :]
                except Exception:
                    ele_gt_ = ele_gt[:, i * self.intervals:, :]
                    ele_pred_ = ele_pred[:, i * self.intervals:, :]
                    ele_mask_ = ele_mask[:, i * self.intervals:, :]

                gt_valid = ele_gt_[ele_mask_].to(torch.float32)
                if gt_valid.numel() == 0:
                    continue
                pred_valid = ele_pred_[ele_mask_].to(torch.float32)
                values = self.compute_values(gt_valid, pred_valid)
                self.metric_wise[i, :] += values
                self.count_wise[i] += 1
