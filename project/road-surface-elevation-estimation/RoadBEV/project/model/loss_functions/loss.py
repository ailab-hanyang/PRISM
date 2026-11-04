from typing import Dict

import torch.nn.functional as F
import torch
from torch import nn

class MyLoss(nn.Module):
    def __init__(self, ele_range, voxel_ele_res, cla_res=1):
        super(MyLoss, self).__init__()
        self.ele_range = ele_range*100  # to cm
        if (self.ele_range*20) % (cla_res*10) != 0:
            print('The class interval is improper')
            exit()
        self.cla_res = cla_res   # in cm
        self.voxel_ele_res = voxel_ele_res*100  # in cm
        self.num_voxels_ele = int(self.ele_range*2 / self.voxel_ele_res)

        self.num_classes = int(2*self.ele_range/cla_res)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def label2class(self, ele_gt):
        # ele_gt: [N,]
        class_label = torch.floor((ele_gt + self.ele_range) / self.cla_res).type(torch.long)
        class_label = self.num_classes - class_label - 1
        # clamp to guard against float32 rounding at boundaries
        class_label = class_label.clamp(0, self.num_classes - 1)
        return class_label

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # ele_pred: [B, num_classes, H, W]  without softmax
        # ele_gt:   [B, H, W]
        # ele_mask: [B, H, W]

        ele_gt = targets["elevation_map"]
        ele_mask = targets["elevation_mask"]
        ele_pred = preds["elevation_map_pred"]

        ele_mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)

        ele_pred_permuted = ele_pred.permute(0, 2, 3, 1)
        ele_pred_masked = ele_pred_permuted[ele_mask, :]
        ele_gt_masked = ele_gt[ele_mask]

        # DDP: return zero loss tied to model params so the gradient reducer stays safe
        if ele_gt_masked.numel() == 0:
            return ele_pred.sum() * 0.0

        class_ele = self.label2class(ele_gt_masked)
        loss_ele = self.loss_func(ele_pred_masked, class_ele)

        return loss_ele
