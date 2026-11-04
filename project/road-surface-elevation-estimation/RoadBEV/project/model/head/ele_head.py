
import torch.nn as nn
import torch.nn.functional as F
from project.model.submodules.submodule import *
from project.model.backbone.efficientnet import efficientnet_cla

class EleCla2D(nn.Module):
    def __init__(self, feat_channel, num_grids, num_classes):
        super(EleCla2D, self).__init__()
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.feat_channel = feat_channel

        # Collapse Z into channel so output spatial plane stays (X, Y), matching GT elevation map.
        self.channel_reshaped = self.feat_channel * self.num_grids_z
        self.num_classes = num_classes

        self.inplanes = int(self.channel_reshaped/8)
        self.first_conv = nn.Sequential(
                        convbn(self.channel_reshaped, self.inplanes, 1, 1, 0, 1),
                        convbn(self.inplanes, self.inplanes, 5, 1, 2, 1),
                        nn.ReLU(inplace=True))

        self.effnet_reg = efficientnet_cla(self.inplanes, self.num_classes)

        self.final_conv = nn.Sequential(convbn(self.num_classes, self.num_classes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1,
                                                  padding=0, bias=False))

    def forward(self, feat_voxel):
        # feat_voxel: [B, C, X, Y, Z] -> [B, Z*C, X, Y]
        B, C, X, Y, Z = feat_voxel.shape
        feat_bev = feat_voxel.permute(0, 4, 1, 2, 3).contiguous().reshape(B, Z * C, X, Y)

        feat_bev = self.first_conv(feat_bev)
        feat_bev = self.effnet_reg(feat_bev)
        ele_cla_prob = self.final_conv(feat_bev)  # [B, num_classes, X, Y]

        return ele_cla_prob
