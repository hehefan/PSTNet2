import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

import pointnet2_utils
from pst_operations import *

class PointNet2(nn.Module):
    def __init__(self, radius=0.9, nsamples=16, num_classes=12):
        super(PointNet2, self).__init__()

        self.encoder1  = PSTOp(in_channels=3,
                               spatial_radius=radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[32, 32, 128],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder2  = PSTOp(in_channels=128,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[64, 64, 256],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder3  = PSTOp(in_channels=256,
                               spatial_radius=4*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=4,
                               spatial_channels=[128, 128, 512],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])

        self.encoder4  = PSTOp(in_channels=512,
                               spatial_radius=8*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[256, 256, 1024],
                               spatial_batch_norm=[True, True, True],
                               spatial_activation=[True, True, True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[],
                               temporal_batch_norm=[],
                               temporal_activation=[])


        self.decoder4 = PSTTransOp(in_channels=1024,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[256, 256],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=512)

        self.decoder3 = PSTTransOp(in_channels=256,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[256, 256],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=256)

        self.decoder2 = PSTTransOp(in_channels=256,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[256, 128],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=128)

        self.decoder1 = PSTTransOp(in_channels=128,
                                   temporal_radius=0,
                                   temporal_stride=1,
                                   temporal_padding=[0, 0],
                                   temporal_channels=[],
                                   temporal_batch_norm=[],
                                   temporal_activation=[],
                                   spatial_channels=[128, 128],
                                   spatial_batch_norm=[True, True],
                                   spatial_activation=[True, True],
                                   original_channels=3)

        self.outconv = nn.Conv2d(in_channels=128, out_channels=12, kernel_size=1, stride=1, padding=0)

    def forward(self, xyzs, rgbs):

        new_xyzs1, new_features1 = self.encoder1(xyzs, rgbs)

        new_xyzs2, new_features2 = self.encoder2(new_xyzs1, new_features1)

        new_xyzs3, new_features3 = self.encoder3(new_xyzs2, new_features2)

        new_xyzs4, new_features4 = self.encoder4(new_xyzs3, new_features3)


        new_xyzsd4, new_featuresd4 = self.decoder4(new_xyzs4, new_xyzs3, new_features4, new_features3)

        new_xyzsd3, new_featuresd3 = self.decoder3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)

        new_xyzsd2, new_featuresd2 = self.decoder2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)

        new_xyzsd1, new_featuresd1 = self.decoder1(new_xyzsd2, xyzs, new_featuresd2, rgbs)


        out = self.outconv(new_featuresd1.transpose(1,2)).transpose(1,2)

        return out

