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

from pst_operations import PSTOp

class MSRAction(nn.Module):
    def __init__(self, radius=0.5, nsamples=3*3, num_classes=20):
        super(MSRAction, self).__init__()

        self.encoder1  = PSTOp(in_channels=0,
                               spatial_radius=radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[45],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_channels=[64],
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder2a = PSTOp(in_channels=64,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[96],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_padding=[1,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[128],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder2b = PSTOp(in_channels=128,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[192],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[1,1],
                               temporal_padding_mode="replicate",
                               temporal_channels=[256],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder3a = PSTOp(in_channels=256,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[384],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_padding=[1,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[512],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder3b = PSTOp(in_channels=512,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[768],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[1,1],
                               temporal_padding_mode="replicate",
                               temporal_channels=[1024],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.encoder4  = PSTOp(in_channels=1024,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[1536],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="replicate",
                               temporal_channels=[2048],
                               temporal_batch_norm=[False],
                               temporal_activation=[False])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.encoder1(xyzs, None)

        new_xys, new_features = self.encoder2a(new_xys, new_features)

        new_xys, new_features = self.encoder2b(new_xys, new_features)

        new_xys, new_features = self.encoder3a(new_xys, new_features)

        new_xys, new_features = self.encoder3b(new_xys, new_features)

        _, new_features = self.encoder4(new_xys, new_features)                      # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)        # (B, L, C)
        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]        # (B, C)

        out = self.fc(new_feature)

        return out

class NTU(nn.Module):
    def __init__(self, radius=0.1, nsamples=3*3, num_classes=60):
        super(NTU, self).__init__()

        self.encoder1  = PSTOp(in_channels=0,
                               spatial_radius=radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[45],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_channels=[64],
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.encoder2a = PSTOp(in_channels=64,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[96],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_channels=[128],
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.encoder2b = PSTOp(in_channels=128,
                               spatial_radius=2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[192],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_channels=[256],
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.encoder3a = PSTOp(in_channels=256,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[384],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=2,
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_channels=[512],
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.encoder3b = PSTOp(in_channels=512,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=1,
                               spatial_channels=[768],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=1,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_channels=[1024],
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.encoder4  = PSTOp(in_channels=1024,
                               spatial_radius=2*2*radius,
                               spatial_neighbours=nsamples,
                               spatial_sampling=2,
                               spatial_channels=[1536],
                               spatial_batch_norm=[True],
                               spatial_activation=[True],
                               temporal_radius=0,
                               temporal_stride=1,
                               temporal_padding=[0,0],
                               temporal_padding_mode="zeros",
                               temporal_channels=[2048],
                               temporal_batch_norm=[True],
                               temporal_activation=[True])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.encoder1(xyzs, None)

        new_xys, new_features = self.encoder2a(new_xys, new_features)

        new_xys, new_features = self.encoder2b(new_xys, new_features)

        new_xys, new_features = self.encoder3a(new_xys, new_features)

        new_xys, new_features = self.encoder3b(new_xys, new_features)

        _, new_features = self.encoder4(new_xys, new_features)                      # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)        # (B, L, C)
        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]        # (B, C)

        out = self.fc(new_feature)

        return out
