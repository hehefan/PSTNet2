""" PSTOp and PSTTrans

From: "Deep Hierarchical Representation of Point Cloud Videos via Spatio-Temporal Decomposition"

Author: Hehe Fan
Date: January 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from typing import List

class PSTOp(nn.Module):
    def __init__(self,
                 in_channels: int,
                 spatial_radius: float,
                 spatial_neighbours: int,
                 spatial_sampling: int,
                 spatial_channels: List[int],
                 spatial_batch_norm: List[bool],
                 spatial_activation: List[bool],
                 temporal_radius: int,
                 temporal_stride: int,
                 temporal_padding: [int, int],
                 temporal_padding_mode: str,
                 temporal_channels: List[int],
                 temporal_batch_norm: List[bool],
                 temporal_activation: List[bool],
                 bias: bool = False):
        """
        Args:
            in_channels: number of point feature channels in the input. it is 0 if point features are not available.
            spatial_radius:
            spatial_neighbours:
            spatial_sampling: spatial down-sampling rate, >= 1
            spatial_channels: list of number of channels of the spatial MLP
            spatial_batch_norm: List[bool],
            spatial_activation: List[bool],
            temporal_radius: >= 0
            temporal_stride: >= 1
            temporal_padding:
            temporal_padding_mode: "zeros" or "replicate"
            temporal_channels: list of number of channels of the temporal MLPs
            temporal_batch_norm: List[bool],
            temporal_activation: List[bool],
            bias:
        """
        super().__init__()

        assert (len(spatial_channels) == len(spatial_batch_norm) and len(spatial_channels) == len(spatial_activation)), "PSTOp: Spatial MLP parameters error!"
        assert (len(temporal_channels) == len(temporal_batch_norm) and len(temporal_channels) == len(temporal_activation)), "PSTOp: Temporal MLP parameters error!"
        assert (temporal_padding_mode in ["zeros", "replicate"]), "PSTOp: 'temporal_padding_mode' should be 'zeros' or 'replicate'!"

        self.in_channels = in_channels

        self.spatial_radius = spatial_radius
        self.spatial_neighbours = spatial_neighbours
        self.spatial_sampling = spatial_sampling
        self.spatial_channels = spatial_channels
        self.spatial_batch_norm = spatial_batch_norm
        self.spatial_activation = spatial_activation

        self.temporal_radius = temporal_radius
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode
        self.temporal_channels = temporal_channels
        self.temporal_batch_norm = temporal_batch_norm
        self.temporal_activation = temporal_activation

        spatial_mlp = []
        for i in range(len(spatial_channels)):
            if i == 0:
                spatial_mlp.append(nn.Conv2d(in_channels=3+in_channels, out_channels=spatial_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))
            else:
                spatial_mlp.append(nn.Conv2d(in_channels=spatial_channels[i-1], out_channels=spatial_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if spatial_batch_norm[i]:
                spatial_mlp.append(nn.BatchNorm2d(num_features=spatial_channels[i]))
            if spatial_activation[i]:
                spatial_mlp.append(nn.ReLU(inplace=True))
        self.spatial_mlp = nn.Sequential(*spatial_mlp)

        for t in range(2*temporal_radius + 1):
            temporal_mlp = []
            for i in range(len(temporal_channels)):
                if i == 0:
                    in_planes = spatial_channels[-1]
                else:
                    in_planes = temporal_channels[i-1]
                temporal_mlp.append(nn.Conv1d(in_channels=in_planes, out_channels=temporal_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))

                if temporal_batch_norm[i]:
                    temporal_mlp.append(nn.BatchNorm1d(num_features=temporal_channels[i]))
                if temporal_activation[i]:
                    temporal_mlp.append(nn.ReLU(inplace=True))
            setattr(self, 'temporal_mlp_%d'%t, nn.Sequential(*temporal_mlp))

    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, L, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, L, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)  # L
        npoints = xyzs.size(2)  # N

        if self.temporal_radius > 0 and self.temporal_stride > 1:
            assert ((nframes + sum(self.temporal_padding) - (2*self.temporal_radius + 1)) % self.temporal_stride == 0), "PSTOp: Temporal parameter error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.in_channels != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        if self.temporal_padding_mode == "zeros":
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]

            if self.in_channels != 0:
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
        else:   # "replicate"
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

            if self.in_channels != 0:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_radius, len(xyzs)-self.temporal_radius, self.temporal_stride):                                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_sampling)                             # (B, N//self.spatial_sampling)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_sampling)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            spatio_temporal_feature = []
            for j, i in enumerate(range(t-self.temporal_radius, t+self.temporal_radius+1)):
                # spatial 
                neighbor_xyz = xyzs[i]

                idx = pointnet2_utils.ball_query(self.spatial_radius, self.spatial_neighbours, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_sampling, spatial_neighbours)

                displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                           # (B, 3, N//spatial_sampling, spatial_neighbours)

                if self.in_channels != 0:
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_channels, N//spatial_sampling, spatial_neighbours)
                    spatial_feature = torch.cat((displacement, neighbor_feature_grouped), dim=1)                                    # (B, 3+in_channels, N//spatial_sampling, spatial_neighbours)
                else:
                    spatial_feature = displacement

                spatial_feature = self.spatial_mlp(spatial_feature)                                                                 # (B, spatial_channels[-1], N//spatial_sampling, spatial_neighbours)

                spatial_feature = torch.max(input=spatial_feature, dim=-1, keepdim=False)[0]                                        # (B, spatial_channels[-1], N//spatial_sampling)

                # temporal 
                spatio_temporal_feature.append(getattr(self, 'temporal_mlp_%d'%j)(spatial_feature))

            spatio_temporal_feature = torch.stack(tensors=spatio_temporal_feature, dim=1, out=None)
            spatio_temporal_feature = torch.sum(input=spatio_temporal_feature, dim=1, keepdim=False)

            new_xyzs.append(anchor_xyz)
            new_features.append(spatio_temporal_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features


class PSTTransOp(nn.Module):
    def __init__(self,
                 in_channels: int,
                 temporal_radius: int,
                 temporal_stride: int,
                 temporal_padding: [int, int],
                 temporal_channels: List[int],
                 temporal_batch_norm: List[bool],
                 temporal_activation: List[bool],
                 spatial_channels: List[int],
                 spatial_batch_norm: List[bool],
                 spatial_activation: List[bool],
                 original_channels: int = 0,
                 bias: bool = False):
        """
        Args:
            in_channels: C'
            temporal_radius: >= 0
            temporal_stride: >= 1
            temporal_padding: <= 0
            temporal_channels: list of number of channels of the temporal MLPs
            temporal_batch_norm: List[bool],
            temporal_activation: List[bool],
            spatial_channels: list of number of channels of the spatial MLP
            spatial_batch_norm: List[bool],
            spatial_activation: List[bool],
            original_channels: C, used for skip connection from original points. when original point features are not available, original_in_channels is 0.
            bias: whether to use bias
        """
        super().__init__()

        assert (len(spatial_channels) == len(spatial_batch_norm) and len(spatial_channels) == len(spatial_activation)), "PSTOp: Spatial MLP parameters error!"
        assert (len(temporal_channels) == len(temporal_batch_norm) and len(temporal_channels) == len(temporal_activation)), "PSTOp: Temporal MLP parameters error!"
        assert (temporal_padding[0] <=0 and temporal_padding[1] <=0), "PSTTransOp: 'temporal_padding' should be <= 0!"

        self.in_channels = in_channels

        # temporal parameters 
        self.temporal_radius = temporal_radius
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_channels = temporal_channels
        self.temporal_batch_norm = temporal_batch_norm
        self.temporal_activation = temporal_activation

        # spatial parameters 
        self.spatial_channels = spatial_channels
        self.spatial_batch_norm = spatial_batch_norm
        self.spatial_activation = spatial_activation

        self.original_channels = original_channels

        for t in range(2*temporal_radius + 1):
            temporal_mlp = []
            for i in range(len(temporal_channels)):
                if i == 0:
                    in_planes = in_channels
                else:
                    in_planes = temporal_channels[i-1]
                temporal_mlp.append(nn.Conv1d(in_channels=in_planes, out_channels=temporal_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))

                if temporal_batch_norm[i]:
                    temporal_mlp.append(nn.BatchNorm1d(num_features=temporal_channels[i]))
                if temporal_activation[i]:
                    temporal_mlp.append(nn.ReLU(inplace=True))
            setattr(self, 'temporal_mlp_%d'%t, nn.Sequential(*temporal_mlp))

        spatial_mlp = []
        for i in range(len(spatial_channels)):
            if i == 0:
                if len(temporal_channels) > 0:
                    spatial_mlp.append(nn.Conv1d(in_channels=temporal_channels[-1]+original_channels, out_channels=spatial_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))
                else:
                    spatial_mlp.append(nn.Conv1d(in_channels=in_channels+original_channels, out_channels=spatial_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))
            else:
                spatial_mlp.append(nn.Conv1d(in_channels=spatial_channels[i-1], out_channels=spatial_channels[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if spatial_batch_norm:
                spatial_mlp.append(nn.BatchNorm1d(num_features=spatial_channels[i]))
            if spatial_activation:
                spatial_mlp.append(nn.ReLU(inplace=True))
        self.spatial_mlp = nn.Sequential(*spatial_mlp)


    def forward(self, xyzs: torch.Tensor, original_xyzs: torch.Tensor, features: torch.Tensor, original_features: torch.Tensor = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        xyzs : torch.Tensor
            (B, L', N', 3) tensor of the xyz positions of the convolved features
        original_xyzs : torch.Tensor
            (B, L,  N,  3) tensor of the xyz positions of the original points

        features : torch.Tensor
            (B, L', C', N') tensor of the features to be propigated to
        original_features : torch.Tensor
            (B, L,  C,  N) tensor of original point features for skip connection

        Returns
        -------
        new_features : torch.Tensor
            (B, L,  C", N) tensor of the features of the unknown features
        """

        L1 = original_xyzs.size(1)
        N1 = original_xyzs.size(2)

        L2 = xyzs.size(1)
        N2 = xyzs.size(2)

        if self.temporal_radius > 0 and self.temporal_stride > 1:
            assert ((L2 - 1) * self.temporal_stride + sum(self.temporal_padding) + (2*self.temporal_radius + 1) == L1), "PSTTransOp: Temporal parameter error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
        features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        new_xyzs = original_xyzs

        original_xyzs = torch.split(tensor=original_xyzs, split_size_or_sections=1, dim=1)
        original_xyzs = [torch.squeeze(input=original_xyz, dim=1).contiguous() for original_xyz in original_xyzs]

        if original_features is not None:
            original_features = torch.split(tensor=original_features, split_size_or_sections=1, dim=1)
            original_features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in original_features]

        # temporal correlation transposed operation
        temporal_trans_features = []
        for feature in features:
            temporal_trans_features.append([getattr(self, 'temporal_mlp_%d'%i)(feature) for i in range(2*self.temporal_radius + 1)])

        # temporal interpolation
        temporal_interpolated_xyzs = []
        temporal_interpolated_features = []

        middles = []
        deltas = []
        for t2 in range(1, L2+1):
            middle = t2 + (t2-1)*(self.temporal_stride-1) + self.temporal_radius + self.temporal_padding[0]
            middles.append(middle)
            delta = range(middle - self.temporal_radius, middle + self.temporal_radius + self.temporal_padding[1] + 1)
            deltas.append(delta)

        for t1 in range(1, L1+1):
            seed_xyzs = []
            seed_features = []
            for t2 in range(L2):
                delta = deltas[t2]
                if t1 in delta:
                    seed_xyzs.append(xyzs[t2])
                    seed_feature = temporal_trans_features[t2][t1-middles[t2]+self.temporal_radius]
                    seed_features.append(seed_feature)
            seed_xyzs = torch.cat(seed_xyzs, dim=1)
            seed_features = torch.cat(seed_features, dim=2)
            temporal_interpolated_xyzs.append(seed_xyzs)
            temporal_interpolated_features.append(seed_features)

        # spatial interpolation
        new_features = []
        for t1 in range(L1):
            neighbor_xyz = temporal_interpolated_xyzs[t1]                                                               # [B, L', 3]
            anchor_xyz = original_xyzs[t1]                                                                              # [B, L,  3]

            dist, idx = pointnet2_utils.three_nn(anchor_xyz, neighbor_xyz)

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(temporal_interpolated_features[t1], idx, weight)

            if original_features is not None:
                new_feature = torch.cat([interpolated_feats, original_features[t1]], dim=1)
            else:
                new_feature = interpolated_feats

            new_feature = self.spatial_mlp(new_feature)

            new_features.append(new_feature)

        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

if __name__ == '__main__':
    xyzs = torch.zeros(4, 8, 512, 3).cuda()
    features = torch.zeros(4, 8, 16, 512).cuda()

    encoder = PSTOp(in_channels=16,
                    spatial_radius=1.0,
                    spatial_neighbours=3,
                    spatial_sampling=2,
                    spatial_channels=[24, 32],
                    spatial_batch_norm=[True, True],
                    spatial_activation=[True, True],
                    temporal_radius=1,
                    temporal_stride=3,
                    temporal_padding=[1, 0],
                    temporal_padding_mode="replicate",
                    temporal_channels=[48, 64],
                    temporal_batch_norm=[True, True],
                    temporal_activation=[True, True]).cuda()

    new_xyzs, new_features = encoder(xyzs, features)

    decoder = PSTTransOp(in_channels=64,
                         temporal_radius=1,
                         temporal_stride=3,
                         temporal_padding=[-1, 0],
                         temporal_channels=[128],
                         temporal_batch_norm=[True],
                         temporal_activation=[True],
                         spatial_channels=[256],
                         spatial_batch_norm=[True],
                         spatial_activation=[True],
                         original_channels=16).cuda()

    out_xyzs, out_features = decoder(new_xyzs, xyzs, new_features, features)
    print("-----------------------------")
    print("Input:")
    print(xyzs.shape)
    print(features.shape)
    print("-----------------------------")
    print("PSTOp Operation:")
    print(new_xyzs.shape)
    print(new_features.shape)
    print("-----------------------------")
    print("PSTTransOp Operation:")
    print(out_xyzs.shape)
    print(out_features.shape)
    print("-----------------------------")
