# Deep Hierarchical Representation of Point Cloud Videos via Spatio-Temporal Decomposition

## Introduction
In point cloud videos, point coordinates are irregular and unordered but point timestamps exhibit regularities and order. Grid-based networks for conventional video processing cannot be directly used to model raw point cloud videos. Therefore, in this work, we propose a point-based network that directly handles raw point cloud videos. First, to preserve the spatio-temporal local structure of point cloud videos, we design a point tube covering a local range along spatial and temporal dimensions. By progressively subsampling frames and points and enlarging the spatial radius as the point features are fed into higher-level layers, the point tube can capture video structure in a spatio-temporally hierarchical manner. Second, to reduce the impact of the spatial irregularity on temporal modeling, we decompose space and time when extracting point tube representations. Specifically, a spatial operation is employed to capture the local structure of each spatial region in a tube and a temporal operation is used to model the dynamics of the spatial regions along the tube.  

## Installation

The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 8.3.1, PyTorch (both v1.4.0 and v1.8.1 are supported), CUDA 10.2 and cuDNN v7.6.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
mv modules-pytorch-1.4.0/modules-pytorch-1.8.1 modules
cd modules
python setup.py install
```
To see if the compilation is successful, try to run python modules/pst_operations.py to see if a forward pass works.
