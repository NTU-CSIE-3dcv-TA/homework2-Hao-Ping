[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lyfclldM)
# Homework2

Dataset: [Download](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ)

## Environment Setup

```bash
conda create -n 3dcv_hw2 python=3.10
conda activate 3dcv_hw2
pip install numpy pandas opencv-python scipy tqdm open3d torch
```
## Data directory
```bash
data/
 ├─ frames/              # input images
 ├─ images.pkl           # ground truth camera poses
 ├─ train.pkl            # training image features
 ├─ points3D.pkl         # 3D map points
 └─ point_desc.pkl       # 2D-3D feature descriptors
```

## Usage
### Camera Pose and Export Estimated Poses
```bash
python 2d3dmatching.py --export_csv my_estimated_poses.csv
```
### Virtual Cube in Augmented Reality
1. Cube Transformation (Manual)
* Run:
```bash
python transform_cube.py
```
2. AR Rendering
```bash
python cube_in_ar.py \
    --use_est my_estimated_poses.csv \
    --cube_transform cube_transform_mat.npy \
    --out ar.mp4 \
    --fps 30 \
    --slow 2 \
    --undistorted_frames \
    --pt_radius 3
```
