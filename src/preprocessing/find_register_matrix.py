import numpy as np
import open3d as o3d

from ..utils.io import create_pcd, transform_pcd
from ..utils.visualization import fuse_visualize


def manual_registration(source_pcd_array, target_pcd_array, initial_params, voxel_size=1.0, visualize=True):
    """Evaluate a hand-specified initial rigid transform (rotation + xy offset) between two sensors."""
    print('Starting Manual Registration...')
    angle = initial_params["angle"]
    x_offset = initial_params["x_offset"]
    y_offset = initial_params["y_offset"]
    angle_rad = np.radians(angle)
    z_rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    x_trans_matrix = np.array([
        [1, 0, 0, x_offset],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    y_trans_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, y_offset],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    init_transform = np.dot(y_trans_matrix, np.dot(x_trans_matrix, z_rot_matrix))
    source_pcd = create_pcd(source_pcd_array, downsample=True, voxel_size=voxel_size)
    target_pcd = create_pcd(target_pcd_array, downsample=True, voxel_size=voxel_size)
    max_correspondence_distance = voxel_size * 1.5
    result = o3d.pipelines.registration.evaluate_registration(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=max_correspondence_distance,
        transformation=init_transform)
    fitness = result.fitness
    rmse = result.inlier_rmse
    score = fitness / (1 + rmse)
    transform = result.transformation
    print(f'Initial Manual Registration matrix:\n{transform.tolist()}')
    print(f'Initial Manual Registration metric: Fitness={fitness:.4f}, RMSE={rmse:.4f}, Score={score:.4f}')
    if visualize:
        source_pcd_array = transform_pcd(source_pcd_array, transform)
        fuse_visualize(source_pcd_array, target_pcd_array)
    return transform


def gicp_registration(source_pcd_array, target_pcd_array, initial_transform, voxel_size=1.0, visualize=True):
    """Refine a registration with Generalized ICP, starting from `initial_transform`."""
    source_pcd, source_fpfh = create_pcd(
        source_pcd_array, downsample=True, voxel_size=voxel_size, estimate_normals=True, fpfh=True)
    target_pcd, target_fpfh = create_pcd(
        target_pcd_array, downsample=True, voxel_size=voxel_size, estimate_normals=True, fpfh=True)
    max_correspondence_distance = voxel_size * 1.5
    print("Starting Generalized ICP Registration...")
    result = o3d.pipelines.registration.registration_generalized_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=initial_transform)
    fitness = result.fitness
    rmse = result.inlier_rmse
    score = fitness / (1 + rmse)
    transform = result.transformation
    print(f'Refined Generalized ICP Registration matrix:\n{transform.tolist()}')
    print(f'Refined Generalized ICP Registration metric: Fitness={fitness:.4f}, RMSE={rmse:.4f}, Score={score:.4f}')
    if visualize:
        source_pcd_array = transform_pcd(source_pcd_array, transform)
        fuse_visualize(source_pcd_array, target_pcd_array)
    return transform
