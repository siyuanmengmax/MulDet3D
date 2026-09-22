# io.py

import open3d as o3d
import pickle
import os
import numpy as np


def ensure_directory(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')


def create_pcd(pcd_array, downsample=False, voxel_size=1.0, estimate_normals=False, fpfh=False, colored=False,
               columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """
    Create an Open3D point cloud from a numpy array.
    :param pcd_array: numpy array of shape (N, M) where N is the number of points and M is the number of features
    :param downsample: whether to downsample the point cloud
    :param voxel_size: voxel size for downsampling
    :param estimate_normals: whether to estimate normals
    :param fpfh: whether to compute FPFH features
    :return: Open3D point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
    if colored:
        colors = np.zeros((pcd_array.shape[0], 3))
        reflectivity = pcd_array[:, columns.index('reflectivity')]
        colors[:, 0] = reflectivity / 255.0
        near_ir = pcd_array[:, columns.index('near_ir')]
        colors[:, 2] = np.clip(near_ir / 592, 0, 1)
        colors[:, 1] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size)
        if estimate_normals:
            radius_normal = voxel_size * 2
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            if fpfh:
                radius_feature = voxel_size * 5
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                return pcd, pcd_fpfh
    return pcd


def read_pkl(file_path):
    """Read a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_pkl(file_path, data):
    """Write a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Saved data to {file_path}')


def write_bin(points_array, output_path):
    """Save a point cloud array to a binary (.bin) file as float32."""
    points_array = points_array.astype(np.float32)
    points_array.tofile(output_path)


def read_bin(bin_path, num_features):
    """Read a binary (.bin) point cloud file into an (N, num_features) array."""
    points_array = np.fromfile(bin_path, dtype=np.float32)
    points = points_array.reshape(-1, num_features)
    return points


def transform_pcd(points, transformation_matrix):
    """Transform point cloud using given transformation matrix."""
    homogeneous = np.column_stack([points[:, :3], np.ones((len(points), 1))])
    transformed = (transformation_matrix @ homogeneous.T).T
    transformed = np.column_stack([transformed[:, :3], points[:, 3:]])
    return transformed
