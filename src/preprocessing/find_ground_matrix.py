# find_ground.py

import numpy as np
import open3d as o3d

from ..utils.visualization import visualize_ground
from ..utils.io import create_pcd

def find_ground_plane(pcd, visualize,ransac_distance_threshold=0.1, ransac_n=3, ransac_num_iterations=10000):
    """
    Find the ground plane
    :param pcd:
    :param ransac_distance_threshold:
    :param ransac_n:
    :param ransac_num_iterations:
    :return:
    """
    # Check if the point cloud has enough points
    if len(pcd.points) < ransac_n:
        raise ValueError(f"Point cloud contains too few points ({len(pcd.points)} < {ransac_n})")
    # Estimate normals if not available
    if not pcd.has_normals():
        pcd.estimate_normals()
    # Segment the ground plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_distance_threshold,
        ransac_n=ransac_n,
        num_iterations=ransac_num_iterations
    )
    # Check if a valid ground plane was found
    if len(inliers) < ransac_n:
        raise ValueError("Failed to find a valid ground plane")
    # Separate ground and non-ground points
    ground_points = pcd.select_by_index(inliers)
    non_ground_points = pcd.select_by_index(inliers, invert=True)
    if visualize:
        visualize_ground(ground_points, non_ground_points)
    # Extract plane parameters
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal_length = np.linalg.norm(normal)
    # Check if the normal vector is valid
    if normal_length < 1e-6:
        raise ValueError("Invalid plane normal vector")
    # Normalize the normal vector
    ground_ratio = len(inliers) / len(pcd.points)
    print(f"Ground points ratio: {ground_ratio:.2%}")
    # Warn if the ground ratio is too low or too high
    if ground_ratio < 0.1:
        print("Warning: Very few ground points detected")
    elif ground_ratio > 0.9:
        print("Warning: Almost all points classified as ground")
    # Print the plane equation and normal vector
    print(f"Ground plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"Normal vector: [{a:.3f}, {b:.3f}, {c:.3f}]")
    # Print the mean and standard deviation of the distances to the plane
    points = np.asarray(ground_points.points)
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_length
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    print(f"Mean distance to plane: {mean_distance:.6f}")
    print(f"Std of distance to plane: {std_distance:.6f}")
    return plane_model, ground_points, non_ground_points

def compute_transformation_matrix(plane_model, translation=True):
    """
    Compute the transformation matrix
    :param plane_model:
    :param translation:
    :return:
    """
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal_length = np.linalg.norm(normal)
    # Normalize the normal vector
    normal = normal / normal_length
    d = d / normal_length
    # Check the direction of the normal vector
    if c < 0:
        normal = -normal
        d = -d
    z_axis = np.array([0, 0, 1])
    # Check if the normal vector is parallel to the z-axis
    if np.allclose(np.abs(np.dot(normal, z_axis)), 1.0):
        rotation_matrix = np.eye(3)
        if np.dot(normal, z_axis) < 0:
            rotation_matrix[2, 2] = -1
    else:
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.clip(np.dot(normal, z_axis), -1.0, 1.0)
        rotation_angle = np.arccos(cos_angle)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    if translation:
        # Compute the translation vector
        point_on_plane = np.array([0, 0, -d / c])
        # Rotate the point to the z=0 plane
        rotated_point = rotation_matrix @ point_on_plane
        # Translate the point to the origin
        transform_matrix[:3, 3] = -rotated_point
    return transform_matrix

def verify_transformation(pcd, transform_matrix, threshold=0.1):
    """
    Verify the transformation matrix
    :param pcd:
    :param transform_matrix:
    :param threshold:
    :return:
    """
    transformed_pcd = pcd.transform(transform_matrix)
    points = np.asarray(transformed_pcd.points)
    z_coords = points[:, 2]
    near_ground = np.abs(z_coords) < threshold
    ground_ratio = np.sum(near_ground) / len(z_coords)
    print(f"Ground points statistics:")
    print(f"Mean z: {np.mean(z_coords):.6f}")
    print(f"Std z: {np.std(z_coords):.6f}")
    print(f"Min z: {np.min(z_coords):.6f}")
    print(f"Max z: {np.max(z_coords):.6f}")
    print(f"Points near ground (±{threshold}m): {ground_ratio * 100:.2f}%")
    return ground_ratio > 0.8

def verify_plane_transformation(plane_model, transform_matrix):
    """
    Verify the transformation of the plane
    :param plane_model:
    :param transform_matrix:
    :return:
    """
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    print("\nTransformation verification:")
    print("1. Original plane normal:", normal)
    # Verify the normal vector
    rotated_normal = transform_matrix[:3, :3] @ normal
    print("2. Transformed normal (should be close to [0,0,1]):", rotated_normal)
    # Verify the z-coordinates of the plane points
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    Z = (-d - a * X - b * Y) / c
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), np.ones_like(X.ravel())])
    transformed_points = transform_matrix @ points
    print("3. Transformed points z-coordinates:")
    print(f"   Mean: {np.mean(transformed_points[2]):.6f}")
    print(f"   Std: {np.std(transformed_points[2]):.6f}")
    print(f"   Min: {np.min(transformed_points[2]):.6f}")
    print(f"   Max: {np.max(transformed_points[2]):.6f}")

def main(pcd_array, visualize=True, voxel_size=0.1, threshold=0.05):
    """
    Main function to find the ground plane and generate transformation matrix
    :param pcd_folder:
    :param trans_matrix_path:
    :param visualize:
    :param voxel_size:
    :return:
    """
    pcd = create_pcd(pcd_array, downsample=True, voxel_size=voxel_size)
    # Find the ground plane
    plane_model, ground_points, non_ground_points = find_ground_plane(pcd,visualize)
    print(f"\nOriginal plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + "
          f"{plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
    # Compute the transformation matrix
    transformation_matrix = compute_transformation_matrix(plane_model)
    # Verify the transformation
    verify_plane_transformation(plane_model, transformation_matrix)
    verify_transformation(ground_points, transformation_matrix, threshold=threshold)
    # Save the transformation matrix
    return transformation_matrix