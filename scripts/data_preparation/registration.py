"""Register two roadside LiDAR sensors into a common coordinate frame.

Two steps, each with results cached to disk:
  1. Ground-plane fitting per sensor (RANSAC), producing a transform that
     levels each sensor's point cloud to a common z=0 ground plane.
  2. Rigid registration between the two ground-leveled point clouds: an
     initial manual guess (rotation + xy offset), optionally refined with
     Generalized ICP.

Produces source_ground_matrix.npy, target_ground_matrix.npy, and
merge_matrix3.npy, consumed by run_preprocessing.py.

The manual initial-registration guess (--initial_params) is
deployment-specific and must be re-estimated for each sensor pair; the
default below matches the Lowell (tilted dual-LiDAR) deployment. The
Amherst (horizontal dual-LiDAR) deployment used angle=0, x_offset=0,
y_offset=40 in this study.
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.io import read_bin, transform_pcd
from src.utils.visualization import fuse_visualize
from src.preprocessing import find_register_matrix, find_ground_matrix


def main(args):
    source_files = sorted(pd.read_csv(args.source_aligned_file_path, header=None).iloc[:, 0])
    target_files = sorted(pd.read_csv(args.target_aligned_file_path, header=None).iloc[:, 0])
    print(f"Found {len(source_files)} source files and {len(target_files)} target files")

    selected_frames = np.random.choice(len(source_files), 1, replace=False)
    source_arrays = []
    target_arrays = []
    for i in selected_frames:
        source_path = os.path.join(args.source_folder_path, source_files[i])
        target_path = os.path.join(args.target_folder_path, target_files[i])
        print(f"Processing frame {i}:")
        print(f"Source: {source_files[i]}")
        print(f"Target: {target_files[i]}")
        columns_len = len(args.columns)
        source_array = read_bin(source_path, columns_len)
        target_array = read_bin(target_path, columns_len)
        print(f'Source shape: {source_array.shape} before filtering')
        print(f'Target shape: {target_array.shape} before filtering')
        source_mask = (source_array[:, args.columns.index('range')] > 0)
        target_mask = (target_array[:, args.columns.index('range')] > 0)
        source_array = source_array[source_mask]
        target_array = target_array[target_mask]
        print(f'Source shape: {source_array.shape} after filtering')
        print(f'Target shape: {target_array.shape} after filtering')
        source_arrays.append(source_array)
        target_arrays.append(target_array)
    source_arrays = np.row_stack(source_arrays)
    target_arrays = np.row_stack(target_arrays)

    source_ground_matrix_path = os.path.join(args.output_folder_path, 'source_ground_matrix.npy')
    target_ground_matrix_path = os.path.join(args.output_folder_path, 'target_ground_matrix.npy')
    if os.path.exists(source_ground_matrix_path):
        source_ground_matrix = np.load(source_ground_matrix_path)
    else:
        source_ground_matrix = find_ground_matrix.main(source_arrays)
        np.save(source_ground_matrix_path, source_ground_matrix)
    if os.path.exists(target_ground_matrix_path):
        target_ground_matrix = np.load(target_ground_matrix_path)
    else:
        target_ground_matrix = find_ground_matrix.main(target_arrays)
        np.save(target_ground_matrix_path, target_ground_matrix)

    source_array = transform_pcd(source_arrays, source_ground_matrix)
    target_array = transform_pcd(target_arrays, target_ground_matrix)

    merge_matrix_path = os.path.join(args.output_folder_path, 'merge_matrix3.npy')
    if os.path.exists(merge_matrix_path):
        merge_matrix = np.load(merge_matrix_path)
        source_array = transform_pcd(source_array, merge_matrix)
        fuse_visualize(source_array, target_array)
    else:
        merge_matrix = find_register_matrix.manual_registration(source_array, target_array, args.initial_params)
        if args.refine:
            merge_matrix = find_register_matrix.gicp_registration(source_array, target_array, merge_matrix)
            np.save(merge_matrix_path, merge_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register source point cloud to target point cloud.")
    parser.add_argument("--output_folder_path", default="bin", help="Folder to save output files")
    parser.add_argument("--source_folder_path", default="bin/one", help="Folder path of source point cloud")
    parser.add_argument("--target_folder_path", default="bin/two", help="Folder path of target point cloud")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    parser.add_argument("--source_aligned_file_path", default="bin/one.txt")
    parser.add_argument("--target_aligned_file_path", default="bin/two.txt")
    initial_params = {
        "angle": 195,  # Rotation angle in degrees, clockwise.
        "x_offset": 18,
        "y_offset": -24
    }
    parser.add_argument("--initial_params", default=initial_params, help="Initial parameters for registration")
    parser.add_argument("--refine", type=bool, default=True, help="Refine the registration with Generalized ICP")
    parser.add_argument("--frame_index", type=int, default=100, help="Index of the frame to process")
    args = parser.parse_args()
    main(args)
