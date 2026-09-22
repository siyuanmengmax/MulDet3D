"""Dual-LiDAR preprocessing: reliability-weighted background modeling + fusion.

For each pair of time-aligned frames from two roadside LiDAR sensors, this
builds a reliability-weighted background model per sensor, runs foreground
detection, transforms both point clouds into a common ground/registration
frame, and fuses them into a single point cloud with per-point foreground
mask, reliability, and source-sensor id columns.

The fused output feeds the detection/clustering pipeline (run_detection.py)
and the object-level AP evaluation (evaluate_detection_ap.py).

Requires the registration matrices produced by
data_preparation/registration.py and the time-aligned frame lists produced
by data_preparation/time_alignment.py.
"""
import os
import sys
import argparse
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.background_model import build_background, detect_foreground
from src.utils.io import ensure_directory, read_bin, transform_pcd, write_bin


def main(args):
    ensure_directory(args.output_folder_path)
    ensure_directory(args.output_pcd_folder_path)

    source_ground_matrix = np.load(args.source_ground_matrix_path)
    target_ground_matrix = np.load(args.target_ground_matrix_path)
    merge_matrix = np.load(args.merge_matrix_path)

    source_bg_model_path = os.path.join(args.output_folder_path, 'source_bg_model.npy')
    target_bg_model_path = os.path.join(args.output_folder_path, 'target_bg_model.npy')
    bg_source_pcd_files = sorted(glob(os.path.join(args.source_pcd_folder_path, "*.bin")))
    bg_target_pcd_files = sorted(glob(os.path.join(args.target_pcd_folder_path, "*.bin")))

    if os.path.exists(source_bg_model_path):
        source_bg_model = np.load(source_bg_model_path)
    else:
        source_bg_model = build_background(bg_source_pcd_files, sample_size=1000, columns=args.columns)
        np.save(source_bg_model_path, source_bg_model)
    if os.path.exists(target_bg_model_path):
        target_bg_model = np.load(target_bg_model_path)
    else:
        target_bg_model = build_background(bg_target_pcd_files, sample_size=1000, columns=args.columns)
        np.save(target_bg_model_path, target_bg_model)

    source_aligned_files = sorted(pd.read_csv(args.source_aligned_file_path, header=None).iloc[:, 0].tolist())
    target_aligned_files = sorted(pd.read_csv(args.target_aligned_file_path, header=None).iloc[:, 0].tolist())
    start_idx = args.start_frame if args.start_frame >= 0 else 0
    end_idx = args.end_frame if args.end_frame >= 0 else len(source_aligned_files)
    source_pcd_files = source_aligned_files[start_idx:end_idx:args.frame_step]
    target_pcd_files = target_aligned_files[start_idx:end_idx:args.frame_step]

    for i, (source_pcd_file, target_pcd_file) in tqdm(enumerate(zip(source_pcd_files, target_pcd_files))):
        start_time = time.time()
        source_pcd_array = read_bin(os.path.join(args.source_pcd_folder_path, source_pcd_file), len(args.columns))
        target_pcd_array = read_bin(os.path.join(args.target_pcd_folder_path, target_pcd_file), len(args.columns))

        source_foreground_mask, source_reliability = detect_foreground(
            source_pcd_array, source_bg_model, columns=args.columns, visualize=False)
        source_lidar_id_array = np.full(len(source_pcd_array), args.source_lidar_id)
        source_pcd_array = np.column_stack(
            [source_pcd_array, source_foreground_mask, source_reliability, source_lidar_id_array])

        target_foreground_mask, target_reliability = detect_foreground(
            target_pcd_array, target_bg_model, columns=args.columns, visualize=False)
        target_lidar_id_array = np.full(len(target_pcd_array), args.target_lidar_id)
        target_pcd_array = np.column_stack(
            [target_pcd_array, target_foreground_mask, target_reliability, target_lidar_id_array])

        source_pcd_array = transform_pcd(source_pcd_array, source_ground_matrix)
        source_pcd_array = transform_pcd(source_pcd_array, merge_matrix)
        target_pcd_array = transform_pcd(target_pcd_array, target_ground_matrix)
        merge_pcd_array = np.row_stack([source_pcd_array, target_pcd_array])

        merge_pcd_name = os.path.basename(source_pcd_file).replace("one", "fused")
        merge_pcd_path = os.path.join(args.output_pcd_folder_path, merge_pcd_name)
        write_bin(merge_pcd_array, merge_pcd_path)

        print(f"Processed {i + 1}/{len(source_pcd_files)}: {source_pcd_file} and {target_pcd_file} "
              f"in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-LiDAR background removal and fusion")
    parser.add_argument("--output_folder_path", default="bin", help="Folder to save background models")
    parser.add_argument("--source_pcd_folder_path", default="bin/one")
    parser.add_argument("--target_pcd_folder_path", default="bin/two")
    parser.add_argument("--output_pcd_folder_path", default="bin/merge_label", help="Folder to save fused point clouds")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame number (inclusive)")
    parser.add_argument("--end_frame", type=int, default=-1,
                         help="Ending frame number (exclusive). Use -1 for all remaining frames")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--merge_matrix_path", default="bin/merge_matrix.npy", help="Path to the fusion matrix")
    parser.add_argument("--source_ground_matrix_path", default="bin/source_ground_matrix.npy",
                         help="Path to the source sensor's ground-alignment matrix")
    parser.add_argument("--target_ground_matrix_path", default="bin/target_ground_matrix.npy",
                         help="Path to the target sensor's ground-alignment matrix")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    parser.add_argument("--source_aligned_file_path", default="bin/one.txt")
    parser.add_argument("--target_aligned_file_path", default="bin/two.txt")
    parser.add_argument("--source_lidar_id", default=0, type=int, help="Sensor id for the source point cloud")
    parser.add_argument("--target_lidar_id", default=1, type=int, help="Sensor id for the target point cloud")
    args = parser.parse_args()
    main(args)
