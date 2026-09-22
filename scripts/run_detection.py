"""MulDet3D detection pipeline: two-stage clustering, classification, and bounding-box generation.

Runs on the fused, foreground-labeled point clouds produced by
run_preprocessing.py: Stage 1 clusters foreground points with the
reliability-weighted density-adaptive algorithm (or an ablation/baseline
variant selected via --clustering_method), Stage 2 optionally merges
nearby clusters under physical constraints, a lightweight geometric
classifier assigns object types, and oriented bounding boxes are fit per
cluster. Results are written to a CSV consumed by evaluate_detection_ap.py.

alpha/rho_min presets from the paper's MOPSO optimization (Table:
"Parameter Optimization Results"), provided here as a starting point --
values differ per dataset and per optimization objective (Cases 1-4):
    Case 1 (coverage-only):    alpha=0,     rho_min=10
    Case 2 (compactness-only): alpha=1,     rho_min=30
    Case 3 (separation, Lowell):  alpha=0.019, rho_min=30
    Case 3 (separation, Amherst): alpha=0.012, rho_min=18.888
    Case 4 (balanced, Lowell):    alpha=0.725, rho_min=30 (Amherst: same as Case 3)
Case 3 is the configuration reported as MulDet3D's headline result in the paper.
"""
import os
import sys
import argparse
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detection.clustering import (
    clustering_stage_1, clustering_stage_1_no_rel, clustering_stage_2,
    dbscan_clustering, hdbscan_clustering,
)
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox
from src.utils.io import read_bin, write_bin, ensure_directory
from src.utils.visualization import visualize_frame, create_point_cloud_video

CLASSES = ['person', 'small vehicle', 'large vehicle', 'other']

CLUSTERING_METHODS = {
    'adaptive': clustering_stage_1,               # MulDet3D Stage 1 (reliability-weighted).
    'adaptive_no_rel': clustering_stage_1_no_rel,  # Ablation: reliability weighting removed.
    'dbscan': dbscan_clustering,                   # Baseline.
    'hdbscan': hdbscan_clustering,                 # Baseline.
}


def process_frame(frame_id, frame_name, pcd_array, filtered_pcd_array, arg):
    """Cluster, (optionally merge), classify, and fit bounding boxes for one frame."""
    cluster_fn = CLUSTERING_METHODS[arg.clustering_method]
    if arg.clustering_method in ('adaptive', 'adaptive_no_rel'):
        clusters, clusters_coords, eps = cluster_fn(
            filtered_pcd_array, arg.columns, rho_min=arg.params['rho_min'],
            alpha=arg.params['alpha'], eps_min=0.5, eps_max=1.0)
    else:
        clusters, clusters_coords, eps = cluster_fn(filtered_pcd_array, arg.columns)

    coarse_classes = classifying_stage_1(clusters_coords)
    if arg.stage_2:
        clusters, clusters_coords, coarse_classes = clustering_stage_2(
            clusters, clusters_coords, coarse_classes, filtered_pcd_array, arg.columns)
    cluster_classes, class_probabilities = classifying_stage_2(clusters_coords, coarse_classes)

    bboxes = []
    reliability = filtered_pcd_array[:, arg.columns.index('reliability')]
    cluster_id = np.ones(len(pcd_array)) * -1
    if clusters:
        tree = cKDTree(pcd_array[:, :arg.columns.index('z') + 1])
        for i, cluster in enumerate(clusters):
            clustered_points = clusters_coords[i]
            dist, idx = tree.query(clustered_points, k=1)
            for j, point_idx in enumerate(idx):
                if dist[j] < 1e-6:
                    cluster_id[point_idx] = i
            bbox = compute_bbox(clusters_coords[i])
            bboxes.append({
                "frame_id": frame_id,
                "frame_name": frame_name,
                'cluster_id': i,
                "label": cluster_classes[i],
                'confidence': class_probabilities[i],
                "x": bbox[0], "y": bbox[1], "z": bbox[2],
                "w": bbox[3], "l": bbox[4], "h": bbox[5], "yaw": bbox[6],
                "avg_eps": np.mean([eps[cluster]]),
                "avg_reliability": np.mean([reliability[cluster]]),
                "num_points": len(cluster)
            })

    pcd_array = np.column_stack([pcd_array, cluster_id])
    if arg.visualize:
        visualize_frame(pcd_array, cluster_id, bboxes, frame_id, CLASSES)
    return bboxes, pcd_array


def main(arg, min_height=0.1, max_height=3):
    print("=== Starting MulDet3D Detection Pipeline ===")
    pcd_file_paths = sorted(glob(os.path.join(arg.input_folder, '*.bin')))
    print(f"Found {len(pcd_file_paths)} point cloud files")

    if arg.create_video:
        video_path = os.path.join(arg.output_folder, 'point_cloud_video.mp4')
        create_point_cloud_video(pcd_file_paths, video_path, process_frame, arg, fps=arg.fps)
        return

    all_bboxes = []
    columns_len = len(arg.columns)
    for i, pcd_file_path in tqdm(enumerate(pcd_file_paths)):
        start_time = time.time()
        frame_name = os.path.basename(pcd_file_path)
        pcd_array = read_bin(pcd_file_path, columns_len)
        filter_mask = (pcd_array[:, arg.columns.index('z')] >= min_height) & (
                pcd_array[:, arg.columns.index('z')] <= max_height) & (
                              pcd_array[:, arg.columns.index('dynamic_mask')] == 1)
        filtered_pcd_array = pcd_array[filter_mask]

        frame_bboxes, pcd_array = process_frame(i, frame_name, pcd_array, filtered_pcd_array, arg)
        all_bboxes.extend(frame_bboxes)

        if arg.save_bin:
            save_folder = os.path.join(arg.output_folder, 'detection')
            ensure_directory(save_folder)
            output_path = os.path.join(save_folder, os.path.basename(pcd_file_path))
            write_bin(pcd_array, output_path)
        print(f"Processed {i + 1}/{len(pcd_file_paths)}: {frame_name} in {time.time() - start_time:.2f} seconds")

    print("\nSaving results...")
    bbox_path = os.path.join(arg.output_folder, arg.output_csv_name)
    bbox_df = pd.DataFrame(all_bboxes, columns=["frame_id", "frame_name", "cluster_id", "label", 'confidence',
                                                 "x", "y", "z", "w", "l", "h", "yaw",
                                                 "avg_eps", "avg_reliability", "num_points"])
    bbox_df.to_csv(bbox_path, index=False)
    print(f"Saved bounding box information to {bbox_path}")
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MulDet3D two-stage clustering, classification, and bbox generation")
    parser.add_argument("--output_folder", default="results/detection", help="Folder to save output files")
    parser.add_argument("--input_folder", default="bin/merge_1000", help="Folder of fused, foreground-labeled point clouds")
    parser.add_argument("--output_csv_name", default="bboxes.csv", help="Output bounding-box CSV file name")
    parser.add_argument("--clustering_method", choices=list(CLUSTERING_METHODS.keys()), default="adaptive",
                         help="Stage-1 clustering algorithm")
    parser.add_argument("--stage_2", type=lambda x: x.lower() != 'false', default=True,
                         help="Whether to run Stage 2 hierarchical merging")
    params = {
        'alpha': 0.019,   # Case 3 (Lowell); see module docstring for other presets.
        'rho_min': 30,
    }
    parser.add_argument("--params", default=params, help="Parameters for the adaptive clustering methods")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize results per frame")
    parser.add_argument("--save_bin", action="store_true", default=False, help="Save processed point clouds with cluster ids")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'dynamic_mask', 'reliability', 'lidar_source']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    parser.add_argument("--create_video", action="store_true", default=False, help="Render a point cloud video instead of computing bounding boxes")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the video")
    args = parser.parse_args()
    main(args)
