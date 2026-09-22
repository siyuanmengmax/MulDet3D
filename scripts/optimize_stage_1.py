"""MOPSO parameter optimization for MulDet3D's Stage-1 clustering (alpha, rho_min).

Optimizes three label-free proxy objectives -- coverage rate, compactness,
and separation (see paper Sec. "Multi-Objective Parameter Optimization") --
over a Cochran's-formula-sized random sample of frames, producing a Pareto
front of non-dominated (alpha, rho_min) solutions. Use optimize_stage_2.py
to select and analyze a best-compromise solution from the resulting front
(e.g. the paper's Case 1-4 configurations, each prioritizing a different
objective combination).
"""
import os
import sys
import argparse
from glob import glob
import numpy as np
import pandas as pd
from math import ceil
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import read_bin, ensure_directory
from src.algorithms.mopso import MOPSO
from src.detection.clustering import clustering_stage_1


def objective_function(solution, obj_func_params, separation_max=50):
    """Compute the three proxy objectives (coverage, compactness, separation) for one (alpha, rho_min)."""
    alpha = solution[0]
    rho_min = solution[1]
    frame_datas = obj_func_params['scene_data']
    columnss = obj_func_params['columns']
    all_objectives = []
    frame_point_len = []
    for frame_data in frame_datas:
        n_points = len(frame_data)
        if n_points > 0:
            clusters, clusters_coord, _ = clustering_stage_1(frame_data, columnss,
                                rho_min=rho_min, alpha=alpha, eps_min=0.5, eps_max=1.0)
            n_clusters = len(clusters)
            if n_clusters > 0:
                frame_point_len.append(n_points)
                # 1. Coverage rate: fraction of points assigned to a cluster (higher is better).
                coverage_rate = sum(len(cluster) for cluster in clusters) / n_points
                # 2. Compactness: point-count-weighted mean intra-cluster distance (lower is better).
                clusters_centroid = np.zeros((n_clusters, 3))
                intra_dist = np.zeros(n_clusters)
                cluster_points_len = np.zeros(n_clusters)
                for i, cluster in enumerate(clusters):
                    min_coord = np.min(clusters_coord[i], axis=0)
                    max_coord = np.max(clusters_coord[i], axis=0)
                    clusters_centroid[i] = (min_coord + max_coord) / 2
                    cluster_points_len[i] = len(cluster)
                    intra_dist[i] = np.mean(np.linalg.norm(clusters_coord[i] - clusters_centroid[i], axis=1))
                compactness = np.sum(intra_dist * cluster_points_len) / np.sum(cluster_points_len)
                # 3. Separation: minimum inter-cluster centroid distance (higher is better).
                if n_clusters > 1:
                    dist_matrix = np.linalg.norm(clusters_centroid[:, np.newaxis] - clusters_centroid, axis=2)
                    np.fill_diagonal(dist_matrix, np.inf)
                    min_inter_dist = np.min(dist_matrix)
                else:
                    min_inter_dist = separation_max  # Single cluster: treat as maximally separated.
                # Objectives are negated where "higher is better" (MOPSO minimizes).
                all_objectives.append([-coverage_rate, compactness, -min_inter_dist])
    # Weight each frame's objective contribution by its point count.
    if len(frame_point_len) > 0:
        total_points = sum(frame_point_len)
        weights = np.array(frame_point_len) / total_points
        weighted_objectives = [np.average([obj[i] for obj in all_objectives], weights=weights)
                                for i in range(len(all_objectives[0]))]
        all_objectives = weighted_objectives
    else:
        big_number = 1000000
        all_objectives = [big_number, big_number, big_number]  # No valid frames: penalize heavily.
    return all_objectives


def calculate_sample_size(population_size, confidence_level=0.5, margin_of_error=0.05, proportion=0.5):
    """
    Compute the Cochran's-formula sample size for frame subsampling.

    Args:
        population_size: total number of available point cloud files.
        confidence_level: desired confidence level (e.g. 0.95 for 95%).
        margin_of_error: allowed margin of error (default 0.05, i.e. +/-5%).
        proportion: assumed population proportion (default 0.5, maximum variance).

    Returns:
        Required sample size, clipped to [30, population_size].
    """
    z = norm.ppf((1 + confidence_level) / 2)
    print(confidence_level, z)
    numerator = z ** 2 * proportion * (1 - proportion) * population_size
    denominator = z ** 2 * proportion * (1 - proportion) + (margin_of_error ** 2 * (population_size - 1))
    sample_size = ceil(numerator / denominator)
    sample_size = max(30, min(sample_size, population_size))
    return sample_size


def main(args, min_height=0.01, max_height=3):
    columns_len = len(args.columns)
    pcd_file_paths = sorted(glob(os.path.join(args.input_folder, '*.bin')))
    print(f"Found {len(pcd_file_paths)} PCD files")
    print("Optimizing parameters...")
    pcd_arrays = []
    total_files = len(pcd_file_paths)
    sample_size = calculate_sample_size(total_files, confidence_level=0.95)
    print(f"Recommended sample size: {sample_size}")
    interval = max(1, total_files // sample_size)
    for i in range(0, total_files, interval):
        pcd_array = read_bin(pcd_file_paths[i], columns_len)
        filter_mask = (pcd_array[:, args.columns.index('z')] >= min_height) & (
                pcd_array[:, args.columns.index('z')] <= max_height) & (
                              pcd_array[:, args.columns.index('dynamic_mask')] == 1)
        filtered_pcd_array = pcd_array[filter_mask]
        pcd_arrays.append(filtered_pcd_array)

    obj_func_params = {'scene_data': pcd_arrays, 'columns': args.columns}
    obj_name = ["coverage_rate", "compactness", "separation"]
    mopso = MOPSO(objective_function, args.param_bounds, obj_func_params, obj_name=obj_name,
                  swarm_size=30, max_iter=30)
    print("=== Starting MOPSO optimization ===")
    pareto_front = mopso.optimize()

    if pareto_front:
        ensure_directory(args.output_folder)
        objective_names = ["coverage_rate", "compactness", "separation"]
        pareto_front_path = os.path.join(args.output_folder, "pareto_front.csv")
        pareto_solutions_path = os.path.join(args.output_folder, "pareto_solutions.csv")

        objectives_data = [particle.objectives for particle in pareto_front]
        objectives_df = pd.DataFrame(objectives_data, columns=objective_names)
        objectives_df.to_csv(pareto_front_path, index=False)
        print(f"Saved Pareto-front objective values to {pareto_front_path}")

        param_names = ['alpha', 'rho_min']
        solutions_data = [particle.position for particle in pareto_front]
        solutions_df = pd.DataFrame(solutions_data, columns=param_names)
        solutions_df.to_csv(pareto_solutions_path, index=False)
        print(f"Saved Pareto-front solutions to {pareto_solutions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOPSO parameter optimization for MulDet3D Stage-1 clustering")
    parser.add_argument("--output_folder", default="results/optimize_stage_1", help="Folder to save output files")
    parser.add_argument("--input_folder", default="bin/merge_1000", help="Folder containing input point cloud files")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'dynamic_mask', 'reliability', 'lidar_source']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    params_bounds = [
        (0, 1),    # alpha, decay rate for adaptive radius
        (10, 30)   # rho_min, minimum density threshold
    ]
    parser.add_argument("--param_bounds", default=params_bounds, help="Bounds for optimization parameters")
    args = parser.parse_args()
    main(args)
