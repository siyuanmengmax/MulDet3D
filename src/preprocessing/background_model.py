"""Reliability-weighted Gaussian background modeling (MulDet3D preprocessing stage).

Shares its physics-based point reliability model and background-modeling
formulation with FRGB3D (Meng et al., 2026, Journal of Computing in Civil
Engineering) -- MulDet3D's own paper explicitly reuses this reliability
metric for its preprocessing stage. Default parameters here follow
MulDet3D's own fixed-parameter table (tau_0 = 0.1 m, c_min = 10 frames),
which differ from FRGB3D's paper (tau_0 = 1 m); the underlying formulas are
otherwise identical.
"""
import numpy as np
from numba import jit, prange
from tqdm import tqdm

from ..utils.io import read_bin
from ..utils.visualization import visualize_dynamic_points


@jit(nopython=True, parallel=True)
def calculate_reliability(ranges, reflectivity):
    """
    Physics-based point reliability model for LiDAR measurements.

    Combines a range-dependent, reflectivity-modulated precision estimate
    into a reliability score in [0, 1]. The reflectivity coefficient a(refl)
    is piecewise-linearly interpolated over the 0-100% reflectivity range,
    anchored at the 10% and 90% calibration points (Ouster OS1 datasheet).

    Args:
        ranges: Point range values (meters).
        reflectivity: Point reflectivity values (0-255, sensor raw units).

    Returns:
        Per-point reliability values in [0, 1].
    """
    reliability = np.zeros_like(ranges, dtype=np.float32)
    min_std = 0.5

    a_low = (3 - min_std) / (90.0 * 90.0)      # coefficient at 10% reflectivity
    a_high = (0.8 - min_std) / (90.0 * 90.0)   # coefficient at 90% reflectivity
    refl_factor = (a_low - a_high) / (90.0 - 10.0)

    a_at_0_percent = a_low + 10.0 * refl_factor
    a_at_100_percent = max(0.0, a_high - 10.0 * refl_factor)

    for i in prange(len(ranges)):
        r = ranges[i]
        refl = (reflectivity[i] / 255.0) * 100.0

        if refl == 0.0:
            a = a_at_0_percent
        elif refl < 10.0:
            t = refl / 10.0
            a = a_at_0_percent * (1.0 - t) + a_low * t
        elif refl == 10.0:
            a = a_low
        elif 10 < refl < 90.0:
            a = a_low - (refl - 10.0) * refl_factor
        elif refl == 90.0:
            a = a_high
        elif refl < 100.0:
            t = (refl - 90.0) / 10.0
            a = a_high * (1.0 - t) + a_at_100_percent * t
        else:
            a = a_at_100_percent

        precision_std = a * (r * r) + min_std
        reliability[i] = max(0.0, 1.0 - 0.01 * precision_std)

    return reliability


@jit(nopython=True, parallel=True)
def update_background_model(mean, std, weight, mean_reflectivity, counts,
                             indices, ranges, reliability, reflectivity, learning_rate, total_points):
    """Reliability-weighted incremental update of the per-point background model.

    mu_i <- mu_i + alpha_i * (d(p_i) - mu_i) / w_i,  alpha_i = eta * r(p_i)
    """
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= total_points:
            continue
        r = ranges[i]
        rel = reliability[i]
        refl = reflectivity[i]
        counts[idx] += 1
        if weight[idx] == 0:
            mean[idx] = r
            weight[idx] = rel
            mean_reflectivity[idx] = refl
        else:
            adaptive_lr = learning_rate * rel
            diff = r - mean[idx]
            diff_refl = refl - mean_reflectivity[idx]
            weight[idx] += adaptive_lr
            mean[idx] += adaptive_lr * diff / weight[idx]
            mean_reflectivity[idx] += adaptive_lr * diff_refl / weight[idx]
            std[idx] = np.sqrt((1 - adaptive_lr) * (std[idx] ** 2) + adaptive_lr * (diff ** 2))
    return mean, std, weight, mean_reflectivity, counts


@jit(nopython=True, parallel=True)
def detect_foreground_points(indices, ranges, reliability, reflectivity, mean, std, mean_reflectivity,
                              counts, min_count, base_threshold, threshold_refl, foreground_mask, range_factor):
    """Reliability-weighted foreground point detection against the background model.

    delta_i = r(p_i) * |d(p_i) - mu_i|;  tau_i = tau_0 + sigma_i;  foreground iff delta_i > tau_i.
    """
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= len(mean):
            continue
        r = ranges[i]
        rel = reliability[i]
        refl = reflectivity[i]
        if counts[idx] >= min_count and mean[idx] > 0:
            diff = abs(mean[idx] - r) * rel
            # Adaptive threshold: tau_i = tau_0 + sigma_i (no reliability term, per paper Eq.).
            adaptive_threshold = base_threshold + std[idx]
            diff_refl = abs(refl - mean_reflectivity[idx])
            if diff > adaptive_threshold and diff_refl > threshold_refl:
                foreground_mask[indices[i]] = True
    return foreground_mask


def build_background(pcd_files, min_range=0.5, max_range=200,
                      learning_rate=0.01, init_std=1, sample_size=1000,
                      columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Build the reliability-weighted background model from a sequence of frames."""
    total_points = len(read_bin(pcd_files[0], len(columns)))
    mean = np.zeros(total_points, dtype=np.float32)
    std = np.ones(total_points, dtype=np.float32) * init_std
    weight = np.zeros(total_points, dtype=np.float32)
    mean_reflectivity = np.zeros(total_points, dtype=np.float32)
    counts = np.zeros(total_points, dtype=np.int32)

    interval = max(1, len(pcd_files) // sample_size)
    for i in tqdm(range(0, len(pcd_files), interval), desc="Building background model"):
        pcd_array = read_bin(pcd_files[i], len(columns))
        ranges = pcd_array[:, columns.index('range')]
        reflectivity = pcd_array[:, columns.index('reflectivity')]
        indices = np.arange(len(pcd_array))
        mask = (ranges > min_range) & (ranges < max_range)
        valid_ranges = ranges[mask]
        valid_reflectivity = reflectivity[mask]
        valid_indices = indices[mask]

        valid_reliability = calculate_reliability(valid_ranges, valid_reflectivity)

        mean, std, weight, mean_reflectivity, counts = update_background_model(
            mean, std, weight, mean_reflectivity, counts,
            valid_indices, valid_ranges, valid_reliability, valid_reflectivity, learning_rate, total_points
        )

    bg_model = np.column_stack([mean, std, mean_reflectivity, counts])
    valid_points = counts > 0
    print(f"Background model built: {np.sum(valid_points)}/{total_points} valid points "
          f"({np.sum(valid_points) / total_points:.2%})")
    return bg_model


def detect_foreground(pcd_array, bg_model, base_threshold=0.1, base_threshold_refl=0, min_count=10, min_range=0.5,
                       max_range=200, range_factor=1,
                       visualize=False, bg_model_columns=['mean', 'std', 'mean_reflectivity', 'counts'],
                       columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Detect foreground points in a frame given a fitted background model.

    base_threshold defaults to 0.1 m, matching MulDet3D's fixed-parameter
    table (tau_0); min_count defaults to 10 frames (c_min).
    """
    mean = bg_model[:, bg_model_columns.index('mean')]
    std = bg_model[:, bg_model_columns.index('std')]
    mean_reflectivity = bg_model[:, bg_model_columns.index('mean_reflectivity')]
    counts = bg_model[:, bg_model_columns.index('counts')]

    ranges = pcd_array[:, columns.index('range')]
    reflectivity = pcd_array[:, columns.index('reflectivity')]
    indices = np.arange(len(pcd_array))
    mask = (ranges > min_range) & (ranges < max_range)
    valid_ranges = ranges[mask]
    valid_reflectivity = reflectivity[mask]
    valid_indices = indices[mask]

    valid_reliability = calculate_reliability(valid_ranges, valid_reflectivity)

    all_reliability = np.zeros(len(pcd_array), dtype=np.float32)
    all_reliability[mask] = valid_reliability

    foreground_mask = np.zeros(len(pcd_array), dtype=np.bool_)
    foreground_mask = detect_foreground_points(
        valid_indices, valid_ranges, valid_reliability, valid_reflectivity,
        mean, std, mean_reflectivity, counts, min_count, base_threshold, base_threshold_refl, foreground_mask,
        range_factor
    )

    if visualize:
        dynamic_points = pcd_array[foreground_mask]
        static_points = pcd_array[~foreground_mask]
        visualize_dynamic_points(dynamic_points, static_points)
    return foreground_mask, all_reliability
