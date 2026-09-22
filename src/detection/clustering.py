import numpy as np
from scipy.spatial import cKDTree
import hdbscan

from .compute_bbox import compute_bbox


def clustering_stage_1(points, columns, rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1, density_unit=1):
    """MulDet3D Stage 1: reliability-weighted, density-adaptive clustering.

    Extends DBSCAN with a per-point adaptive search radius that shrinks in
    high-utility (well-observed, high-density) regions and grows towards
    eps_max elsewhere, and with a reliability-weighted density estimate
    (points are weighted by their FRGB3D-style reliability score rather
    than counted uniformly).

    Args:
        points: (N, M) point array, expected to include a 'reliability' column.
        columns: column name list used to index into `points`.
        rho_min: minimum reliability-weighted density threshold for a core point.
        alpha: adaptive-radius decay rate in [0, 1].
        eps_min, eps_max: bounds for the adaptive search radius.
        density_unit: scaling factor applied to the density estimate.

    Returns:
        clusters: list of per-cluster point index lists.
        clusters_coord: list of per-cluster point coordinate arrays.
        adaptive_radius: per-point adaptive radius used during clustering.
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]
    tree = cKDTree(point_coord)
    point_reliability = points[:, columns.index('reliability')]

    basic_point_neighbor_id = tree.query_ball_point(point_coord, r=eps_max)
    basic_point_density = np.array(
        [density_unit * np.sum(point_reliability[ids]) for ids in basic_point_neighbor_id])
    utility = 1 - np.clip(rho_min / basic_point_density, 0, 1)
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max)

    clusters = []
    clusters_coord = []
    processed = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        if not processed[i]:
            adaptive_point_neighbor_id = tree.query_ball_point(point_coord[i], r=adaptive_radius[i])
            adaptive_point_density = density_unit * np.sum(point_reliability[adaptive_point_neighbor_id])
            if adaptive_point_density >= rho_min:
                cluster_point_id = [i]
                points_to_process = list(adaptive_point_neighbor_id)
                processed[i] = True
                while points_to_process:
                    j = points_to_process.pop()
                    if not processed[j]:
                        processed[j] = True
                        new_adaptive_point_neighbor_id = tree.query_ball_point(point_coord[j], r=adaptive_radius[j])
                        new_adaptive_point_density = np.sum(point_reliability[new_adaptive_point_neighbor_id])
                        if new_adaptive_point_density >= rho_min:
                            new_points = [k for k in new_adaptive_point_neighbor_id if not processed[k]]
                            points_to_process.extend(new_points)
                        cluster_point_id.append(j)
                clusters.append(cluster_point_id)
                clusters_coord.append(point_coord[cluster_point_id])
    return clusters, clusters_coord, adaptive_radius


def clustering_stage_1_no_rel(points, columns, rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1, density_unit=1):
    """Ablation variant of Stage 1 with reliability weighting removed (all points weight 1).

    Used in the paper's ablation study to isolate the contribution of
    reliability-weighted density estimation (see paper Sec. "Ablation Study").
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]
    tree = cKDTree(point_coord)
    point_reliability = np.ones(n_points)  # Reliability weighting disabled.

    basic_point_neighbor_id = tree.query_ball_point(point_coord, r=eps_max)
    basic_point_density = np.array(
        [density_unit * np.sum(point_reliability[ids]) for ids in basic_point_neighbor_id])
    utility = 1 - np.clip(rho_min / basic_point_density, 0, 1)
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max)

    clusters = []
    clusters_coord = []
    processed = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        if not processed[i]:
            adaptive_point_neighbor_id = tree.query_ball_point(point_coord[i], r=adaptive_radius[i])
            adaptive_point_density = density_unit * np.sum(point_reliability[adaptive_point_neighbor_id])
            if adaptive_point_density >= rho_min:
                cluster_point_id = [i]
                points_to_process = list(adaptive_point_neighbor_id)
                processed[i] = True
                while points_to_process:
                    j = points_to_process.pop()
                    if not processed[j]:
                        processed[j] = True
                        new_adaptive_point_neighbor_id = tree.query_ball_point(point_coord[j], r=adaptive_radius[j])
                        new_adaptive_point_density = np.sum(point_reliability[new_adaptive_point_neighbor_id])
                        if new_adaptive_point_density >= rho_min:
                            new_points = [k for k in new_adaptive_point_neighbor_id if not processed[k]]
                            points_to_process.extend(new_points)
                        cluster_point_id.append(j)
                clusters.append(cluster_point_id)
                clusters_coord.append(point_coord[cluster_point_id])
    return clusters, clusters_coord, adaptive_radius


def clustering_stage_2(clusters, clusters_coord, cluster_classes, points, columns):
    """MulDet3D Stage 2: physically constrained hierarchical merging.

    Greedily merges nearby cluster pairs (excluding pairs both classified as
    "other") when the merged cluster satisfies volume and aspect-ratio
    constraints, closest-distance first, iterating until no further merge
    is possible. Prevents over-segmentation of vehicle parts into multiple
    clusters while avoiding merging distinct nearby objects.
    """
    params = {
        'max_volume': 120,
        'min_aspect_ratio': 1.5,
        'max_aspect_ratio': 5,
        'min_distance': 1.5
    }
    while True:
        n_clusters = len(clusters)
        if n_clusters <= 1:
            break
        merge_candidates = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if cluster_classes[i] == 0 and cluster_classes[j] == 0:
                    continue  # Only clusters with at least one "vehicle part" class are eligible.
                class_id = cluster_classes[i]
                min_distance = params['min_distance']
                max_volume = params['max_volume']
                min_aspect_ratio = params['min_aspect_ratio']
                max_aspect_ratio = params['max_aspect_ratio']

                if len(clusters_coord[i]) < len(clusters_coord[j]):
                    tree = cKDTree(clusters_coord[i])
                    dist, _ = tree.query(clusters_coord[j], k=1)
                else:
                    tree = cKDTree(clusters_coord[j])
                    dist, _ = tree.query(clusters_coord[i], k=1)
                min_dist = np.min(dist)
                if min_dist <= min_distance:
                    merged_points = clusters[i] + clusters[j]
                    merged_coords = np.row_stack([clusters_coord[i], clusters_coord[j]])
                    bbox = compute_bbox(merged_coords)
                    volume = bbox[3] * bbox[4] * bbox[5]
                    if volume <= max_volume:
                        width = min(bbox[3], bbox[4])
                        length = max(bbox[3], bbox[4])
                        if width > 0:
                            aspect_ratio = length / width
                            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                                merge_candidates.append({
                                    'distance': min_dist,
                                    'i': i,
                                    'j': j,
                                    'merged_points': merged_points,
                                    'merged_coords': merged_coords,
                                    'class_id': class_id
                                })

        if not merge_candidates:
            break

        # Greedy merge: closest pairs first, each cluster merged at most once per iteration.
        merge_candidates.sort(key=lambda x: x['distance'])
        merged_cluster_ids = set()
        new_clusters = []
        new_clusters_coord = []
        new_cluster_classes = []
        for candidate in merge_candidates:
            i, j = candidate['i'], candidate['j']
            if i not in merged_cluster_ids and j not in merged_cluster_ids:
                new_clusters.append(candidate['merged_points'])
                new_clusters_coord.append(candidate['merged_coords'])
                new_cluster_classes.append(candidate['class_id'])
                merged_cluster_ids.add(i)
                merged_cluster_ids.add(j)
        for i in range(n_clusters):
            if i not in merged_cluster_ids:
                new_clusters.append(clusters[i])
                new_clusters_coord.append(clusters_coord[i])
                new_cluster_classes.append(cluster_classes[i])

        if len(new_clusters) == n_clusters:
            break

        clusters = new_clusters
        clusters_coord = new_clusters_coord
        cluster_classes = new_cluster_classes
    return clusters, clusters_coord, cluster_classes


def dbscan_clustering(points, columns, eps=1, min_points=50):
    """
    Standard DBSCAN clustering (fixed eps / min_points), used as a baseline.

    Args:
        points: point cloud array, including coordinates and other attributes.
        columns: column name list, used to locate x/y/z fields.
        eps: DBSCAN neighborhood radius.
        min_points: DBSCAN minimum-points threshold.

    Returns:
        clusters: list of per-cluster point index lists.
        clusters_coord: list of per-cluster point coordinates.
        eps: per-point eps value (constant here, kept for interface compatibility).
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]
    tree = cKDTree(point_coord)
    labels = np.zeros(n_points, dtype=int) - 1  # -1: noise/unprocessed, >0: cluster id.
    cluster_id = 0

    for i in range(n_points):
        if labels[i] != -1:
            continue
        neighbor_indices = tree.query_ball_point(point_coord[i], r=eps)
        if len(neighbor_indices) < min_points:
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neighbor_indices)
        seeds.remove(i)
        j = 0
        while j < len(seeds):
            current_point = seeds[j]
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            elif labels[current_point] == 0:
                labels[current_point] = cluster_id
                current_neighbors = tree.query_ball_point(point_coord[current_point], r=eps)
                if len(current_neighbors) >= min_points:
                    for neighbor in current_neighbors:
                        if labels[neighbor] == 0 or labels[neighbor] == -1:
                            if neighbor not in seeds:
                                seeds.append(neighbor)
            j += 1

    clusters = []
    clusters_coord = []
    for c in range(1, cluster_id + 1):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) > 0:
            clusters.append(cluster_indices.tolist())
            clusters_coord.append(point_coord[cluster_indices])
    eps = np.full(n_points, eps)
    return clusters, clusters_coord, eps


def hdbscan_clustering(points, columns, min_cluster_size=50, min_samples=None,
                        cluster_selection_epsilon=0.5, alpha=1.0):
    """
    HDBSCAN clustering, used as a baseline.

    Args:
        points: point cloud array, including coordinates and other attributes.
        columns: column name list, used to locate x/y/z fields.
        min_cluster_size: minimum cluster size.
        min_samples: minimum neighbors for a core point (default: min_cluster_size).
        cluster_selection_epsilon: HDBSCAN cluster-selection epsilon.
        alpha: distance-metric alpha parameter.

    Returns:
        clusters: list of per-cluster point index lists.
        clusters_coord: list of per-cluster point coordinates.
        density_scores: per-point density proxy (from HDBSCAN's outlier scores).
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]

    if min_samples is None:
        min_samples = min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        metric='euclidean',
        algorithm='best',
        leaf_size=40,
    )
    cluster_labels = clusterer.fit_predict(point_coord)

    clusters = []
    clusters_coord = []
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Exclude the noise label (-1).
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        if len(cluster_indices) > 0:
            clusters.append(cluster_indices.tolist())
            clusters_coord.append(point_coord[cluster_indices])

    if hasattr(clusterer, 'outlier_scores_'):
        # Higher outlier_scores_ means more noise-like; use (1 - score) as a density proxy.
        density_scores = 1.0 - clusterer.outlier_scores_
        density_scores = np.maximum(density_scores, 0.1)
    else:
        density_scores = np.full(n_points, 0.5)

    return clusters, clusters_coord, density_scores
