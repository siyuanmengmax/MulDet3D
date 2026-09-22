import numpy as np

from .compute_bbox import compute_bbox


def classifying_stage_1(clusters_coords):
    """
    Coarse two-way classification of clusters into "vehicle part" vs "other",
    using simple geometric/density heuristics on each cluster's bounding box.

    Args:
        clusters_coords: list of per-cluster point coordinates.

    Returns:
        coarse_classes: list of labels (0: other, 1: vehicle part).
    """
    coarse_classes = []

    for cluster_points in clusters_coords:
        try:
            bbox = compute_bbox(cluster_points)
            n_points = len(cluster_points)
            x, y, z, width, length, height, _ = bbox
            volume = width * length * height
            density = n_points / volume if volume > 0.0001 else 0
            aspect_ratio = max(width, length) / (min(width, length) + 1e-6)
            min_height = np.min(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
            max_height = np.max(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
            height_mean = 0.5 * (min_height + max_height) if len(cluster_points) > 0 else 0
            height_std = np.std(cluster_points[:, 2]) if len(cluster_points) > 1 else 0
            if len(cluster_points) >= 3:
                cov_matrix = np.cov(cluster_points[:, :2].T)  # XY plane only
                if cov_matrix.shape == (2, 2):
                    eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
                    xy_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)  # principal-axis ratio in the XY plane
                else:
                    xy_ratio = 1.0
            else:
                xy_ratio = 1.0

            vehicle_part_score = 0
            # Density: very sparse clusters are likely scattered foliage etc.
            if density < 10:
                vehicle_part_score -= 10
            elif density > 100:
                vehicle_part_score += 1.0
            # 1. Volume range check.
            if 8 <= volume <= 120.0:
                vehicle_part_score += 2.0
            elif volume < 1.0:
                vehicle_part_score -= 5.0
            # 2. Aspect ratio check (vehicles are typically elongated).
            if 1.5 <= aspect_ratio <= 4:
                vehicle_part_score += 1.5
            elif aspect_ratio > 4:
                vehicle_part_score -= 1.0
            # 3. Height check (vehicles fall within a typical height range).
            if 0.5 <= height <= 3.0:
                vehicle_part_score += 1.0
            else:
                vehicle_part_score -= 5
            # 4. Height distribution (vehicle parts usually have low height variance).
            if height_std < 0.3 * height:
                vehicle_part_score += 1.0
            # 5. Principal-axis ratio (vehicles usually have a clear XY-plane orientation).
            if xy_ratio > 2.0:
                vehicle_part_score += 1.5
            # 6. Centroid height (vehicles are typically close to the ground).
            if 0.5 <= height_mean < 1.8:
                vehicle_part_score += 1.0
            else:
                vehicle_part_score -= 10
            # 7. Typical vehicle size combinations.
            if (1.5 <= width <= 2.5 and 3.5 <= length <= 6.0) or \
                    (2.0 <= width <= 3.0 and 5.0 <= length <= 15.0):
                vehicle_part_score += 2.0

            is_vehicle_part = vehicle_part_score >= 5.0
            coarse_classes.append(1 if is_vehicle_part else 0)
        except Exception as e:
            print(f"Error in coarse classification: {e}")
            coarse_classes.append(0)  # Default to "other" on error.
    return coarse_classes


def classifying_stage_2(clusters_coords, coarse_classes=None):
    """
    Fine-grained classification using a hand-crafted utility function per
    class, converted to a probability distribution via softmax.

    Args:
        clusters_coords: per-cluster point coordinates.
        coarse_classes: optional coarse classification results (unused here,
            kept for interface compatibility with the calling pipeline).

    Returns:
        cluster_classes: predicted class label per cluster
            (0: person, 1: small vehicle, 2: large vehicle, 3: other).
        class_probabilities: predicted-class probability per cluster.
    """
    cluster_classes = []
    class_probabilities = []

    for i, cluster_points in enumerate(clusters_coords):
        try:
            bbox = compute_bbox(cluster_points)
            x, y, z, width, length, height, _ = bbox
            volume = width * length * height
            n_points = len(cluster_points)
            density = n_points / volume if volume > 0.0001 else 0
            length = max(length, width)
            width = min(length, width)
            aspect_ratio = length / (width + 1e-6)
            height_ratio = height / (length + 1e-6)

            utility_scores = np.zeros(4)  # [person, small vehicle, large vehicle, other]

            # ======== Person utility ========
            if 1 <= height <= 2.0:
                utility_scores[0] += 2
            if width <= 0.8 and length <= 0.8:
                utility_scores[0] += 3
            if height_ratio >= 1.5:
                utility_scores[0] += 3
            if volume < 1.0:
                utility_scores[0] += 2
            if density >= 10:
                utility_scores[0] += 1
            else:
                utility_scores[0] -= 2
            if 0.5 <= z < 1.5:
                utility_scores[0] += 1
            else:
                utility_scores[0] -= 10

            # ======== Small vehicle utility ========
            if 1.0 <= height <= 2.2:
                utility_scores[1] += 1
            if 1.5 <= width <= 2.5:
                utility_scores[1] += 2
            if 2.5 <= length <= 4:
                utility_scores[1] += 2
            if 1.5 <= aspect_ratio <= 4.0:
                utility_scores[1] += 2
            if 8 < volume <= 40:
                utility_scores[1] += 3
            if density >= 10:
                utility_scores[1] += 1
            else:
                utility_scores[1] -= 2
            if 0.5 <= z < 1.5:
                utility_scores[1] += 1
            else:
                utility_scores[1] -= 10

            # ======== Large vehicle utility ========
            if 2.5 <= height <= 4.5:
                utility_scores[2] += 1
            if 2.0 <= width <= 3.0:
                utility_scores[2] += 2
            if 5.0 <= length <= 15.0:
                utility_scores[2] += 2
            if aspect_ratio >= 2.0:
                utility_scores[2] += 2
            if volume > 40.0:
                utility_scores[2] += 3
            if density >= 10:
                utility_scores[2] += 1
            else:
                utility_scores[2] -= 2
            if 1 <= z < 2.0:
                utility_scores[2] += 1
            else:
                utility_scores[2] -= 10

            # If the top class score is too low, fall back to "other".
            max_class = np.argmax(utility_scores)
            if utility_scores[max_class] < 6:
                utility_scores[3] += 12

            # Softmax over utility scores to obtain class probabilities.
            temperature = 1.0
            utility_scores = utility_scores / temperature
            utility_scores = utility_scores - np.max(utility_scores)  # for numerical stability
            exp_scores = np.exp(utility_scores)
            probs = exp_scores / np.sum(exp_scores)

            class_label = np.argmax(probs)
            cluster_classes.append(class_label)
            class_probabilities.append(probs)

        except Exception as e:
            print(f"Error classifying cluster {i}: {e}")
            cluster_classes.append(3)  # Default to "other" on error.
            class_probabilities.append(np.array([0.0, 0.0, 0.0, 1.0]))

    if class_probabilities:
        class_probabilities = np.max(class_probabilities, axis=1)
    else:
        class_probabilities = np.array([])
    return cluster_classes, class_probabilities
