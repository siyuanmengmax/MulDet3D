import numpy as np


def compute_bbox(points_coords):
    """Compute the minimum-area oriented bounding box (rotating calipers over
    the xy-plane projection) that strictly encloses all points in a cluster.

    Returns [x, y, z, w, l, h, yaw] (center, size, and yaw in radians).
    """
    if len(points_coords) == 0:
        return [0, 0, 0, 0, 0, 0, 0]

    min_coords = np.min(points_coords, axis=0)
    max_coords = np.max(points_coords, axis=0)
    centroid_coords = (min_coords + max_coords) / 2
    centered_points_coords = points_coords - centroid_coords
    centered_points_2d_coords = centered_points_coords[:, :2]

    best_yaw = 0.0
    min_area = float('inf')
    for angle_deg in np.arange(0, 90, 0.1):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        rotated_points_coords = np.dot(centered_points_2d_coords, R)
        min_x, min_y = np.min(rotated_points_coords, axis=0)
        max_x, max_y = np.max(rotated_points_coords, axis=0)
        width = max_x - min_x
        length = max_y - min_y
        area = width * length
        if area < min_area:
            min_area = area
            best_yaw = angle

    # Build the full 3D bounding box using the optimal yaw.
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    R_3d = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    rotated_points_3d_coords = np.dot(centered_points_coords, R_3d)
    rot_min_coords = np.min(rotated_points_3d_coords, axis=0)
    rot_max_coords = np.max(rotated_points_3d_coords, axis=0)
    size = rot_max_coords - rot_min_coords
    bbox_center_rotated_coords = (rot_min_coords + rot_max_coords) / 2
    bbox_center_coords = centroid_coords + np.dot(bbox_center_rotated_coords, R_3d.T)

    # Verify the box fully contains all points; expand slightly if floating
    # point rounding leaves any point marginally outside.
    tolerance = 1e-6
    points_bbox_coords = np.dot(points_coords - bbox_center_coords, R_3d)
    half_size = size / 2
    points_outside = np.any(np.abs(points_bbox_coords) > half_size + tolerance, axis=1)
    if np.any(points_outside):
        min_vals = np.min(points_bbox_coords, axis=0)
        max_vals = np.max(points_bbox_coords, axis=0)
        size_adjustment = np.zeros(3)
        for i in range(3):
            if min_vals[i] < -half_size[i] - tolerance:
                size_adjustment[i] += 2 * ((-half_size[i] - tolerance) - min_vals[i])
            if max_vals[i] > half_size[i] + tolerance:
                size_adjustment[i] += 2 * (max_vals[i] - (half_size[i] + tolerance))
        size += size_adjustment

    bbox = [
        bbox_center_coords[0],
        bbox_center_coords[1],
        bbox_center_coords[2],
        size[0],  # width
        size[1],  # length
        size[2],  # height
        best_yaw
    ]
    return bbox
