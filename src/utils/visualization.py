# visualization.py

import os
from glob import glob

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .io import create_pcd, read_bin


def visualize_dynamic_points(dynamic_points_array, static_points_array):
    """Visualize foreground ("dynamic", red) and background ("static", gray) points."""
    dynamic_points = create_pcd(dynamic_points_array[:, :3])
    static_points = create_pcd(static_points_array[:, :3])
    dynamic_points.paint_uniform_color([1, 0, 0])
    static_points.paint_uniform_color([0.5, 0.5, 0.5])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(dynamic_points)
    vis.add_geometry(static_points)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()


def visualize_ground(ground_points, non_ground_points):
    """Visualize RANSAC-segmented ground (red) vs. non-ground (gray) points."""
    ground_points.paint_uniform_color([1, 0, 0])
    non_ground_points.paint_uniform_color([0.5, 0.5, 0.5])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(ground_points)
    vis.add_geometry(non_ground_points)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()


def fuse_visualize(pcd_source_array, pcd_target_array):
    """Visualize two point clouds (e.g. before/after registration) with a
    ground grid and coordinate frame for reference."""
    pcd_source = create_pcd(pcd_source_array)
    pcd_target = create_pcd(pcd_target_array)
    pcd_source.paint_uniform_color([1, 0, 0])  # Source cloud in red.
    pcd_target.paint_uniform_color([0, 0, 1])  # Target cloud in blue.
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.5
    vis.add_geometry(pcd_source)
    vis.add_geometry(pcd_target)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    vis.add_geometry(target_frame)

    # Ground grid (XY plane), spaced 5 units apart.
    grid_size = 5
    grid_extent = 50
    grid_color = [0.5, 0.5, 0.5]
    for i in range(-grid_extent, grid_extent + 1, grid_size):
        line_x = o3d.geometry.LineSet()
        line_x.points = o3d.utility.Vector3dVector([[i, -grid_extent, 0], [i, grid_extent, 0]])
        line_x.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_x.colors = o3d.utility.Vector3dVector([grid_color])
        vis.add_geometry(line_x)
        line_y = o3d.geometry.LineSet()
        line_y.points = o3d.utility.Vector3dVector([[-grid_extent, i, 0], [grid_extent, i, 0]])
        line_y.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_y.colors = o3d.utility.Vector3dVector([grid_color])
        vis.add_geometry(line_y)

    view_control = vis.get_view_control()
    front_view = [0, 0, 1]
    up_vector = [0, 1, 0]
    zoom = 0.2
    vis.poll_events()
    vis.update_renderer()
    view_control.set_front(front_view)
    view_control.set_up(up_vector)
    view_control.set_zoom(zoom)
    vis.run()
    vis.destroy_window()


def visualize_frame(pcd_array, cluster_ids, bboxes, frame_id, classes):
    """2D top-down matplotlib visualization of one frame's clusters and bounding boxes."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]

    mask_noise = cluster_ids == -1
    ax.scatter(pcd_array[mask_noise, 0], pcd_array[mask_noise, 1], c='gray', s=0.5, alpha=0.3)

    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = colors[i % len(colors)]
        ax.scatter(pcd_array[mask, 0], pcd_array[mask, 1], c=[color], s=0.5, alpha=0.5)

    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x'] - bbox['w'] / 2, bbox['y'] - bbox['l'] / 2),
            bbox['w'], bbox['l'],
            angle=np.rad2deg(bbox['yaw']),
            edgecolor='red', facecolor='none', alpha=1, linewidth=1.0,
            rotation_point='center'
        )
        ax.add_patch(rect)
        text = f"{classes[bbox['label']]},{bbox['confidence']:.6f}"
        ax.text(bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['l'] / 2,
                text, color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='bottom')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Frame: {frame_id}', fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_frame_for_video(pcd_array, cluster_ids, bboxes, frame_id, classes, fig, ax):
    """Per-frame 2D visualization used when assembling a point cloud video."""
    ax.clear()
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]

    mask_noise = cluster_ids == -1
    ax.scatter(pcd_array[mask_noise, 0], pcd_array[mask_noise, 1], c='gray', s=0.5, alpha=0.3)

    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = colors[i % len(colors)]
        ax.scatter(pcd_array[mask, 0], pcd_array[mask, 1], c=[color], s=0.5, alpha=0.5)

    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x'] - bbox['w'] / 2, bbox['y'] - bbox['l'] / 2),
            bbox['w'], bbox['l'],
            angle=np.rad2deg(bbox['yaw']),
            edgecolor='red', facecolor='none', alpha=1, linewidth=1.0,
            rotation_point='center'
        )
        ax.add_patch(rect)
        text = f"{classes[bbox['label']]},{bbox['confidence']:.2f}"
        ax.text(bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['l'] / 2,
                text, color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='bottom')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Frame: {frame_id}', fontsize=10)
    return fig


def create_point_cloud_video(pcd_files, output_video_path, process_frame_func, args, fps=10):
    """Render a sequence of processed frames (via `process_frame_func`) into an MP4 video."""
    print("=== Creating Point Cloud Video ===")

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    plt.tight_layout()
    canvas = FigureCanvas(fig)

    temp_frames_dir = os.path.join(output_dir, 'temp_frames')
    if not os.path.exists(temp_frames_dir):
        os.makedirs(temp_frames_dir)

    columns_len = len(args.columns)
    classes = ['person', 'small vehicle', 'large vehicle', 'other']

    print(f"Processing {len(pcd_files)} frames...")
    for i, pcd_file_path in tqdm(enumerate(pcd_files)):
        frame_name = os.path.basename(pcd_file_path)
        pcd_array = read_bin(pcd_file_path, columns_len)

        filter_mask = (pcd_array[:, args.columns.index('z')] >= 0.1) & (
                pcd_array[:, args.columns.index('z')] <= 3) & (
                              pcd_array[:, args.columns.index('dynamic_mask')] == 1)
        filtered_pcd_array = pcd_array[filter_mask]

        frame_bboxes, processed_pcd_array = process_frame_func(i, frame_name, pcd_array, filtered_pcd_array, args)

        cluster_id = processed_pcd_array[:, -1]  # Cluster id is appended as the last column.
        fig = visualize_frame_for_video(processed_pcd_array, cluster_id, frame_bboxes, i, classes, fig, ax)

        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
        cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print("Creating video...")
    frame_sample = cv2.imread(os.path.join(temp_frames_dir, f'frame_{0:06d}.jpg'))
    height, width, _ = frame_sample.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i in tqdm(range(len(pcd_files))):
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    plt.close(fig)

    import shutil
    shutil.rmtree(temp_frames_dir)

    print(f"Video created successfully: {output_video_path}")
    return output_video_path


def visualize_frame_3d(pcd_array, cluster_ids, bboxes, frame_id):
    """
    3D point cloud and bounding-box visualization for one frame using Open3D.

    Args:
        pcd_array: (N, 3+) array; the first three columns are x, y, z.
        cluster_ids: per-point cluster id, shape (N,); unclustered points are -1.
        bboxes: list of bounding boxes, each a dict or [x,y,z,w,l,h,yaw] sequence.
        frame_id: current frame id, used in the window title.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id}", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])

    colors = np.ones((len(pcd_array), 3)) * 0.5  # Default gray for unclustered points.
    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    color_map = plt.cm.get_cmap('tab10', max(10, len(unique_clusters)))
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = color_map(i % 10)[:3]
        colors[mask] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, dict):
            center = [bbox['x'], bbox['y'], bbox['z']]
            size = [bbox['w'], bbox['l'], bbox['h']]
            yaw = bbox['yaw']
        else:
            center = [bbox[0], bbox[1], bbox[2]]
            size = [bbox[3], bbox[4], bbox[5]]
            yaw = bbox[6]
        bbox_o3d = create_oriented_bounding_box(center, size, yaw)
        color_idx = i % len(unique_clusters) if len(unique_clusters) > 0 else 0
        bbox_color = color_map(color_idx)[:3]
        # LineSet colors are set per line, not per vertex.
        bbox_o3d.colors = o3d.utility.Vector3dVector(
            [[bbox_color[0], bbox_color[1], bbox_color[2]] for _ in range(len(bbox_o3d.lines))])
        vis.add_geometry(bbox_o3d)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.1)
    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.9, 0.9, 0.9])
    opt.point_size = 3
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def create_oriented_bounding_box(center, size, yaw):
    """
    Build an oriented bounding box as an Open3D LineSet.

    Args:
        center: box center [x, y, z].
        size: box dimensions [width, length, height].
        yaw: rotation about the z-axis (radians).

    Returns:
        o3d.geometry.LineSet representing the box wireframe.
    """
    w, l, h = size
    w_half, l_half, h_half = w / 2, l / 2, h / 2
    vertices = [
        [-w_half, -l_half, -h_half],
        [w_half, -l_half, -h_half],
        [w_half, l_half, -h_half],
        [-w_half, l_half, -h_half],
        [-w_half, -l_half, h_half],
        [w_half, -l_half, h_half],
        [w_half, l_half, h_half],
        [-w_half, l_half, h_half]
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
    ]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    rotated_vertices = []
    for vertex in vertices:
        rotated = np.dot(R, vertex)
        translated = [rotated[0] + center[0], rotated[1] + center[1], rotated[2] + center[2]]
        rotated_vertices.append(translated)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    return line_set


def oriented_bounding_box(center, size, yaw):
    """
    Manually construct an OrientedBoundingBox, compatible across Open3D versions.

    Args:
        center: box center [x, y, z].
        size: box dimensions [width (x-axis), length (y-axis), height (z-axis)].
        yaw: rotation about the z-axis (radians).

    Returns:
        o3d.geometry.OrientedBoundingBox (or a LineSet fallback if construction fails).
    """
    center = np.array(center, dtype=np.float64)
    size = np.array(size, dtype=np.float64)
    yaw = float(yaw)

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [-w, -l, -h],
        [w, -l, -h],
        [w, l, -h],
        [-w, l, -h],
        [-w, -l, h],
        [w, -l, h],
        [w, l, h],
        [-w, l, h]
    ], dtype=np.float64)
    corners = (R @ corners.T).T + center

    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))

    if obb.volume() <= 0:
        try:
            obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=size)
        except Exception:
            # Fall back to a plain wireframe if the OBB constructor rejects a
            # degenerate box (e.g. a single-point cluster).
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(corners)
            lines.lines = o3d.utility.Vector2iVector([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ])
            return lines

    return obb


def create_bbox_lineset(center, size, yaw, color=[1, 0, 0]):
    """Build a colored wireframe LineSet for a bounding box (see `create_oriented_bounding_box`)."""
    center = np.array(center, dtype=np.float64)
    size = np.array(size, dtype=np.float64)
    yaw = float(yaw)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [-w, -l, -h],
        [w, -l, -h],
        [w, l, -h],
        [-w, l, -h],
        [-w, -l, h],
        [w, -l, h],
        [w, l, h],
        [-w, l, h]
    ], dtype=np.float64)
    corners = (R @ corners.T).T + center
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    lines.colors = o3d.utility.Vector3dVector([color] * len(lines.lines))
    return lines


def parse_bbox(bbox):
    """Parse a bounding box (dict or [x,y,z,w,l,h,yaw] sequence) into (center, size, yaw)."""
    if isinstance(bbox, dict):
        if 'position' in bbox:
            center = bbox['position']
            size = bbox['dimensions']
            yaw = bbox['rotation']
        else:
            center = [bbox.get('x', 0), bbox.get('y', 0), bbox.get('z', 0)]
            size = [bbox.get('w', 0), bbox.get('l', 0), bbox.get('h', 0)]
            yaw = bbox.get('yaw', 0)
    else:
        center = [bbox[0], bbox[1], bbox[2]]
        size = [bbox[3], bbox[4], bbox[5]]
        yaw = bbox[6]
    center = [float(c) for c in center]
    size = [float(s) for s in size]
    yaw = float(yaw)
    return center, size, yaw


def visualize_3d_bboxes(gt_data, det_data, pcd_folder, column_len):
    """
    Visualize, frame by frame, ground-truth (red) vs. detected (blue) 3D
    bounding boxes overlaid on the point cloud.

    Args:
        gt_data: {frame_name: [bbox, ...]} ground-truth boxes.
        det_data: {frame_name: [bbox, ...]} detected boxes.
        pcd_folder: folder containing the corresponding .bin point cloud frames.
        column_len: number of columns per point in the .bin files.
    """
    pcd_files = sorted(glob(os.path.join(pcd_folder, '*.bin')))
    for frame_id, pcd_file in enumerate(pcd_files):
        frame_name = os.path.basename(pcd_file)
        print(frame_name)
        gt_bboxes = gt_data.get(frame_name, [])
        det_bboxes = det_data.get(frame_name, [])
        if not gt_bboxes and not det_bboxes:
            continue

        pcd_array = read_bin(pcd_file, column_len)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Frame {frame_name}", width=1280, height=720)
        pcd = create_pcd(pcd_array)
        colors = np.ones((len(pcd_array), 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)

        all_centers = []
        for i, bbox in enumerate(gt_bboxes):
            center, size, yaw = parse_bbox(bbox)
            all_centers.append(center)
            lines = create_bbox_lineset(center, size, yaw, [1, 0, 0])  # Red: ground truth.
            vis.add_geometry(lines)
            bbox_o3d = oriented_bounding_box(center, size, yaw)
            bbox_o3d.color = [1, 0, 0]
            vis.add_geometry(bbox_o3d)
        for i, bbox in enumerate(det_bboxes):
            center, size, yaw = parse_bbox(bbox)
            all_centers.append(center)
            lines = create_bbox_lineset(center, size, yaw, [0, 0, 1])  # Blue: detections.
            vis.add_geometry(lines)
            bbox_o3d = oriented_bounding_box(center, size, yaw)
            bbox_o3d.color = [0, 0, 1]
            vis.add_geometry(bbox_o3d)

        if all_centers:
            center_array = np.array(all_centers)
            center_mean = np.mean(center_array, axis=0)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.3)
            ctr.set_front([0, 0, 1])
            ctr.set_lookat(center_mean)
            ctr.set_up([0, 1, 0])
        else:
            ctr = vis.get_view_control()
            ctr.set_zoom(0.1)
            ctr.set_front([0, 0, 1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.9, 0.9, 0.9])
        opt.point_size = 1
        opt.line_width = 5.0
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        import time
        time.sleep(0.1)
