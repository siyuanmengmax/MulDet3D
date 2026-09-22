"""Object-level Average Precision (AP) evaluation.

Compares detected bounding boxes (from run_detection.py) against manually
annotated ground truth (Segments.ai export format) using rotated 2D IoU
matching, and reports AP per category plus the mean AP across categories.
Matches the paper's per-class thresholds (pedestrian AP@0.1, small vehicle
AP@0.3, large vehicle AP@0.5) when run three times with --category_id and
the matching --iou_threshold, or reports the aggregate "Vehicle" AP@0.3
column (small + large vehicle merged) with --unify_vehicles.

Example:
    python scripts/evaluate_detection_ap.py \\
        --groundtruth_json bin/lowell_1000_label.json \\
        --detection_csv results/detection/bboxes.csv \\
        --iou_threshold 0.3
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.visualization import visualize_3d_bboxes

# Ground-truth category ids (Segments.ai export).
CATEGORY_NAMES = {1: 'Small Vehicle', 2: 'Large Vehicle', 3: 'Pedestrian'}
UNIFIED_CATEGORY_NAMES = {1: 'Vehicle (Small + Large)', 3: 'Pedestrian'}

# Detection label ids (from the classifier in classifying_stage_2) mapped to
# the ground-truth category ids above.
DETECTION_LABEL_TO_CATEGORY = {
    1: 1,  # small vehicle -> Small Vehicle
    2: 2,  # large vehicle -> Large Vehicle
    0: 3,  # person -> Pedestrian
}


def load_groundtruth(json_file, unify_vehicles=False):
    """Load ground-truth boxes from a Segments.ai export.

    Returns {frame_id: [{'category_id', 'position', 'dimensions', 'rotation'}]}.
    """
    print(f"Loading groundtruth from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    gt_data = {}
    samples = data.get('dataset', {}).get('samples', [])
    for sample in samples:
        frame_id = sample.get('name', '')
        labels = sample.get('labels', {}).get('ground-truth', {}).get('attributes', {}).get('annotations', [])
        if not frame_id or not labels:
            continue
        frame_bboxes = []
        for label in labels:
            category_id = label.get('category_id')
            if category_id not in (1, 2, 3):
                continue
            position = label.get('position', {})
            dimensions = label.get('dimensions', {})
            yaw = label.get('yaw', {})
            if not all([category_id, position, dimensions, yaw]):
                continue
            if unify_vehicles and category_id in (1, 2):
                category_id = 1  # Merge Small Vehicle + Large Vehicle -> Vehicle.
            frame_bboxes.append({
                'category_id': category_id,
                'position': [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
                'dimensions': [dimensions.get('x', 0), dimensions.get('y', 0), dimensions.get('z', 0)],
                'rotation': yaw
            })
        if frame_bboxes:
            gt_data[frame_id] = frame_bboxes
    print(f"Loaded {len(gt_data)} frames with groundtruth annotations")
    return gt_data


def load_detections(csv_file, unify_vehicles=False):
    """Load detected boxes from run_detection.py's output CSV.

    Only small vehicle / large vehicle / person detections are kept and
    mapped onto the ground-truth category ids; 'other' detections are ignored.

    Returns {frame_id: [{'category_id', 'position', 'dimensions', 'rotation', 'confidence'}]}.
    """
    print(f"Loading detections from {csv_file}...")
    df = pd.read_csv(csv_file)
    required_columns = ['frame_id', 'label', 'confidence', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in detection CSV")

    det_data = {}
    for frame_name, group in df.groupby('frame_name'):
        frame_detections = []
        for _, row in group.iterrows():
            label_id = int(row['label'])
            category_id = DETECTION_LABEL_TO_CATEGORY.get(label_id)
            if category_id is None:
                continue
            if unify_vehicles and category_id in (1, 2):
                category_id = 1
            frame_detections.append({
                'category_id': category_id,
                'position': [row['x'], row['y'], row['z']],
                'dimensions': [row['w'], row['l'], row['h']],
                'rotation': row['yaw'],
                'confidence': row['confidence']
            })
        if frame_detections:
            det_data[frame_name] = frame_detections
    print(f"Loaded {len(det_data)} frames with detections")
    return det_data


def calculate_iou(bbox1, bbox2):
    """2D rotated-rectangle IoU between two boxes (dicts with position/dimensions/rotation)."""
    x1, y1 = bbox1['position'][0], bbox1['position'][1]
    w1, l1 = bbox1['dimensions'][0], bbox1['dimensions'][1]
    yaw1 = bbox1['rotation']

    x2, y2 = bbox2['position'][0], bbox2['position'][1]
    w2, l2 = bbox2['dimensions'][0], bbox2['dimensions'][1]
    yaw2 = bbox2['rotation']

    rect1 = ((x1, y1), (w1, l1), np.rad2deg(yaw1))
    rect2 = ((x2, y2), (w2, l2), np.rad2deg(yaw2))
    area1 = w1 * l1
    area2 = w2 * l2

    ret, int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if ret == cv2.INTERSECT_NONE:
        return 0
    elif ret == cv2.INTERSECT_FULL:
        return min(area1, area2) / max(area1, area2)
    if int_pts is None:
        return 0
    int_area = abs(cv2.contourArea(int_pts))
    union_area = area1 + area2 - int_area
    iou = int_area / union_area if union_area > 0 else 0
    return np.clip(iou, 0, 1)


def evaluate_detections(gt_data, det_data, iou_threshold, target_categories):
    """Evaluate detections against ground truth; returns (ap_per_class, mAP)."""
    print(f"Evaluating detections with IoU threshold {iou_threshold}...")

    all_detections = defaultdict(list)
    num_gt_per_class = defaultdict(int)
    frame_ids = sorted(gt_data.keys())

    for frame_id in tqdm(frame_ids):
        gt_bboxes = gt_data.get(frame_id, [])
        det_bboxes = det_data.get(frame_id, [])
        if not gt_bboxes or not det_bboxes:
            continue

        for gt_bbox in gt_bboxes:
            num_gt_per_class[gt_bbox['category_id']] += 1

        matched_gt_indices = {}
        det_bboxes_sorted = sorted(det_bboxes, key=lambda x: x['confidence'], reverse=True)

        for det_bbox in det_bboxes_sorted:
            category_id = det_bbox['category_id']
            confidence = det_bbox['confidence']
            if category_id not in matched_gt_indices:
                matched_gt_indices[category_id] = set()

            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_bbox['category_id'] == category_id and gt_idx not in matched_gt_indices[category_id]:
                    iou = calculate_iou(det_bbox, gt_bbox)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                is_match = True
                matched_gt_indices[category_id].add(best_gt_idx)
            else:
                is_match = False

            all_detections[category_id].append({'confidence': confidence, 'is_match': is_match})

    ap_per_class = {}
    for category_id in target_categories:
        detections = all_detections[category_id]
        num_gt = num_gt_per_class[category_id]
        if not detections or num_gt == 0:
            ap_per_class[category_id] = 0.0
            continue

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        for i, detection in enumerate(detections):
            if detection['is_match']:
                tp[i] = 1
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        recall = tp / max(num_gt, np.finfo(np.float64).eps)

        # 11-point interpolated AP.
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = 0 if np.sum(recall >= t) == 0 else np.max(precision[recall >= t])
            ap += p / 11
        ap_per_class[category_id] = ap

    valid_aps = [ap for cat, ap in ap_per_class.items() if cat in target_categories]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    return ap_per_class, mAP


def main(args):
    gt_data = load_groundtruth(args.groundtruth_json, unify_vehicles=args.unify_vehicles)
    det_data = load_detections(args.detection_csv, unify_vehicles=args.unify_vehicles)

    target_categories = [1, 3] if args.unify_vehicles else [1, 2, 3]
    names = UNIFIED_CATEGORY_NAMES if args.unify_vehicles else CATEGORY_NAMES

    ap_per_class, mAP = evaluate_detections(gt_data, det_data, args.iou_threshold, target_categories)

    print("\nEvaluation Results:")
    for category_id, ap in ap_per_class.items():
        category_name = names.get(category_id, f'Category {category_id}')
        print(f"{category_name} AP: {ap * 100:.2f}%")
    print(f"mAP: {mAP * 100:.2f}%")

    if args.visualize:
        columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'dynamic_mask', 'reliability', 'lidar_source']
        visualize_3d_bboxes(gt_data, det_data, args.pcd_folder, column_len=len(columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object-level AP evaluation")
    parser.add_argument("--pcd_folder", default="bin/merge_1000", help="Folder of point cloud frames (only used for --visualize)")
    parser.add_argument("--detection_csv", default="results/detection/bboxes.csv", help="Detection results CSV")
    parser.add_argument("--groundtruth_json", default="bin/lowell_1000_label.json", help="Ground-truth annotation JSON")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for a true positive")
    parser.add_argument("--unify_vehicles", action="store_true", default=False,
                         help="Merge Small Vehicle + Large Vehicle into a single 'Vehicle' category "
                              "(matches the paper's aggregate 'Vehicle AP@0.3' column)")
    parser.add_argument("--visualize", action="store_true", default=False, help="Show a 3D ground-truth vs. detection comparison")
    args = parser.parse_args()
    main(args)
