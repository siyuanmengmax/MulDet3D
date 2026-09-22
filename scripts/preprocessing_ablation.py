"""
Preprocessing Ablation Study.
Isolates the impact of (1) background modeling and (2) multi-LiDAR registration
on detection AP, using Case 3 parameters for both datasets.

Four preprocessing configurations (2x2):
  A: Full          - both sensors + dynamic_mask filter      (existing Case 3)
  B: No BG         - both sensors + no dynamic_mask filter
  C: Single sensor - lidar_source==0 only + dynamic_mask filter
  D: Single+NoBG   - lidar_source==0 only + no dynamic_mask filter

All configs use the same Case 3 (alpha, rho_min) and Stage-2 merging.
Evaluated on the same GT and metrics as the main paper.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import cv2
from glob import glob
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import read_bin
from src.detection.clustering import clustering_stage_1, clustering_stage_2
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox


# -----------------------------------------------------------------------
# Columns in the merged bin files
# -----------------------------------------------------------------------
COLUMNS = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir',
           'dynamic_mask', 'reliability', 'lidar_source']


# -----------------------------------------------------------------------
# Single-frame detection (with reliability-weighted clustering = full method)
# -----------------------------------------------------------------------
def process_frame(frame_id, frame_name, pcd_array, filtered_array,
                  alpha, rho_min, stage_2=True):
    clusters, clusters_coord, eps = clustering_stage_1(
        filtered_array, COLUMNS,
        rho_min=rho_min, alpha=alpha,
        eps_min=0.5, eps_max=1.0)

    coarse_classes = classifying_stage_1(clusters_coord)

    if stage_2 and clusters:
        clusters, clusters_coord, coarse_classes = clustering_stage_2(
            clusters, clusters_coord, coarse_classes, filtered_array, COLUMNS)

    cluster_classes, class_probs = classifying_stage_2(clusters_coord, coarse_classes)

    bboxes = []
    reliability = filtered_array[:, COLUMNS.index('reliability')]
    for i in range(len(clusters)):
        bbox = compute_bbox(clusters_coord[i])
        bboxes.append({
            'frame_id':        frame_id,
            'frame_name':      frame_name,
            'cluster_id':      i,
            'label':           cluster_classes[i],
            'confidence':      float(class_probs[i]) if i < len(class_probs) else 0.0,
            'x': bbox[0], 'y': bbox[1], 'z': bbox[2],
            'w': bbox[3], 'l': bbox[4], 'h': bbox[5],
            'yaw':             bbox[6],
            'avg_eps':         float(np.mean(eps[clusters[i]])) if eps is not None else 0.0,
            'avg_reliability': float(np.mean(reliability[clusters[i]])),
            'num_points':      len(clusters[i]),
        })
    return bboxes


# -----------------------------------------------------------------------
# Run detection over all bin files under a given preprocessing config
# -----------------------------------------------------------------------
def run_config(bin_files, alpha, rho_min,
               use_bg_mask=True, single_sensor=False,
               min_height=0.1, max_height=3.0,
               stage_2=True, desc=''):
    """
    use_bg_mask  : if True, filter points where dynamic_mask == 1
    single_sensor: if True, use only points where lidar_source == 0
    """
    all_bboxes = []
    for i, fpath in enumerate(tqdm(bin_files, desc=desc)):
        frame_name = os.path.basename(fpath)
        pcd = read_bin(fpath, len(COLUMNS))

        # Height filter (always applied)
        mask = (
            (pcd[:, COLUMNS.index('z')] >= min_height) &
            (pcd[:, COLUMNS.index('z')] <= max_height)
        )
        # Background modeling filter
        if use_bg_mask:
            mask &= (pcd[:, COLUMNS.index('dynamic_mask')] == 1)
        # Single-sensor filter
        if single_sensor:
            mask &= (pcd[:, COLUMNS.index('lidar_source')] == 0)

        filtered = pcd[mask]
        if len(filtered) < 5:
            continue

        bboxes = process_frame(i, frame_name, pcd, filtered,
                               alpha=alpha, rho_min=rho_min, stage_2=stage_2)
        all_bboxes.extend(bboxes)
    return all_bboxes


# -----------------------------------------------------------------------
# AP evaluation (same IoU/AP logic as iou_sensitivity.py)
# -----------------------------------------------------------------------
LABEL_MAP_DEFAULT = {0: (3, 0.1), 1: (1, 0.3), 2: (2, 0.5)}
LABEL_MAP_DEFAULT_AMHERST = {0: (3, 0.1), 1: (1, 0.3)}  # no large vehicle


def rotated_iou_2d(det, gt):
    x1, y1 = det['x'], det['y']
    w1, l1 = det['w'], det['l']
    x2, y2 = gt['position'][0], gt['position'][1]
    w2, l2 = gt['dimensions'][0], gt['dimensions'][1]
    rect1 = ((x1, y1), (w1, l1), float(np.rad2deg(det['yaw'])))
    rect2 = ((x2, y2), (w2, l2), float(np.rad2deg(gt['rotation'])))
    area1, area2 = w1 * l1, w2 * l2
    ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if ret == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    if ret == cv2.INTERSECT_FULL:
        return float(min(area1, area2) / max(area1, area2))
    int_area = abs(cv2.contourArea(pts))
    union = area1 + area2 - int_area
    return float(np.clip(int_area / union if union > 0 else 0, 0, 1))


def compute_ap(prec, rec):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11
    return ap


def evaluate(bboxes, gt_dict, label_map):
    det_by_frame = defaultdict(list)
    for b in bboxes:
        if int(b['label']) in label_map:
            det_by_frame[b['frame_name']].append(b)

    n_gt = defaultdict(int)
    for blist in gt_dict.values():
        for g in blist:
            n_gt[g['category_id']] += 1

    all_dets = defaultdict(list)
    for frame_name, gt_bboxes in gt_dict.items():
        dets = sorted(det_by_frame.get(frame_name, []),
                      key=lambda x: x['confidence'], reverse=True)
        matched = defaultdict(set)
        for det in dets:
            gt_cat, iou_thr = label_map[int(det['label'])]
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt_bboxes):
                if g['category_id'] != gt_cat or gi in matched[gt_cat]:
                    continue
                iou = rotated_iou_2d(det, g)
                if iou > best_iou and iou >= iou_thr:
                    best_iou, best_gi = iou, gi
            is_tp = best_gi >= 0
            if is_tp:
                matched[gt_cat].add(best_gi)
            all_dets[gt_cat].append({'confidence': det['confidence'], 'is_tp': is_tp})

    ap_per_cat = {}
    for gt_cat, dets in all_dets.items():
        total_gt = n_gt[gt_cat]
        if total_gt == 0:
            continue
        dets.sort(key=lambda x: x['confidence'], reverse=True)
        tp = np.cumsum([1.0 if d['is_tp'] else 0.0 for d in dets])
        fp = np.cumsum([0.0 if d['is_tp'] else 1.0 for d in dets])
        prec = tp / np.maximum(tp + fp, 1e-10)
        rec  = tp / float(total_gt)
        ap_per_cat[gt_cat] = compute_ap(prec, rec)

    active = [c for c in [1, 2, 3] if n_gt[c] > 0]
    mAP = float(np.mean([ap_per_cat.get(c, 0.0) for c in active])) if active else 0.0
    return mAP, ap_per_cat


def load_gt(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    gt_dict = {}
    for sample in raw.get('dataset', {}).get('samples', []):
        fn = sample.get('name', '')
        anns = (sample.get('labels', {})
                      .get('ground-truth', {})
                      .get('attributes', {})
                      .get('annotations', []))
        bboxes = []
        for ann in anns:
            cat = ann.get('category_id')
            if cat not in [1, 2, 3]:
                continue
            pos, dim = ann['position'], ann['dimensions']
            bboxes.append({
                'category_id': cat,
                'position':    [pos['x'], pos['y'], pos['z']],
                'dimensions':  [dim['x'], dim['y'], dim['z']],
                'rotation':    ann['yaw'],
            })
        if bboxes:
            gt_dict[fn] = bboxes
    return gt_dict


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    bin_files = sorted(glob(os.path.join(args.input_folder, '*.bin')))
    print(f"Found {len(bin_files)} bin files")

    gt_dict = load_gt(args.gt_json)
    print(f"GT frames: {len(gt_dict)}")

    has_large_veh = any(b['category_id'] == 2
                        for blist in gt_dict.values() for b in blist)
    label_map = LABEL_MAP_DEFAULT if has_large_veh else LABEL_MAP_DEFAULT_AMHERST
    print(f"Dataset has large vehicle: {has_large_veh}")

    # Four preprocessing configs
    configs = [
        dict(name='Full',          use_bg_mask=True,  single_sensor=False),
        dict(name='No_BG',         use_bg_mask=False, single_sensor=False),
        dict(name='Single_Sensor', use_bg_mask=True,  single_sensor=True),
        dict(name='Single_NoBG',   use_bg_mask=False, single_sensor=True),
    ]

    results = []
    for cfg in configs:
        print(f"\n=== Config: {cfg['name']} ===")
        bboxes = run_config(
            bin_files,
            alpha=args.alpha,
            rho_min=args.rho_min,
            use_bg_mask=cfg['use_bg_mask'],
            single_sensor=cfg['single_sensor'],
            stage_2=True,
            desc=cfg['name'],
        )
        print(f"  Detections: {len(bboxes)}")

        # Save detections
        out_csv = os.path.join(args.output_folder,
                               f"bboxes_ablation_{cfg['name'].lower()}.csv")
        pd.DataFrame(bboxes).to_csv(out_csv, index=False)

        # Evaluate
        mAP, ap_dict = evaluate(bboxes, gt_dict, label_map)

        row = {
            'config':   cfg['name'],
            'use_bg':   cfg['use_bg_mask'],
            'single':   cfg['single_sensor'],
            'mAP':      round(mAP * 100, 2),
            'SmVeh':    round(ap_dict.get(1, 0.0) * 100, 2),
            'LgVeh':    round(ap_dict.get(2, 0.0) * 100, 2),
            'Ped':      round(ap_dict.get(3, 0.0) * 100, 2),
        }
        results.append(row)
        print(f"  mAP={row['mAP']:.2f}%  SmVeh={row['SmVeh']:.2f}%  "
              f"LgVeh={row['LgVeh']:.2f}%  Ped={row['Ped']:.2f}%")

    df = pd.DataFrame(results)
    out_summary = os.path.join(args.output_folder, 'ablation_preprocessing.csv')
    df.to_csv(out_summary, index=False)
    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_summary}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--gt_json',      required=True)
    parser.add_argument('--output_folder', default='results/ablation_preprocessing')
    parser.add_argument('--alpha',   type=float, required=True)
    parser.add_argument('--rho_min', type=float, required=True)
    args = parser.parse_args()
    main(args)
