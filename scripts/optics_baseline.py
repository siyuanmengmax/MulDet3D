"""
OPTICS baseline detection.

Runs OPTICS clustering on preprocessed point cloud .bin files using the same
Stage-2 merging, classification, and bbox computation as MulDet3D and the
DBSCAN/HDBSCAN baselines in run_detection.py.

Grid search over (min_samples, xi) on a small validation subset to find the
best OPTICS parameters, then runs on the full 1000-frame test set.

Output CSV columns match run_detection.py's output.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.cluster import OPTICS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import read_bin
from src.detection.clustering import clustering_stage_2
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox


# -----------------------------------------------------------------------
# OPTICS clustering (same interface as dbscan_clustering / hdbscan_clustering)
# -----------------------------------------------------------------------
def optics_clustering(points, columns, min_samples=30, xi=0.05, max_eps=1.0,
                      min_cluster_size=None):
    """
    OPTICS-based point cloud clustering.
    Parameters
    ----------
    min_samples : int
        Core neighborhood size (same meaning as in DBSCAN).
    xi : float
        Steepness value for cluster extraction (sklearn xi method).
    max_eps : float
        Maximum reachability distance (upper bound on eps).
    min_cluster_size : int or None
        Minimum cluster size; defaults to min_samples.
    Returns
    -------
    clusters, clusters_coord, labels_array
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]   # x, y, z

    if min_cluster_size is None:
        min_cluster_size = min_samples

    clust = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_method='xi',
        n_jobs=1,
    )
    cluster_labels = clust.fit_predict(point_coord)

    clusters = []
    clusters_coord = []
    for lbl in np.unique(cluster_labels):
        if lbl < 0:           # noise
            continue
        idx = np.where(cluster_labels == lbl)[0]
        clusters.append(idx.tolist())
        clusters_coord.append(point_coord[idx])

    return clusters, clusters_coord, cluster_labels


# -----------------------------------------------------------------------
# Single-frame processing (mirrors run_detection.py:process_frame)
# -----------------------------------------------------------------------
def process_frame_optics(frame_id, frame_name, pcd_array, filtered_pcd_array,
                         columns, params, stage_2=True):
    clusters, clusters_coord, _ = optics_clustering(
        filtered_pcd_array, columns,
        min_samples=params['min_samples'],
        xi=params['xi'],
        max_eps=params.get('max_eps', 1.0),
        min_cluster_size=params.get('min_cluster_size', params['min_samples']),
    )

    coarse_classes = classifying_stage_1(clusters_coord)

    if stage_2 and clusters:
        clusters, clusters_coord, coarse_classes = clustering_stage_2(
            clusters, clusters_coord, coarse_classes, filtered_pcd_array, columns)

    cluster_classes, class_probabilities = classifying_stage_2(clusters_coord, coarse_classes)

    bboxes = []
    for i in range(len(clusters)):
        bbox = compute_bbox(clusters_coord[i])
        bboxes.append({
            'frame_id':        frame_id,
            'frame_name':      frame_name,
            'cluster_id':      i,
            'label':           cluster_classes[i],
            'confidence':      float(class_probabilities[i]) if len(class_probabilities) > i else 0.0,
            'x': bbox[0], 'y': bbox[1], 'z': bbox[2],
            'w': bbox[3], 'l': bbox[4], 'h': bbox[5],
            'yaw':             bbox[6],
            'avg_eps':         params.get('max_eps', 1.0),
            'avg_reliability': 1.0,
            'num_points':      len(clusters[i]),
        })
    return bboxes


# -----------------------------------------------------------------------
# Run full detection on a list of bin files
# -----------------------------------------------------------------------
def run_detection(bin_files, columns, params, min_height=0.1, max_height=3.0,
                  stage_2=True, desc='Detecting'):
    all_bboxes = []
    for i, fpath in enumerate(tqdm(bin_files, desc=desc)):
        frame_name = os.path.basename(fpath)
        pcd = read_bin(fpath, len(columns))
        mask = (
            (pcd[:, columns.index('z')] >= min_height) &
            (pcd[:, columns.index('z')] <= max_height) &
            (pcd[:, columns.index('dynamic_mask')] == 1)
        )
        filtered = pcd[mask]
        if len(filtered) == 0:
            continue
        bboxes = process_frame_optics(i, frame_name, pcd, filtered, columns,
                                      params, stage_2=stage_2)
        all_bboxes.extend(bboxes)
    return all_bboxes


# -----------------------------------------------------------------------
# Grid search: evaluate mAP on val_frames for each param combo
# Uses the same IoU / AP logic as evaluate_detection_ap.py
# -----------------------------------------------------------------------
def grid_search(bin_files_val, columns, gt_dict, param_grid, stage_2=True):
    import cv2
    from collections import defaultdict

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

    LABEL_MAP = {0: (3, 0.1), 1: (1, 0.3), 2: (2, 0.5)}   # Same per-class IoU thresholds as the full method.

    def compute_ap(prec, rec):
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p / 11
        return ap

    def eval_bboxes(bboxes, frame_names_set):
        det_by_frame = defaultdict(list)
        for b in bboxes:
            if b['frame_name'] in frame_names_set and b['label'] in LABEL_MAP:
                det_by_frame[b['frame_name']].append(b)

        n_gt = defaultdict(int)
        for fn in frame_names_set:
            for g in gt_dict.get(fn, []):
                n_gt[g['category_id']] += 1

        all_dets = defaultdict(list)
        for fn in frame_names_set:
            gt_bboxes = gt_dict.get(fn, [])
            dets = sorted(det_by_frame[fn], key=lambda x: x['confidence'], reverse=True)
            matched = defaultdict(set)
            for det in dets:
                gt_cat, iou_thr = LABEL_MAP[det['label']]
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

        ap_dict = {}
        for gt_cat, dets in all_dets.items():
            total_gt = n_gt[gt_cat]
            if total_gt == 0:
                continue
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            tp = np.cumsum([1.0 if d['is_tp'] else 0.0 for d in dets])
            fp = np.cumsum([0.0 if d['is_tp'] else 1.0 for d in dets])
            prec = tp / np.maximum(tp + fp, 1e-10)
            rec  = tp / float(total_gt)
            ap_dict[gt_cat] = compute_ap(prec, rec)

        cats = [c for c in [1, 2, 3] if n_gt[c] > 0]
        mAP = float(np.mean([ap_dict.get(c, 0.0) for c in cats])) if cats else 0.0
        return mAP, ap_dict

    val_frame_names = set(os.path.basename(f) for f in bin_files_val)
    best_mAP, best_params = -1, None

    print(f"\nGrid search over {len(param_grid)} param combos on {len(bin_files_val)} val frames:")
    for params in param_grid:
        bboxes = run_detection(bin_files_val, columns, params, stage_2=stage_2,
                               desc=f"  min_samples={params['min_samples']}, xi={params['xi']}")
        mAP, ap_dict = eval_bboxes(bboxes, val_frame_names)
        print(f"    min_samples={params['min_samples']:3d}  xi={params['xi']:.3f}  "
              f"max_eps={params['max_eps']:.1f}  ->  mAP={mAP*100:.2f}%  "
              f"per-cat: {', '.join(f'cat{k}={v*100:.1f}%' for k,v in sorted(ap_dict.items()))}")
        if mAP > best_mAP:
            best_mAP = mAP
            best_params = params

    print(f"\nBest: {best_params}  mAP={best_mAP*100:.2f}%")
    return best_params, best_mAP


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args):
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir',
               'dynamic_mask', 'reliability', 'lidar_source']

    bin_files = sorted(glob(os.path.join(args.input_folder, '*.bin')))
    print(f"Found {len(bin_files)} bin files in {args.input_folder}")

    os.makedirs(args.output_folder, exist_ok=True)

    if args.grid_search:
        import json
        with open(args.gt_json) as f:
            gt_raw = json.load(f)
        gt_dict = {}
        for sample in gt_raw.get('dataset', {}).get('samples', []):
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
                bboxes.append({'category_id': cat,
                               'position': [pos['x'], pos['y'], pos['z']],
                               'dimensions': [dim['x'], dim['y'], dim['z']],
                               'rotation': ann['yaw']})
            if bboxes:
                gt_dict[fn] = bboxes

        val_files = bin_files[:args.n_val]
        param_grid = [
            {'min_samples': ms, 'xi': xi, 'max_eps': me, 'min_cluster_size': ms}
            for ms in [20, 30, 50]
            for xi in [0.01, 0.05, 0.10]
            for me in [0.7, 1.0]
        ]
        best_params, _ = grid_search(val_files, columns, gt_dict, param_grid,
                                     stage_2=args.stage_2)
        pd.DataFrame([best_params]).to_csv(
            os.path.join(args.output_folder, 'best_optics_params.csv'), index=False)
        print("Best params saved. Use --min_samples/--xi/--max_eps or edit the CSV to run full detection.")
        return

    params = {
        'min_samples':    args.min_samples,
        'xi':             args.xi,
        'max_eps':        args.max_eps,
        'min_cluster_size': args.min_cluster_size if args.min_cluster_size else args.min_samples,
    }
    print(f"Running OPTICS detection with params: {params}")

    all_bboxes = run_detection(bin_files, columns, params, stage_2=args.stage_2,
                               desc='OPTICS detection')

    out_path = os.path.join(args.output_folder, args.out_csv)
    pd.DataFrame(all_bboxes,
                 columns=['frame_id', 'frame_name', 'cluster_id', 'label', 'confidence',
                           'x', 'y', 'z', 'w', 'l', 'h', 'yaw',
                           'avg_eps', 'avg_reliability', 'num_points']
                 ).to_csv(out_path, index=False)
    print(f"Saved {len(all_bboxes)} detections to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OPTICS baseline detection')
    parser.add_argument('--input_folder',  default='bin/merge_1000')
    parser.add_argument('--output_folder', default='results/optics_baseline')
    parser.add_argument('--out_csv',       default='bboxes_optics.csv')
    parser.add_argument('--stage_2',       type=bool, default=True)
    # Grid search
    parser.add_argument('--grid_search',   action='store_true')
    parser.add_argument('--gt_json',       default='bin/lowell_1000_label.json')
    parser.add_argument('--n_val',         type=int, default=50,
                        help='Number of frames for grid search validation')
    # OPTICS params (used when not doing grid search)
    parser.add_argument('--min_samples',    type=int,   default=30)
    parser.add_argument('--xi',             type=float, default=0.05)
    parser.add_argument('--max_eps',        type=float, default=1.0)
    parser.add_argument('--min_cluster_size', type=int, default=None)
    args = parser.parse_args()
    main(args)
