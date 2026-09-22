"""
IoU Threshold Sensitivity Analysis.
Evaluates all methods at multiple uniform IoU thresholds to demonstrate that
MulDet3D's advantage holds under stricter thresholds.

Thresholds tested:
  Default  : pedestrian@0.1, small_veh@0.3, large_veh@0.5  (paper default)
  Uniform@0.1: all classes @ 0.1
  Uniform@0.3: all classes @ 0.3  (stricter for pedestrian)
  Uniform@0.5: all classes @ 0.5  (stricter for all)

Outputs:
  results/iou_sensitivity/sensitivity_results.csv
  results/iou_sensitivity/sensitivity_summary.txt
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict


# -----------------------------------------------------------------------
# Label → GT category mappings
# -----------------------------------------------------------------------
def build_label_map(threshold, unified=False):
    """Build label→(gt_cat, iou_threshold) at a single uniform threshold."""
    if unified:
        return {
            0: (3, threshold),   # pedestrian
            1: (1, threshold),   # vehicle
            2: (1, threshold),
            3: (1, threshold),   # baseline "other" → vehicle
        }
    else:
        return {
            0: (3, threshold),   # pedestrian
            1: (1, threshold),   # small vehicle
            2: (2, threshold),   # large vehicle
        }


# Default per-class thresholds (paper standard)
LABEL_MAP_DEFAULT = {
    0: (3, 0.1),
    1: (1, 0.3),
    2: (2, 0.5),
}
LABEL_MAP_DEFAULT_UNIFIED = {
    0: (3, 0.1),
    1: (1, 0.3),
    2: (1, 0.3),
    3: (1, 0.3),
}


# -----------------------------------------------------------------------
# Ground-truth loading
# -----------------------------------------------------------------------
def load_groundtruth(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    gt_data = {}
    samples = data.get('dataset', {}).get('samples', [])
    for sample in samples:
        frame_name = sample.get('name', '')
        annotations = (sample.get('labels', {})
                              .get('ground-truth', {})
                              .get('attributes', {})
                              .get('annotations', []))
        bboxes = []
        for ann in annotations:
            cat = ann.get('category_id')
            if cat not in [1, 2, 3]:
                continue
            pos = ann['position']
            dim = ann['dimensions']
            bboxes.append({
                'category_id': cat,
                'position':   [pos['x'], pos['y'], pos['z']],
                'dimensions': [dim['x'], dim['y'], dim['z']],
                'rotation':   ann['yaw'],
            })
        if bboxes:
            gt_data[frame_name] = bboxes
    return gt_data


# -----------------------------------------------------------------------
# 2D rotated-rectangle IoU
# -----------------------------------------------------------------------
def rotated_iou_2d(det, gt):
    x1, y1 = det['position'][0], det['position'][1]
    w1, l1 = det['dimensions'][0], det['dimensions'][1]
    x2, y2 = gt['position'][0], gt['position'][1]
    w2, l2 = gt['dimensions'][0], gt['dimensions'][1]
    rect1 = ((x1, y1), (w1, l1), float(np.rad2deg(det['rotation'])))
    rect2 = ((x2, y2), (w2, l2), float(np.rad2deg(gt['rotation'])))
    area1, area2 = w1 * l1, w2 * l2
    ret, int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if ret == cv2.INTERSECT_NONE or int_pts is None:
        return 0.0
    if ret == cv2.INTERSECT_FULL:
        return float(min(area1, area2) / max(area1, area2))
    int_area = abs(cv2.contourArea(int_pts))
    union_area = area1 + area2 - int_area
    return float(np.clip(int_area / union_area if union_area > 0 else 0, 0, 1))


# -----------------------------------------------------------------------
# 11-point interpolated AP
# -----------------------------------------------------------------------
def compute_ap_from_pr(precisions, recalls):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0
        ap += p / 11
    return ap


# -----------------------------------------------------------------------
# Full-dataset evaluation at a given label map
# -----------------------------------------------------------------------
def evaluate_full(gt_data, det_df, label_to_gt, unified_gt=False):
    """
    Evaluate over all frames in gt_data.
    unified_gt: if True, merge GT cat 1+2 → cat 1.
    Returns: {gt_cat: ap}, mAP
    """
    # Optionally merge GT vehicle categories
    if unified_gt:
        chunk_gt = {}
        for fn, blist in gt_data.items():
            merged = []
            for b in blist:
                bc = dict(b)
                if bc['category_id'] in [1, 2]:
                    bc['category_id'] = 1
                merged.append(bc)
            chunk_gt[fn] = merged
    else:
        chunk_gt = gt_data

    # Build per-frame detection dict
    det_by_frame = defaultdict(list)
    for _, row in det_df.iterrows():
        det_label = int(row['label'])
        if det_label not in label_to_gt:
            continue
        gt_cat_mapped, _ = label_to_gt[det_label]
        det_by_frame[row['frame_name']].append({
            'label':      det_label,
            'gt_cat':     gt_cat_mapped,
            'confidence': float(row['confidence']),
            'position':   [float(row['x']), float(row['y']), float(row['z'])],
            'dimensions': [float(row['w']), float(row['l']), float(row['h'])],
            'rotation':   float(row['yaw']),
        })

    # Count GT per class
    n_gt = defaultdict(int)
    for blist in chunk_gt.values():
        for b in blist:
            n_gt[b['category_id']] += 1

    all_dets = defaultdict(list)

    for frame_name, gt_bboxes in chunk_gt.items():
        det_bboxes = sorted(det_by_frame.get(frame_name, []),
                            key=lambda x: x['confidence'], reverse=True)
        matched = defaultdict(set)

        for det in det_bboxes:
            gt_cat, iou_thresh = label_to_gt[det['label']]
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt_bboxes):
                if g['category_id'] != gt_cat or gi in matched[gt_cat]:
                    continue
                iou = rotated_iou_2d(det, g)
                if iou > best_iou and iou >= iou_thresh:
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
        tp_cum = np.cumsum([1.0 if d['is_tp'] else 0.0 for d in dets])
        fp_cum = np.cumsum([0.0 if d['is_tp'] else 1.0 for d in dets])
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
        rec  = tp_cum / float(total_gt)
        ap_per_cat[gt_cat] = compute_ap_from_pr(prec, rec)

    evaluated = [ap_per_cat.get(c, 0.0) for c in [1, 2, 3] if n_gt[c] > 0]
    mAP = float(np.mean(evaluated)) if evaluated else 0.0
    return ap_per_cat, mAP


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args):
    print("Loading ground truth...")
    gt_dict = load_groundtruth(args.gt_json)
    all_frames = sorted(gt_dict.keys())
    print(f"  GT annotated frames: {len(all_frames)}")

    cat_counts = defaultdict(int)
    for blist in gt_dict.values():
        for b in blist:
            cat_counts[b['category_id']] += 1
    print(f"  GT class counts: ped={cat_counts[3]}, small_veh={cat_counts[1]}, large_veh={cat_counts[2]}")

    # Load methods
    methods = {}
    for item in (args.methods or []):
        name, csv_path = item.split(':', 1)
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {name}")
            continue
        df = pd.read_csv(csv_path)
        methods[name] = df
        print(f"  Loaded {name}: {len(df)} detections")

    if not methods:
        print("No methods loaded.")
        return

    baseline_set = set(args.baseline_methods or [])

    # Thresholds to evaluate
    # 'default' = per-class thresholds; others = uniform
    thresholds = ['default', 0.1, 0.3, 0.5]
    threshold_labels = {
        'default': 'Default\n(ped@0.1,sveh@0.3,lveh@0.5)',
        0.1: 'Uniform\n@0.1',
        0.3: 'Uniform\n@0.3',
        0.5: 'Uniform\n@0.5',
    }

    # Determine active GT classes
    has_large_veh = cat_counts[2] > 0
    if args.unified:
        active_cats = [c for c in [1, 3] if (cat_counts[c] + cat_counts.get(2, 0) > 0 if c == 1 else cat_counts[c] > 0)]
        cat_display = {1: 'Vehicle', 3: 'Pedestrian'}
    else:
        active_cats = [c for c in [1, 2, 3] if cat_counts[c] > 0]
        cat_display = {1: 'SmVeh', 2: 'LgVeh', 3: 'Ped'}

    print("\n" + "="*80)
    print("IoU Threshold Sensitivity Analysis")
    print("="*80)

    rows = []
    for thr in thresholds:
        thr_label = threshold_labels[thr]
        for name, df in methods.items():
            is_baseline = (name in baseline_set)
            use_unified = (is_baseline or args.unified)

            if thr == 'default':
                lmap = LABEL_MAP_DEFAULT_UNIFIED if use_unified else LABEL_MAP_DEFAULT
            else:
                lmap = build_label_map(thr, unified=use_unified)

            ap_dict, mAP = evaluate_full(gt_dict, df, lmap, unified_gt=use_unified)

            row = {'threshold': thr_label, 'method': name, 'mAP': mAP * 100}
            for c in active_cats:
                row[cat_display[c]] = ap_dict.get(c, 0.0) * 100
            rows.append(row)

    results_df = pd.DataFrame(rows)

    # Print summary table
    print(f"\n{'Threshold':<35} {'Method':<25} {'mAP':>7}", end='')
    for c in active_cats:
        print(f"  {cat_display[c]:>8}", end='')
    print()
    print("-"*80)

    for thr in thresholds:
        thr_label = threshold_labels[thr]
        subset = results_df[results_df['threshold'] == thr_label]
        for _, r in subset.iterrows():
            print(f"{thr_label.replace(chr(10), ' '):<35} {r['method']:<25} {r['mAP']:>6.2f}%", end='')
            for c in active_cats:
                print(f"  {r[cat_display[c]]:>7.2f}%", end='')
            print()
        print()

    # Save
    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, 'sensitivity_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    # Also produce a clean pivot table: rows=method, cols=threshold×metric
    # Useful for the LaTeX table
    print("\n" + "="*80)
    print("Clean summary for LaTeX table (mAP only)")
    print("="*80)
    thr_short = {
        threshold_labels['default']: 'Default',
        threshold_labels[0.1]: '@0.1',
        threshold_labels[0.3]: '@0.3',
        threshold_labels[0.5]: '@0.5',
    }
    pivot = results_df.pivot_table(index='method', columns='threshold', values='mAP')
    pivot.columns = [thr_short.get(c, c) for c in pivot.columns]
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}%"))

    sum_path = os.path.join(args.output_folder, 'sensitivity_summary.txt')
    with open(sum_path, 'w') as f:
        f.write("IoU Threshold Sensitivity Analysis\n")
        f.write("="*80 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nmAP Pivot (rows=method, cols=threshold):\n")
        f.write(pivot.to_string(float_format=lambda x: f"{x:.2f}%"))
    print(f"Summary saved to {sum_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IoU Threshold Sensitivity Analysis')
    parser.add_argument('--gt_json',        required=True)
    parser.add_argument('--output_folder',  default='results/iou_sensitivity')
    parser.add_argument('--methods',        nargs='+', metavar='NAME:CSV')
    parser.add_argument('--baseline_methods', nargs='*', default=[],
                        help='Method names using unified label mapping (label 3 → vehicle)')
    parser.add_argument('--unified',        action='store_true',
                        help='Use unified mapping for ALL methods')
    args = parser.parse_args()
    main(args)
