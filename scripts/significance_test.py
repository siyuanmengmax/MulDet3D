"""
Statistical Significance Testing.
Divides the 1000-frame test set into N_CHUNKS sequential blocks.
Computes per-class AP for each chunk using the same 2D rotated-rect IoU
and per-class thresholds as correlation_analysis.py / metric_cal.py.
Reports mean ± std, 95% CI, and paired t-tests vs baselines.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from tqdm import tqdm
from collections import defaultdict


# Detection label → (GT category_id, IoU threshold)  — same as correlation_analysis.py
LABEL_TO_GT_FULL = {
    0: (3, 0.1),   # person  → gt_cat=3 (pedestrian), AP@0.1
    1: (1, 0.3),   # small vehicle → gt_cat=1,         AP@0.3
    2: (2, 0.5),   # large vehicle → gt_cat=2,         AP@0.5
}

# Unified mode (vehicle classes merged, used for baselines that output label 3):
# GT: cat 1+2 → merged to cat 1 ("vehicle")
# Det: label 1+2+3 → mapped to gt_cat 1 ("vehicle") at @0.3; label 0 → ped at @0.1
LABEL_TO_GT_UNIFIED = {
    0: (3, 0.1),   # person  → pedestrian, AP@0.1
    1: (1, 0.3),   # small/large vehicle → unified vehicle, AP@0.3
    2: (1, 0.3),
    3: (1, 0.3),   # baseline "other" treated as vehicle
}


# ---------------------------------------------------------------------------
# Ground-truth loading  (identical to correlation_analysis.py)
# ---------------------------------------------------------------------------
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
                'position':    [pos['x'], pos['y'], pos['z']],
                'dimensions':  [dim['x'], dim['y'], dim['z']],
                'rotation':    ann['yaw'],
            })
        if bboxes:
            gt_data[frame_name] = bboxes
    return gt_data


# ---------------------------------------------------------------------------
# IoU  (identical to correlation_analysis.py / metric_cal.py)
# ---------------------------------------------------------------------------
def rotated_iou_2d(det, gt):
    """2D rotated-rectangle IoU."""
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


# ---------------------------------------------------------------------------
# AP computation  (11-point interpolation, identical to correlation_analysis.py)
# ---------------------------------------------------------------------------
def compute_ap_from_pr(precisions, recalls):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0
        ap += p / 11
    return ap


def evaluate_chunk(gt_data, det_df, frame_names, label_to_gt, unified_gt=False):
    """
    Compute per-class AP for a subset of frames.
    label_to_gt: mapping from det_label → (gt_cat_id, iou_threshold)
    unified_gt:  if True, merge GT cat 1+2 → cat 1 before evaluation
    Returns: {gt_cat_id: ap}, mAP
    """
    frame_set = set(frame_names)
    chunk_gt_raw = {fn: v for fn, v in gt_data.items() if fn in frame_set}

    # Optionally merge vehicle GT categories
    if unified_gt:
        chunk_gt = {}
        for fn, blist in chunk_gt_raw.items():
            merged = []
            for b in blist:
                bc = dict(b)
                if bc['category_id'] in [1, 2]:
                    bc['category_id'] = 1  # unified vehicle
                merged.append(bc)
            chunk_gt[fn] = merged
    else:
        chunk_gt = chunk_gt_raw

    # Build per-frame detection dict from the DataFrame subset
    chunk_det_df = det_df[det_df['frame_name'].isin(frame_set)]
    det_by_frame = defaultdict(list)
    for _, row in chunk_det_df.iterrows():
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

    # Count GT per class (over all chunk frames, including unannotated ones)
    n_gt = defaultdict(int)
    for blist in chunk_gt.values():
        for b in blist:
            n_gt[b['category_id']] += 1

    all_dets = defaultdict(list)   # gt_cat → [{confidence, is_tp}]

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

    # mAP over categories that have GT (default 0.0 if no detections for that class)
    evaluated = [ap_per_cat.get(c, 0.0) for c in [1, 2, 3] if n_gt[c] > 0]
    mAP = float(np.mean(evaluated)) if evaluated else 0.0
    return ap_per_cat, mAP


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading ground truth...")
    gt_dict = load_groundtruth(args.gt_json)
    all_frames = sorted(gt_dict.keys())
    print(f"  GT annotated frames: {len(all_frames)}")

    # Determine active GT categories
    cat_counts = defaultdict(int)
    for blist in gt_dict.values():
        for b in blist:
            cat_counts[b['category_id']] += 1
    if args.unified:
        cat_names = {1: 'Vehicle_AP03', 3: 'Pedestrian_AP01'}
        active_cats = [c for c in [1, 3] if (cat_counts[c] + cat_counts.get(2, 0) > 0 if c == 1 else cat_counts[c] > 0)]
    else:
        cat_names = {1: 'SmallVeh_AP03', 2: 'LargeVeh_AP05', 3: 'Pedestrian_AP01'}
        active_cats = [c for c in [1, 2, 3] if cat_counts[c] > 0]
    print(f"  Active GT classes: {[cat_names[c] for c in active_cats]}")

    # Chunk frames
    n_chunks = args.n_chunks
    chunk_size = len(all_frames) // n_chunks
    chunks = [all_frames[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    print(f"  Chunks: {n_chunks} × {chunk_size} frames")

    # Load detection CSVs
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

    # Decide evaluation mode per method
    # Methods listed in --baseline_methods use UNIFIED label mapping (label 3 → vehicle)
    baseline_set = set(args.baseline_methods or [])
    unified_gt = args.unified

    # Evaluate per chunk
    print("\nEvaluating chunks (this may take a few minutes)...")
    chunk_maps = {name: [] for name in methods}
    chunk_aps  = {name: {c: [] for c in active_cats} for name in methods}

    for ci, chunk_frames in enumerate(tqdm(chunks, desc='Chunks')):
        for name, df in methods.items():
            lmap = LABEL_TO_GT_UNIFIED if (name in baseline_set or unified_gt) else LABEL_TO_GT_FULL
            use_unified_gt = (name in baseline_set or unified_gt)
            ap_dict, mAP = evaluate_chunk(gt_dict, df, chunk_frames,
                                          label_to_gt=lmap, unified_gt=use_unified_gt)
            chunk_maps[name].append(mAP)
            for c in active_cats:
                chunk_aps[name][c].append(ap_dict.get(c, 0.0))

    # -----------------------------------------------------------------------
    # Report statistics
    # -----------------------------------------------------------------------
    print("\n" + "="*75)
    print("Statistical Summary — mAP across chunks")
    print("="*75)
    print(f"{'Method':<30} {'Mean':>8} {'Std':>8} {'95% CI':>22}")
    print("-"*75)
    stats_table = {}
    for name in methods:
        arr = np.array(chunk_maps[name])
        mean = np.nanmean(arr)
        std  = np.nanstd(arr, ddof=1)
        ci   = stats.t.interval(0.95, df=len(arr)-1, loc=mean, scale=stats.sem(arr))
        stats_table[name] = {'mean': mean, 'std': std, 'ci': ci, 'aps': arr}
        print(f"{name:<30} {mean*100:>7.2f}% {std*100:>7.2f}%  [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")

    print("\nPer-class AP (mean ± std across chunks):")
    for name in methods:
        parts = [f"{name}: "]
        for c in active_cats:
            vals = np.array(chunk_aps[name][c])
            parts.append(f"  {cat_names[c]}={np.nanmean(vals)*100:.2f}±{np.nanstd(vals,ddof=1)*100:.2f}%")
        print(''.join(parts))

    # -----------------------------------------------------------------------
    # Paired t-tests
    # -----------------------------------------------------------------------
    if args.main_method not in stats_table:
        print(f"\nMain method '{args.main_method}' not in results.")
        return

    print("\n" + "="*75)
    print(f"Paired t-test: {args.main_method} vs baselines (mAP, {n_chunks} chunks)")
    print("="*75)
    main_arr = stats_table[args.main_method]['aps']
    for name, s in stats_table.items():
        if name == args.main_method:
            continue
        t_stat, p_val = stats.ttest_rel(main_arr, s['aps'])
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        delta = (stats_table[args.main_method]['mean'] - s['mean']) * 100
        print(f"  vs {name:<28}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}  "
              f"MulDet3D-baseline={delta:+.2f}%")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(args.output_folder, exist_ok=True)
    out_df = pd.DataFrame({'chunk': list(range(1, n_chunks+1))})
    for name in methods:
        out_df[f'{name}_mAP'] = [v*100 for v in chunk_maps[name]]
        for c in active_cats:
            out_df[f'{name}_{cat_names[c]}'] = [v*100 for v in chunk_aps[name][c]]
    out_path = os.path.join(args.output_folder, 'chunk_results.csv')
    out_df.to_csv(out_path, index=False)

    # Summary
    rows = []
    for name, s in stats_table.items():
        row = {'method': name,
               'mean_mAP': s['mean']*100,
               'std_mAP':  s['std']*100,
               'ci_lo':    s['ci'][0]*100,
               'ci_hi':    s['ci'][1]*100}
        for c in active_cats:
            vals = np.array(chunk_aps[name][c])
            row[f'mean_{cat_names[c]}'] = np.nanmean(vals)*100
            row[f'std_{cat_names[c]}']  = np.nanstd(vals, ddof=1)*100
        rows.append(row)
    sum_path = os.path.join(args.output_folder, 'significance_summary.csv')
    pd.DataFrame(rows).to_csv(sum_path, index=False)
    print(f"\nResults saved to {args.output_folder}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Significance testing via chunk-wise AP evaluation')
    parser.add_argument('--gt_json',       required=True)
    parser.add_argument('--n_chunks',      type=int, default=10)
    parser.add_argument('--main_method',   default='MulDet3D_Case3')
    parser.add_argument('--output_folder', default='results/significance')
    parser.add_argument('--methods', nargs='+', metavar='NAME:CSV')
    parser.add_argument('--baseline_methods', nargs='*', default=[],
                        help='Method names that use unified label mapping (label 3 → vehicle)')
    parser.add_argument('--unified', action='store_true',
                        help='Use unified mapping for ALL methods')
    args = parser.parse_args()
    main(args)
