"""
Correlation analysis between the MOPSO proxy objectives (coverage,
compactness, separation) and downstream detection AP -- i.e. whether the
label-free objectives optimized during parameter search actually track
labeled detection accuracy.

Experiment:
  1. Load Pareto-front solutions (objectives already computed).
  2. Add extra random/grid samples across the full parameter space.
  3. For all parameter combinations:
       a. Compute proxy objectives on optimization frames (label-free).
       b. Run full detection on test frames and evaluate AP against GT.
  4. Generate scatter plots and correlation statistics.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import cv2
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import read_bin, ensure_directory
from src.detection.clustering import clustering_stage_1, clustering_stage_2
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox

# Detection label → (GT category_id, IoU threshold)
LABEL_TO_GT = {
    0: (3, 0.1),   # person  → pedestrian,    AP@0.1
    1: (1, 0.3),   # small vehicle,            AP@0.3
    2: (2, 0.5),   # large vehicle,            AP@0.5
}
GT_CAT_NAMES = {1: 'Small Vehicle', 2: 'Large Vehicle', 3: 'Pedestrian'}


# ─── Proxy objective computation ─────────────────────────────────────────────

def compute_objectives(alpha, rho_min, opt_frames, columns, separation_max=50):
    """
    Compute Coverage, Compactness, Separation on optimization frames.
    Returns actual physical values (all positive, larger-is-better for
    Coverage and Separation, smaller-is-better for Compactness).
    Returns None if no valid frames were processed.
    """
    all_objs = []
    frame_weights = []

    for frame_data in opt_frames:
        n_points = len(frame_data)
        if n_points == 0:
            continue

        clusters, clusters_coord, _ = clustering_stage_1(
            frame_data, columns,
            rho_min=rho_min, alpha=alpha,
            eps_min=0.5, eps_max=1.0,
        )
        n_clusters = len(clusters)
        if n_clusters == 0:
            continue

        # Coverage: fraction of foreground points assigned to clusters
        coverage = sum(len(c) for c in clusters) / n_points

        # Compactness: point-weighted mean intra-cluster distance
        centroids = np.zeros((n_clusters, 3))
        intra = np.zeros(n_clusters)
        sizes = np.zeros(n_clusters)
        for i, cluster in enumerate(clusters):
            lo = np.min(clusters_coord[i], axis=0)
            hi = np.max(clusters_coord[i], axis=0)
            centroids[i] = (lo + hi) / 2.0
            sizes[i] = len(cluster)
            intra[i] = np.mean(np.linalg.norm(clusters_coord[i] - centroids[i], axis=1))
        compactness = np.sum(intra * sizes) / np.sum(sizes)

        # Separation: minimum inter-cluster centroid distance
        if n_clusters > 1:
            diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_mat, np.inf)
            separation = float(np.min(dist_mat))
        else:
            separation = float(separation_max)

        all_objs.append([coverage, compactness, separation])
        frame_weights.append(n_points)

    if not frame_weights:
        return None

    weights = np.array(frame_weights, dtype=float)
    weights /= weights.sum()
    weighted = np.average(all_objs, weights=weights, axis=0)
    return {
        'coverage_rate': float(weighted[0]),
        'compactness':   float(weighted[1]),
        'separation':    float(weighted[2]),
    }


# ─── Detection pipeline ───────────────────────────────────────────────────────

def detect_frame(filtered_pcd, columns, alpha, rho_min, use_stage2=True):
    """Run the full detection pipeline for one pre-filtered frame."""
    clusters, clusters_coord, _ = clustering_stage_1(
        filtered_pcd, columns,
        rho_min=rho_min, alpha=alpha,
        eps_min=0.5, eps_max=1.0,
    )
    coarse_classes = classifying_stage_1(clusters_coord)

    if use_stage2 and len(clusters) > 1:
        clusters, clusters_coord, coarse_classes = clustering_stage_2(
            clusters, clusters_coord, coarse_classes, filtered_pcd, columns
        )

    cluster_classes, class_probs = classifying_stage_2(clusters_coord, coarse_classes)

    bboxes = []
    for i in range(len(clusters)):
        bbox = compute_bbox(clusters_coord[i])
        bboxes.append({
            'label':      cluster_classes[i],
            'confidence': class_probs[i],
            'position':   [bbox[0], bbox[1], bbox[2]],
            'dimensions': [bbox[3], bbox[4], bbox[5]],
            'rotation':   bbox[6],
        })
    return bboxes


# ─── AP computation ───────────────────────────────────────────────────────────

def rotated_iou_2d(det, gt):
    """2D rotated-rectangle IoU, matching metric_cal.py."""
    x1, y1 = det['position'][0], det['position'][1]
    w1, l1 = det['dimensions'][0], det['dimensions'][1]
    x2, y2 = gt['position'][0], gt['position'][1]
    w2, l2 = gt['dimensions'][0], gt['dimensions'][1]
    rect1 = ((x1, y1), (w1, l1), np.rad2deg(det['rotation']))
    rect2 = ((x2, y2), (w2, l2), np.rad2deg(gt['rotation']))
    area1, area2 = w1 * l1, w2 * l2
    ret, int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if ret == cv2.INTERSECT_NONE or int_pts is None:
        return 0.0
    if ret == cv2.INTERSECT_FULL:
        return float(min(area1, area2) / max(area1, area2))
    int_area = abs(cv2.contourArea(int_pts))
    union_area = area1 + area2 - int_area
    return float(np.clip(int_area / union_area if union_area > 0 else 0, 0, 1))


def compute_ap_from_pr(precisions, recalls):
    """11-point interpolated AP, matching metric_cal.py."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        ap += (np.max(precisions[mask]) if mask.any() else 0.0)
    return ap / 11.0


def evaluate_ap(gt_data, det_by_frame):
    """
    Compute per-class AP using per-class IoU thresholds.
    Returns dict {gt_cat_id: ap} and mAP over detected categories.
    """
    all_dets = defaultdict(list)      # gt_cat → list of {confidence, is_tp}
    n_gt = defaultdict(int)           # gt_cat → count

    for frame_id, gt_bboxes in gt_data.items():
        for g in gt_bboxes:
            n_gt[g['category_id']] += 1

        det_bboxes = det_by_frame.get(frame_id, [])
        det_sorted = sorted(det_bboxes, key=lambda x: x['confidence'], reverse=True)
        matched = defaultdict(set)

        for det in det_sorted:
            det_label = det['label']
            if det_label not in LABEL_TO_GT:
                continue
            gt_cat, iou_thresh = LABEL_TO_GT[det_label]

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
    for gt_cat in [1, 2, 3]:
        dets = sorted(all_dets[gt_cat], key=lambda x: x['confidence'], reverse=True)
        total_gt = n_gt[gt_cat]
        if not dets or total_gt == 0:
            ap_per_cat[gt_cat] = 0.0
            continue
        tp_cum = np.cumsum([1.0 if d['is_tp'] else 0.0 for d in dets])
        fp_cum = np.cumsum([0.0 if d['is_tp'] else 1.0 for d in dets])
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
        rec  = tp_cum / float(total_gt)
        ap_per_cat[gt_cat] = compute_ap_from_pr(prec, rec)

    evaluated = [ap_per_cat[c] for c in [1, 2, 3] if n_gt[c] > 0]
    mAP = float(np.mean(evaluated)) if evaluated else 0.0
    return ap_per_cat, mAP


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_correlation(results_df, best_params, output_folder):
    """
    Generate scatter plots: proxy objective vs AP (mAP and per-class).
    Pareto solutions are highlighted; best compromise is specially marked.
    """
    pareto_mask = results_df['is_pareto'].values.astype(bool)
    extra_mask  = ~pareto_mask

    obj_configs = [
        ('coverage_rate', 'Coverage Rate ↑', True),
        ('compactness',   'Compactness ↓',   False),
        ('separation',    'Separation ↑',    True),
    ]
    ap_configs = [
        ('mAP',              'mAP'),
        ('AP_pedestrian',    'AP Pedestrian @0.1'),
        ('AP_small_vehicle', 'AP Small Vehicle @0.3'),
        ('AP_large_vehicle', 'AP Large Vehicle @0.5'),
    ]

    for ap_col, ap_label in ap_configs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Proxy Objectives vs {ap_label}', fontsize=13)

        for ax, (obj_col, obj_label, higher_better) in zip(axes, obj_configs):
            x_all  = results_df[obj_col].values
            y_all  = results_df[ap_col].values * 100  # → percentage

            x_par  = results_df.loc[pareto_mask, obj_col].values
            y_par  = results_df.loc[pareto_mask, ap_col].values * 100
            x_ext  = results_df.loc[extra_mask,  obj_col].values
            y_ext  = results_df.loc[extra_mask,  ap_col].values * 100

            # All solutions
            ax.scatter(x_ext, y_ext, c='steelblue',  alpha=0.55, s=40,
                       label='Non-Pareto', zorder=2)
            ax.scatter(x_par, y_par, c='tomato',     alpha=0.80, s=55,
                       label='Pareto front', zorder=3)

            # Best compromise solution
            if best_params is not None:
                # Find closest row to best params
                dist = ((results_df['alpha']   - best_params['alpha']).abs() +
                        (results_df['rho_min'] - best_params['rho_min']).abs())
                best_idx = dist.idxmin()
                bx = results_df.loc[best_idx, obj_col]
                by = results_df.loc[best_idx, ap_col] * 100
                ax.scatter(bx, by, c='gold', edgecolors='black', s=150,
                           marker='*', label='Best compromise', zorder=5)

            # Correlation annotation
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            if valid.sum() > 2:
                r_p, p_p = pearsonr(x_all[valid], y_all[valid])
                r_s, p_s = spearmanr(x_all[valid], y_all[valid])
                ax.annotate(
                    f'Pearson r={r_p:.2f} (p={p_p:.3f})\n'
                    f'Spearman ρ={r_s:.2f} (p={p_s:.3f})',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    va='top', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                )

            ax.set_xlabel(obj_label, fontsize=10)
            ax.set_ylabel(f'{ap_label} (%)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = ap_col.replace('@', '_').replace(' ', '_')
        path = os.path.join(output_folder, f'correlation_{safe_name}.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')

    # 3-D Pareto front coloured by mAP
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection='3d')
    sc = ax3.scatter(
        results_df.loc[pareto_mask, 'coverage_rate'],
        results_df.loc[pareto_mask, 'compactness'],
        results_df.loc[pareto_mask, 'separation'],
        c=results_df.loc[pareto_mask, 'mAP'] * 100,
        cmap='RdYlGn', s=60, alpha=0.85,
    )
    plt.colorbar(sc, ax=ax3, label='mAP (%)', shrink=0.6)
    ax3.set_xlabel('Coverage Rate ↑')
    ax3.set_ylabel('Compactness ↓')
    ax3.set_zlabel('Separation ↑')
    ax3.set_title('Pareto Front Coloured by mAP')
    plt.tight_layout()
    path3d = os.path.join(output_folder, 'pareto_front_mAP.pdf')
    plt.savefig(path3d, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path3d}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    columns = args.columns
    ensure_directory(args.output_folder)
    results_path = os.path.join(args.output_folder, 'correlation_results.csv')

    # ── 1. Ground truth ──────────────────────────────────────────────────────
    print('Loading ground truth...')
    with open(args.gt_json) as f:
        raw = json.load(f)
    gt_data = {}
    for sample in raw['dataset']['samples']:
        frame_id = sample.get('name', '')
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
            gt_data[frame_id] = bboxes
    print(f'  {len(gt_data)} frames with GT annotations')

    # ── 2. Pre-load test frames ───────────────────────────────────────────────
    print('Pre-loading test frames...')
    test_paths = sorted(glob(os.path.join(args.test_folder, '*.bin')))
    test_frames = []  # list of (frame_name, filtered_pcd_array)
    for path in tqdm(test_paths, desc='  test frames'):
        pcd = read_bin(path, len(columns))
        mask = (
            (pcd[:, columns.index('z')] >= 0.1) &
            (pcd[:, columns.index('z')] <= 3.0) &
            (pcd[:, columns.index('dynamic_mask')] == 1)
        )
        test_frames.append((os.path.basename(path), pcd[mask]))
    print(f'  {len(test_frames)} test frames loaded')

    # ── 3. Pareto solutions (objectives already stored) ───────────────────────
    print('Loading Pareto solutions...')
    sol_df = pd.read_csv(os.path.join(args.pareto_folder, 'pareto_solutions.csv'))
    obj_df = pd.read_csv(os.path.join(args.pareto_folder, 'pareto_front.csv'))
    # Sign correction: stored as (-coverage, compactness, -separation)
    combos = []
    for i in range(len(sol_df)):
        combos.append({
            'alpha':         float(sol_df.iloc[i]['alpha']),
            'rho_min':       float(sol_df.iloc[i]['rho_min']),
            'coverage_rate': float(-obj_df.iloc[i]['coverage_rate']),
            'compactness':   float( obj_df.iloc[i]['compactness']),
            'separation':    float(-obj_df.iloc[i]['separation']),
            'is_pareto':     True,
        })
    print(f'  {len(combos)} Pareto solutions')

    # Best compromise solution (for plot marker)
    best_params = None
    best_sol_path = os.path.join(args.pareto_folder, 'best_solutions.csv')
    if os.path.exists(best_sol_path):
        best_df = pd.read_csv(best_sol_path)
        best_params = {'alpha': float(best_df.iloc[0]['alpha']),
                       'rho_min': float(best_df.iloc[0]['rho_min'])}
        print(f'  Best compromise: alpha={best_params["alpha"]:.4f}, '
              f'rho_min={best_params["rho_min"]:.4f}')

    # ── 4. Extra samples across the parameter space ───────────────────────────
    print(f'Generating {args.n_extra} additional parameter samples...')
    rng = np.random.default_rng(42)
    extra_alphas   = rng.uniform(0.0,  1.0, args.n_extra)
    extra_rho_mins = rng.uniform(10.0, 30.0, args.n_extra)

    # Pre-load optimization frames (for proxy-objective computation on extras)
    print('Loading optimization frames...')
    opt_paths = sorted(glob(os.path.join(args.opt_frames_folder, '*.bin')))
    total_opt = len(opt_paths)
    interval  = max(1, total_opt // 278)
    opt_frames = []
    for i in range(0, total_opt, interval):
        pcd = read_bin(opt_paths[i], len(columns))
        mask = (
            (pcd[:, columns.index('z')] >= 0.01) &
            (pcd[:, columns.index('z')] <= 3.0) &
            (pcd[:, columns.index('dynamic_mask')] == 1)
        )
        opt_frames.append(pcd[mask])
    print(f'  {len(opt_frames)} optimization frames loaded')

    print('Computing proxy objectives for extra samples...')
    for alpha, rho_min in tqdm(zip(extra_alphas, extra_rho_mins),
                                total=args.n_extra, desc='  objectives'):
        objs = compute_objectives(alpha, rho_min, opt_frames, columns)
        if objs is not None:
            combos.append({
                'alpha':    alpha,
                'rho_min':  rho_min,
                'is_pareto': False,
                **objs,
            })
    print(f'  Total parameter combinations: {len(combos)}')

    # ── 5. Compute AP for each combination ───────────────────────────────────
    # Resume: skip combinations already in saved results
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        done_keys = set(
            zip(existing['alpha'].round(6), existing['rho_min'].round(6))
        )
        print(f'Resuming: {len(existing)} combinations already computed')
        results = existing.to_dict('records')
    else:
        done_keys = set()
        results = []

    pending = [c for c in combos
               if (round(c['alpha'], 6), round(c['rho_min'], 6)) not in done_keys]
    print(f'Computing AP for {len(pending)} remaining combinations '
          f'({len(test_frames)} test frames each)...')

    for idx, combo in enumerate(tqdm(pending, desc='  param combos')):
        alpha, rho_min = combo['alpha'], combo['rho_min']
        det_by_frame = {}
        for frame_name, filtered_pcd in test_frames:
            if len(filtered_pcd) == 0:
                continue
            bboxes = detect_frame(filtered_pcd, columns, alpha, rho_min,
                                  use_stage2=args.stage_2)
            if bboxes:
                det_by_frame[frame_name] = bboxes

        ap_per_cat, mAP = evaluate_ap(gt_data, det_by_frame)
        results.append({
            'alpha':            alpha,
            'rho_min':          rho_min,
            'coverage_rate':    combo['coverage_rate'],
            'compactness':      combo['compactness'],
            'separation':       combo['separation'],
            'AP_pedestrian':    ap_per_cat.get(3, 0.0),
            'AP_small_vehicle': ap_per_cat.get(1, 0.0),
            'AP_large_vehicle': ap_per_cat.get(2, 0.0),
            'mAP':              mAP,
            'is_pareto':        combo['is_pareto'],
        })
        # Save incrementally every 10 combinations
        if (idx + 1) % 10 == 0 or (idx + 1) == len(pending):
            pd.DataFrame(results).to_csv(results_path, index=False)

    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f'Results saved to {results_path}')

    # ── 6. Print correlation summary ──────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print('\n=== Pearson Correlation (proxy objective vs mAP) ===')
    for obj_col in ['coverage_rate', 'compactness', 'separation']:
        r, p = pearsonr(results_df[obj_col], results_df['mAP'])
        print(f'  {obj_col:20s}: r={r:+.3f}, p={p:.4f}')

    # ── 7. Generate plots ─────────────────────────────────────────────────────
    print('\nGenerating plots...')
    plot_correlation(results_df, best_params, args.output_folder)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proxy-objective vs AP correlation analysis')

    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir',
               'dynamic_mask', 'reliability', 'lidar_source']

    parser.add_argument('--pareto_folder',      default='results/optimize_stage_1',
                        help='Folder with pareto_solutions.csv and pareto_front.csv')
    parser.add_argument('--opt_frames_folder',  default='bin/merge_optimization',
                        help='Folder with optimization point-cloud frames')
    parser.add_argument('--test_folder',        default='bin/merge_1000',
                        help='Folder with test point-cloud frames (1000 frames)')
    parser.add_argument('--gt_json',            default='bin/lowell_1000_label.json',
                        help='Ground-truth JSON file')
    parser.add_argument('--output_folder',      default='results/correlation',
                        help='Folder to save results and plots')
    parser.add_argument('--n_extra',   type=int, default=40,
                        help='Number of extra random parameter samples')
    parser.add_argument('--stage_2',   type=bool, default=True,
                        help='Include Stage-2 hierarchical merging in detection')
    parser.add_argument('--columns',   default=columns)

    args = parser.parse_args()
    main(args)
