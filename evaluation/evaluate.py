"""
Evaluation harness.

Runs each target-selection method (from evaluation.methods) on every
recorded clip, logs per-frame predictions to CSV, then computes metrics
by comparing predictions against ground truth. Outputs:

- Per-clip CSVs: frame_number, method, pred_x, pred_y, gt_x, gt_y, error_m
- Summary CSV:   clip, lighting_condition, method, mean_error_m,
                 jitter_m_per_sec, fps

Metrics computed:
1. Targeting Accuracy — mean Euclidean distance (meters) between predicted
   and ground-truth floor coordinates.
2. Target Stability (Jitter) — total path length of predictions per second.
3. Robustness Drop-off — accuracy ratio between each condition and baseline.
4. Throughput — average FPS per method on the evaluation hardware.

Usage:
    python -m evaluation.evaluate --clips data/clips/ --gt data/gt/ \
                                  --calibration calibration/homography.json \
                                  --output results/
"""
