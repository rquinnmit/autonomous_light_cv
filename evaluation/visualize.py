"""
Visualization and figure generation for the research report.

Reads the summary and per-clip CSVs produced by evaluate.py and generates
publication-ready matplotlib figures:

1. Accuracy vs. Illumination (grouped bar / line chart)
   — X: lighting condition, Y: mean targeting error, series: method.
2. Trajectory Smoothing (2D floor-plan path plot)
   — Overlays GT path with each method's predicted path for a selected
     5-second window, showing relative jitter.
3. Qualitative Failure Modes (image grid)
   — Selects frames with highest error per method, renders the frame with
     the method's internal state (heatmap, flow field, bounding boxes) and
     the erroneous aim point annotated.

All figures are saved as both PNG (for the report) and PDF (for LaTeX).

Usage:
    python -m evaluation.visualize --results results/ --output figures/
"""
