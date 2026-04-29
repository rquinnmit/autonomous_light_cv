"""
Ground truth extraction from recorded video.

Detects a known tracking marker (bright retroreflector or specific-color
LED) in each frame of a recorded clip and outputs the marker's (x, y)
pixel position per frame as a CSV. Supports two modes:

1. Color thresholding — HSV range isolation for a distinctly colored marker
   (e.g., bright green LED not present in stage lighting palette).
2. Brightness peak — For a retroreflector that appears as the brightest
   small blob in an otherwise dim scene.

When the marker is not detected in a frame (occlusion, failure), the
frame is flagged and linearly interpolated from neighbors. The output
CSV has columns: frame_number, timestamp_ms, x_px, y_px, detected (bool).

Usage:
    python -m evaluation.ground_truth --video data/clips/ambient_01.mp4 \
                                      --mode color --output data/gt/
"""
