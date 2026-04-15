"""
Person detection, tracking, and motion analysis module.

YOLOv8-nano detects people in each frame (optionally every Nth frame for
throughput). ByteTrack associates detections across frames via Kalman
filtering and IoU matching, assigning persistent IDs. Frame differencing
produces a per-pixel motion signal that is masked to tracked person
bounding boxes, then Gaussian-blurred into a spatial heatmap. The peak
of the person-masked heatmap identifies the most active dancer.

Beam masking prevents the system from chasing its own light, and a
configurable min-motion threshold suppresses noise when the floor is idle.
"""
