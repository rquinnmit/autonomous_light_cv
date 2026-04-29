"""
Target selection methods for comparative evaluation.

Each method implements the same interface: given a pair of consecutive
frames (and optionally prior state), return a predicted target (x, y)
in pixel space representing "where the light should aim." Methods:

1. FrameDiffMethod      — Raw frame differencing (cv2.absdiff), threshold,
                          Gaussian blur, peak of heatmap. No person filtering.
2. FarnebackFlowMethod  — Dense optical flow (cv2.calcOpticalFlowFarneback),
                          magnitude map, peak of motion energy.
3. DetectionMethod      — YOLOv8n + ByteTrack, target = center of most
                          persistent / most central tracked person.
4. HybridMethod         — YOLOv8n bounding boxes mask a frame-differencing
                          heatmap; peak of the person-masked heatmap. This is
                          the current VisionBeam design.

All methods return results in pixel coordinates; the evaluation harness
applies the homography transform to floor coordinates before computing
metrics against ground truth.
"""
