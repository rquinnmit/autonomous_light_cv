# VisionBeam

## Introduction
VisionBeam is a spatially-aware autonomous party light that combines computer vision, spatial mapping, and live-event hardware. Rather than relying on pre-programmed lighting cues, the system uses person detection and multi-person tracking to identify who is dancing most and steers a moving head light to follow the action in real time.

## System Pipeline

### 1. Spatial Calibration (one-time, pre-event)
Two-step calibration performed once at the venue before an event:
* **Camera-to-floor homography:** Four ArUco markers are placed at known positions on the floor. OpenCV detects them automatically and computes a homography matrix (`cv2.findHomography`) mapping camera pixels to top-down floor coordinates.
* **Light position triangulation:** The light is aimed at 3+ known floor points and the pan/tilt angles are recorded at each. A nonlinear least-squares solver (`scipy.optimize.least_squares`) triangulates the light's 3D mount position (x, y, z) in the same floor coordinate system.

### 2. Person Detection
YOLOv8-nano produces bounding boxes for every person in the scene. Non-human motion (doors, curtains, fog machines, reflections) is discarded at this stage, so downstream tracking operates exclusively on people. Detection can run every Nth frame (e.g., every 2nd or 3rd) to maintain throughput, with the tracker's Kalman predictions filling the gaps.

### 3. Multi-Person Tracking
ByteTrack associates detections across frames using a Kalman filter and IoU-based matching, assigning each person a persistent ID. This enables identity-aware behaviors: the system can follow a specific dancer, smoothly hand off between individuals, and ignore people who have been stationary.

### 4. Motion Heatmap & Target Selection
Frame differencing (`cv2.absdiff`) on downscaled grayscale frames (~320x240) produces a per-pixel motion signal. This signal is masked to only include regions covered by tracked person bounding boxes, then Gaussian-blurred to form a spatial heatmap. The peak of the heatmap identifies the most active dancer on the floor.

**Beam masking:** A mask is applied around the light's current aim point before computing motion, preventing the system from chasing its own beam.

**Temporal smoothing:** An exponential moving average on the target coordinates prevents frame-to-frame jitter and produces smooth light sweeps.

### 5. Spatial Translation (Floor → Pan/Tilt)
Given the light's known mount position from calibration, converting a floor target to pan/tilt angles is direct trigonometry (`atan2`). The homography first maps the heatmap peak from camera pixels to floor coordinates, then the IK step computes the required pan and tilt to aim from the light's mount point to that floor position.

### 6. DMX Hardware Actuation
Pan/tilt angles are scaled to DMX channel values (0–255 coarse, 0–255 fine for 16-bit resolution) and transmitted over USB-to-DMX512 serial at ~40 FPS. A fixture profile config maps logical channels (pan, tilt, dimmer, color, etc.) to DMX addresses, supporting different moving head models.

### 7. Director's Station UI
A real-time operator interface built initially with OpenCV's `imshow` for prototyping, with a planned migration to PySide6 for production use.

**Live monitoring view:**
* Warped top-down floor plan with motion heatmap overlay
* Current light aim and smoothed target indicators
* Camera and DMX connection status

**Manual override:** Click-to-aim on the floor plan, with configurable behavior (permanent override until auto is re-enabled, or timed return to autonomous mode).

**Calibration wizard:** Guided step-by-step UI for the ArUco homography and light triangulation setup.

**Architecture:** The pipeline (camera → CV → IK → DMX) runs on a dedicated thread, pushing display frames and metadata via `queue.Queue` to the UI thread. Manual overrides and parameter changes flow back through shared state.

## Hardware Requirements
1. **DMX Lighting Fixture:** 1x moving head light with DMX512 pan/tilt control (16-bit recommended)
2. **Communication Interface:** 1x USB-to-DMX512 adapter (e.g., Enttec Open DMX)
3. **Camera:** 1x standard USB webcam (720p sufficient)
4. **Calibration Markers:** 4x printed ArUco markers (generated via OpenCV)
5. **Computer:** Any modern laptop (a discrete or integrated GPU is recommended for real-time YOLO inference but not strictly required — CPU inference at ~25-30 FPS is sufficient)

## Software Dependencies
* `python 3.10+`
* `opencv-contrib-python` — camera capture, ArUco detection, homography, frame differencing, display (includes base OpenCV)
* `ultralytics` — YOLOv8-nano person detection and ByteTrack multi-object tracking
* `torch` — PyTorch runtime required by ultralytics (CPU or CUDA)
* `scipy` — light position triangulation via least-squares optimization
* `pyserial` — USB-to-DMX512 serial communication
* `PySide6` — Director's Station UI (production)