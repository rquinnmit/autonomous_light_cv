# VisionBeam Research Plan: Motion-Driven Target Selection Under Dynamic Stage Illumination

## 1. Research Question

**For motion-driven target selection in autonomous stage lighting, how do detection-based, classical optical flow, and hybrid approaches compare under dynamic stage illumination—and at what point does adding deep learning actually help?**

This study aims to evaluate the robustness and accuracy of different computer vision techniques for identifying "where to point the light" in challenging, realistic live-event environments (low light, fog, strobes, and color-shifting illumination). 

## 2. Methods to Compare

Our core contribution is the **Hybrid Detection + Motion** method — the target-selection pipeline native to VisionBeam. It combines deep learning (YOLOv8n person detection + ByteTrack multi-object tracking) with classical computer vision (frame-differencing motion heatmap masked to tracked bounding boxes). This hybrid design is motivated by the hypothesis that DL-based person filtering prevents the motion signal from being corrupted by non-human illumination artifacts (beam reflections, strobe flashes, fog scatter), while the classical motion signal within those regions identifies *which* person is most active without requiring pose estimation or action recognition.

To evaluate whether this hybrid design is justified — and to isolate the contribution of each component — we compare it against three baseline methods that each omit one or both of the hybrid's components. All methods implement the same interface: given a frame, output a 2D floor coordinate $(x,y)$ representing the target aim point.

### Baselines (implemented in `evaluation/methods.py`)

1. **Classical Motion (Frame Differencing):** Computes `cv2.absdiff` between consecutive frames, applies a threshold, and finds the spatial peak of a Gaussian-blurred heatmap. No person filtering — the motion signal includes non-human sources.
2. **Classical Dense Flow (Farneback):** Computes dense optical flow (`cv2.calcOpticalFlowFarneback`). The target is derived from the region with the highest motion magnitude. More robust to global brightness shifts than frame differencing, but still susceptible to illumination-induced apparent motion.
3. **Deep Learning Detection Only (YOLOv8 + ByteTrack):** Uses YOLOv8n to detect persons and ByteTrack to track them. The target is the center of the bounding box with the highest track persistence or nearest to the center. No motion signal — cannot distinguish a stationary person from an active dancer.

### Core Method (implemented in `visionbeam/tracker.py`)

4. **Hybrid Detection + Motion (VisionBeam):** Uses YOLOv8n + ByteTrack to generate tracked bounding boxes, then masks a frame-differencing heatmap to only include motion *inside* those boxes. The peak of the person-masked heatmap identifies the most active person. Includes beam masking (to prevent chasing the system's own light) and a configurable min-motion threshold.

### Optional Upper Bound

5. *(Stretch goal)* **Deep Flow (RAFT):** Offline evaluation using a DL-based optical flow model to see how much classical flow degrades compared to state-of-the-art learned flow under stage lighting.

## 3. Evaluation Framework (Dataset Collection)

Since real multi-person stage environments are hard to stage, we will leverage a controlled, single-person studio environment to systematically vary illumination.

### Independent Variable: Lighting Conditions

We will record 30–60 second video clips for each of the following controlled lighting sweeps:

1. **Baseline Ambient:** Standard room lighting (easy CV condition).
2. **Static Color:** One additional stage fixture turned on, fixed color (e.g., deep red or blue).
3. **Slow Drift:** Slow color and intensity shifts (simulating a slow scene transition).
4. **Strobe / Fast Change:** Rapid color shifts and strobing (worst-case scenario).
5. **Moving Beam:** The VisionBeam fixture actively sweeping the scene (testing self-induced illumination changes).

### Ground Truth Generation

To evaluate "targeting accuracy," we need a ground-truth $(x,y)$ floor coordinate for each frame.

- **Method:** Wear a bright, distinct tracking marker (e.g., a retroreflector or highly specific color patch/LED).
- **Extraction:** Use a simple offline color/brightness thresholding script (separate from the evaluated pipelines) to extract the $(x,y)$ center of the marker for every frame. 
- *Fallback:* If a marker fails under colored stage lighting, use manual sparse labeling (clicking the target location every 15-30 frames and interpolating).

## 4. Metrics Collection

For each frame $t$, the ground truth is $GT_t$ and the predicted target from method $M$ is $P_{M, t}$. 

1. **Targeting Accuracy (Mean Error):**
  - The Euclidean distance in meters on the floor plane between the predicted target and the ground truth.
  - $Error = \frac{1}{N} \sum || P_{M,t} - GT_t ||$
2. **Target Stability (Jitter):**
  - The total path length the predicted target travels per second. A high value indicates erratic, jittery aiming.
  - $Jitter = \sum || P_{M,t} - P_{M,t-1} ||$
3. **Robustness Drop-off:**
  - The percentage increase in Targeting Error from the Baseline Ambient condition to the Strobe/Fast Change condition.
4. **Throughput / Latency:**
  - Average Frames Per Second (FPS) on the target hardware (laptop CPU/GPU).

## 5. Visualization Plan

We will generate the following figures for the final report:

### Figure 1: Accuracy vs. Illumination (The Headline Plot)

- **Type:** Grouped Bar Chart or Line Graph.
- **X-axis:** Lighting Conditions (Ambient $\rightarrow$ Static $\rightarrow$ Drift $\rightarrow$ Strobe $\rightarrow$ Moving Beam).
- **Y-axis:** Mean Targeting Error (meters).
- **Series:** One line/bar color per method (Frame Diff, Farneback, YOLO, Hybrid).
- **Hypothesis:** Classical methods degrade sharply under Strobe/Drift; YOLO remains stable; Hybrid offers the best overall accuracy.

### Figure 2: Trajectory Smoothing (Jitter Analysis)

- **Type:** 2D Floor Plan Scatter/Path Plot.
- **Visual:** Plot the 2D path of the $GT_t$ (smooth line) overlaid with the 2D path of $P_{M,t}$ for different methods over a 5-second window.
- **Purpose:** Visually demonstrates how "jittery" classical flow or raw detection might be compared to smoothed hybrid tracking.

### Figure 3: Qualitative Failure Modes

- **Type:** Image Grid.
- **Visual:** Select 3-4 specific frames where methods failed (e.g., frame differencing locking onto a strobe reflection on the wall; YOLO failing to detect a person in deep blue light). Show the frame, the algorithm's internal state (flow field, bounding boxes, or heatmap), and the resulting erroneous aim point.

