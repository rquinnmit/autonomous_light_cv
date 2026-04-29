# VisionBeam — Related Work & References

A curated collection of papers, benchmarks, and systems relevant to the VisionBeam research project. Organized by topic area with brief annotations on why each is relevant.

---

## 1. Optical Flow — Classical Methods

### Farneback (2003) — Dense Optical Flow via Polynomial Expansion
- **Citation:** G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," *Proc. 13th Scandinavian Conference on Image Analysis (SCIA)*, LNCS 2749, pp. 363–370, 2003.
- **Link:** https://doi.org/10.1007/3-540-45103-X_50
- **Relevance:** The dense flow baseline in our comparison. Approximates each pixel neighborhood with a quadratic polynomial and estimates displacement from coefficient changes. Implemented in OpenCV as `cv2.calcOpticalFlowFarneback`. Known to be sensitive to illumination changes — quantifying this degradation under stage lighting is a core part of our study.

### Lucas & Kanade (1981) — Sparse Optical Flow
- **Citation:** B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision," *Proc. 7th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 674–679, 1981.
- **Relevance:** Foundation of the KLT (Kanade-Lucas-Tomasi) sparse tracker. Assumes brightness constancy and small displacements — both assumptions break under stage lighting. Implemented in OpenCV as `cv2.calcOpticalFlowPyrLK`.

### Shi & Tomasi (1994) — Good Features to Track
- **Citation:** J. Shi and C. Tomasi, "Good Features to Track," *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 593–600, 1994.
- **Link:** https://cecas.clemson.edu/~stb/klt/shi-tomasi-good-features-cvpr1994.pdf
- **Relevance:** Defines the feature selection criterion used with Lucas-Kanade flow. If we include sparse flow in our comparison, this is the feature detector. Implemented in OpenCV as `cv2.goodFeaturesToTrack`.

---

## 2. Optical Flow — Deep Learning Methods

### Teed & Deng (2020) — RAFT
- **Citation:** Z. Teed and J. Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow," *Proc. European Conference on Computer Vision (ECCV)*, pp. 402–419, 2020.
- **Link:** https://arxiv.org/abs/2003.12039
- **Code:** https://github.com/princeton-vl/RAFT
- **Relevance:** State-of-the-art deep optical flow. Extracts per-pixel features, builds 4D correlation volumes, and iteratively refines flow via a recurrent GRU. Achieves 5.10% F1-all on KITTI and 2.855 EPE on Sintel. Useful as an offline upper bound in our comparison — too slow for real-time on CPU, but demonstrates how much accuracy classical flow leaves on the table.

---

## 3. Optical Flow — Robustness & Illumination

### Fortun et al. (2015) — Optical Flow Survey
- **Citation:** D. Fortun, P. Bouthemy, and C. Kervrann, "Optical Flow Modeling and Computation: A Survey," *Computer Vision and Image Understanding*, vol. 134, pp. 1–21, 2015.
- **Link:** https://doi.org/10.1016/j.cviu.2014.09.005
- **Relevance:** Comprehensive survey covering data constancy assumptions, regularization, and the impact of illumination changes on flow estimation. Good for framing why classical flow degrades and what invariants help.

### Alfarano et al. (2024) — Optical Flow Review (Illumination Section)
- **Citation:** A. Alfarano et al., "Estimating Optical Flow: A Comprehensive Review of the State of the Art," 2024.
- **Link:** https://iris.uniroma1.it/retrieve/3f23fbb7-2c19-4b91-8821-7082ff204f5b/Alfarano_Estimating-optical_2024.pdf
- **Relevance:** Up-to-date review with a dedicated section (§3.3) on illumination variation in optical flow. Covers robust feature descriptors (HOG, SIFT), domain-change methods, photometric invariant color spaces, and Laplacian filtering — all candidate preprocessing steps for making classical flow more robust under stage lighting.

### Kim et al. (2005) — Robust Motion Estimation Under Varying Illumination
- **Citation:** S. J. Kim, J. M. Frahm, and M. Pollefeys, "Robust Motion Estimation Under Varying Illumination," *Image and Vision Computing*, vol. 23, no. 4, pp. 365–375, 2005.
- **Link:** https://engineering.purdue.edu/RVL/Publications/Kim05Robust.pdf
- **Relevance:** Directly addresses our core challenge. Proposes a framework that handles both multiplicative and additive illumination changes simultaneously with motion discontinuities. Shows that naïve brightness constancy fails under large illumination variation — exactly the stage-lighting scenario.

### Mileva et al. (2007) — Illumination-Robust Variational Optical Flow
- **Citation:** Y. Mileva, A. Bruhn, and J. Weickert, "Illumination-Robust Variational Optical Flow with Photometric Invariants," *Proc. DAGM (Pattern Recognition)*, LNCS 4713, pp. 152–162, 2007.
- **Link:** https://doi.org/10.1007/978-3-540-74936-3_16
- **Relevance:** Uses photometric invariants (spherical/conical color transforms, log-differentiation) embedded in a variational flow framework to achieve robustness under strong illumination changes. Demonstrates that color-space transformations can partially decouple motion from lighting — a technique we could apply as preprocessing before Farneback flow.

### Schmalfuss et al. (2024) — FlowBench
- **Citation:** J. Schmalfuss et al., "FlowBench: A Robustness Benchmark for Optical Flow," *OpenReview*, 2024.
- **Link:** https://openreview.net/pdf/8e159018a5c024da7e73dc7490953b76805423d9.pdf
- **Relevance:** Benchmarks 57 optical flow model checkpoints across adversarial attacks and common image corruptions. Key finding: methods with state-of-the-art i.i.d. performance often lack robustness to distribution shifts — directly relevant to our hypothesis that DL flow trained on clean data may not generalize to stage lighting without fine-tuning.

---

## 4. Object Detection — Architecture & Low-Light Robustness

### Jocher et al. (2023) — YOLOv8
- **Citation:** G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," version 8.0.0, 2023.
- **Link:** https://github.com/ultralytics/ultralytics
- **Relevance:** The detection backbone in our current pipeline (YOLOv8-nano). Anchor-free split head, CSPNet backbone, FPN+PAN neck. No formal paper published; cite via the software reference above per Ultralytics' recommendation.

### Loh & Chan (2019) — ExDark Dataset
- **Citation:** Y. P. Loh and C. S. Chan, "Getting to Know Low-light Images with the Exclusively Dark Dataset," *Computer Vision and Image Understanding*, vol. 178, pp. 30–42, 2019.
- **Link:** https://arxiv.org/abs/1805.11227
- **Code/Data:** https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
- **Relevance:** The standard benchmark for low-light object detection. 7,363 images across 10 illumination types with 12 object classes. Key insight: "the effects of low-light reach far deeper into the features than can be solved by simple illumination invariance." If we extend our study to evaluate detection mAP under different lighting, ExDark provides context for comparison.

### Dark-YOLO (2025)
- **Citation:** "Dark-YOLO: A Low-Light Object Detection Algorithm Integrating Multiple Attention Mechanisms," *Applied Sciences*, vol. 15, no. 9, 5170, 2025.
- **Link:** https://www.mdpi.com/2076-3417/15/9/5170
- **Relevance:** Achieves 71.3% mAP@50 on ExDark by integrating SCINet enhancement, dynamic feature extraction, and attention mechanisms into a YOLOv8 backbone. Demonstrates the gap between vanilla YOLO and low-light-optimized variants — helps contextualize how much our YOLOv8n baseline might degrade under stage lighting.

### POP-YOLOv8 (2026) — Nighttime Pedestrian Detection
- **Citation:** "POP-YOLOv8," *Scientific Reports*, 2026.
- **Link:** https://doi.org/10.1038/s41598-026-35146-9
- **Relevance:** Integrates SCI brightness enhancement (via TensorRT FP32/FP16), Feature Enhancement Module, and Partial Occlusion Pedestrian Attention Module into YOLOv8n. Outperforms YOLOv10 for partially occluded pedestrians under nighttime conditions. Demonstrates the SCI preprocessing → YOLO pipeline that could serve as a comparison point.

---

## 5. Low-Light Image Enhancement (Preprocessing)

### Guo et al. (2020) / Li et al. (2021) — Zero-DCE / Zero-DCE++
- **Citation (CVPR 2020):** C. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1780–1789, 2020.
- **Citation (TPAMI 2021):** C. Li, C. Guo, and C. C. Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.
- **Link:** https://arxiv.org/abs/2103.00860
- **Code:** https://github.com/Li-Chongyi/Zero-DCE
- **Relevance:** Zero-reference low-light enhancement using learned pixel-wise curves. Zero-DCE++ has only 10K parameters and runs at 1000 FPS on GPU / 11 FPS on CPU — fast enough to use as a real-time preprocessing step. Could be tested as an enhancement stage before frame differencing or flow to improve robustness.

### Zuiderveld (1994) — CLAHE
- **Citation:** K. J. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV* (P. S. Heckbert, ed.), pp. 474–485, Elsevier, 1994.
- **Link:** https://doi.org/10.1016/b978-0-12-336156-1.50061-6
- **Relevance:** Classical contrast enhancement via tile-based adaptive histogram equalization with a clip limit to prevent noise amplification. Available in OpenCV as `cv2.createCLAHE`. The simplest preprocessing candidate to test before flow or detection under low/uneven stage lighting.

---

## 6. Multi-Object Tracking

### Zhang et al. (2022) — ByteTrack
- **Citation:** Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *Proc. European Conference on Computer Vision (ECCV)*, pp. 1–21, 2022.
- **Link:** https://arxiv.org/abs/2110.06864
- **Code:** https://github.com/ifzhang/ByteTrack
- **Relevance:** The tracker used in our current pipeline. Key idea: associate *every* detection box (including low-confidence ones) rather than only high-score boxes, recovering occluded targets. Achieves 80.3 MOTA, 77.3 IDF1 on MOT17 at 30 FPS. Important to cite since our hybrid method relies on ByteTrack for persistent IDs and bounding box masks.

### Sun et al. (2022) — DanceTrack Benchmark
- **Citation:** P. Sun et al., "DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 20961–20970, 2022.
- **Link:** https://arxiv.org/abs/2111.14690
- **Code/Data:** https://github.com/DanceTrack/DanceTrack
- **Relevance:** The closest existing benchmark to our domain. 100 group dance videos with uniform appearance and diverse motion. Key finding: state-of-the-art trackers show significant performance drops compared to MOT17, especially on association (IDF1). Useful for contextualizing tracking difficulty in dance/performance scenarios and as a potential multi-person evaluation dataset.

---

## 7. Autonomous Stage Lighting Systems

### Naostage K SYSTEM (Commercial)
- **Link:** https://www.naostage.com/en
- **Relevance:** The closest commercial system to VisionBeam. Uses multi-spectral cameras (visible, near-IR, thermal) with a proprietary deep learning model trained on thousands of hours of shows. Provides beaconless 3D tracking of up to 16 performers with centimeter accuracy. Outputs tracking data via PosiStageNet/OSC to lighting consoles. Differs from VisionBeam in that it requires specialized hardware (KAPTA sensor rig + KORE server) and separates tracking from lighting control.

### Li et al. (2022) — Intelligent Stage Light Actor Tracking
- **Citation:** "An Intelligent Stage Light-based Actor Identification and Positioning System," *International Journal of Information and Computer Security (IJICS)*, vol. 18, no. 1/2, pp. 204–218, 2022.
- **Link:** https://doi.org/10.1504/IJICS.2022.122920
- **Relevance:** Proposes a DCNN + particle filter pipeline for automatic stage light tracking of actors. The closest published academic work to VisionBeam's goal. However, it does not address the perception-under-own-illumination problem or compare flow-based vs. detection-based approaches.

---

## 8. Spatial Calibration

### Garrido-Jurado et al. (2014) — ArUco Markers
- **Citation:** S. Garrido-Jurado et al., "Automatic Generation and Detection of Highly Reliable Fiducial Markers Under Occlusion," *Pattern Recognition*, vol. 47, no. 6, pp. 2280–2292, 2014.
- **Link:** https://doi.org/10.1016/j.patcog.2014.01.005
- **Relevance:** Defines the ArUco marker system used in our calibration module. Covers marker dictionary design, detection algorithm, and handling of partial occlusion.

### OpenCV ArUco Module Documentation
- **Link:** https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- **Relevance:** Reference documentation for the ArUco detection and camera calibration APIs used in `visionbeam/calibration.py` (`cv2.aruco.ArucoDetector`, `cv2.findHomography`, `cv2.perspectiveTransform`).

### Martinez et al. (2023) — ArUco-Based Workspace Calibration
- **Citation:** D. S. Martinez et al., "Automatic Workspace Calibration Using Homography for Pick and Place," *Proc. IEEE International Conference on Automation Science and Engineering (CASE)*, Auckland, New Zealand, 2023.
- **Link:** https://doi.org/10.1109/CASE56687.2023.10260601
- **Code:** https://github.com/david-s-martinez/Automatic-Workspace-Calibration-Based-on-Aruco
- **Relevance:** Closely mirrors our calibration approach: ArUco markers placed at known real-world positions, homography computation, and perspective warping for a top-down view. Provides a citable precedent for our `FloorCalibration` implementation.