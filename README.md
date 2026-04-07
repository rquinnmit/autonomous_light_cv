# VisionBeam

## Introduction
VisionBeam is a spatially-aware autonomous party light that combines computer vision, spatial mapping, and live-event hardware. Rather than relying on pre-programmed lighting cues, this system is able to direct creative lighting patterns based on dance floor movement.

## Core Features
* **Spatial Calibration:** Uses OpenCV to calculate a perspective transform mapping the camera feed to a top-down 2D floor plan for the UI.
* **Autonomous Tracking:** Utilizes dense optical flow / background subtraction to identify the highest density of movement on the dance floor.
* **Spatial Translation:** Instantly converts 2D screen coordinates into physical inverse kinematics, calculating the required angles to aim the physical light.
* **Hardware Actuation:** Communicates directly with standard DMX512 event lighting via serial communication.
* **Director's Station UI:** A custom graphical interface that displays the calibrated 2D floor plan and allows users to interact with the light.

## Hardware Requirements
1. **1x DMX Lighting Fixture** 1x moving head light (must support DMX and pan/tilt channels)
2. **Communication Interface:** 1x USB-to-DMX512 Adapter
3. **Camera:** 1x standard webcam
4. **Computer:** Any modern laptop

## Software Dependencies
* `python 3.8+`
* `opencv-python`
* `pyserial` (For transmitting DMX data to USB)
* `numpy`