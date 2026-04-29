"""
Dataset recording tool.

Captures video clips from the webcam under each of the 5 controlled
lighting conditions (ambient, static color, slow drift, strobe, moving
beam). Saves each clip as a timestamped video file alongside a metadata
JSON containing the lighting condition label, camera index, resolution,
FPS, and duration. Optionally triggers DMX fixture states to automate
the lighting changes between conditions.

Usage:
    python -m evaluation.record --camera 0 --output data/clips/
"""
