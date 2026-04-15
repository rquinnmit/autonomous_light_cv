"""
VisionBeam entry point.

Launches either the calibration wizard or the live pipeline with the
PySide6 Director's Station UI (or an OpenCV fallback for development).
The live view renders tracked person bounding boxes, persistent IDs,
and the person-masked motion heatmap alongside light aim indicators.

Usage:
    python main.py                          # Launch Director's Station
    python main.py --calibrate              # Run calibration wizard
    python main.py --no-dmx                 # Run without DMX hardware
    python main.py --camera 1               # Use a specific camera index
    python main.py --fixture path/to.json   # Use a specific fixture profile
"""
