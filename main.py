"""
VisionBeam entry point.

Usage:
    python main.py                          # Launch Director's Station
    python main.py --calibrate              # Run calibration wizard
    python main.py --no-dmx                 # Run without DMX hardware
    python main.py --camera 1               # Use a specific camera index
    python main.py --fixture path/to.json   # Use a specific fixture profile
"""

import argparse
import queue
import signal
import sys

import cv2

from visionbeam.calibration import FloorCalibration
from visionbeam.dmx import DMXConnection, FixtureProfile
from visionbeam.ik import LightMount
from visionbeam.pipeline import Pipeline


DEFAULT_FLOOR_CAL_PATH = "calibration/floor.json"
DEFAULT_MOUNT_PATH = "calibration/light_mount.json"
DEFAULT_FIXTURE_PATH = "config/fixture_default.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionBeam — autonomous party light")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the calibration wizard instead of the live pipeline")
    parser.add_argument("--no-dmx", action="store_true",
                        help="Run without DMX hardware (CV-only mode)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--dmx-port", type=str, default="/dev/ttyUSB0",
                        help="Serial port for the USB-to-DMX adapter")
    parser.add_argument("--fixture", type=str, default=DEFAULT_FIXTURE_PATH,
                        help="Path to fixture profile JSON")
    return parser.parse_args()


def run_calibration(args: argparse.Namespace):
    """Interactive calibration workflow.

    ---------------------------------------------------------------
    UI TEAM: Replace this with the calibration wizard in ui.py.
    This is a minimal CLI/OpenCV fallback so calibration can be
    tested before the full PyQt5 UI is built.
    ---------------------------------------------------------------
    """
    from visionbeam.calibration import generate_marker_sheet, triangulate_light

    print("=== VisionBeam Calibration ===\n")

    # Step 0: Generate printable markers if needed
    print("Generating marker sheet → calibration/markers.png")
    generate_marker_sheet([0, 1, 2, 3], "calibration/markers.png")

    # Step 1: Camera-to-floor homography
    print("\nStep 1: Camera-to-floor homography")
    print("Place ArUco markers 0-3 at known floor positions.")
    print("Press 'c' to capture when all 4 markers are visible.\n")

    cal = FloorCalibration()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        sys.exit(1)

    captured = False
    while not captured:
        ret, frame = cap.read()
        if not ret:
            continue

        markers = cal.detect_markers(frame)
        display = frame.copy()

        for mid, center in markers.items():
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(display, str(mid), (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = f"Markers detected: {sorted(markers.keys())}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration - Step 1", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and len(markers) >= 4:
            captured = True
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # --- UI TEAM: Replace this input block with a GUI form ---
    import numpy as np
    print("\nEnter floor coordinates (meters) for each marker.")
    pixel_pts = []
    floor_pts = []
    for mid in sorted(markers.keys())[:4]:
        pixel_pts.append(markers[mid])
        x = float(input(f"  Marker {mid} — floor X: "))
        y = float(input(f"  Marker {mid} — floor Y: "))
        floor_pts.append([x, y])

    cal.compute_homography(np.array(pixel_pts), np.array(floor_pts))
    cal.save(DEFAULT_FLOOR_CAL_PATH)
    print(f"\nHomography saved to {DEFAULT_FLOOR_CAL_PATH}")

    # Step 2: Light position triangulation
    print("\nStep 2: Light position triangulation")
    print("Aim the light at 3+ known floor points and record pan/tilt readings.\n")

    # --- UI TEAM: Replace this input block with a GUI form ---
    aim_points = []
    while True:
        resp = input("Add aim point? (y/n): ").strip().lower()
        if resp != "y":
            break
        fx = float(input("  Floor X (meters): "))
        fy = float(input("  Floor Y (meters): "))
        pan = float(input("  Pan reading (degrees): "))
        tilt = float(input("  Tilt reading (degrees): "))
        aim_points.append({
            "floor_x": fx, "floor_y": fy,
            "pan_deg": pan, "tilt_deg": tilt,
        })

    if len(aim_points) < 3:
        print("Error: Need at least 3 aim points.")
        sys.exit(1)

    mount = triangulate_light(aim_points)
    mount.save(DEFAULT_MOUNT_PATH)
    print(f"Light mount saved to {DEFAULT_MOUNT_PATH}")
    print(f"  Position: ({mount.x:.2f}, {mount.y:.2f}, {mount.z:.2f})")
    print(f"  Pan offset: {mount.pan_offset:.1f}°, Tilt offset: {mount.tilt_offset:.1f}°")
    print("\nCalibration complete.")


def run_live(args: argparse.Namespace):
    """Launch the live pipeline with a display loop.

    ---------------------------------------------------------------
    UI TEAM: Replace the cv2.imshow preview below with the full
    Director's Station UI (ui.py). The Pipeline object and its
    display_queue are ready to use:

        pipeline.display_queue  → read PipelineState each frame
        pipeline.set_manual_target(fx, fy) → manual override
        pipeline.set_auto()     → return to autonomous mode
        pipeline.floor_cal.warp_frame(frame, size) → top-down view

    The PyQt5 UI should:
        1. Create its own display_queue and pass it to Pipeline()
        2. Run a QTimer that polls the queue at ~30 Hz
        3. Render the frame/heatmap/indicators in a QLabel or QGraphicsView
        4. Handle mouse clicks on the floor plan for manual override
    ---------------------------------------------------------------
    """
    floor_cal = FloorCalibration.load(DEFAULT_FLOOR_CAL_PATH)
    mount = LightMount.load(DEFAULT_MOUNT_PATH)

    dmx = None
    if not args.no_dmx:
        fixture = FixtureProfile(args.fixture)
        dmx = DMXConnection(args.dmx_port, fixture)
        dmx.set_defaults(dimmer=255)
        dmx.start()

    display_queue = queue.Queue(maxsize=2)

    pipeline = Pipeline(
        camera_id=args.camera,
        floor_cal=floor_cal,
        mount=mount,
        dmx=dmx,
        display_queue=display_queue,
    )

    pipeline.start()
    print("VisionBeam running. Press 'q' to quit, 'a' for auto, 'm' for manual.")

    # --- UI TEAM: Replace this cv2.imshow loop with the PyQt5 UI ---
    try:
        while True:
            try:
                state = display_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            display = state.frame.copy()

            if state.heatmap is not None:
                cv2.addWeighted(display, 0.7, state.heatmap, 0.3, 0, display)

            if state.aim_pixel is not None:
                cx, cy = int(state.aim_pixel[0]), int(state.aim_pixel[1])
                cv2.circle(display, (cx, cy), 12, (0, 0, 255), 2)
                cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)

            info = f"FPS: {state.fps:.0f} | Mode: {state.mode.name}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("VisionBeam", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                pipeline.set_auto()
            elif key == ord("m"):
                pipeline.set_manual_target(2.5, 4.0)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if dmx is not None:
            dmx.blackout()
            dmx.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


def main():
    args = parse_args()

    if args.calibrate:
        run_calibration(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
