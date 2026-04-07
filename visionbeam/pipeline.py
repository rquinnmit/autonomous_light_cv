"""
Main pipeline loop.

Camera capture → motion detection → spatial translation → DMX output.
Runs on a dedicated thread, pushes display data to the UI via queue.
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from visionbeam.calibration import FloorCalibration
from visionbeam.dmx import DMXConnection
from visionbeam.ik import LightMount, TargetSmoother, floor_to_pan_tilt
from visionbeam.tracker import MotionTracker


class Mode(Enum):
    AUTO = auto()
    MANUAL = auto()


@dataclass
class PipelineState:
    """Snapshot of pipeline state pushed to the UI each frame."""
    frame: np.ndarray
    heatmap: np.ndarray | None
    aim_floor: tuple[float, float] | None
    target_floor: tuple[float, float] | None
    aim_pixel: tuple[float, float] | None
    mode: Mode
    fps: float


class Pipeline:
    """Ties together camera, tracker, IK, and DMX into a real-time loop.

    Runs on a dedicated thread. The UI reads display state from
    the display_queue and sends commands via set_manual_target / set_auto.

    Args:
        camera_id: OpenCV camera index (usually 0).
        floor_cal: Loaded FloorCalibration with a valid homography.
        mount: Calibrated LightMount with position and offsets.
        dmx: Optional DMXConnection. If None, the pipeline runs without
             hardware output (useful for testing CV in isolation).
        tracker: Optional MotionTracker. Uses defaults if not provided.
        smoother: Optional TargetSmoother. Uses defaults if not provided.
        display_queue: Optional queue for pushing PipelineState to the UI.
    """

    def __init__(
        self,
        camera_id: int,
        floor_cal: FloorCalibration,
        mount: LightMount,
        dmx: DMXConnection | None = None,
        tracker: MotionTracker | None = None,
        smoother: TargetSmoother | None = None,
        display_queue: queue.Queue | None = None,
    ):
        self.camera_id = camera_id
        self.floor_cal = floor_cal
        self.mount = mount
        self.dmx = dmx
        self.tracker = tracker or MotionTracker()
        self.smoother = smoother or TargetSmoother()
        self.display_queue = display_queue or queue.Queue(maxsize=2)

        self.mode = Mode.AUTO
        self._manual_target: tuple[float, float] | None = None
        self._manual_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._aim_floor: tuple[float, float] | None = None
        self._aim_pixel: tuple[float, float] | None = None

    def set_manual_target(self, floor_x: float, floor_y: float):
        """Override autonomous tracking with a specific floor coordinate."""
        with self._manual_lock:
            self._manual_target = (floor_x, floor_y)
            self.mode = Mode.MANUAL

    def set_auto(self):
        """Return to autonomous tracking mode."""
        with self._manual_lock:
            self._manual_target = None
            self.mode = Mode.AUTO
        self.smoother.reset()
        self.tracker.reset()

    def _push_state(self, state: PipelineState):
        """Push state to the UI, dropping the oldest frame if the queue is full."""
        try:
            self.display_queue.put_nowait(state)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            self.display_queue.put_nowait(state)

    def _loop(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        prev_time = time.monotonic()
        fps = 0.0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                now = time.monotonic()
                dt = now - prev_time
                fps = 1.0 / dt if dt > 0 else 0.0
                prev_time = now

                target_floor = None

                with self._manual_lock:
                    current_mode = self.mode
                    manual = self._manual_target

                if current_mode == Mode.AUTO:
                    peak = self.tracker.update(frame, beam_pixel=self._aim_pixel)
                    if peak is not None:
                        fx, fy = self.floor_cal.pixel_to_floor(peak[0], peak[1])
                        target_floor = (fx, fy)
                        sx, sy = self.smoother.update(fx, fy)
                        self._aim_floor = (sx, sy)
                        self._aim_pixel = peak

                elif current_mode == Mode.MANUAL and manual is not None:
                    target_floor = manual
                    self._aim_floor = manual

                if self._aim_floor is not None:
                    pan, tilt = floor_to_pan_tilt(
                        self._aim_floor[0], self._aim_floor[1], self.mount,
                    )
                    if self.dmx is not None:
                        self.dmx.aim(pan, tilt)

                heatmap = self.tracker.heatmap_color(
                    size=(frame.shape[1], frame.shape[0]),
                )

                self._push_state(PipelineState(
                    frame=frame,
                    heatmap=heatmap,
                    aim_floor=self._aim_floor,
                    target_floor=target_floor,
                    aim_pixel=self._aim_pixel,
                    mode=current_mode,
                    fps=fps,
                ))
        finally:
            cap.release()

    def start(self):
        """Start the pipeline loop on a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the pipeline and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
