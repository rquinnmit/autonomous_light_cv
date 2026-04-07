"""
Motion detection module.

Frame differencing, heatmap generation, beam masking,
and temporal smoothing to produce a stable floor target.
"""

import cv2
import numpy as np


class MotionTracker:
    """Detects the densest motion zone in a camera feed.

    Operates at a reduced resolution internally for performance.
    All returned coordinates are in the original frame's pixel space.

    Args:
        process_width: Internal processing width in pixels. The frame is
            downscaled to this width (maintaining aspect ratio) before any
            CV work. 320 is plenty for zone-level detection.
        blur_kernel: Size of the Gaussian kernel applied to the motion mask
            to form the heatmap. Larger = broader/smoother hotspots.
            Must be odd.
        threshold: Pixel intensity difference (0-255) that counts as motion.
            Lower = more sensitive, higher = ignores subtle changes.
        mask_radius: Radius (in processing pixels) of the circular mask
            applied around the beam's current position.
        min_motion: Minimum heatmap peak value to count as real motion.
            Below this, update() returns None (avoids chasing noise).
    """

    def __init__(
        self,
        process_width: int = 320,
        blur_kernel: int = 31,
        threshold: int = 25,
        mask_radius: int = 50,
        min_motion: float = 5.0,
    ):
        if blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be odd, got {blur_kernel}")

        self.process_width = process_width
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self.mask_radius = mask_radius
        self.min_motion = min_motion

        self._prev_gray: np.ndarray | None = None
        self._heatmap: np.ndarray | None = None
        self._scale: float = 1.0

    def update(
        self,
        frame: np.ndarray,
        beam_pixel: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Process one camera frame and return the peak motion location.

        Args:
            frame: BGR camera frame at full resolution.
            beam_pixel: Current light aim position in full-resolution pixel
                coordinates. A circular region around this point is masked
                out to prevent the system from tracking its own beam.

        Returns:
            (x, y) pixel coordinates of the motion peak in the original
            frame's resolution, or None if no significant motion is detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        orig_h, orig_w = gray.shape
        self._scale = orig_w / self.process_width
        process_h = int(orig_h / self._scale)
        small = cv2.resize(gray, (self.process_width, process_h))

        if self._prev_gray is None:
            self._prev_gray = small
            return None

        diff = cv2.absdiff(self._prev_gray, small)
        self._prev_gray = small

        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        if beam_pixel is not None:
            bx = int(beam_pixel[0] / self._scale)
            by = int(beam_pixel[1] / self._scale)
            cv2.circle(motion_mask, (bx, by), self.mask_radius, 0, -1)

        heatmap = cv2.GaussianBlur(
            motion_mask.astype(np.float32),
            (self.blur_kernel, self.blur_kernel),
            0,
        )
        self._heatmap = heatmap

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)

        if max_val < self.min_motion:
            return None

        peak_x = max_loc[0] * self._scale
        peak_y = max_loc[1] * self._scale
        return peak_x, peak_y

    @property
    def heatmap(self) -> np.ndarray | None:
        """The latest motion heatmap at processing resolution.

        A float32 grayscale image where brighter = more motion.
        Returns None before the first pair of frames is processed.
        """
        return self._heatmap

    def heatmap_color(self, size: tuple[int, int] | None = None) -> np.ndarray | None:
        """The latest heatmap as a colorized BGR image for UI overlay.

        Args:
            size: Optional (width, height) to resize the output.
                  If None, returns at the internal processing resolution.

        Returns:
            BGR image with JET colormap applied, or None if no heatmap yet.
        """
        if self._heatmap is None:
            return None

        normalized = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

        if size is not None:
            colored = cv2.resize(colored, size)

        return colored

    def reset(self):
        """Clear state. Call when switching camera or restarting."""
        self._prev_gray = None
        self._heatmap = None
