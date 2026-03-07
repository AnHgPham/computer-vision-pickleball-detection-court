"""
Court keypoint detection and court polygon filtering.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


class CourtDetector:
    """
    Detects 12 court keypoints using YOLOv8-Pose model.
    Features temporal smoothing to reduce jitter between frames.
    """

    def __init__(self, model_path, conf=0.5, every_n=5, smooth=5):
        """
        Args:
            model_path: Path to YOLOv8-Pose court keypoint model (.pt)
            conf: Confidence threshold for detection
            every_n: Run detection every N frames (use cached result otherwise)
            smooth: Number of frames for temporal smoothing
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.every_n = every_n
        self._hist = deque(maxlen=smooth)
        self._last = None
        self._n = 0

    def detect(self, frame):
        """
        Detect court keypoints in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            numpy array of shape (12, 3) with [x, y, confidence] per keypoint,
            or None if no court detected
        """
        self._n += 1

        # Use cached result between detection intervals
        if self.every_n > 1 and self._n % self.every_n != 0:
            return self._last.copy() if self._last is not None else None

        res = self.model(frame, verbose=False, conf=self.conf, half=True)

        if not res or res[0].keypoints is None or res[0].keypoints.shape[0] == 0:
            return self._last.copy() if self._last is not None else None

        # Select the detection with highest confidence
        idx = 0
        if res[0].boxes and len(res[0].boxes.conf) > 0:
            idx = res[0].boxes.conf.argmax().item()

        xy = res[0].keypoints.xy[idx].cpu().numpy()
        cf = res[0].keypoints.conf[idx].cpu().numpy()

        kps = np.zeros((len(xy), 3), dtype=np.float32)
        kps[:, :2] = xy
        kps[:, 2] = cf

        self._hist.append(kps.copy())
        self._last = kps.copy()

        # Temporal smoothing with weighted average
        if len(self._hist) <= 1:
            return kps

        ws = np.linspace(0.5, 1.0, len(self._hist))
        ws /= ws.sum()

        sm = np.zeros_like(kps)
        tw = np.zeros(len(sm))

        for w, k in zip(ws, self._hist):
            v = (k[:, 2] > 0) & (k[:, 0] > 0)
            sm[v, :2] += w * k[v, :2]
            sm[v, 2] += w * k[v, 2]
            tw[v] += w

        nz = tw > 0
        sm[nz, :2] /= tw[nz, None]
        sm[nz, 2] /= tw[nz]

        return sm


class CourtPolygonFilter:
    """
    Uses 4 corner keypoints (0, 2, 9, 11) to create a polygon.
    Filters out players/objects outside the court area (spectators, referees).
    """

    def __init__(self, margin_ratio=0.15):
        """
        Args:
            margin_ratio: How much to expand the polygon beyond court corners
        """
        self.margin = margin_ratio
        self.polygon = None

    def update(self, kps, conf_thresh=0.3):
        """Update the court polygon from detected keypoints."""
        if kps is None or len(kps) < 12:
            return

        corners_idx = [0, 2, 9, 11]
        pts = []
        for idx in corners_idx:
            if kps[idx][2] >= conf_thresh and kps[idx][0] > 0:
                pts.append(kps[idx][:2])

        if len(pts) == 4:
            poly = np.array(pts, dtype=np.float32)
            cx, cy = poly.mean(axis=0)
            expanded = []
            for p in poly:
                dx, dy = p[0] - cx, p[1] - cy
                expanded.append([p[0] + dx * self.margin, p[1] + dy * self.margin])
            self.polygon = np.array(expanded, dtype=np.int32)

    def is_on_court(self, foot_pos):
        """Check if a point (foot position) is inside the court polygon."""
        if self.polygon is None:
            return True  # No polygon = accept all
        return cv2.pointPolygonTest(
            self.polygon,
            (float(foot_pos[0]), float(foot_pos[1])),
            False,
        ) >= 0
