"""
Zone-based projection, optical flow estimation, and bounce detection.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

import cv2
import numpy as np
from collections import deque

from .constants import COURT_DST, CW, CH, ZONES


class ZoneProjector:
    """
    Projects points from camera space to 2D court space using zone-based
    homography. Divides the court into 6 zones, each with its own
    perspective transform for higher accuracy.
    """

    def __init__(self):
        self._zone_Hs = {}
        self._zone_polys = {}
        self._global_H = None
        self._valid = False
        self._n_valid = 0

    def update(self, kps, conf_thresh=0.3):
        """Update homography matrices from detected keypoints."""
        if kps is None or len(kps) < 12:
            return

        v = (kps[:, 2] >= conf_thresh) & (kps[:, 0] > 0) & (kps[:, 1] > 0)

        # Global homography from all valid keypoints
        if v.sum() >= 6:
            H, _ = cv2.findHomography(
                kps[v, :2].astype(np.float32),
                COURT_DST[v].astype(np.float32),
                cv2.RANSAC, 5.0,
            )
            if H is not None and abs(np.linalg.det(H)) > 1e-6:
                self._global_H = H

        # Per-zone perspective transforms
        for name, indices in ZONES.items():
            if all(v[i] for i in indices):
                src = kps[indices, :2].astype(np.float32)
                dst = COURT_DST[indices].astype(np.float32)
                H_l = cv2.getPerspectiveTransform(src, dst)
                if H_l is not None:
                    self._zone_Hs[name] = H_l
                    self._zone_polys[name] = src.astype(np.int32)

        self._n_valid = int(v.sum())
        self._valid = len(self._zone_Hs) > 0 or self._global_H is not None

    def is_reliable(self, kps=None, min_kps=6, min_conf=0.4, min_area=5000):
        """Check if projection is reliable enough to use."""
        if not hasattr(self, '_n_valid') or self._n_valid < min_kps:
            return False
        if kps is None:
            return self._valid

        # Check average confidence
        confs = [kps[i][2] for i in range(len(kps)) if kps[i][2] > 0.1 and kps[i][0] > 0]
        if not confs or np.mean(confs) < min_conf:
            return False

        # Check if corners form a reasonable quadrilateral
        corners = [0, 2, 9, 11]
        cpts = [(kps[i][0], kps[i][1]) for i in corners if kps[i][2] > 0.2 and kps[i][0] > 0]
        if len(cpts) >= 4:
            pts = np.array(cpts)
            area = cv2.contourArea(pts.astype(np.float32))
            if area < min_area:
                return False

        return self._valid

    def find_zone(self, point):
        """Find which zone a camera-space point belongs to."""
        for name, poly in self._zone_polys.items():
            if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                return name
        return None

    def project(self, point):
        """
        Project a camera-space point to 2D court space.

        Args:
            point: (x, y) in camera pixels

        Returns:
            (x, y) in court coordinates or None if projection fails
        """
        if not self._valid:
            return None

        zone = self.find_zone(point)
        H = self._zone_Hs.get(zone) if zone else None
        if H is None:
            H = self._global_H
        if H is None:
            return None

        o = cv2.perspectiveTransform(
            np.array([[[point[0], point[1]]]], dtype=np.float32), H,
        )
        x, y = float(o[0][0][0]), float(o[0][0][1])

        # Reject points too far outside court
        if x < -80 or x > CW + 80 or y < -80 or y > CH + 80:
            return None
        return (x, y)


class OpticalFlowEstimator:
    """
    Lucas-Kanade optical flow estimator for ball velocity tracking.
    Used to improve Kalman filter predictions when ball is not detected.
    """

    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray = None

    def estimate_velocity(self, frame, ball_pos):
        """
        Estimate ball velocity using optical flow.

        Args:
            frame: Current BGR frame
            ball_pos: Last known ball (x, y) position

        Returns:
            (vx, vy) velocity or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or ball_pos is None:
            self.prev_gray = gray
            return None

        cx, cy = ball_pos
        offsets = np.array([
            [0, 0], [-2, -2], [2, -2], [-2, 2], [2, 2],
            [-1, 0], [1, 0], [0, -1], [0, 1],
        ], dtype=np.float32)
        pts = (np.array([[cx, cy]], dtype=np.float32) + offsets).reshape(-1, 1, 2)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **self.lk_params,
        )
        self.prev_gray = gray

        if new_pts is None:
            return None
        status = status.flatten().astype(bool)
        if not np.any(status):
            return None

        vel = (new_pts.reshape(-1, 2) - pts.reshape(-1, 2))[status]
        return tuple(np.median(vel, axis=0).tolist())

    def update_frame(self, frame):
        """Update the previous frame for next optical flow computation."""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class BounceDetector:
    """
    Detects ball bounces from Y-trajectory pattern (local maxima in camera Y).
    Filters out hits (ball near player) vs actual bounces.
    """

    def __init__(self, window=7, min_gap=8):
        self.y_hist = deque(maxlen=window)
        self.last_bounce = -999
        self.min_gap = min_gap
        self.bounces = []

    def _is_hit_not_bounce(self, ball_pos, players):
        """Check if ball is near a player (likely a hit, not a bounce)."""
        if not players:
            return False
        bx, by = ball_pos
        for p in players:
            x1, y1, x2, y2 = p['bbox']
            foot_y = y2
            bh = y2 - y1
            if not (x1 - 30 <= bx <= x2 + 30):
                continue
            if by >= foot_y - bh * 0.1:
                return False
            if by >= foot_y - bh * 0.35:
                return True
            if by < foot_y - bh * 0.35:
                return True
        return False

    def update(self, ball_det, frame_idx, players=None):
        """
        Check if current frame has a bounce.

        Args:
            ball_det: Ball detection dict or None
            frame_idx: Current frame index
            players: List of player dicts (for hit filtering)

        Returns:
            True if bounce detected
        """
        if ball_det is None:
            return False

        self.y_hist.append(ball_det['pos'][1])
        if len(self.y_hist) < 5 or frame_idx - self.last_bounce < self.min_gap:
            return False

        mid = len(self.y_hist) // 2
        ys = list(self.y_hist)
        mid_y = ys[mid]
        before, after = ys[:mid], ys[mid + 1:]

        if not before or not after:
            return False

        # Check for local maximum in Y (bounce = ball goes down then up)
        if mid_y > max(before) - 3 and mid_y > max(after) - 3:
            if np.mean(before) < mid_y and np.mean(after) < mid_y:
                if players and self._is_hit_not_bounce(ball_det['pos'], players):
                    return False
                self.last_bounce = frame_idx
                self.bounces.append({'frame': frame_idx, 'pos': ball_det['pos']})
                return True

        return False
