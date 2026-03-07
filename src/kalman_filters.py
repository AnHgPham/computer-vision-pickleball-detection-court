"""
Kalman filters for ball tracking in camera and court space.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

import numpy as np
from filterpy.kalman import KalmanFilter as KF

from .constants import CW, CH


class BallKalmanOF:
    """
    Kalman filter for ball tracking in camera space.
    Integrates optical flow velocity estimates when ball is not detected.
    """

    def __init__(self, proc_noise=5.0, meas_noise=2.0, max_miss=10, of_weight=0.3):
        self.proc_noise = proc_noise
        self.meas_noise = meas_noise
        self.max_miss = max_miss
        self.of_weight = of_weight
        self.kf = None
        self.miss = 0
        self.ready = False

    def _create_kf(self, x0, y0):
        """Initialize a constant-velocity Kalman filter."""
        kf = KF(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        q = self.proc_noise ** 2
        kf.Q = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2],
        ], dtype=np.float64)
        kf.R = np.eye(2, dtype=np.float64) * self.meas_noise ** 2
        kf.P *= 100
        kf.x = np.array([[x0], [y0], [0], [0]], dtype=np.float64)
        return kf

    def update(self, detection, of_vel=None):
        """
        Update the Kalman filter with a new detection or predict.

        Args:
            detection: Ball detection dict (with 'pos' key) or None
            of_vel: Optical flow velocity (dx, dy) or None

        Returns:
            Smoothed (x, y) position or None if lost
        """
        if detection is not None:
            cx, cy = detection['pos']
            if not self.ready:
                self.kf = self._create_kf(cx, cy)
                self.ready = True
                self.miss = 0
                return (cx, cy)
            self.kf.predict()
            self.kf.update(np.array([[cx], [cy]]))
            self.miss = 0
        else:
            if not self.ready:
                return None
            self.miss += 1
            if self.miss > self.max_miss:
                return None
            # Integrate optical flow velocity
            if of_vel is not None:
                a = self.of_weight
                self.kf.x[2, 0] = (1 - a) * self.kf.x[2, 0] + a * of_vel[0]
                self.kf.x[3, 0] = (1 - a) * self.kf.x[3, 0] + a * of_vel[1]
            self.kf.predict()

        return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))


class Court2DKalman:
    """
    Kalman filter for ball position in 2D court space.
    Handles trust levels and prevents wild teleportation on the minimap.
    """

    def __init__(self, proc_noise=5.0, meas_noise=10.0, max_pred=15):
        self.kf = KF(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.kf.Q = np.diag([proc_noise, proc_noise, proc_noise * 3, proc_noise * 3])
        self.kf.R = np.eye(2) * meas_noise
        self.kf.P *= 200
        self.ready = False
        self.miss = 0
        self.max_pred = max_pred

    def predict_or_update(self, meas_2d, is_bounce=False,
                          trust_measurement=False, margin=80, max_jump=120):
        """
        Update with 2D measurement or predict next position.

        Args:
            meas_2d: (x, y) in court space or None
            is_bounce: If True, force update (high trust)
            trust_measurement: If True, accept measurement if within bounds
            margin: Court boundary margin
            max_jump: Maximum allowed jump distance

        Returns:
            Smoothed (x, y) in court space or None
        """
        if is_bounce and meas_2d is not None:
            mx, my = meas_2d
            if not self.ready:
                self.kf.x = np.array([[mx], [my], [0], [0]])
                self.ready = True
                self.miss = 0
                return (mx, my)
            self.kf.predict()
            self.kf.update([[mx], [my]])
            self.miss = 0
            return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))

        elif meas_2d is not None and not self.ready:
            mx, my = meas_2d
            self.kf.x = np.array([[mx], [my], [0], [0]])
            self.ready = True
            self.miss = 0
            return (mx, my)

        elif meas_2d is not None and self.ready and trust_measurement:
            mx, my = meas_2d
            self.kf.predict()
            px, py = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
            inside = (-margin <= mx <= CW + margin) and (-margin <= my <= CH + margin)
            close = np.hypot(mx - px, my - py) <= max_jump
            if inside and close:
                self.kf.update([[mx], [my]])
                self.miss = 0
                return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))
            self.miss += 1
            if self.miss > self.max_pred:
                return None
            return (px, py)

        else:
            if not self.ready:
                return None
            self.miss += 1
            if self.miss > self.max_pred:
                return None
            self.kf.predict()
            return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))


def check_in_out(pt, margin=5):
    """Check if a 2D court point is IN or OUT of bounds."""
    if pt is None:
        return None
    x, y = pt
    return 'IN' if -margin <= x <= CW + margin and -margin <= y <= CH + margin else 'OUT'
