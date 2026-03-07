"""
Ball detection with 4-stage cascade pipeline.
Extracted from notebook 03_pipeline_inference.ipynb.

Cascade stages:
  1. YOLO ball model (dedicated)
  2. YOLO player model (pickleball class)
  3. Classical CV (HSV color + contour)
  4. Trajectory interpolation (polynomial fitting)
"""

import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO


class ClassicalBallDetector:
    """
    Classical CV ball detection using color segmentation + contour analysis.
    Fallback when both YOLO models miss the ball.
    """

    def __init__(self, min_area=4, max_area=800, min_circ=0.3):
        self.lower_hsv = np.array([20, 80, 150])
        self.upper_hsv = np.array([55, 255, 255])
        self.min_area = min_area
        self.max_area = max_area
        self.min_circ = min_circ
        self.prev_gray = None

    def detect(self, frame, court_mask=None):
        """Detect ball using color segmentation and motion detection."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Motion mask from frame difference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
            mask = cv2.bitwise_and(mask, motion_mask)
        self.prev_gray = gray.copy()

        if court_mask is not None:
            mask = cv2.bitwise_and(mask, court_mask)

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circ:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            # Skip top region (scoreboard area)
            if cy < h * 0.15:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            candidates.append({
                'pos': (float(cx), float(cy)),
                'bbox': (x, y, x + bw, y + bh),
                'conf': min(0.5, circularity * 0.5 + area / self.max_area * 0.3),
                'area': area,
                'circ': circularity,
                'method': 'classical',
                'ground_anchor': (float(cx), float(cy)),
                'size': (float(bw), float(bh)),
            })

        if candidates:
            candidates.sort(key=lambda c: c['circ'], reverse=True)
            return candidates[0]
        return None


class TrajectoryInterpolator:
    """
    Polynomial trajectory fitting for ball position prediction.
    Predicts ball position when missed for 1-5 frames.
    """

    def __init__(self, history_len=15, max_predict=5, poly_deg=2):
        self.history = deque(maxlen=history_len)
        self.max_predict = max_predict
        self.poly_deg = poly_deg
        self.miss_count = 0

    def add(self, frame_idx, pos):
        """Add a detected position to history."""
        self.history.append((frame_idx, pos[0], pos[1]))
        self.miss_count = 0

    def predict(self, frame_idx):
        """Predict ball position from trajectory history."""
        self.miss_count += 1
        if self.miss_count > self.max_predict or len(self.history) < 4:
            return None

        frames = np.array([h[0] for h in self.history])
        xs = np.array([h[1] for h in self.history])
        ys = np.array([h[2] for h in self.history])

        try:
            deg = min(self.poly_deg, len(self.history) - 1)
            px = np.polyfit(frames, xs, deg)
            py = np.polyfit(frames, ys, deg)
            pred_x = float(np.polyval(px, frame_idx))
            pred_y = float(np.polyval(py, frame_idx))

            # Sanity check
            last = self.history[-1]
            max_move = 60 * self.miss_count
            if abs(pred_x - last[1]) > max_move or abs(pred_y - last[2]) > max_move:
                return None

            return {
                'pos': (pred_x, pred_y),
                'bbox': (int(pred_x - 5), int(pred_y - 5), int(pred_x + 5), int(pred_y + 5)),
                'conf': max(0.1, 0.4 - self.miss_count * 0.08),
                'method': 'interpolation',
                'ground_anchor': (pred_x, pred_y),
                'size': (10.0, 10.0),
            }
        except (np.RankWarning, np.linalg.LinAlgError):
            return None

    def reset(self):
        self.history.clear()
        self.miss_count = 0


class CascadeBallDetector:
    """
    4-stage cascade ball detection:
      1. YOLO ball model (dedicated ball detector)
      2. YOLO player model (may also detect pickleball class)
      3. Classical CV (HSV color + contour + motion)
      4. Trajectory interpolation (polynomial prediction)
    """

    def __init__(self, ball_model_path, player_model_path,
                 ball_conf=0.15, player_ball_conf=0.20,
                 imgsz=640, max_r=0.06):
        # Stage 1: Dedicated ball YOLO model
        self.ball_model = YOLO(ball_model_path)
        self.ball_conf = ball_conf

        # Stage 2: Player model (also detects Pickleball class)
        self.player_model = YOLO(player_model_path)
        self.player_ball_conf = player_ball_conf

        # Stage 3: Classical CV
        self.classical = ClassicalBallDetector()

        # Stage 4: Trajectory interpolation
        self.interpolator = TrajectoryInterpolator()

        self.imgsz = imgsz
        self.max_r = max_r
        self._dets = []
        self._last_pos = None
        self._methods = {
            'yolo_ball': 0, 'yolo_player': 0,
            'classical': 0, 'interpolation': 0, 'miss': 0,
        }

    def _filter_boxes(self, boxes, frame_shape):
        """Filter out boxes that are too large or too small."""
        fh, fw = frame_shape[:2]
        valid = []
        for i in range(len(boxes)):
            xy = boxes.xyxy[i].cpu().numpy()
            w = xy[2] - xy[0]
            h = xy[3] - xy[1]
            if w > fw * self.max_r or h > fh * self.max_r:
                continue
            if w < 2 or h < 2:
                continue
            valid.append(i)
        return valid

    def _best_det(self, boxes, valid, frame_shape):
        """Select the best detection from valid candidates."""
        if not valid:
            return None

        fh, fw = frame_shape[:2]

        # Prefer closest to last known position
        if self._last_pos and len(valid) > 1:
            dists = []
            for vi in valid:
                xy = boxes.xyxy[vi].cpu().numpy()
                cx = (xy[0] + xy[2]) / 2
                cy = (xy[1] + xy[3]) / 2
                dists.append(np.sqrt((cx - self._last_pos[0])**2 + (cy - self._last_pos[1])**2))
            best = valid[np.argmin(dists)]
        else:
            best = valid[np.argmax([boxes.conf[vi].item() for vi in valid])]

        xy = boxes.xyxy[best].cpu().numpy()
        cx, cy = float((xy[0] + xy[2]) / 2), float((xy[1] + xy[3]) / 2)

        # Jump filter
        if self._last_pos:
            dist = np.sqrt((cx - self._last_pos[0])**2 + (cy - self._last_pos[1])**2)
            if dist > max(fw, fh) * 0.25:
                return None

        return {
            'pos': (cx, cy),
            'bbox': (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])),
            'conf': boxes.conf[best].item(),
            'ground_anchor': (cx, float(xy[3])),
            'size': (float(xy[2] - xy[0]), float(xy[3] - xy[1])),
        }

    def detect(self, frame, frame_idx=0, court_mask=None):
        """
        Run 4-stage cascade ball detection.

        Returns:
            Detection dict with keys: pos, bbox, conf, method, ground_anchor, size
            or None if no ball found
        """
        det = None

        # === Stage 1: YOLO Ball Model ===
        res = self.ball_model(frame, verbose=False, imgsz=self.imgsz,
                              conf=self.ball_conf, half=True)
        if res and res[0].boxes and len(res[0].boxes) > 0:
            valid = self._filter_boxes(res[0].boxes, frame.shape)
            det = self._best_det(res[0].boxes, valid, frame.shape)
            if det:
                det['method'] = 'yolo_ball'
                self._methods['yolo_ball'] += 1

        # === Stage 2: YOLO Player Model (Pickleball class) ===
        if det is None:
            res2 = self.player_model(frame, verbose=False, imgsz=self.imgsz,
                                     conf=self.player_ball_conf, half=True)
            if res2 and res2[0].boxes and len(res2[0].boxes) > 0:
                # Filter for ball class only (not person = class 0)
                best_ball = None
                best_conf = 0
                for i in range(len(res2[0].boxes)):
                    cls_id = int(res2[0].boxes.cls[i].item())
                    if cls_id == 0:  # Skip person class
                        continue
                    xy = res2[0].boxes.xyxy[i].cpu().numpy()
                    w, h = xy[2] - xy[0], xy[3] - xy[1]
                    if w > frame.shape[1] * self.max_r or h > frame.shape[0] * self.max_r:
                        continue
                    conf = res2[0].boxes.conf[i].item()
                    cx, cy = float((xy[0] + xy[2]) / 2), float((xy[1] + xy[3]) / 2)
                    if self._last_pos:
                        dist = np.sqrt((cx - self._last_pos[0])**2 + (cy - self._last_pos[1])**2)
                        if dist > max(frame.shape[1], frame.shape[0]) * 0.25:
                            continue
                    if conf > best_conf:
                        best_conf = conf
                        best_ball = i

                if best_ball is not None:
                    xy = res2[0].boxes.xyxy[best_ball].cpu().numpy()
                    cx, cy = float((xy[0] + xy[2]) / 2), float((xy[1] + xy[3]) / 2)
                    det = {
                        'pos': (cx, cy),
                        'bbox': (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])),
                        'conf': best_conf,
                        'method': 'yolo_player',
                        'ground_anchor': (cx, float(xy[3])),
                        'size': (float(xy[2] - xy[0]), float(xy[3] - xy[1])),
                    }
                    self._methods['yolo_player'] += 1

        # === Stage 3: Classical CV ===
        if det is None:
            det = self.classical.detect(frame, court_mask)
            if det:
                det['method'] = 'classical'
                self._methods['classical'] += 1

        # === Stage 4: Trajectory Interpolation ===
        if det is None:
            det = self.interpolator.predict(frame_idx)
            if det:
                det['method'] = 'interpolation'
                self._methods['interpolation'] += 1

        # Update state
        if det is not None:
            self._last_pos = det['pos']
            self.interpolator.add(frame_idx, det['pos'])
        else:
            self._methods['miss'] += 1
            self.interpolator.miss_count += 1

        self._dets.append(det)
        return det

    def rate(self):
        """Get overall detection rate."""
        if not self._dets:
            return 0
        return sum(1 for d in self._dets if d is not None) / len(self._dets)

    def method_stats(self):
        """Print detection method breakdown."""
        total = sum(self._methods.values())
        print('  Detection methods breakdown:')
        for m, c in sorted(self._methods.items(), key=lambda x: -x[1]):
            bar = '█' * (c * 30 // max(total, 1))
            print(f'    {m:15s}: {c:5d} ({c / max(total, 1) * 100:5.1f}%) {bar}')

    def dataframe(self):
        """Export detections as a pandas DataFrame."""
        rows = []
        for i, d in enumerate(self._dets):
            rows.append({
                'frame': i,
                'x': d['pos'][0] if d else np.nan,
                'y': d['pos'][1] if d else np.nan,
                'conf': d['conf'] if d else 0,
                'det': d is not None,
                'method': d.get('method', '') if d else 'miss',
            })
        return pd.DataFrame(rows)
