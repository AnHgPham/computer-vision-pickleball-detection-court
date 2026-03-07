"""
Visualization functions for court keypoints, ball, players, and minimap.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

import cv2
import numpy as np

from .constants import (
    COURT_DST, CW, CH, COURT_LINES,
    C_BALL, C_KP, C_W, C_IN, C_OUT, C_TEAM_A, C_TEAM_B,
)


def draw_keypoints(frame, kps, alpha=0.8):
    """Draw court keypoints on the frame with numbered labels."""
    ov = frame.copy()
    for i, kp in enumerate(kps):
        if kp[2] < 0.3 or kp[0] <= 0:
            continue
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(ov, (x, y), 8, C_KP, -1)
        cv2.circle(ov, (x, y), 9, C_W, 2)
        cv2.putText(ov, str(i), (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_W, 2, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)


def draw_ball_bbox(frame, det):
    """Draw ball bounding box with confidence label."""
    if det and det.get('method', '') != 'interpolation':
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2)
        lbl = f"Ball {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), C_BALL, -1)
        cv2.putText(frame, lbl, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_bounce_indicator(frame, is_bounce, in_out, ball_det):
    """Draw bounce indicator (IN/OUT circle) at ball position."""
    if is_bounce and ball_det:
        x, y = int(ball_det['pos'][0]), int(ball_det['pos'][1])
        color = C_IN if in_out == 'IN' else C_OUT
        cv2.circle(frame, (x, y), 25, color, 3)
        cv2.putText(frame, in_out, (x + 30, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame


def draw_player_bbox(frame, players):
    """Draw player bounding boxes with team labels and foot points."""
    for p in players:
        x1, y1, x2, y2 = p['bbox']
        team = p.get('team', 'A')
        label = p.get('team_label', '?')
        c = C_TEAM_A if team == 'A' else C_TEAM_B
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        if 'foot' in p:
            fx, fy = int(p['foot'][0]), int(p['foot'][1])
            cv2.circle(frame, (fx, fy), 4, c, -1)
            cv2.circle(frame, (fx, fy), 5, C_W, 1)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return frame


def _draw_court_base(size=(150, 330)):
    """Draw a mini court base image for the minimap."""
    sx, sy = size[0] / CW, size[1] / CH
    pad = 8
    W = size[0] + pad * 2
    H = size[1] + pad * 2

    cv = np.full((H, W, 3), (30, 30, 30), dtype=np.uint8)
    oy, ox = pad, pad

    # Court surface
    cv2.rectangle(cv, (ox, oy), (ox + size[0], oy + size[1]), (25, 80, 25), -1)
    # Kitchen zone
    cv2.rectangle(cv, (ox, oy + int(300 * sy)),
                  (ox + size[0], oy + int(580 * sy)), (25, 60, 100), -1)

    # Court lines
    for (i, j) in COURT_LINES:
        cv2.line(
            cv,
            (ox + int(COURT_DST[i][0] * sx), oy + int(COURT_DST[i][1] * sy)),
            (ox + int(COURT_DST[j][0] * sx), oy + int(COURT_DST[j][1] * sy)),
            C_W, 1, cv2.LINE_AA,
        )

    # Net line
    net_y = oy + int(440 * sy)
    cv2.line(cv, (ox, net_y), (ox + size[0], net_y), (200, 200, 200), 2)

    return cv, ox, oy, sx, sy


def create_triple_minimap(ball_2d, trail_2d, players_2d, all_bounce_pts,
                          latest_bounce, size=(150, 330)):
    """
    Create a triple-panel minimap showing:
      1. Ball trail + player positions
      2. All bounce positions (IN/OUT)
      3. Latest bounce only

    Returns:
        numpy array (BGR image) of the panel
    """
    sx, sy = size[0] / CW, size[1] / CH
    pad = 8
    mW = size[0] + pad * 2
    mH = size[1] + pad * 2
    title_h = 24
    total_H = title_h + mH * 3 + 6

    panel = np.full((total_H, mW, 3), (40, 40, 40), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (mW, title_h), (50, 50, 50), -1)
    cv2.putText(panel, 'COURT VIEW', (mW // 2 - 42, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_W, 1, cv2.LINE_AA)

    # === MAP 1: Ball Trail + Players ===
    m1, ox, oy, _, _ = _draw_court_base(size)
    cv2.putText(m1, 'TRAIL', (ox + 2, oy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    if trail_2d and len(trail_2d) > 1:
        for k in range(1, len(trail_2d)):
            al = (k + 1) / len(trail_2d)
            t = max(1, int(2 * al))
            c = (0, int(140 + 115 * al), int(200 + 55 * al))
            x1 = ox + max(0, min(int(trail_2d[k - 1][0] * sx), size[0] - 1))
            y1 = oy + max(0, min(int(trail_2d[k - 1][1] * sy), size[1] - 1))
            x2 = ox + max(0, min(int(trail_2d[k][0] * sx), size[0] - 1))
            y2 = oy + max(0, min(int(trail_2d[k][1] * sy), size[1] - 1))
            cv2.line(m1, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)

    if ball_2d is not None:
        bx = ox + max(0, min(int(ball_2d[0] * sx), size[0] - 1))
        by = oy + max(0, min(int(ball_2d[1] * sy), size[1] - 1))
        cv2.circle(m1, (bx, by), 6, (0, 100, 100), -1)
        cv2.circle(m1, (bx, by), 4, C_BALL, -1)
        cv2.circle(m1, (bx, by), 5, C_W, 1)

    for p2d in players_2d:
        px = ox + max(0, min(int(p2d['pos'][0] * sx), size[0] - 1))
        py = oy + max(0, min(int(p2d['pos'][1] * sy), size[1] - 1))
        c = C_TEAM_A if p2d['team'] == 'A' else C_TEAM_B
        cv2.circle(m1, (px, py), 7, c, -1)
        cv2.circle(m1, (px, py), 8, C_W, 1)
        cv2.putText(m1, p2d['label'], (px - 6, py + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_W, 1)

    # === MAP 2: All Bounces ===
    m2, ox2, oy2, _, _ = _draw_court_base(size)
    cv2.putText(m2, 'ALL BOUNCES', (ox2 + 2, oy2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    n_in = n_out = 0
    for bp in all_bounce_pts:
        bx = ox2 + max(0, min(int(bp['pos'][0] * sx), size[0] - 1))
        by = oy2 + max(0, min(int(bp['pos'][1] * sy), size[1] - 1))
        color = C_IN if bp['in_out'] == 'IN' else C_OUT
        sz = 5
        cv2.line(m2, (bx - sz, by - sz), (bx + sz, by + sz), color, 2)
        cv2.line(m2, (bx + sz, by - sz), (bx - sz, by + sz), color, 2)
        if bp['in_out'] == 'IN':
            n_in += 1
        else:
            n_out += 1

    cv2.putText(m2, f'IN:{n_in} OUT:{n_out}', (ox2 + 2, oy2 + size[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_W, 1)

    # === MAP 3: Latest Bounce ===
    m3, ox3, oy3, _, _ = _draw_court_base(size)
    cv2.putText(m3, 'LATEST', (ox3 + 2, oy3 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    if latest_bounce is not None:
        bx = ox3 + max(0, min(int(latest_bounce['pos'][0] * sx), size[0] - 1))
        by = oy3 + max(0, min(int(latest_bounce['pos'][1] * sy), size[1] - 1))
        color = C_IN if latest_bounce['in_out'] == 'IN' else C_OUT
        sz = 8
        cv2.line(m3, (bx - sz, by - sz), (bx + sz, by + sz), color, 3)
        cv2.line(m3, (bx + sz, by - sz), (bx - sz, by + sz), color, 3)
        cv2.circle(m3, (bx, by), 12, color, 2)
        cv2.putText(m3, latest_bounce['in_out'], (bx + 14, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    # Stack vertically
    y_off = title_h
    panel[y_off:y_off + mH, 0:mW] = m1
    y_off += mH + 3
    panel[y_off:y_off + mH, 0:mW] = m2
    y_off += mH + 3
    panel[y_off:y_off + mH, 0:mW] = m3

    return panel


def overlay_minimap(frame, minimap, margin=10):
    """Overlay the minimap on the top-right corner of the frame."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    mh, mw = minimap.shape[:2]

    x, y = fw - mw - margin, margin
    if y + mh > fh:
        mh = fh - y - 2

    cv2.rectangle(out, (x - 2, y - 2),
                  (x + mw + 2, y + min(mh, minimap.shape[0]) + 2),
                  (60, 60, 60), -1)
    out[y:y + min(mh, minimap.shape[0]), x:x + mw] = \
        minimap[:min(mh, minimap.shape[0]), :mw]
    return out
