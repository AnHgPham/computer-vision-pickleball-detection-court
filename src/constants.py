"""
Court geometry constants, zone definitions, and visualization colors.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

import numpy as np

# ========== COURT GEOMETRY ==========
# 12 keypoints defining the pickleball court (2D top-down coordinates)
#
#     LEFT        CENTER       RIGHT
#     ┌────────────┬────────────┐
#  0  │            │            │  2       ← Background Baseline
#     │            1            │
#     │   Service  │  Service   │
#     │    Box L   │   Box R    │
#  3  │            │            │  5       ← Background Kitchen Line
#     ├────────────4────────────┤
#     │       KITCHEN (NVZ)     │
#     ├─────────── NET ─────────┤
#     │       KITCHEN (NVZ)     │
#  6  ├────────────┼────────────┤  8       ← Foreground Kitchen Line
#     │            7            │
#     │   Service  │  Service   │
#     │    Box L   │   Box R    │
#  9  │           10            │ 11       ← Foreground Baseline
#     └────────────┴────────────┘

COURT_DST = np.array([
    [0, 0], [200, 0], [400, 0],
    [400, 300], [200, 300], [0, 300],
    [0, 580], [200, 580], [400, 580],
    [400, 880], [200, 880], [0, 880],
], dtype=np.float32)

CW, CH = 400, 880  # Court width and height in 2D space

# Court line connections (keypoint index pairs)
COURT_LINES = [
    (0, 2), (9, 11), (3, 5), (6, 8),  # horizontal lines
    (0, 11), (2, 9),                    # sidelines
    (1, 4), (7, 10),                    # center lines
]

# Six zones of the court (each defined by 4 corner keypoint indices)
ZONES = {
    'A': [0, 1, 4, 5], 'B': [1, 2, 3, 4],
    'C': [5, 4, 7, 6], 'D': [4, 3, 8, 7],
    'E': [6, 7, 10, 11], 'F': [7, 8, 9, 10],
}

# ========== VISUALIZATION COLORS (BGR) ==========
C_BALL = (0, 255, 255)       # Yellow
C_KP = (0, 0, 255)          # Red
C_W = (255, 255, 255)       # White
C_IN = (0, 255, 0)          # Green
C_OUT = (0, 0, 255)         # Red
C_TEAM_A = (255, 150, 50)   # Near team — orange
C_TEAM_B = (50, 150, 255)   # Far team — blue
