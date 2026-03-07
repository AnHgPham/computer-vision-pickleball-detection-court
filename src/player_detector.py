"""
Player detection and team assignment.
Extracted from notebook 03_pipeline_inference.ipynb.
"""

from ultralytics import YOLO


class PlayerDetector:
    """
    Detects players using a YOLO object detection model.
    Filters detections using court polygon to exclude spectators/referees.
    """

    def __init__(self, model_path, conf=0.5, max_players=6):
        """
        Args:
            model_path: Path to player detection model (.pt)
            conf: Confidence threshold
            max_players: Maximum number of players to return
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.max_p = max_players

    def detect(self, frame, court_filter=None):
        """
        Detect players in a frame.

        Args:
            frame: BGR image
            court_filter: Optional CourtPolygonFilter to exclude off-court detections

        Returns:
            List of player dicts with keys: bbox, conf, foot
        """
        res = self.model(frame, verbose=False, conf=self.conf, half=True)

        if not res or not res[0].boxes or len(res[0].boxes) == 0:
            return []

        players = []
        for i in range(min(len(res[0].boxes), 15)):
            xy = res[0].boxes.xyxy[i].cpu().numpy()
            c = res[0].boxes.conf[i].item()
            cls_id = int(res[0].boxes.cls[i].item()) if res[0].boxes.cls is not None else 0

            # Only keep Person class (class 0)
            if cls_id != 0:
                continue

            foot = (float((xy[0] + xy[2]) / 2), float(xy[3]))

            # Court polygon filter
            if court_filter and not court_filter.is_on_court(foot):
                continue

            players.append({
                'bbox': (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])),
                'conf': c,
                'foot': foot,
            })

        players.sort(key=lambda p: p['conf'], reverse=True)
        return players[:self.max_p]


def assign_teams(players, net_y_cam=None):
    """
    Assign players to teams based on their Y position (camera view).

    - Near team (closer to camera, larger Y): Team A (orange)
    - Far team (farther from camera, smaller Y): Team B (blue)

    Args:
        players: List of player dicts (must have 'foot' key)
        net_y_cam: Optional Y coordinate of the net in camera space

    Returns:
        Updated players list with 'team' and 'team_label' keys
    """
    if not players:
        return []

    # Sort by foot Y (ascending = far first)
    sorted_p = sorted(enumerate(players), key=lambda x: x[1]['foot'][1])

    if len(sorted_p) >= 4:
        for rank, (orig_idx, _) in enumerate(sorted_p):
            if rank < 2:
                players[orig_idx]['team'] = 'B'
                players[orig_idx]['team_label'] = f'B{rank + 1}'
            else:
                players[orig_idx]['team'] = 'A'
                players[orig_idx]['team_label'] = f'A{rank - 1}'
    elif len(sorted_p) >= 2:
        mid = len(sorted_p) // 2
        for rank, (orig_idx, _) in enumerate(sorted_p):
            if rank < mid:
                players[orig_idx]['team'] = 'B'
                players[orig_idx]['team_label'] = f'B{rank + 1}'
            else:
                players[orig_idx]['team'] = 'A'
                players[orig_idx]['team_label'] = f'A{rank - mid + 1}'
    else:
        players[0]['team'] = 'A'
        players[0]['team_label'] = 'A1'

    return players
