"""
Main pipeline orchestrating all components for pickleball match analysis.
Extracted from notebook 03_pipeline_inference.ipynb (Section 3: Run Pipeline).
"""

import cv2
import numpy as np
import time
from collections import deque

from .constants import CW, CH
from .court_detector import CourtDetector, CourtPolygonFilter
from .ball_tracker import CascadeBallDetector
from .player_detector import PlayerDetector, assign_teams
from .kalman_filters import BallKalmanOF, Court2DKalman, check_in_out
from .projection import ZoneProjector, OpticalFlowEstimator, BounceDetector
from .visualization import (
    draw_keypoints, draw_ball_bbox, draw_bounce_indicator,
    draw_player_bbox, create_triple_minimap, overlay_minimap,
)


class Pipeline:
    """
    Pickleball match analysis pipeline.

    Combines court detection, ball tracking (cascade), player detection,
    Kalman filtering, bounce detection, and visualization into a single
    video processing pipeline.
    """

    def __init__(self, court_model_path, ball_model_path, player_model_path,
                 court_conf=0.5, ball_conf=0.15, player_conf=0.5,
                 court_every_n=5, player_every_n=2):
        """
        Args:
            court_model_path: Path to court keypoint model (.pt)
            ball_model_path: Path to ball detection model (.pt)
            player_model_path: Path to player detection model (.pt)
            court_conf: Court detection confidence threshold
            ball_conf: Ball detection confidence threshold
            player_conf: Player detection confidence threshold
            court_every_n: Run court detection every N frames
            player_every_n: Run player detection every N frames
        """
        self.court_det = CourtDetector(court_model_path, conf=court_conf,
                                       every_n=court_every_n)
        self.ball_det = CascadeBallDetector(ball_model_path, player_model_path,
                                            ball_conf=ball_conf, player_ball_conf=0.20)
        self.player_det = PlayerDetector(player_model_path, conf=player_conf)
        self.zone_proj = ZoneProjector()
        self.court_filter = CourtPolygonFilter(margin_ratio=0.15)
        self.ball_kf_cam = BallKalmanOF(proc_noise=5.0, meas_noise=2.0,
                                         max_miss=10, of_weight=0.3)
        self.court_kf = Court2DKalman(proc_noise=5.0, meas_noise=10.0, max_pred=15)
        self.bounce_det = BounceDetector(window=7, min_gap=8)
        self.of_est = OpticalFlowEstimator()
        self.player_every_n = player_every_n

        # State
        self.heatmap = None
        self.all_bounces = []

    def process(self, input_video, output_video, max_frames=None, show_preview=False):
        """
        Process a video through the full analysis pipeline.

        Args:
            input_video: Path to input video (.mp4)
            output_video: Path to save output video (.mp4)
            max_frames: Maximum frames to process (None = all)
            show_preview: Show preview window during processing

        Returns:
            dict with processing statistics
        """
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        to_proc = max_frames or total

        print(f'📐 {VW}x{VH} @ {fps:.0f}fps | {total} frames')

        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (VW, VH))

        self.heatmap = np.zeros((CH, CW), dtype=np.float32)
        self.all_bounces = []
        trail_2d = deque(maxlen=60)
        last_players = []
        pcnt = 0
        last_ball_pos_cam = None
        t0 = time.time()
        n = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and n >= max_frames:
                break

            # 1) Court keypoints
            kps = self.court_det.detect(frame)

            # 2) Update zone projector + court polygon filter
            self.zone_proj.update(kps)
            self.court_filter.update(kps)

            # 3) Ball detection (cascade)
            ball = self.ball_det.detect(frame, frame_idx=n)

            # 4) Players (filtered by court polygon)
            pcnt += 1
            if pcnt % self.player_every_n == 0:
                raw_players = self.player_det.detect(frame, court_filter=self.court_filter)
                raw_players = raw_players[:4]

                # Net Y estimation from kitchen keypoints
                net_y_cam = None
                if kps is not None and len(kps) >= 8:
                    ky = [kps[i][1] for i in [3, 4, 5, 6, 7, 8]
                          if kps[i][2] > 0.3 and kps[i][1] > 0]
                    if ky:
                        net_y_cam = np.mean(ky)
                last_players = assign_teams(raw_players, net_y_cam=net_y_cam)

            # 5) Optical Flow
            of_vel = self.of_est.estimate_velocity(frame, last_ball_pos_cam)
            if ball is not None:
                self.of_est.update_frame(frame)

            # 6) Camera Kalman + OF
            cam_pos = self.ball_kf_cam.update(ball, of_vel)
            if cam_pos:
                last_ball_pos_cam = cam_pos

            # 7) Bounce detection
            is_bounce = self.bounce_det.update(ball, n, players=last_players)

            # 8) Ball → 2D court projection
            raw_2d = None
            if ball and self.zone_proj.is_reliable(kps):
                raw_2d = self.zone_proj.project(np.array(ball['ground_anchor']))

            trust_2d = False
            if ball is not None and raw_2d is not None:
                bw, bh = ball.get('size', (0.0, 0.0))
                trust_2d = is_bounce or (bh <= max(10.0, VH * 0.03))

            smooth_2d = self.court_kf.predict_or_update(
                raw_2d, is_bounce=is_bounce, trust_measurement=trust_2d,
            )
            if smooth_2d is not None:
                trail_2d.append(smooth_2d)

            # 9) In/Out at bounce
            in_out = None
            if is_bounce and smooth_2d is not None:
                in_out = check_in_out(smooth_2d)
                bp = {'frame': n, 'pos': smooth_2d, 'in_out': in_out}
                self.all_bounces.append(bp)
                bx, by = int(smooth_2d[0]), int(smooth_2d[1])
                if 0 <= bx < CW and 0 <= by < CH:
                    self.heatmap[by, bx] += 1

            # 10) Draw annotations
            out = frame.copy()
            if kps is not None:
                out = draw_keypoints(out, kps)
            out = draw_ball_bbox(out, ball)
            out = draw_bounce_indicator(out, is_bounce, in_out, ball)
            out = draw_player_bbox(out, last_players)

            # 11) Players → 2D with team info
            players_2d = []
            for p in last_players:
                if self.zone_proj.is_reliable(kps):
                    p2d = self.zone_proj.project(np.array(p['foot']))
                else:
                    p2d = None
                if p2d is not None:
                    players_2d.append({
                        'pos': p2d,
                        'team': p.get('team', 'A'),
                        'label': p.get('team_label', '?'),
                    })

            # Re-label players by position on minimap
            near_2d = sorted([p for p in players_2d if p['team'] == 'A'],
                             key=lambda p: p['pos'][0])
            far_2d = sorted([p for p in players_2d if p['team'] == 'B'],
                            key=lambda p: p['pos'][0])
            for idx, p in enumerate(near_2d, start=1):
                p['label'] = f'A{idx}'
            for idx, p in enumerate(far_2d):
                p['label'] = f'B{len(far_2d) - idx}'

            # 12) Court View minimap
            latest_b = self.all_bounces[-1] if self.all_bounces else None
            cv_map = create_triple_minimap(
                smooth_2d, list(trail_2d), players_2d,
                self.all_bounces, latest_b,
            )
            out = overlay_minimap(out, cv_map)

            writer.write(out)
            n += 1

            if n % 200 == 0:
                el = time.time() - t0
                spd = n / el
                eta = (to_proc - n) / spd if spd > 0 else 0
                print(f'  [{n / to_proc * 100:5.1f}%] {n}/{to_proc} | '
                      f'{spd:.1f}fps | ETA:{int(eta // 60)}m{int(eta % 60)}s')

            if show_preview:
                preview = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2))
                cv2.imshow('Pickleball Analysis', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        el = time.time() - t0
        n_in = sum(1 for b in self.all_bounces if b['in_out'] == 'IN')
        n_out = sum(1 for b in self.all_bounces if b['in_out'] == 'OUT')

        print(f'\n{"=" * 55}')
        print(f'  ✅ DONE! {n} frames / {el:.1f}s ({n / el:.1f} fps)')
        print(f'  🎯 Ball detection: {self.ball_det.rate():.1%}')
        print(f'  🏓 Bounces: {len(self.all_bounces)} (IN:{n_in} OUT:{n_out})')
        print(f'  📁 {output_video}')
        print(f'{"=" * 55}')

        self.ball_det.method_stats()

        return {
            'frames_processed': n,
            'time_seconds': el,
            'fps': n / el if el > 0 else 0,
            'detection_rate': self.ball_det.rate(),
            'bounces_total': len(self.all_bounces),
            'bounces_in': n_in,
            'bounces_out': n_out,
        }

    def generate_heatmap_image(self, output_path):
        """Save bounce heatmap as an image."""
        if self.heatmap is None:
            return
        hm = cv2.GaussianBlur(self.heatmap, (31, 31), 0)
        if hm.max() > 0:
            hm_c = cv2.applyColorMap(
                (hm / hm.max() * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            cv2.imwrite(output_path, hm_c)
            print(f'🔥 Heatmap saved: {output_path}')

    def export_ball_data(self, output_path):
        """Export ball detection data to CSV."""
        df = self.ball_det.dataframe()
        df.to_csv(output_path, index=False)
        print(f'📊 Ball data exported: {output_path}')
