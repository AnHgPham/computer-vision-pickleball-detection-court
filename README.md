# 🏓 Pickleball Match Analysis

A computer vision pipeline for analyzing pickleball match videos. The system detects court keypoints, tracks the ball, identifies players and their teams, and produces an annotated video with a 2D court minimap.

## ✨ Features

- **Court Detection** — 12 keypoint detection using YOLOv8-Pose
- **Ball Tracking** — 4-stage cascade: YOLO ball → YOLO player → Classical CV → Trajectory interpolation
- **Player Detection** — YOLO-based player detection with court polygon filtering
- **Team Assignment** — Automatic near/far team classification
- **Homography Projection** — Zone-based perspective transform to 2D court view
- **Bounce Detection** — Y-trajectory analysis with hit/bounce classification
- **IN/OUT Judgment** — Bounce position vs court boundary check
- **Kalman Filtering** — Temporal smoothing with optical flow integration
- **Triple Minimap** — Ball trail, all bounces, and latest bounce visualization
- **Heatmap** — Shot placement heatmap on 2D court

## 📁 Project Structure

```
pickleball-analysis/
├── notebooks/                          # Google Colab training & inference
│   ├── 01_court_keypoint_training.ipynb   # Train court keypoint model
│   ├── 02_ball_detection_training.ipynb   # Train ball detection model
│   ├── 03_pipeline_inference.ipynb        # Full pipeline (Colab demo)
│   └── 04_player_detection_training.ipynb # Train player detection model
├── src/                                # Clean Python source code
│   ├── __init__.py
│   ├── constants.py                    # Court geometry & color constants
│   ├── court_detector.py              # Court keypoint detection
│   ├── ball_tracker.py                # 4-stage cascade ball detection
│   ├── player_detector.py            # Player detection & team assignment
│   ├── kalman_filters.py             # Kalman filters (camera & court space)
│   ├── projection.py                 # Zone projection, optical flow, bounce
│   ├── visualization.py             # Drawing functions & minimap
│   └── pipeline.py                  # Main pipeline orchestration
├── models/                           # Model weights (not in Git)
│   ├── court_keypoint_best.pt
│   ├── ball_tracker_best.pt
│   └── player_detection_best.pt
├── configs/                          # Training data configs
├── report/                           # LaTeX graduation report
├── main.py                          # CLI entry point
├── requirements.txt
└── .gitignore
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (Google Colab)

Use the notebooks in `notebooks/` to train the three models on Google Colab:

| Notebook | Model | Dataset |
|----------|-------|---------|
| `01_court_keypoint_training.ipynb` | YOLOv8l-Pose | pb-9bsin (2,064 images, 12 keypoints) |
| `02_ball_detection_training.ipynb` | YOLOv8m | annotations_pickleball (18,408 images) |
| `04_player_detection_training.ipynb` | YOLO11m | Pickleball with Players (4,119 images) |

After training, download `*.pt` files to the `models/` directory.

### 3. Run Pipeline

```bash
python main.py \
    --input video.mp4 \
    --court-model models/court_keypoint_best.pt \
    --ball-model models/ball_tracker_best.pt \
    --player-model models/player_detection_best.pt \
    --heatmap outputs/heatmap.png \
    --export-csv outputs/ball_data.csv
```

Or use `notebooks/03_pipeline_inference.ipynb` directly on Google Colab.

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Input video path |
| `--court-model` | (required) | Court keypoint model |
| `--ball-model` | (required) | Ball detection model |
| `--player-model` | (required) | Player detection model |
| `--output` | `outputs/<name>_analyzed.mp4` | Output video path |
| `--max-frames` | all | Limit frames to process |
| `--court-conf` | 0.5 | Court detection confidence |
| `--ball-conf` | 0.15 | Ball detection confidence |
| `--player-conf` | 0.5 | Player detection confidence |
| `--preview` | off | Show live preview window |
| `--heatmap` | none | Save bounce heatmap |
| `--export-csv` | none | Export ball data CSV |

## 🏗️ Architecture

```
Video Frame
    │
    ├── Court Detector (YOLOv8-Pose) ──→ 12 Keypoints
    │       │
    │       ├── Court Polygon Filter ──→ On-court mask
    │       └── Zone Projector ──→ 6 zone homographies
    │
    ├── Cascade Ball Detector ──→ Ball position
    │       │
    │       ├── Kalman Filter (camera) + Optical Flow
    │       └── Kalman Filter (court 2D)
    │
    ├── Player Detector ──→ Team Assignment (A/B)
    │
    ├── Bounce Detector ──→ IN/OUT classification
    │
    └── Visualization ──→ Annotated Frame + Triple Minimap
```

## 📝 License

This project is for educational purposes (graduation project).
