"""
Pickleball Match Analysis - Main Entry Point
=============================================
Analyzes pickleball match videos using computer vision:
- Court detection (12 keypoints via YOLOv8-Pose)
- Ball tracking (4-stage cascade: YOLO + Classical CV + Interpolation)
- Player detection and team assignment
- Homography projection to bird's-eye court view
- Bounce detection with IN/OUT classification
- Analytics & visualization (minimap, heatmap, etc.)

Usage:
    python main.py --input video.mp4 \
                   --court-model models/court_keypoint_best.pt \
                   --ball-model models/ball_tracker_best.pt \
                   --player-model models/player_detection_best.pt
"""

import argparse
import os
import sys

from src.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pickleball Match Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input video file (.mp4)')
    parser.add_argument('--court-model', required=True,
                        help='Path to court keypoint detection model (.pt)')
    parser.add_argument('--ball-model', required=True,
                        help='Path to ball detection model (.pt)')
    parser.add_argument('--player-model', required=True,
                        help='Path to player detection model (.pt)')

    # Optional arguments
    parser.add_argument('--output', '-o', default=None,
                        help='Output video path (default: outputs/<input>_analyzed.mp4)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (default: all)')
    parser.add_argument('--court-conf', type=float, default=0.5,
                        help='Court detection confidence (default: 0.5)')
    parser.add_argument('--ball-conf', type=float, default=0.15,
                        help='Ball detection confidence (default: 0.15)')
    parser.add_argument('--player-conf', type=float, default=0.5,
                        help='Player detection confidence (default: 0.5)')
    parser.add_argument('--court-every-n', type=int, default=5,
                        help='Court detection interval in frames (default: 5)')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview window during processing')
    parser.add_argument('--heatmap', default=None,
                        help='Path to save bounce heatmap image')
    parser.add_argument('--export-csv', default=None,
                        help='Path to export ball positions CSV')

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    for name, path in [('Input video', args.input),
                       ('Court model', args.court_model),
                       ('Ball model', args.ball_model),
                       ('Player model', args.player_model)]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Default output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join('outputs', f'{base}_analyzed.mp4')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("=" * 60)
    print("  🏓 Pickleball Match Analysis Pipeline")
    print("=" * 60)

    # Create and run pipeline
    pipeline = Pipeline(
        court_model_path=args.court_model,
        ball_model_path=args.ball_model,
        player_model_path=args.player_model,
        court_conf=args.court_conf,
        ball_conf=args.ball_conf,
        player_conf=args.player_conf,
        court_every_n=args.court_every_n,
    )

    stats = pipeline.process(
        input_video=args.input,
        output_video=args.output,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )

    # Optional exports
    if args.heatmap:
        pipeline.generate_heatmap_image(args.heatmap)
    if args.export_csv:
        pipeline.export_ball_data(args.export_csv)

    print("\n" + "=" * 60)
    print("  ✅ Processing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
