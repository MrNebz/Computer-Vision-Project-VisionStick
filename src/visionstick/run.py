"""
VisionStick V2 — CLI entry point.

Usage examples:
    # Webcam (default)
    python -m visionstick.run

    # Specific webcam index
    python -m visionstick.run --source 1

    # Video file
    python -m visionstick.run --source data/videos/test.mp4

    # Custom YOLO weights
    python -m visionstick.run --yolo yolov8m.pt

    # Override confidence threshold
    python -m visionstick.run --conf 0.50
"""

import argparse
import sys
import os

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath("Ultralytics"))

# Allow running as `python run.py` from within src/visionstick/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visionstick.pipeline import pipeline_create, pipeline_run
import visionstick.config as C


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="visionstick",
        description="VisionStick V2 — real-time obstacle detection for the visually impaired.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", default=0,
        help="Video source: 0/1/2 for webcam index, or a path to a video file. (default: 0)"
    )
    p.add_argument(
        "--yolo", default=C.YOLO_MODEL,
        help=f"Path to COCO YOLO weights file. (default: {C.YOLO_MODEL})"
    )
    p.add_argument(
        "--custom-yolo", default=C.CUSTOM_YOLO_MODEL,
        help=f"Path to custom Door/Tree/Stairs YOLO weights file. (default: {C.CUSTOM_YOLO_MODEL})"
    )
    p.add_argument(
        "--no-custom-yolo", action="store_true",
        help="Disable the custom Door/Tree/Stairs YOLO model."
    )
    p.add_argument(
        "--conf", type=float, default=C.YOLO_CONF,
        help=f"YOLO detection confidence threshold. (default: {C.YOLO_CONF})"
    )
    p.add_argument(
        "--depth-skip", type=int, default=C.DEPTH_SKIP,
        help=f"Run depth estimation every N frames. Higher = faster but less frequent depth updates. (default: {C.DEPTH_SKIP})"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Apply runtime overrides to config module
    C.YOLO_CONF   = args.conf
    C.DEPTH_SKIP  = args.depth_skip
    custom_yolo = None if args.no_custom_yolo else args.custom_yolo

    # Resolve source: try to cast to int (webcam index) first
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass   # keep as string path

    print("=" * 60)
    print("VisionStick V2 — Active Configuration")
    print("=" * 60)
    print(f"  Source       : {source} ({'webcam' if isinstance(source, int) else 'video file'})")
    print(f"  YOLO model   : {args.yolo}")
    print(f"  Custom YOLO  : {custom_yolo or 'disabled'}")
    print(f"  Confidence   : {args.conf}")
    print(f"  Depth skip   : {args.depth_skip}")
    print("=" * 60)

    p = pipeline_create(
        source=source,
        yolo_model=args.yolo,
        custom_yolo_model=custom_yolo,
        depth_skip=args.depth_skip,
    )
    pipeline_run(p)


if __name__ == "__main__":
    main()
