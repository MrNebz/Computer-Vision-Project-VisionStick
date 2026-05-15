"""
VisionStick V2 — configuration constants.

All tunable knobs live here. Import with:
    from visionstick.config import *
or
    from visionstick import config
    config.YOLO_CONF = 0.50   # override at runtime
"""

import torch

# ---------------------------------------------------------------------------
# Hardware / device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Depth model device.
# RTX 4060 has 8 GB VRAM — enough for both YOLO + Depth Anything V2 Small simultaneously.
# Falls back to CPU on machines without CUDA.
DEPTH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8s.pt"                                      # COCO-pretrained small model
CUSTOM_YOLO_MODEL = "models/best.pt"                           # Door/Tree/Stairs custom model
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"      # HuggingFace hub id

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Confidence threshold — raised to 0.45 to suppress hallucinations in noisy scenes
YOLO_CONF = 0.45

# IOU threshold for NMS
YOLO_IOU = 0.45

# YOLO inference image size (long edge). Smaller = faster inference.
# 480 gives ~35% speedup over 640 with minimal accuracy drop for large-obstacle detection.
YOLO_IMGSZ = 480

# Minimum bounding-box fill (bbox_area / frame_area) to even consider a detection.
# Filters out distant noise / tiny false positives that are not a real hazard.
MIN_BBOX_FILL = 0.003   # ~0.3% of frame ≈ roughly 10×10 px on 640×480

# COCO classes that are navigation-relevant for a blind person
OBSTACLE_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
}

# Classes learned by the custom navigation model.
CUSTOM_OBSTACLE_CLASSES = {
    0: "door",
    1: "tree",
    2: "stairs",
}

CUSTOM_TRACK_ID_OFFSET = 100000

# Per-class hazard weight (higher = more dangerous even at same distance)
CLASS_HAZARD = {
    "person":       1.00,
    "bicycle":      0.90,
    "motorcycle":   0.90,
    "car":          0.95,
    "bus":          0.95,
    "truck":        0.95,
    "traffic light":0.50,
    "stop sign":    0.50,
    "chair":        0.70,
    "couch":        0.65,
    "potted plant": 0.60,
    "bed":          0.55,
    "dining table": 0.70,
    "tv":           0.55,
    "door":         0.65,
    "tree":         0.85,
    "stairs":       0.95,
}
DEFAULT_HAZARD = 0.60

# ---------------------------------------------------------------------------
# Zone thresholds — based on BBOX FILL (physics-based, always correct)
#
#   bbox_fill = (bbox_width * bbox_height) / (frame_width * frame_height)
#
#   Rationale:
#     A person at arm's length fills ~20-30% of a typical frame.
#     A person 2m away fills ~6-10%.
#     A person 4-5m away fills ~1.5-3%.
#     Smaller than 0.3% → too far to be an immediate hazard.
# ---------------------------------------------------------------------------
ZONE_DANGER_SCORE = 0.76
ZONE_CLOSE_SCORE  = 0.56
ZONE_MEDIUM_SCORE = 0.36

# Bbox fill is kept only as a weak fallback / supporting closeness cue.
ZONE_DANGER_FILL = 0.18

ZONE_NAMES = {
    "danger": "DANGER",
    "close":  "CLOSE",
    "medium": "MEDIUM",
    "safe":   "SAFE",
}

# Minimum risk score for primary obstacle to be selected
MIN_RISK_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Risk score weights
# Must sum to 1.0
# ---------------------------------------------------------------------------
W_CLOSENESS  = 0.45   # depth-heavy object proximity
W_PATH       = 0.30   # overlap with the likely walking corridor
W_HAZARD     = 0.15   # class hazard weight
W_CONFIDENCE = 0.10   # YOLO detection confidence

# ---------------------------------------------------------------------------
# Closeness computation — how to blend bbox fill with depth
# ---------------------------------------------------------------------------
BBOX_WEIGHT  = 0.25   # bbox-fill component; deliberately weak to reduce size bias
DEPTH_WEIGHT = 0.75   # depth component; primary closeness cue

# Fallback blend used before the async depth worker has produced a map.
NO_DEPTH_BBOX_WEIGHT = 0.30
NO_DEPTH_NEUTRAL_WEIGHT = 0.70

# Walking corridor in normalised frame coordinates.
# Objects centered in this lower-middle region are treated as more relevant.
PATH_X_MIN = 0.30
PATH_X_MAX = 0.70
PATH_Y_MIN = 0.35
PATH_CENTER_WEIGHT = 0.55
PATH_OVERLAP_WEIGHT = 0.30
PATH_BOTTOM_WEIGHT = 0.15

# Zone signal blend. Closeness dominates, but being in the walking path can
# escalate the warning.
ZONE_CLOSENESS_WEIGHT = 0.70
ZONE_PATH_WEIGHT = 0.30

# Depth input resolution — cap long edge to this to keep inference fast
DEPTH_MAX_SIDE = 320   # pixels; good quality/speed balance on RTX 4060

# Run depth estimation every N frames (skip frames for speed)
# At 30+ FPS, every 5 frames = ~6 depth updates/sec — enough for walking pace
DEPTH_SKIP = 5

# Percentile bounds for robust depth normalization (avoids outlier pixels)
DEPTH_P_NEAR = 10    # percentile representing "close" depth reference
DEPTH_P_FAR  = 90    # percentile representing "far" depth reference

# Inner-crop fraction for extracting depth under a bounding box
# Using the central 60% of the box avoids background bleed at edges
DEPTH_CROP_FRAC = 0.60

# EMA alpha for per-track depth smoothing (higher = faster response)
EMA_ALPHA = 0.45

# ---------------------------------------------------------------------------
# Temporal tracking
# ---------------------------------------------------------------------------
TREND_WINDOW = 6    # rolling window length (frames) for approach/recession detection
TREND_THRESH = 0.08 # normalised fill-change threshold to call "approaching"

# ---------------------------------------------------------------------------
# Audio / TTS
# ---------------------------------------------------------------------------
# Zone-specific cooldowns (seconds) — danger fires more often
COOLDOWN_DANGER = 1.5
COOLDOWN_CLOSE  = 3.0
COOLDOWN_MEDIUM = 5.0
COOLDOWN_SAFE   = 12.0

# Track stability: how many consecutive frames a track must exist
# before its audio is trusted. Suppresses ghost detections.
MIN_TRACK_AGE = 3

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
BYTETRACK_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Display colours (BGR)
# ---------------------------------------------------------------------------
COLOR_DANGER = (0,   0,   220)   # red
COLOR_CLOSE  = (0,  140,  255)   # orange
COLOR_MEDIUM = (0,  220,  220)   # yellow
COLOR_SAFE   = (0,  200,   80)   # green
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,     0)
COLOR_GRAY   = (120, 120,  120)

ZONE_COLORS = {
    "danger": COLOR_DANGER,
    "close":  COLOR_CLOSE,
    "medium": COLOR_MEDIUM,
    "safe":   COLOR_SAFE,
}
