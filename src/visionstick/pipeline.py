"""
VisionStick V2 — pipeline orchestration (procedural, no classes).

    pipeline_create()       — allocate pipeline state dict
    pipeline_open(p)        — load models, open source
    pipeline_close(p)       — release all resources
    pipeline_process_frame(p, frame)  — run one frame through all stages
    pipeline_run(p)         — full loop (open → loop → close)

    render_frame(...)       — draw detections + HUD onto a frame copy
"""

from __future__ import annotations

import time
import threading
import os
from typing import Optional

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath("Ultralytics"))

import cv2
import numpy as np
from ultralytics import YOLO

from . import config as C
from .core import (
    video_open, video_read, video_release, video_width, video_height, video_fps,
    preprocess_frame,
    track_custom_obstacles, track_obstacles,
    depth_estimator_create, depth_estimate,
    depth_smoother_create, depth_smoother_prune,
    compute_risk_scores,
    select_primary_obstacle,
    temporal_tracker_create, temporal_update, temporal_get_age, temporal_prune,
    alert_decider_create, alert_decide, alert_prune,
    tts_create, tts_speak, tts_stop,
)


# =============================================================================
# Async depth worker
#
#   dw = depth_worker_create(estimator_dict)
#   depth_worker_start(dw)
#   depth_worker_submit(dw, frame)     — non-blocking, drops if busy
#   map = depth_worker_latest(dw)      — most recent result (or None)
#   depth_worker_stop(dw)
# =============================================================================

def depth_worker_create(estimator: dict) -> dict:
    """Allocate async depth worker state. Call depth_worker_start() after."""
    dw = {
        "estimator":     estimator,
        "input_lock":    threading.Lock(),
        "pending_frame": None,
        "latest_map":    None,
        "result_lock":   threading.Lock(),
        "has_work":      threading.Event(),
        "stop_flag":     threading.Event(),
        "thread":        None,
    }
    dw["thread"] = threading.Thread(target=_depth_worker_run, args=(dw,), daemon=True)
    return dw


def depth_worker_start(dw: dict):
    dw["thread"].start()


def depth_worker_stop(dw: dict):
    dw["stop_flag"].set()
    dw["has_work"].set()   # unblock thread if it is waiting
    dw["thread"].join(timeout=5.0)


def depth_worker_submit(dw: dict, frame: np.ndarray):
    """Queue a new frame for depth inference (drops silently if already busy)."""
    with dw["input_lock"]:
        dw["pending_frame"] = frame.copy()
    dw["has_work"].set()


def depth_worker_latest(dw: dict) -> Optional[np.ndarray]:
    with dw["result_lock"]:
        return dw["latest_map"]


def _depth_worker_run(dw: dict):
    """Background thread: runs depth inference whenever a frame is submitted."""
    while not dw["stop_flag"].is_set():
        dw["has_work"].wait()
        dw["has_work"].clear()
        if dw["stop_flag"].is_set():
            break
        with dw["input_lock"]:
            frame = dw["pending_frame"]
            dw["pending_frame"] = None
        if frame is None:
            continue
        try:
            result = depth_estimate(dw["estimator"], frame)
            with dw["result_lock"]:
                dw["latest_map"] = result
        except Exception:
            pass   # never crash the background thread


# =============================================================================
# Frame renderer
# =============================================================================

def render_frame(
    frame: np.ndarray,
    ranked: list[dict],
    primary: Optional[dict],
    fps: float,
    frame_idx: int,
    depth_skip: int = C.DEPTH_SKIP,
) -> np.ndarray:
    """
    Draw bounding boxes, zone labels, risk scores and a HUD.

    Primary obstacle: thick border + filled label bar.
    Secondary obstacles: thin border + short label.
    """
    out = frame.copy()

    # Secondary obstacles (drawn first so primary renders on top)
    for r in ranked:
        if primary and r["detection"]["track_id"] == primary["detection"]["track_id"]:
            continue
        color = C.ZONE_COLORS.get(r["zone"], C.COLOR_GRAY)
        x1, y1, x2, y2 = r["detection"]["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = (
            f"{r['detection']['class_name'][:6]} "
            f"{r['zone'][0].upper()} "
            f"{r['risk_score']:.2f}"
        )
        cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Primary obstacle
    if primary:
        color = C.ZONE_COLORS.get(primary["zone"], C.COLOR_GRAY)
        x1, y1, x2, y2 = primary["detection"]["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label = (
            f"{primary['detection']['class_name']}  "
            f"{primary['zone'].upper()}  "
            f"{primary['direction']}  "
            f"risk={primary['risk_score']:.2f}"
        )
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bar_y1 = max(y1 - lh - baseline - 6, 0)
        bar_y2 = max(y1, lh + baseline + 6)
        cv2.rectangle(out, (x1, bar_y1), (x1 + lw + 8, bar_y2), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 4, bar_y2 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C.COLOR_BLACK, 2, cv2.LINE_AA)

    # HUD — top-left info bar
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_idx}",
        f"Objects: {len(ranked)}",
        f"Depth skip: {depth_skip}",
    ]
    if primary:
        hud_lines.append(
            f"PRIMARY: {primary['detection']['class_name']} "
            f"({primary['zone']}, {primary['direction']}, "
            f"close={primary['closeness']:.2f}, path={primary['path_score']:.2f})"
        )
    for i, line in enumerate(hud_lines):
        y = 20 + i * 20
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C.COLOR_BLACK, 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C.COLOR_WHITE, 1, cv2.LINE_AA)

    return out


# =============================================================================
# Pipeline — state dict + lifecycle functions
#
#   p = pipeline_create(source=0, yolo_model="yolov8s.pt")
#   pipeline_run(p)          # opens, loops, closes automatically
#
#   Or frame-by-frame:
#   pipeline_open(p)
#   while True:
#       frame = video_read(p["source"])
#       if frame is None: break
#       result = pipeline_process_frame(p, frame)
#   pipeline_close(p)
# =============================================================================

def pipeline_create(
    source: int | str = 0,
    yolo_model: str = C.YOLO_MODEL,
    custom_yolo_model: Optional[str] = C.CUSTOM_YOLO_MODEL,
    preprocess_kwargs: Optional[dict] = None,
    depth_skip: int = C.DEPTH_SKIP,
) -> dict:
    """
    Allocate a pipeline state dict. Does NOT load models yet.
    Call pipeline_open(p) or pipeline_run(p) to initialise.
    """
    return {
        # Configuration
        "source_path":       source,
        "yolo_model_path":   yolo_model,
        "custom_yolo_model_path": custom_yolo_model,
        "preprocess_kwargs": preprocess_kwargs or {},
        "depth_skip":        depth_skip,
        # Runtime state (filled by pipeline_open)
        "source":            None,
        "yolo":              None,
        "custom_yolo":       None,
        "depth_worker":      None,
        "depth_counter":     0,
        "frame_idx":         0,
        "smoother":          None,
        "temporal":          None,
        "alert_decider":     None,
        "tts":               None,
        # FPS tracking
        "fps_times":         [],
        "fps":               0.0,
    }


def pipeline_open(p: dict):
    """Load all models and initialise every stage."""
    print("[VisionStick] Loading YOLO model...")
    p["yolo"] = YOLO(p["yolo_model_path"])
    if C.DEVICE == "cuda":
        p["yolo"].fuse()
        p["yolo"].model.half()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(2):
        p["yolo"].predict(dummy, imgsz=C.YOLO_IMGSZ, verbose=False, device=C.DEVICE)

    if p["custom_yolo_model_path"]:
        print("[VisionStick] Loading custom Door/Tree/Stairs YOLO model...")
        p["custom_yolo"] = YOLO(p["custom_yolo_model_path"])
        if C.DEVICE == "cuda":
            p["custom_yolo"].fuse()
            p["custom_yolo"].model.half()
        for _ in range(2):
            p["custom_yolo"].predict(dummy, imgsz=C.YOLO_IMGSZ, verbose=False, device=C.DEVICE)

    print("[VisionStick] Loading depth model...")
    de = depth_estimator_create(model_id=C.DEPTH_MODEL, device=C.DEPTH_DEVICE)
    p["depth_worker"] = depth_worker_create(de)
    depth_worker_start(p["depth_worker"])

    print("[VisionStick] Opening video source...")
    p["source"] = video_open(p["source_path"])

    p["smoother"]      = depth_smoother_create()
    p["temporal"]      = temporal_tracker_create()
    p["alert_decider"] = alert_decider_create()
    p["tts"]           = tts_create()

    w   = video_width(p["source"])
    h   = video_height(p["source"])
    fps = video_fps(p["source"])
    print(f"[VisionStick] Ready. Source: {w}×{h} @ {fps:.1f} fps. Device: {C.DEVICE}")
    print(f"[VisionStick] Custom model: {p['custom_yolo_model_path'] or 'disabled'}")


def pipeline_close(p: dict):
    """Release all resources."""
    if p["depth_worker"]:
        depth_worker_stop(p["depth_worker"])
    if p["source"]:
        video_release(p["source"])
    tts_stop(p["tts"])


def pipeline_process_frame(p: dict, frame: np.ndarray) -> dict:
    """
    Run the full pipeline on one frame.

    Returns a dict with: ranked, primary, state, fps, frame_idx.
    """
    t0 = time.perf_counter()
    p["frame_idx"]     += 1
    p["depth_counter"] += 1

    # Stage 2 — preprocessing
    proc = (
        preprocess_frame(frame, **p["preprocess_kwargs"])
        if p["preprocess_kwargs"] else frame
    )

    # Stage 4 — detect + track
    detections = track_obstacles(proc, p["yolo"])
    if p["custom_yolo"] is not None:
        detections += track_custom_obstacles(proc, p["custom_yolo"])

    # Stage 5 — depth (every depth_skip frames)
    if p["depth_counter"] % p["depth_skip"] == 0:
        depth_worker_submit(p["depth_worker"], proc)
    depth_map = depth_worker_latest(p["depth_worker"])

    # Stage 6-8 — closeness + smooth + risk
    ranked = compute_risk_scores(detections, depth_map, frame.shape, p["smoother"])

    # Stage 9 — primary
    primary = select_primary_obstacle(ranked)

    # Stage 10 — temporal
    state = None
    if primary:
        state = temporal_update(
            p["temporal"],
            primary["detection"]["track_id"],
            primary["closeness"],
            primary["zone"],
            primary["direction"],
        )

    # Stage 11-12 — alert + TTS
    if primary and state:
        age = temporal_get_age(p["temporal"], primary["detection"]["track_id"])
        msg = alert_decide(p["alert_decider"], primary, state, age)
        if msg:
            tts_speak(p["tts"], msg, priority=(state["zone"] == "danger"))

    # Prune stale tracks
    active = {d["track_id"] for d in detections}
    depth_smoother_prune(p["smoother"], active)
    temporal_prune(p["temporal"], active)
    alert_prune(p["alert_decider"], active)

    # FPS
    elapsed = time.perf_counter() - t0
    p["fps_times"].append(elapsed)
    if len(p["fps_times"]) > 30:
        p["fps_times"].pop(0)
    p["fps"] = 1.0 / (sum(p["fps_times"]) / len(p["fps_times"]))

    return {
        "ranked":    ranked,
        "primary":   primary,
        "state":     state,
        "fps":       p["fps"],
        "frame_idx": p["frame_idx"],
    }


def pipeline_run(p: dict):
    """
    Open the source, run the main loop, then close everything.

    Keyboard shortcuts:
        Q / ESC — quit
        P       — pause / resume
    """
    pipeline_open(p)
    try:
        paused = False

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print("[VisionStick]", "Paused" if paused else "Resumed")

            if paused:
                continue

            frame = video_read(p["source"])
            if frame is None:
                print("[VisionStick] End of source.")
                break

            result = pipeline_process_frame(p, frame)

            rendered = render_frame(
                frame,
                result["ranked"],
                result["primary"],
                result["fps"],
                result["frame_idx"],
                depth_skip=p["depth_skip"],
            )
            cv2.imshow("VisionStick V2", rendered)

    finally:
        pipeline_close(p)
        cv2.destroyAllWindows()
