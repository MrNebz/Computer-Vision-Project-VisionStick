"""
VisionStick V2 — core pipeline stages (procedural, no classes).

Stateful components use plain dicts as state containers.
  create_X(...)  → allocates and returns a state dict
  x_do_thing(state, ...)  → operates on that dict

Stage breakdown:
    Stage 1  video_open / video_read / video_release
    Stage 2  preprocess_frame
    Stage 3  detect_obstacles
    Stage 4  track_obstacles
    Stage 5  depth_estimator_create / depth_estimate
    Stage 6  compute_object_closeness
    Stage 7  depth_smoother_create / depth_smoother_update
    Stage 8  compute_risk_scores
    Stage 9  select_primary_obstacle
    Stage 10 temporal_tracker_create / temporal_update
    Stage 11 alert_decider_create / alert_decide
    Stage 12 tts_create / tts_speak / tts_stop
"""

from __future__ import annotations

import time
import threading
import queue
import collections
import subprocess
from typing import Optional

import cv2
import numpy as np
import torch

from . import config as C


# =============================================================================
# Stage 1 — Video source
#
#   vs = video_open(source)
#   frame = video_read(vs)
#   video_release(vs)
# =============================================================================

def video_open(source: int | str = 0) -> dict:
    """Open a webcam or video file. Returns a video-state dict."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return {
        "_cap":    cap,
        "is_file": isinstance(source, str),
    }


def video_width(vs: dict) -> int:
    return int(vs["_cap"].get(cv2.CAP_PROP_FRAME_WIDTH))


def video_height(vs: dict) -> int:
    return int(vs["_cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))


def video_fps(vs: dict) -> float:
    fps = vs["_cap"].get(cv2.CAP_PROP_FPS)
    return fps if fps > 0 else 30.0


def video_total_frames(vs: dict) -> int:
    return int(vs["_cap"].get(cv2.CAP_PROP_FRAME_COUNT)) if vs["is_file"] else -1


def video_read(vs: dict) -> Optional[np.ndarray]:
    ok, frame = vs["_cap"].read()
    return frame if ok else None


def video_release(vs: dict):
    vs["_cap"].release()


# =============================================================================
# Stage 2 — Preprocessing (pure function, no state)
# =============================================================================

def preprocess_frame(
    frame: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    clahe: bool = False,
    gamma: float = 1.0,
    sharpen_strength: float = 0.0,
) -> np.ndarray:
    """
    Optional frame corrections applied BEFORE YOLO.
    All defaults are no-ops — only activate what you need.
    """
    out = frame.copy()

    if alpha != 1.0 or beta != 0.0:
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((cl.apply(l), a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if gamma != 1.0:
        inv = 1.0 / gamma
        table = (np.arange(256) / 255.0) ** inv * 255.0
        out = cv2.LUT(out, table.astype(np.uint8))

    if sharpen_strength > 0.0:
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)

    return out


# =============================================================================
# Stage 3 — Detection
#
#   Detection dict keys:
#       class_id, class_name, confidence, bbox (x1,y1,x2,y2)
# =============================================================================

def det_center_x(det: dict) -> float:
    x1, _, x2, _ = det["bbox"]
    return (x1 + x2) / 2.0


def det_center_y(det: dict) -> float:
    _, y1, _, y2 = det["bbox"]
    return (y1 + y2) / 2.0


def detect_obstacles(
    frame: np.ndarray,
    model,
    conf: float = C.YOLO_CONF,
    iou: float = C.YOLO_IOU,
) -> list[dict]:
    """Run YOLO on *frame* and return only navigation-relevant detections."""
    frame_area = frame.shape[0] * frame.shape[1]

    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=C.YOLO_IMGSZ,
        classes=list(C.OBSTACLE_CLASSES.keys()),
        verbose=False,
        device=C.DEVICE,
    )

    detections: list[dict] = []
    if not results or results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        cid = int(box.cls[0])
        if cid not in C.OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        fill = ((x2 - x1) * (y2 - y1)) / max(frame_area, 1)
        if fill < C.MIN_BBOX_FILL:
            continue
        detections.append({
            "class_id":   cid,
            "class_name": C.OBSTACLE_CLASSES[cid],
            "confidence": float(box.conf[0]),
            "bbox":       (x1, y1, x2, y2),
        })

    return detections


# =============================================================================
# Stage 4 — Tracking
#
#   Tracked detection dict — same keys as detection + track_id
# =============================================================================

def track_obstacles(
    frame: np.ndarray,
    model,
    conf: float = C.YOLO_CONF,
    iou: float = C.YOLO_IOU,
) -> list[dict]:
    """Run YOLO + ByteTrack. Returns detections with persistent track IDs."""
    frame_area = frame.shape[0] * frame.shape[1]

    results = model.track(
        frame,
        persist=True,
        tracker=C.BYTETRACK_CONFIG,
        conf=conf,
        iou=iou,
        imgsz=C.YOLO_IMGSZ,
        classes=list(C.OBSTACLE_CLASSES.keys()),
        verbose=False,
        device=C.DEVICE,
    )

    tracked: list[dict] = []
    if not results or results[0].boxes is None:
        return tracked

    boxes     = results[0].boxes
    track_ids = boxes.id

    for i, box in enumerate(boxes):
        cid = int(box.cls[0])
        if cid not in C.OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        fill = ((x2 - x1) * (y2 - y1)) / max(frame_area, 1)
        if fill < C.MIN_BBOX_FILL:
            continue
        tid = int(track_ids[i]) if track_ids is not None else -1
        tracked.append({
            "class_id":   cid,
            "class_name": C.OBSTACLE_CLASSES[cid],
            "confidence": float(box.conf[0]),
            "bbox":       (x1, y1, x2, y2),
            "track_id":   tid,
        })

    return tracked


def track_custom_obstacles(
    frame: np.ndarray,
    model,
    conf: float = C.YOLO_CONF,
    iou: float = C.YOLO_IOU,
) -> list[dict]:
    """Run the custom Door/Tree/Stairs YOLO model and return tracked detections."""
    frame_area = frame.shape[0] * frame.shape[1]
    custom_names = {name.lower() for name in C.CUSTOM_OBSTACLE_CLASSES.values()}

    results = model.track(
        frame,
        persist=True,
        tracker=C.BYTETRACK_CONFIG,
        conf=conf,
        iou=iou,
        imgsz=C.YOLO_IMGSZ,
        verbose=False,
        device=C.DEVICE,
    )

    tracked: list[dict] = []
    if not results or results[0].boxes is None:
        return tracked

    boxes = results[0].boxes
    track_ids = boxes.id

    for i, box in enumerate(boxes):
        cid = int(box.cls[0])
        model_name = getattr(model, "names", {}).get(cid, str(cid))
        class_name = C.CUSTOM_OBSTACLE_CLASSES.get(cid, str(model_name).lower())
        class_name = class_name.lower()
        if class_name not in custom_names:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        fill = ((x2 - x1) * (y2 - y1)) / max(frame_area, 1)
        if fill < C.MIN_BBOX_FILL:
            continue

        raw_tid = int(track_ids[i]) if track_ids is not None else i
        tracked.append({
            "class_id":   cid,
            "class_name": class_name,
            "confidence": float(box.conf[0]),
            "bbox":       (x1, y1, x2, y2),
            "track_id":   C.CUSTOM_TRACK_ID_OFFSET + raw_tid,
            "source":     "custom",
        })

    return tracked


# =============================================================================
# Stage 5 — Depth estimator
#
#   de = depth_estimator_create()
#   depth_map = depth_estimate(de, frame)
# =============================================================================

def depth_estimator_create(
    model_id: str = C.DEPTH_MODEL,
    device: str = C.DEPTH_DEVICE,
) -> dict:
    """Load Depth Anything V2 and return an estimator state dict."""
    from transformers import pipeline as hf_pipeline

    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    pipe = hf_pipeline(
        task="depth-estimation",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )
    return {"pipe": pipe, "device": device}


def depth_estimate(
    de: dict,
    frame_bgr: np.ndarray,
    max_side: int = C.DEPTH_MAX_SIDE,
) -> np.ndarray:
    """
    Returns a float32 depth map (same H×W as input frame).
    Higher value = closer.
    """
    from PIL import Image

    h, w = frame_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr

    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        out = de["pipe"](pil_img)

    depth_np = np.array(out["depth"], dtype=np.float32)
    if depth_np.shape[:2] != (h, w):
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth_np


# =============================================================================
# Stage 6 — Object closeness (pure functions, no state)
# =============================================================================

def bbox_fill(bbox: tuple, frame_shape: tuple) -> float:
    """Fraction of frame area covered by this bounding box."""
    x1, y1, x2, y2 = bbox
    box_area   = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    return box_area / max(frame_area, 1)


def get_zone(score: float) -> str:
    """Map a blended proximity/path danger score to a zone string."""
    if score >= C.ZONE_DANGER_SCORE:
        return "danger"
    if score >= C.ZONE_CLOSE_SCORE:
        return "close"
    if score >= C.ZONE_MEDIUM_SCORE:
        return "medium"
    return "safe"


def get_direction(bbox: tuple, frame_w: int) -> str:
    """Classify horizontal position as 'left', 'ahead', or 'right'."""
    x1, _, x2, _ = bbox
    cx    = (x1 + x2) / 2.0
    third = frame_w / 3.0
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "ahead"


def walking_path_score(bbox: tuple, frame_shape: tuple) -> float:
    """
    Score how much an object matters to the user's likely walking corridor.

    This intentionally looks beyond raw bbox size:
      - bbox center near the middle of the frame
      - bbox overlap with the lower-middle walking corridor
      - bbox bottom reaching into the lower frame
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    box_area = max(bw * bh, 1)

    path_x1 = int(frame_w * C.PATH_X_MIN)
    path_x2 = int(frame_w * C.PATH_X_MAX)
    path_y1 = int(frame_h * C.PATH_Y_MIN)
    path_y2 = frame_h

    ix1 = max(x1, path_x1)
    iy1 = max(y1, path_y1)
    ix2 = min(x2, path_x2)
    iy2 = min(y2, path_y2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    overlap_score = inter_area / box_area

    cx_norm = ((x1 + x2) / 2.0) / max(frame_w, 1)
    path_half_width = max((C.PATH_X_MAX - C.PATH_X_MIN) / 2.0, 1e-6)
    center_score = 1.0 - abs(cx_norm - 0.5) / path_half_width
    center_score = float(np.clip(center_score, 0.0, 1.0))

    bottom_norm = y2 / max(frame_h, 1)
    bottom_score = (bottom_norm - C.PATH_Y_MIN) / max(1.0 - C.PATH_Y_MIN, 1e-6)
    bottom_score = float(np.clip(bottom_score, 0.0, 1.0))

    score = (
        C.PATH_CENTER_WEIGHT * center_score
        + C.PATH_OVERLAP_WEIGHT * float(np.clip(overlap_score, 0.0, 1.0))
        + C.PATH_BOTTOM_WEIGHT * bottom_score
    )
    return float(np.clip(score, 0.0, 1.0))


def class_specific_risk_adjustment(
    class_name: str,
    risk: float,
    closeness: float,
    path_score: float,
) -> float:
    """Apply small navigation-specific risk corrections by object class."""
    name = class_name.lower()

    if name == "stairs":
        risk *= 0.90 + 0.30 * path_score
        risk += 0.08 * path_score
    elif name == "tree":
        risk *= 0.65 + 0.45 * path_score
        if path_score > 0.55 and closeness > 0.45:
            risk += 0.04
    elif name == "door":
        risk *= 0.70 + 0.35 * path_score
        if path_score < 0.35:
            risk -= 0.05
    elif name in {"car", "bus", "truck", "motorcycle", "bicycle"}:
        risk *= 0.80 + 0.35 * path_score
        if path_score > 0.50 and closeness > 0.45:
            risk += 0.05
    elif name in {"traffic light", "stop sign", "tv"}:
        risk *= 0.70 + 0.20 * path_score

    return float(np.clip(risk, 0.0, 1.0))


def extract_depth_closeness(
    bbox: tuple,
    depth_map: Optional[np.ndarray],
    frame_shape: tuple,
) -> float:
    """
    Returns a [0,1] closeness score from the depth map under *bbox*.
    Returns 0.5 (neutral) when depth map is unavailable.
    Higher = closer.
    """
    if depth_map is None:
        return 0.5

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.5

    margin_x = int(bw * (1 - C.DEPTH_CROP_FRAC) / 2)
    margin_y = int(bh * (1 - C.DEPTH_CROP_FRAC) / 2)
    cx1 = max(0, x1 + margin_x)
    cy1 = max(0, y1 + margin_y)
    cx2 = min(depth_map.shape[1] - 1, x2 - margin_x)
    cy2 = min(depth_map.shape[0] - 1, y2 - margin_y)

    crop = depth_map[cy1:cy2+1, cx1:cx2+1]
    if crop.size == 0:
        crop = depth_map[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return 0.5

    obj_depth = float(np.median(crop))
    p_near    = float(np.percentile(depth_map, C.DEPTH_P_NEAR))
    p_far     = float(np.percentile(depth_map, C.DEPTH_P_FAR))
    drange    = p_far - p_near

    if drange < 1e-6:
        return 0.5

    closeness = np.clip((obj_depth - p_near) / drange, 0.0, 1.0)
    return float(closeness)


def compute_object_closeness(
    bbox: tuple,
    depth_map: Optional[np.ndarray],
    frame_shape: tuple,
) -> float:
    """Blend depth closeness with a weak bbox-size fallback."""
    fill           = bbox_fill(bbox, frame_shape)
    fill_closeness = np.clip(fill / C.ZONE_DANGER_FILL, 0.0, 1.0)
    if depth_map is None:
        return (
            C.NO_DEPTH_BBOX_WEIGHT * float(fill_closeness)
            + C.NO_DEPTH_NEUTRAL_WEIGHT * 0.5
        )

    depth_closeness = extract_depth_closeness(bbox, depth_map, frame_shape)
    return C.BBOX_WEIGHT * float(fill_closeness) + C.DEPTH_WEIGHT * depth_closeness


# =============================================================================
# Stage 7 — Depth smoother (per-track EMA)
#
#   ds = depth_smoother_create()
#   smoothed = depth_smoother_update(ds, track_id, raw_closeness)
#   depth_smoother_prune(ds, active_ids)
# =============================================================================

def depth_smoother_create(alpha: float = C.EMA_ALPHA) -> dict:
    return {"alpha": alpha, "state": {}}


def depth_smoother_update(ds: dict, track_id: int, raw_closeness: float) -> float:
    s = ds["state"]
    if track_id not in s:
        s[track_id] = raw_closeness
    else:
        s[track_id] = ds["alpha"] * raw_closeness + (1.0 - ds["alpha"]) * s[track_id]
    return s[track_id]


def depth_smoother_prune(ds: dict, active_ids: set):
    stale = [tid for tid in ds["state"] if tid not in active_ids]
    for tid in stale:
        del ds["state"][tid]


# =============================================================================
# Stage 8 — Risk scoring
#
#   Ranked obstacle dict keys:
#       detection, fill, closeness, zone, direction, risk_score
# =============================================================================

def compute_risk_scores(
    detections: list[dict],
    depth_map: Optional[np.ndarray],
    frame_shape: tuple,
    depth_smoother: dict,
) -> list[dict]:
    """Score every tracked detection; return sorted highest-risk first."""
    frame_w = frame_shape[1]
    ranked: list[dict] = []

    for det in detections:
        raw_closeness = compute_object_closeness(det["bbox"], depth_map, frame_shape)
        smoothed      = depth_smoother_update(depth_smoother, det["track_id"], raw_closeness)

        fill      = bbox_fill(det["bbox"], frame_shape)
        direction = get_direction(det["bbox"], frame_w)
        path_score = walking_path_score(det["bbox"], frame_shape)

        hazard = C.CLASS_HAZARD.get(det["class_name"], C.DEFAULT_HAZARD)

        risk = (
            C.W_CLOSENESS   * smoothed
            + C.W_PATH      * path_score
            + C.W_HAZARD    * hazard
            + C.W_CONFIDENCE * det["confidence"]
        )
        risk = class_specific_risk_adjustment(
            det["class_name"],
            risk,
            smoothed,
            path_score,
        )

        zone_signal = (
            C.ZONE_CLOSENESS_WEIGHT * smoothed
            + C.ZONE_PATH_WEIGHT * path_score
        )
        zone = get_zone(zone_signal)

        ranked.append({
            "detection":  det,
            "fill":       fill,
            "closeness":  smoothed,
            "path_score": path_score,
            "zone_signal": zone_signal,
            "zone":       zone,
            "direction":  direction,
            "risk_score": float(np.clip(risk, 0.0, 1.0)),
        })

    ranked.sort(key=lambda r: r["risk_score"], reverse=True)
    return ranked


# =============================================================================
# Stage 9 — Primary obstacle selection (pure function)
# =============================================================================

def select_primary_obstacle(
    ranked: list[dict],
    min_risk: float = C.MIN_RISK_THRESHOLD,
) -> Optional[dict]:
    """Return the highest-risk obstacle above *min_risk*, else None."""
    for r in ranked:
        if r["risk_score"] >= min_risk:
            return r
    return None


# =============================================================================
# Stage 10 — Temporal tracker
#
#   tt = temporal_tracker_create()
#   state_dict = temporal_update(tt, track_id, proximity, zone, direction)
#   age = temporal_get_age(tt, track_id)
#   temporal_prune(tt, active_ids)
#
#   Temporal state dict keys: zone, trend, direction
# =============================================================================

def temporal_tracker_create(window: int = C.TREND_WINDOW) -> dict:
    return {
        "window":  window,
        "history": {},   # {track_id: deque of fill values}
        "ages":    {},   # {track_id: int}
    }


def _temporal_compute_trend(tt: dict, track_id: int) -> str:
    hist = list(tt["history"][track_id])
    if len(hist) < max(2, tt["window"] // 2):
        return "stable"
    mid      = len(hist) // 2
    old_mean = float(np.mean(hist[:mid]))
    new_mean = float(np.mean(hist[mid:]))
    if old_mean < 1e-6:
        return "stable"
    rel_change = (new_mean - old_mean) / old_mean
    if rel_change > C.TREND_THRESH:
        return "approaching"
    if rel_change < -C.TREND_THRESH:
        return "receding"
    return "stable"


def temporal_update(
    tt: dict,
    track_id: int,
    proximity: float,
    zone: str,
    direction: str,
) -> dict:
    """Update history for *track_id*; return a temporal-state dict."""
    if track_id not in tt["history"]:
        tt["history"][track_id] = collections.deque(maxlen=tt["window"])
        tt["ages"][track_id] = 0

    tt["history"][track_id].append(proximity)
    tt["ages"][track_id] += 1

    trend = _temporal_compute_trend(tt, track_id)
    return {"zone": zone, "trend": trend, "direction": direction}


def temporal_get_age(tt: dict, track_id: int) -> int:
    return tt["ages"].get(track_id, 0)


def temporal_prune(tt: dict, active_ids: set):
    stale = [tid for tid in tt["history"] if tid not in active_ids]
    for tid in stale:
        del tt["history"][tid]
        tt["ages"].pop(tid, None)


# =============================================================================
# Stage 11 — Alert decider
#
#   ad = alert_decider_create()
#   msg = alert_decide(ad, primary, state, track_age)  → str or None
#   alert_prune(ad, active_ids)
# =============================================================================

_COOLDOWNS = {
    "danger": C.COOLDOWN_DANGER,
    "close":  C.COOLDOWN_CLOSE,
    "medium": C.COOLDOWN_MEDIUM,
    "safe":   C.COOLDOWN_SAFE,
}

_ZONE_ORDER = {"safe": 0, "medium": 1, "close": 2, "danger": 3}


def alert_decider_create() -> dict:
    return {
        "last":             {},    # {track_id: (zone, alert_time)}
        "last_primary_tid": None,
    }


def _alert_build_message(primary: dict, state: dict) -> str:
    name      = primary["detection"]["class_name"]
    zone      = state["zone"]
    trend     = state["trend"]
    direction = state["direction"]

    zone_phrase = {
        "danger": "very close",
        "close":  "close",
        "medium": "nearby",
        "safe":   "ahead",
    }.get(zone, "ahead")

    trend_phrase = {
        "approaching": ", approaching",
        "receding":    ", moving away",
        "stable":      "",
    }.get(trend, "")

    dir_phrase = {
        "left":  "on the left",
        "right": "on the right",
        "ahead": "ahead",
    }.get(direction, "ahead")

    return f"{name} {zone_phrase} {dir_phrase}{trend_phrase}"


def alert_decide(
    ad: dict,
    primary: dict,
    state: dict,
    track_age: int,
) -> Optional[str]:
    """Returns a TTS message string, or None if should stay silent."""
    if track_age < C.MIN_TRACK_AGE:
        return None

    tid  = primary["detection"]["track_id"]
    zone = state["zone"]
    now  = time.time()

    primary_changed     = (tid != ad["last_primary_tid"])
    ad["last_primary_tid"] = tid

    prev_zone, last_time = ad["last"].get(tid, (None, 0.0))
    cooldown = _COOLDOWNS.get(zone, C.COOLDOWN_MEDIUM)

    escalated = (
        prev_zone is not None
        and _ZONE_ORDER.get(zone, 0) > _ZONE_ORDER.get(prev_zone, 0)
    )

    if primary_changed or escalated:
        should_fire = True
    elif zone == "safe":
        should_fire = False
    else:
        should_fire = (now - last_time) >= cooldown

    if not should_fire:
        return None

    msg = _alert_build_message(primary, state)
    ad["last"][tid] = (zone, now)
    return msg


def alert_prune(ad: dict, active_ids: set):
    stale = [tid for tid in ad["last"] if tid not in active_ids]
    for tid in stale:
        del ad["last"][tid]
    if ad["last_primary_tid"] not in active_ids:
        ad["last_primary_tid"] = None


# =============================================================================
# Stage 12 — TTS engine
#
#   tts = tts_create()
#   tts_speak(tts, "person ahead")
#   tts_stop(tts)
# =============================================================================

_TTS_POISON = object()   # sentinel that tells the worker thread to exit


def _tts_start_worker(tts: dict):
    """Launch the PowerShell speech process."""
    sapi_rate = max(-10, min(10, round((tts["rate"] - 130) / 15)))
    ps_cmd = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {sapi_rate}; "
        "$s.Speak(' '); "        # warmup: forces COM + audio device init now
        "while ($true) { "
        "    $t = [Console]::In.ReadLine(); "
        "    if ($null -eq $t -or $t -eq '__STOP__') { break }; "
        "    if ($t.Trim().Length -gt 0) { $s.Speak($t) } "
        "}"
    )
    return subprocess.Popen(
        ["powershell", "-NonInteractive", "-NoProfile",
         "-WindowStyle", "Hidden", "-Command", ps_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _tts_run(tts: dict):
    """Worker thread: reads from queue and writes to PowerShell stdin."""
    tts["proc"] = _tts_start_worker(tts)

    while True:
        item = tts["queue"].get()
        if item is _TTS_POISON:
            break

        # Restart worker if it exited unexpectedly
        if tts["proc"].poll() is not None:
            tts["proc"] = _tts_start_worker(tts)

        safe = item.replace("\n", " ").replace("\r", " ")
        try:
            tts["proc"].stdin.write(safe + "\n")
            tts["proc"].stdin.flush()
        except Exception:
            try:
                tts["proc"].terminate()
            except Exception:
                pass
            tts["proc"] = _tts_start_worker(tts)
            try:
                tts["proc"].stdin.write(safe + "\n")
                tts["proc"].stdin.flush()
            except Exception:
                pass   # give up on this message


def tts_create(rate: int = 160, volume: float = 1.0) -> dict:
    """Create and start the TTS engine. Returns a state dict."""
    tts = {
        "rate":   rate,
        "volume": volume,
        "queue":  queue.Queue(maxsize=1),
        "proc":   None,
        "thread": None,
    }
    tts["thread"] = threading.Thread(target=_tts_run, args=(tts,), daemon=True)
    tts["thread"].start()
    return tts


def tts_speak(tts: dict, text: str, *, priority: bool = False):
    """
    Queue *text* for speech.
    priority=True (danger): drains pending queue so danger is next.
    A phrase being spoken is never cut off.
    """
    if priority:
        while not tts["queue"].empty():
            try:
                tts["queue"].get_nowait()
            except queue.Empty:
                break
    try:
        tts["queue"].put_nowait(text)
    except queue.Full:
        pass   # drop — freshness over backlog


def tts_stop(tts: dict):
    """Shut down the TTS thread and PowerShell worker process."""
    try:
        tts["queue"].put_nowait(_TTS_POISON)
    except queue.Full:
        pass
    if tts["thread"]:
        tts["thread"].join(timeout=5.0)
    proc = tts.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.stdin.write("__STOP__\n")
            proc.stdin.flush()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
