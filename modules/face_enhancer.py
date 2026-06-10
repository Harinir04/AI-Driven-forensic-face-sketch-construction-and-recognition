"""
Module 5: Forensic Mark Renderer
Renders scars, moles, birthmarks, and other distinguishing marks
onto generated faces at PRECISE locations using face landmarks.

Two-stage approach:
  Stage 1 (face_generator) produces a clean base face.
  Stage 2 (this module) adds marks at exact pixel positions
  derived from 5-point face landmarks.

No extra model downloads. Uses OpenCV compositing for fast,
deterministic, realistic mark rendering.
"""

import math
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from modules.onnx_face import OnnxFaceProcessor


# ─── Singleton face processor for landmark detection ─────────

_processor: Optional[OnnxFaceProcessor] = None


def _get_processor() -> OnnxFaceProcessor:
    global _processor
    if _processor is None:
        _processor = OnnxFaceProcessor()
        _processor.load_detection()
    return _processor


# ═══════════════════════════════════════════════════════════════
# Face Region Mapping
# Maps verbal location descriptions → pixel coordinates
# using 5-point landmarks.
#
# IMPORTANT — coordinate convention:
#   RetinaFace returns landmarks in VIEWER-perspective order:
#     lmks[0] is the point at the viewer's LEFT side of the image,
#             which is the SUBJECT's RIGHT eye.
#     lmks[1] is the viewer's RIGHT = SUBJECT's LEFT eye.
#     lmks[3], lmks[4] = mouth corners in the same order.
#
#   Witness descriptions ("mole below her right eye", "scar on left
#   cheek") are always in the SUBJECT's perspective. Rendering moles
#   on the wrong side of the face is a serious forensic error, so we
#   bind the landmarks to SUBJECT-pov variables here and build every
#   region dict entry from those names. From this point on, "left" and
#   "right" in region keys ALWAYS mean the subject's left / right.
# ═══════════════════════════════════════════════════════════════

def _compute_face_regions(bbox: np.ndarray, lmks: np.ndarray) -> dict[str, tuple[int, int]]:
    """
    Compute named face regions from bbox + 5-point landmarks.

    All region keys use SUBJECT-perspective left/right (matches the
    witness's wording). Values are (x, y) pixels in original image coords.
    """
    # Subject-perspective landmark binding (see file-level note above).
    subj_right_eye   = lmks[0]     # at viewer's left
    subj_left_eye    = lmks[1]     # at viewer's right
    nose             = lmks[2]
    subj_right_mouth = lmks[3]     # at viewer's left
    subj_left_mouth  = lmks[4]     # at viewer's right

    eye_center   = (subj_right_eye + subj_left_eye) / 2
    mouth_center = (subj_right_mouth + subj_left_mouth) / 2
    eye_dist     = np.linalg.norm(subj_left_eye - subj_right_eye)

    # Image extents — these are coordinate-space, not body-side.
    # Renamed to make it obvious they're image edges, not face sides.
    img_left_edge  = bbox[0]       # lowest x in the face bbox
    img_right_edge = bbox[2]       # highest x in the face bbox
    face_top       = bbox[1]
    face_bottom    = bbox[3]
    face_cx        = (img_left_edge + img_right_edge) / 2

    # Proportional offsets based on eye distance (most stable reference)
    unit = eye_dist / 4  # ~25px for a typical 512px face

    regions = {
        # ── Eye area ───────────────────────────────────────────
        # "left eye" = SUBJECT's left eye, which is at viewer's RIGHT
        # of the image, which is lmks[1] = subj_left_eye.
        "left eye":          _pt(subj_left_eye),
        "right eye":         _pt(subj_right_eye),
        "below left eye":    _pt(subj_left_eye  + [0, unit * 1.2]),
        "below right eye":   _pt(subj_right_eye + [0, unit * 1.2]),
        "under left eye":    _pt(subj_left_eye  + [0, unit * 1.0]),
        "under right eye":   _pt(subj_right_eye + [0, unit * 1.0]),
        "above left eye":    _pt(subj_left_eye  + [0, -unit * 1.0]),
        "above right eye":   _pt(subj_right_eye + [0, -unit * 1.0]),
        # "near X eye" drifts toward the face center in x
        "near left eye":     _pt(subj_left_eye  + [-unit * 0.8, unit * 0.5]),
        "near right eye":    _pt(subj_right_eye + [ unit * 0.8, unit * 0.5]),
        "left eyebrow":      _pt(subj_left_eye  + [0, -unit * 1.5]),
        "right eyebrow":     _pt(subj_right_eye + [0, -unit * 1.5]),
        "between eyebrows":  _pt(eye_center     + [0, -unit * 0.8]),
        # Unspecified-side eyebrow — default to subject's LEFT eyebrow.
        # Witness phrases like "a scar on her eyebrow" almost always
        # mean the hair line of one brow, not the glabella.
        "eyebrow":           _pt(subj_left_eye  + [0, -unit * 1.5]),
        "eyebrows":          _pt(subj_left_eye  + [0, -unit * 1.5]),
        "brow":              _pt(subj_left_eye  + [0, -unit * 1.5]),

        # ── Cheeks ─────────────────────────────────────────────
        # Subject's left cheek sits to the OUTSIDE of subj_left_eye,
        # i.e. moving further in the +x direction (viewer's right).
        "left cheek":        _pt([subj_left_eye[0]  + unit * 1.5,
                                  (subj_left_eye[1]  + subj_left_mouth[1])  / 2]),
        "right cheek":       _pt([subj_right_eye[0] - unit * 1.5,
                                  (subj_right_eye[1] + subj_right_mouth[1]) / 2]),
        "left side of face": _pt([img_right_edge - unit,
                                  (subj_left_eye[1]  + subj_left_mouth[1])  / 2]),
        "right side of face":_pt([img_left_edge  + unit,
                                  (subj_right_eye[1] + subj_right_mouth[1]) / 2]),
        "left face":         _pt([subj_left_eye[0]  + unit * 1.5,
                                  (subj_left_eye[1]  + subj_left_mouth[1])  / 2]),
        "right face":        _pt([subj_right_eye[0] - unit * 1.5,
                                  (subj_right_eye[1] + subj_right_mouth[1]) / 2]),

        # ── Forehead ───────────────────────────────────────────
        "forehead":              _pt([face_cx, (face_top + eye_center[1]) / 2]),
        "center of forehead":    _pt([face_cx, (face_top + eye_center[1]) / 2]),
        # Subject's left temple = viewer's right = image's RIGHT edge side
        "left temple":           _pt([img_right_edge - unit, subj_left_eye[1]  - unit]),
        "right temple":          _pt([img_left_edge  + unit, subj_right_eye[1] - unit]),
        "left side of forehead": _pt([subj_left_eye[0]  + unit,
                                      (face_top + subj_left_eye[1])  / 2]),
        "right side of forehead":_pt([subj_right_eye[0] - unit,
                                      (face_top + subj_right_eye[1]) / 2]),

        # ── Nose ───────────────────────────────────────────────
        # Nose-side and nostril: subject's left = +x, subject's right = -x
        "nose":              _pt(nose),
        "bridge of nose":    _pt(eye_center + [0, unit * 0.8]),
        "tip of nose":       _pt(nose       + [0, unit * 0.5]),
        "left side of nose": _pt(nose       + [ unit * 0.8, 0]),
        "right side of nose":_pt(nose       + [-unit * 0.8, 0]),
        "left nostril":      _pt(nose       + [ unit * 0.7, unit * 0.4]),
        "right nostril":     _pt(nose       + [-unit * 0.7, unit * 0.4]),

        # ── Mouth / Lip ────────────────────────────────────────
        "upper lip":            _pt(mouth_center + [0, -unit * 0.5]),
        "lower lip":            _pt(mouth_center + [0,  unit * 0.5]),
        "above lip":            _pt(mouth_center + [0, -unit * 1.0]),
        "above mouth":          _pt(mouth_center + [0, -unit * 1.2]),
        "corner of mouth":      _pt(subj_left_mouth),   # default: subject's left
        "left corner of mouth": _pt(subj_left_mouth),
        "right corner of mouth":_pt(subj_right_mouth),

        # ── Chin / Jaw ─────────────────────────────────────────
        "chin":              _pt([face_cx, face_bottom - unit * 0.8]),
        "left chin":         _pt([subj_left_mouth[0],  face_bottom - unit * 0.5]),
        "right chin":        _pt([subj_right_mouth[0], face_bottom - unit * 0.5]),
        "left side of chin": _pt([subj_left_mouth[0],  face_bottom - unit * 0.5]),
        "right side of chin":_pt([subj_right_mouth[0], face_bottom - unit * 0.5]),
        "left jaw":          _pt([img_right_edge - unit,
                                  (subj_left_mouth[1]  + face_bottom) / 2]),
        "right jaw":         _pt([img_left_edge  + unit,
                                  (subj_right_mouth[1] + face_bottom) / 2]),
        "jawline":           _pt([face_cx, (mouth_center[1] + face_bottom) / 2]),

        # ── Ear / Neck ─────────────────────────────────────────
        "left ear":          _pt([img_right_edge, subj_left_eye[1]]),
        "right ear":         _pt([img_left_edge,  subj_right_eye[1]]),
        "neck":              _pt([face_cx, face_bottom + unit]),
        "throat":            _pt([face_cx, face_bottom + unit * 0.5]),
    }

    return regions


def _pt(arr) -> tuple[int, int]:
    """Convert array-like to integer pixel coordinates."""
    if isinstance(arr, np.ndarray):
        return (int(arr[0]), int(arr[1]))
    return (int(arr[0]), int(arr[1]))


def resolve_location(location_str: str, bbox: np.ndarray, landmarks: np.ndarray) -> Optional[tuple[int, int]]:
    """
    Map a verbal location description to pixel coordinates.

    Args:
        location_str: e.g. "below left eye", "right cheek"
        bbox: face bounding box [x1, y1, x2, y2]
        landmarks: 5-point face landmarks

    Returns:
        (x, y) pixel coordinates or None if location not recognized
    """
    if not location_str:
        return None

    regions = _compute_face_regions(bbox, landmarks)

    # Clean the location string
    loc = location_str.lower().strip()
    loc = loc.replace("his ", "").replace("her ", "").replace("their ", "").replace("the ", "")

    # Direct match
    if loc in regions:
        return regions[loc]

    # Partial match — find the best matching region
    best_match = None
    best_score = 0
    for region_name, coords in regions.items():
        # Check word overlap
        loc_words = set(loc.split())
        region_words = set(region_name.split())
        overlap = len(loc_words & region_words)
        if overlap > best_score:
            best_score = overlap
            best_match = coords

    if best_score >= 1:
        return best_match

    return None


# ═══════════════════════════════════════════════════════════════
# Mark Renderers
# Realistic OpenCV compositing for each mark type
# ═══════════════════════════════════════════════════════════════

def _sample_skin_color(image: np.ndarray, center: tuple[int, int], radius: int = 8) -> np.ndarray:
    """Sample the average skin color around a point."""
    h, w = image.shape[:2]
    x, y = center
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([140, 120, 100], dtype=np.uint8)
    return patch.mean(axis=(0, 1)).astype(np.uint8)


def render_mole(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "small",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """
    Render a realistic mole/beauty mark on the face.

    Uses skin color sampling to create a darker-than-skin ellipse
    with soft edges and slight 3D highlight effect.
    """
    img = image.copy()
    h, w = img.shape[:2]
    x, y = max(5, min(w - 5, center[0])), max(5, min(h - 5, center[1]))

    # Size based on descriptor and face scale
    scale = eye_dist / 100.0
    size_map = {
        "tiny": max(2, int(2 * scale)),
        "small": max(3, int(4 * scale)),
        "medium": max(4, int(6 * scale)),
        "large": max(6, int(9 * scale)),
        "big": max(6, int(9 * scale)),
    }

    # Check descriptors for size hints
    radius = size_map.get(size, size_map["small"])
    if descriptors:
        for d in descriptors:
            if d in size_map:
                radius = size_map[d]
                break

    # Determine mole color (darker than surrounding skin)
    skin_color = _sample_skin_color(img, (x, y))
    darkness = -50 if descriptors and "dark" in descriptors else -35

    mole_color = np.clip(skin_color.astype(int) + darkness, 10, 180)
    # Make clearly reddish-brown
    mole_color[2] = min(mole_color[2] + 15, 180)  # R channel (BGR)
    mole_color[0] = max(mole_color[0] - 10, 10)   # Less blue

    # Create soft mask for the mole
    mask = np.zeros((h, w), dtype=np.float32)
    # Slightly elongated ellipse with random angle for naturalness
    angle = random.randint(0, 180)
    axes = (radius, max(2, int(radius * 0.85)))
    cv2.ellipse(mask, (x, y), axes, angle, 0, 360, 1.0, -1)

    # Gaussian blur the mask for soft edges
    blur_k = max(3, radius * 2 + 1)
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    # Apply: blend mole color with original image using mask
    mole_layer = np.full_like(img, mole_color, dtype=np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    img = (img.astype(np.float32) * (1 - mask_3ch * 0.92) +
           mole_layer.astype(np.float32) * mask_3ch * 0.92).astype(np.uint8)

    # Subtle highlight on top edge (3D effect)
    if radius >= 4:
        highlight_y = max(0, y - int(radius * 0.3))
        highlight_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(highlight_mask, (x, highlight_y),
                    (max(1, radius // 2), max(1, radius // 3)),
                    0, 0, 360, 0.3, -1)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (blur_k, blur_k), 0)
        highlight_3ch = np.stack([highlight_mask] * 3, axis=-1)
        img = np.clip(img.astype(np.float32) + highlight_3ch * 30, 0, 255).astype(np.uint8)

    return img


def render_scar(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "small",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """
    Render a realistic scar on the face.

    Creates a line-based scar with texture, appropriate coloring
    (slightly lighter/pinker than skin), and natural edge blending.
    """
    img = image.copy()
    h, w = img.shape[:2]
    x, y = max(10, min(w - 10, center[0])), max(10, min(h - 10, center[1]))

    scale = eye_dist / 100.0
    descriptors = descriptors or []

    # Determine scar length
    if "long" in descriptors or "deep" in descriptors:
        length = int(30 * scale)
    elif "small" in descriptors or "short" in descriptors:
        length = int(12 * scale)
    else:
        length = int(20 * scale)

    # Determine scar width
    if "deep" in descriptors or "thick" in descriptors:
        thickness = max(3, int(4 * scale))
    elif "thin" in descriptors or "faint" in descriptors:
        thickness = max(1, int(2 * scale))
    else:
        thickness = max(2, int(3 * scale))

    # Determine scar orientation
    if "vertical" in descriptors:
        angle = math.radians(90 + random.randint(-10, 10))
    elif "horizontal" in descriptors:
        angle = math.radians(0 + random.randint(-10, 10))
    elif "diagonal" in descriptors:
        angle = math.radians(45 + random.randint(-15, 15))
    else:
        angle = math.radians(random.choice([30, 60, 90, 120, 150]))

    half_len = length // 2

    # Calculate start and end points
    dx = int(half_len * math.cos(angle))
    dy = int(half_len * math.sin(angle))
    pt1 = (x - dx, y - dy)
    pt2 = (x + dx, y + dy)

    # Scar color — slightly lighter and pinker than surrounding skin
    skin_color = _sample_skin_color(img, (x, y), radius=15)

    if "old" in descriptors or "healed" in descriptors or "faint" in descriptors:
        # Old/healed scar: lighter, less pink
        scar_color = np.clip(skin_color.astype(int) + [10, 15, 30], 0, 255).astype(np.uint8)
        opacity = 0.55
    elif "fresh" in descriptors:
        # Fresh scar: redder, very visible
        scar_color = np.clip(skin_color.astype(int) + [-15, -15, 50], 0, 255).astype(np.uint8)
        opacity = 0.8
    elif "deep" in descriptors:
        # Deep scar: darker center, very prominent
        scar_color = np.clip(skin_color.astype(int) + [-25, -10, 40], 0, 255).astype(np.uint8)
        opacity = 0.8
    else:
        scar_color = np.clip(skin_color.astype(int) + [-5, 10, 35], 0, 255).astype(np.uint8)
        opacity = 0.65

    # Create scar mask
    scar_mask = np.zeros((h, w), dtype=np.float32)

    # Generate slightly jagged path for natural look
    if "jagged" in descriptors:
        num_points = max(4, length // 5)
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            px = int(pt1[0] + t * (pt2[0] - pt1[0]) + random.randint(-3, 3))
            py = int(pt1[1] + t * (pt2[1] - pt1[1]) + random.randint(-3, 3))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(scar_mask, points[i], points[i + 1], 1.0, thickness)
    else:
        # Slight natural curve
        mid_x = (pt1[0] + pt2[0]) // 2 + random.randint(-2, 2)
        mid_y = (pt1[1] + pt2[1]) // 2 + random.randint(-2, 2)
        pts = np.array([pt1, (mid_x, mid_y), pt2], dtype=np.int32)

        # Draw as polyline for slight curve
        cv2.polylines(scar_mask, [pts], False, 1.0, thickness)

    # Blur edges for natural blending
    blur_k = max(3, thickness * 2 + 1)
    if blur_k % 2 == 0:
        blur_k += 1
    scar_mask = cv2.GaussianBlur(scar_mask, (blur_k, blur_k), 0)

    # Apply scar
    scar_layer = np.full_like(img, scar_color, dtype=np.uint8)
    mask_3ch = np.stack([scar_mask] * 3, axis=-1) * opacity
    img = (img.astype(np.float32) * (1 - mask_3ch) +
           scar_layer.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Add slight indentation effect (dark edge + light center)
    if thickness >= 3 and "deep" in descriptors:
        # Dark edges
        edge_mask = scar_mask.copy()
        inner = cv2.erode(edge_mask, np.ones((2, 2), np.uint8))
        edge_only = np.clip(edge_mask - inner, 0, 1)
        edge_3ch = np.stack([edge_only] * 3, axis=-1) * 0.2
        img = np.clip(img.astype(np.float32) - edge_3ch * 40, 0, 255).astype(np.uint8)

    return img


def render_birthmark(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "medium",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render a birthmark (irregular darker patch)."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = max(5, min(w - 5, center[0])), max(5, min(h - 5, center[1]))

    scale = eye_dist / 100.0
    radius = int(8 * scale)

    skin_color = _sample_skin_color(img, (x, y))
    mark_color = np.clip(skin_color.astype(int) + [-20, -15, 5], 10, 220).astype(np.uint8)

    # Irregular shape using multiple overlapping ellipses
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        ox = random.randint(-radius // 2, radius // 2)
        oy = random.randint(-radius // 2, radius // 2)
        rx = max(3, radius + random.randint(-3, 3))
        ry = max(3, int(rx * random.uniform(0.6, 1.0)))
        ang = random.randint(0, 180)
        cv2.ellipse(mask, (x + ox, y + oy), (rx, ry), ang, 0, 360, 1.0, -1)

    blur_k = max(5, radius + 1)
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = np.clip(mask, 0, 1)

    mark_layer = np.full_like(img, mark_color, dtype=np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1) * 0.5
    img = (img.astype(np.float32) * (1 - mask_3ch) +
           mark_layer.astype(np.float32) * mask_3ch).astype(np.uint8)

    return img


def render_wrinkles(
    image: np.ndarray,
    center: tuple[int, int],
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render wrinkle lines (crow's feet, forehead lines)."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = center

    scale = eye_dist / 100.0
    num_lines = 4 if descriptors and "deep" in descriptors else 2

    skin_color = _sample_skin_color(img, (x, y))
    wrinkle_color = np.clip(skin_color.astype(int) - 15, 0, 200).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(num_lines):
        offset_y = int((i - num_lines // 2) * 4 * scale)
        length = int(15 * scale)
        pt1 = (x - length, y + offset_y + random.randint(-1, 1))
        pt2 = (x + length, y + offset_y + random.randint(-1, 1))
        cv2.line(mask, pt1, pt2, 0.6, 1)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    wrinkle_layer = np.full_like(img, wrinkle_color, dtype=np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    img = (img.astype(np.float32) * (1 - mask_3ch * 0.4) +
           wrinkle_layer.astype(np.float32) * mask_3ch * 0.4).astype(np.uint8)

    return img


def render_dimple(
    image: np.ndarray,
    center: tuple[int, int],
    eye_dist: float = 100.0,
    **kwargs,
) -> np.ndarray:
    """Render a subtle dimple depression."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = center
    scale = eye_dist / 100.0
    radius = max(3, int(5 * scale))

    # Dimple = subtle shadow (slightly darker circle)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (x, y), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)

    img = np.clip(img.astype(np.float32) - np.stack([mask] * 3, axis=-1) * 12, 0, 255).astype(np.uint8)
    return img


# ═══════════════════════════════════════════════════════════════
# Accessory Renderers
# Bindi, earrings, nose rings — critical for Indian forensic context
# ═══════════════════════════════════════════════════════════════

def render_bindi(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "medium",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render a bindi (forehead dot) — classic Indian identifying feature."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = center
    scale = eye_dist / 100.0
    descriptors = descriptors or []

    # Size
    if "large" in descriptors or "big" in descriptors:
        radius = max(5, int(7 * scale))
    elif "small" in descriptors or "tiny" in descriptors:
        radius = max(3, int(4 * scale))
    else:
        radius = max(4, int(5 * scale))

    # Color — traditionally red, but can vary
    if "black" in descriptors:
        bindi_color = (15, 15, 15)       # BGR
    elif "maroon" in descriptors:
        bindi_color = (20, 10, 120)
    elif "golden" in descriptors or "gold" in descriptors:
        bindi_color = (30, 180, 220)
    else:
        bindi_color = (30, 30, 200)      # Classic red (BGR)

    # Draw solid circle with slight 3D effect
    # Dark base
    cv2.circle(img, (x, y), radius, bindi_color, -1, cv2.LINE_AA)

    # Highlight for 3D dome effect
    highlight_pos = (x - radius // 4, y - radius // 4)
    highlight_r = max(1, radius // 3)
    highlight_color = tuple(min(c + 60, 255) for c in bindi_color)
    cv2.circle(img, highlight_pos, highlight_r, highlight_color, -1, cv2.LINE_AA)

    # Subtle outer glow/shadow
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (x, y), radius + 1, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    shadow_3ch = np.stack([mask] * 3, axis=-1) * 0.15
    img = np.clip(img.astype(np.float32) - shadow_3ch * 30, 0, 255).astype(np.uint8)

    # Re-draw the bindi on top (shadow shouldn't cover it)
    cv2.circle(img, (x, y), radius, bindi_color, -1, cv2.LINE_AA)
    cv2.circle(img, highlight_pos, highlight_r, highlight_color, -1, cv2.LINE_AA)

    return img


def render_nose_ring(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "medium",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render a nose ring/stud — common Indian identifying feature."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = center
    scale = eye_dist / 100.0
    descriptors = descriptors or []

    is_ring = any(w in descriptors for w in ["ring", "hoop"]) or "ring" in " ".join(descriptors)
    is_stud = any(w in descriptors for w in ["stud", "pin", "stone"])

    if is_ring or (not is_stud and "ring" in (size or "")):
        # Nose ring (hoop) — on the side of the nostril
        ring_radius = max(5, int(8 * scale))

        # Determine color
        if "silver" in descriptors or "white" in descriptors:
            ring_color = (200, 200, 210)   # Silver (BGR)
        else:
            ring_color = (50, 190, 230)    # Gold (BGR)

        # Draw ring arc (half circle hanging from nostril)
        ring_center = (x, y + ring_radius // 2)
        cv2.ellipse(img, ring_center, (ring_radius, ring_radius),
                    0, 10, 170, ring_color, max(2, int(2 * scale)), cv2.LINE_AA)

        # Small attachment dot at nostril
        cv2.circle(img, (x, y), max(2, int(2 * scale)), ring_color, -1, cv2.LINE_AA)

    else:
        # Nose stud (small jewel dot)
        stud_radius = max(2, int(3 * scale))

        if "diamond" in descriptors or "white" in descriptors:
            stud_color = (230, 230, 240)   # Diamond/silver
        elif "red" in descriptors or "ruby" in descriptors:
            stud_color = (40, 40, 200)     # Red stone
        else:
            stud_color = (60, 200, 240)    # Gold

        cv2.circle(img, (x, y), stud_radius, stud_color, -1, cv2.LINE_AA)

        # Highlight
        h_pos = (x - 1, y - 1)
        cv2.circle(img, h_pos, max(1, stud_radius // 2),
                   tuple(min(c + 40, 255) for c in stud_color), -1, cv2.LINE_AA)

    return img


def render_earring(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "medium",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render an earring — stud, hoop, or dangling."""
    img = image.copy()
    h, w = img.shape[:2]
    x, y = center
    scale = eye_dist / 100.0
    descriptors = descriptors or []

    is_hoop = any(w in descriptors for w in ["hoop", "ring", "loop"])
    is_dangle = any(w in descriptors for w in ["dangling", "hanging", "long", "drop"])

    # Determine color
    if "silver" in descriptors or "white" in descriptors:
        metal_color = (200, 200, 210)
    else:
        metal_color = (50, 190, 230)  # Gold

    # Position: slightly below ear center (earlobe)
    lobe_y = y + int(8 * scale)

    if is_hoop:
        ring_r = max(6, int(10 * scale))
        cv2.ellipse(img, (x, lobe_y), (ring_r, ring_r),
                    0, -30, 210, metal_color, max(2, int(2 * scale)), cv2.LINE_AA)
    elif is_dangle:
        # Stud at top + dangling element
        cv2.circle(img, (x, lobe_y), max(2, int(3 * scale)), metal_color, -1, cv2.LINE_AA)
        dangle_len = int(15 * scale)
        cv2.line(img, (x, lobe_y + 3), (x, lobe_y + dangle_len),
                 metal_color, max(1, int(1.5 * scale)), cv2.LINE_AA)
        # Small gem at bottom
        gem_r = max(2, int(3 * scale))
        cv2.circle(img, (x, lobe_y + dangle_len + gem_r),
                   gem_r, metal_color, -1, cv2.LINE_AA)
    else:
        # Simple stud
        stud_r = max(3, int(4 * scale))
        cv2.circle(img, (x, lobe_y), stud_r, metal_color, -1, cv2.LINE_AA)
        # Highlight
        cv2.circle(img, (x - 1, lobe_y - 1), max(1, stud_r // 2),
                   tuple(min(c + 40, 255) for c in metal_color), -1, cv2.LINE_AA)

    return img


def render_piercing(
    image: np.ndarray,
    center: tuple[int, int],
    size: str = "small",
    descriptors: list = None,
    eye_dist: float = 100.0,
) -> np.ndarray:
    """Render a facial piercing (lip, eyebrow, etc.)."""
    img = image.copy()
    x, y = center
    scale = eye_dist / 100.0

    if "silver" in (descriptors or []):
        color = (200, 200, 210)
    else:
        color = (180, 180, 190)  # Silver metal default

    stud_r = max(2, int(3 * scale))
    cv2.circle(img, (x, y), stud_r, color, -1, cv2.LINE_AA)
    # Metallic highlight
    cv2.circle(img, (x - 1, y - 1), max(1, stud_r // 2),
               (230, 230, 240), -1, cv2.LINE_AA)

    return img


# ─── Mark Type Router ─────────────────────────────────────────

_MARK_RENDERERS = {
    "scar": render_scar,
    "deep scar": render_scar,
    "long scar": render_scar,
    "small scar": render_scar,
    "vertical scar": render_scar,
    "horizontal scar": render_scar,
    "diagonal scar": render_scar,
    "jagged scar": render_scar,
    "surgical scar": render_scar,
    "burn scar": render_scar,
    "mole": render_mole,
    "large mole": render_mole,
    "small mole": render_mole,
    "dark mole": render_mole,
    "raised mole": render_mole,
    "flat mole": render_mole,
    "birthmark": render_birthmark,
    "port wine stain": render_birthmark,
    "wrinkles": render_wrinkles,
    "deep wrinkles": render_wrinkles,
    "crow's feet": render_wrinkles,
    "dimple": render_dimple,
    "dimples": render_dimple,
    # Accessories
    "bindi": render_bindi,
    "nose ring": render_nose_ring,
    "nose stud": render_nose_ring,
    "nose piercing": render_nose_ring,
    "earring": render_earring,
    "ear ring": render_earring,
    "ear piercing": render_earring,
    "hoop earring": render_earring,
    "stud earring": render_earring,
    "piercing": render_piercing,
    "lip piercing": render_piercing,
    "eyebrow piercing": render_piercing,
}


# ═══════════════════════════════════════════════════════════════
# Main API: Add Marks to Face
# ═══════════════════════════════════════════════════════════════

def add_marks_to_face(
    image: Image.Image | np.ndarray | str,
    marks: list[dict],
) -> Image.Image:
    """
    Add distinguishing marks to a generated face image.

    Args:
        image: PIL Image, numpy array (BGR), or file path
        marks: list of mark dicts from attribute_parser, each with:
               {type, location, descriptors, raw}

    Returns:
        PIL Image with marks rendered at precise locations
    """
    # Convert to BGR numpy array
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    if not marks:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Detect face landmarks
    proc = _get_processor()
    faces = proc.detect_faces(img, score_thresh=0.3)

    if not faces:
        print("  [enhancer] No face detected — returning image without marks")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    face = faces[0]
    bbox = face["bbox"]
    landmarks = face["landmarks_5"]
    eye_dist = float(np.linalg.norm(landmarks[1] - landmarks[0]))

    # Render each mark
    for mark in marks:
        mark_type = mark.get("type", "")
        location_str = mark.get("location", "")
        descriptors = mark.get("descriptors", [])

        # Resolve location to pixel coordinates
        center = resolve_location(location_str, bbox, landmarks)
        if center is None:
            # Fallback: try to find a reasonable default
            if "cheek" in mark_type or "cheek" in location_str:
                center = resolve_location("right cheek", bbox, landmarks)
            elif "eye" in location_str:
                center = resolve_location("below right eye", bbox, landmarks)
            else:
                center = resolve_location("right cheek", bbox, landmarks)

        if center is None:
            print(f"  [enhancer] Could not resolve location '{location_str}' for {mark_type}")
            continue

        # Clamp to image bounds
        h, w = img.shape[:2]
        center = (max(5, min(w - 5, center[0])), max(5, min(h - 5, center[1])))

        # Find renderer for this mark type
        renderer = _MARK_RENDERERS.get(mark_type)
        if renderer is None:
            # Try partial match
            for key, func in _MARK_RENDERERS.items():
                if key in mark_type or mark_type in key:
                    renderer = func
                    break

        if renderer is None:
            # Default to scar for unknown types
            if "scar" in mark_type:
                renderer = render_scar
            elif "mole" in mark_type:
                renderer = render_mole
            else:
                print(f"  [enhancer] No renderer for mark type: {mark_type}")
                continue

        # Merge mark_type descriptors into descriptor list
        type_descriptors = list(descriptors)
        for word in mark_type.split():
            if word not in ["scar", "mole", "birthmark", "dimple", "wrinkles"]:
                type_descriptors.append(word)

        print(f"  [enhancer] Rendering {mark_type} at ({center[0]}, {center[1]}) "
              f"[{location_str or 'no location'}]")

        img = renderer(
            img, center,
            size=type_descriptors[0] if type_descriptors else "medium",
            descriptors=type_descriptors,
            eye_dist=eye_dist,
        )

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ─── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    print("=== Module 5: Face Mark Renderer Test ===\n")

    test_img = Path(__file__).parent.parent / "data" / "generated_faces" / "generated_1.png"
    if not test_img.exists():
        print(f"No test image at {test_img}")
    else:
        test_marks = [
            {"type": "deep scar", "location": "below left eye",
             "descriptors": ["deep", "vertical"], "raw": "deep scar below left eye"},
            {"type": "small mole", "location": "right cheek",
             "descriptors": ["small", "dark"], "raw": "small dark mole on right cheek"},
            {"type": "mole", "location": "chin",
             "descriptors": [], "raw": "mole on chin"},
        ]

        result = add_marks_to_face(str(test_img), test_marks)
        out_path = test_img.parent / "test_marks.png"
        result.save(str(out_path))
        print(f"\n  Saved: {out_path}")
        print(f"  Open it to verify mark placement!")
