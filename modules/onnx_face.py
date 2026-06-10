"""
Pure ONNX Face Processor — Zero insightface dependency.

Uses RetinaFace (det_10g.onnx) for detection + 5-point landmarks,
and ArcFace (w600k_r50.onnx) for 512-d embedding extraction.

Dependencies: onnxruntime, numpy, opencv-python-headless, Pillow
No Cython, no scikit-learn, no ABI issues. Works on Python 3.10-3.14+.
"""

import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ─── Model paths ──────────────────────────────────────────────

_DEFAULT_MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"
_DET_MODEL_NAME = "det_10g.onnx"
_REC_MODEL_NAME = "w600k_r50.onnx"

# RetinaFace constants
_FPN_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2  # 2 anchors per grid position
_DET_SIZE = (640, 640)

# ArcFace alignment reference — standard 5-point landmarks
# for warping a detected face into 112×112 aligned crop
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class OnnxFaceProcessor:
    """
    Standalone face detection + recognition using ONNX models directly.
    Drop-in replacement for insightface FaceAnalysis — same models, no wrapper.
    """

    def __init__(self, model_dir: str | Path = None):
        self.model_dir = Path(model_dir or _DEFAULT_MODEL_DIR)
        self._det_session = None
        self._rec_session = None
        self._anchor_cache: dict[tuple, np.ndarray] = {}

    # ─── Model Loading ────────────────────────────────────────

    def _get_providers(self) -> list[str]:
        """Return available ONNX execution providers, preferring GPU."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def load_detection(self):
        """Load RetinaFace detection model."""
        if self._det_session is not None:
            return

        import onnxruntime as ort

        det_path = self.model_dir / _DET_MODEL_NAME
        if not det_path.exists():
            raise FileNotFoundError(
                f"Detection model not found: {det_path}\n"
                f"Download buffalo_l model pack from insightface and extract to {self.model_dir}"
            )

        self._det_session = ort.InferenceSession(
            str(det_path), providers=self._get_providers()
        )

    def load_recognition(self):
        """Load ArcFace recognition model."""
        if self._rec_session is not None:
            return

        import onnxruntime as ort

        rec_path = self.model_dir / _REC_MODEL_NAME
        if not rec_path.exists():
            raise FileNotFoundError(
                f"Recognition model not found: {rec_path}\n"
                f"Download buffalo_l model pack from insightface and extract to {self.model_dir}"
            )

        self._rec_session = ort.InferenceSession(
            str(rec_path), providers=self._get_providers()
        )

    def load(self):
        """Load both models."""
        t = time.time()
        self.load_detection()
        self.load_recognition()
        print(f"  [onnx_face] Models loaded in {time.time() - t:.1f}s")

    def unload(self):
        """Free memory."""
        self._det_session = None
        self._rec_session = None
        self._anchor_cache.clear()

    # ─── Face Detection (RetinaFace) ──────────────────────────

    def _generate_anchors(self, height: int, width: int) -> np.ndarray:
        """Generate anchor centers for all FPN levels."""
        cache_key = (height, width)
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]

        all_anchors = []
        for stride in _FPN_STRIDES:
            grid_h = height // stride
            grid_w = width // stride
            # Create grid of anchor centers
            yy, xx = np.mgrid[:grid_h, :grid_w]
            centers = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
            # Repeat for each anchor at this position
            centers = np.repeat(centers, _NUM_ANCHORS, axis=0)
            all_anchors.append((centers, stride))

        self._anchor_cache[cache_key] = all_anchors
        return all_anchors

    def _preprocess_det(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """Resize + normalize image for detection. Returns (blob, scale, padding)."""
        h, w = img_bgr.shape[:2]
        det_h, det_w = _DET_SIZE

        # Scale to fit within det_size, preserving aspect ratio
        scale = min(det_h / h, det_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))

        # Pad to det_size
        det_img = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        # Normalize: (img - 127.5) / 128.0, then CHW
        blob = (det_img.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

        return blob, scale, (new_h, new_w)

    def detect_faces(
        self,
        img_bgr: np.ndarray,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> list[dict]:
        """
        Detect faces in an image.

        Returns list of dicts: [{bbox, score, landmarks_5}, ...]
          bbox: [x1, y1, x2, y2] in original image coords
          landmarks_5: [[x,y], ...] 5 points in original image coords
        """
        self.load_detection()

        blob, scale, (pad_h, pad_w) = self._preprocess_det(img_bgr)
        det_h, det_w = _DET_SIZE

        # Run detection
        input_name = self._det_session.get_inputs()[0].name
        outputs = self._det_session.run(None, {input_name: blob})

        # Parse outputs: 9 arrays, 3 per FPN level (scores, boxes, landmarks)
        anchors = self._generate_anchors(det_h, det_w)

        all_scores = []
        all_boxes = []
        all_landmarks = []

        for level_idx, (centers, stride) in enumerate(anchors):
            scores = outputs[level_idx]            # [N, 1]
            boxes = outputs[level_idx + 3]         # [N, 4]
            lmks = outputs[level_idx + 6]          # [N, 10]

            # Filter by score
            mask = scores[:, 0] > score_thresh
            if not mask.any():
                continue

            filtered_scores = scores[mask, 0]
            filtered_boxes = boxes[mask]
            filtered_lmks = lmks[mask]
            filtered_centers = centers[mask]

            # Decode boxes: distance from anchor center
            # boxes are [left, top, right, bottom] distances from anchor center
            x1 = (filtered_centers[:, 0] - filtered_boxes[:, 0]) * stride
            y1 = (filtered_centers[:, 1] - filtered_boxes[:, 1]) * stride
            x2 = (filtered_centers[:, 0] + filtered_boxes[:, 2]) * stride
            y2 = (filtered_centers[:, 1] + filtered_boxes[:, 3]) * stride
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=1)

            # Decode landmarks: offset from anchor center
            decoded_lmks = np.zeros_like(filtered_lmks)
            for k in range(5):
                decoded_lmks[:, k * 2] = (filtered_centers[:, 0] + filtered_lmks[:, k * 2]) * stride
                decoded_lmks[:, k * 2 + 1] = (filtered_centers[:, 1] + filtered_lmks[:, k * 2 + 1]) * stride

            all_scores.append(filtered_scores)
            all_boxes.append(decoded_boxes)
            all_landmarks.append(decoded_lmks)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes)
        landmarks = np.concatenate(all_landmarks)

        # NMS
        keep = self._nms(boxes, scores, nms_thresh)
        scores = scores[keep]
        boxes = boxes[keep]
        landmarks = landmarks[keep]

        # Scale back to original image coordinates
        results = []
        for i in range(len(scores)):
            bbox = boxes[i] / scale
            lmk = landmarks[i].reshape(5, 2) / scale
            results.append({
                "bbox": bbox.astype(np.float32),
                "score": float(scores[i]),
                "landmarks_5": lmk.astype(np.float32),
            })

        # Sort by face area (largest first)
        results.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
        """Non-maximum suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]

        return keep

    # ─── Face Alignment ───────────────────────────────────────

    @staticmethod
    def align_face(img_bgr: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
        """
        Warp a detected face into 112×112 aligned crop using 5-point landmarks.
        Uses the standard ArcFace alignment transform.
        """
        src = landmarks_5.astype(np.float32)
        dst = _ARCFACE_DST.copy()

        # Estimate similarity transform (scale + rotation + translation)
        tform = _estimate_similarity_transform(src, dst)
        aligned = cv2.warpAffine(img_bgr, tform, (112, 112), borderValue=0)
        return aligned

    # ─── Face Recognition (ArcFace) ───────────────────────────

    def get_embedding(self, img_bgr: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
        """
        Extract 512-d ArcFace embedding from a face.

        Args:
            img_bgr: Full image (BGR)
            landmarks_5: 5-point face landmarks from detect_faces()

        Returns:
            512-d normalized embedding
        """
        self.load_recognition()

        # Align face
        aligned = self.align_face(img_bgr, landmarks_5)

        # Preprocess for ArcFace: (img - 127.5) / 127.5, CHW
        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 112, 112]

        # Run recognition
        input_name = self._rec_session.get_inputs()[0].name
        embedding = self._rec_session.run(None, {input_name: blob})[0][0]

        return embedding

    # ─── High-Level API ───────────────────────────────────────

    def get_face_embedding(self, image_input) -> Optional[np.ndarray]:
        """
        Detect the largest face in an image and return its 512-d embedding.

        Args:
            image_input: file path (str/Path), PIL Image, or BGR numpy array

        Returns:
            512-d embedding or None if no face found
        """
        # Convert input to BGR numpy array
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                print(f"  [onnx_face] Could not read image: {image_input}")
                return None
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        # Detect faces
        faces = self.detect_faces(img_bgr, score_thresh=0.3)
        if not faces:
            return None

        # Use largest face
        face = faces[0]
        return self.get_embedding(img_bgr, face["landmarks_5"])

    def get_all_face_data(self, image_input) -> Optional[dict]:
        """
        Detect the largest face and return embedding + metadata.

        Returns dict with: embedding, bbox, landmarks, score
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                return None
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        faces = self.detect_faces(img_bgr, score_thresh=0.3)
        if not faces:
            return None

        face = faces[0]
        embedding = self.get_embedding(img_bgr, face["landmarks_5"])

        return {
            "embedding": embedding,
            "bbox": face["bbox"],
            "landmarks_5": face["landmarks_5"],
            "det_score": face["score"],
        }

    # ─── v5 helpers ──────────────────────────────────────────
    # The matcher v5 needs the aligned 112×112 crop, the embedding,
    # and a per-face quality score so it can weight candidates
    # by how trustworthy each detection is.

    def get_face_data_full(self, image_input) -> Optional[dict]:
        """
        Run the full pipeline and return everything the matcher
        needs in one call. None if no face detected.

        Returns:
            {
                "img_bgr": original BGR image,
                "aligned": 112×112 aligned face crop (BGR),
                "embedding": 512-d ArcFace embedding (raw, not L2-normalized),
                "bbox": [x1,y1,x2,y2] in original coords,
                "landmarks_5": 5×2 array in original coords,
                "det_score": detection confidence (0–1),
                "img_h": height, "img_w": width,
            }
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                return None
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            return None

        faces = self.detect_faces(img_bgr, score_thresh=0.3)
        if not faces:
            return None

        face = faces[0]
        aligned = self.align_face(img_bgr, face["landmarks_5"])

        # Embed the aligned crop directly (skip the redundant warp inside
        # get_embedding by calling the underlying ONNX session here).
        self.load_recognition()
        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        input_name = self._rec_session.get_inputs()[0].name
        embedding = self._rec_session.run(None, {input_name: blob})[0][0]

        h, w = img_bgr.shape[:2]
        return {
            "img_bgr": img_bgr,
            "aligned": aligned,
            "embedding": embedding,
            "bbox": face["bbox"],
            "landmarks_5": face["landmarks_5"],
            "det_score": float(face["score"]),
            "img_h": h,
            "img_w": w,
        }

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embed an already-aligned 112×112 BGR face crop. Returns raw 512-d vector."""
        self.load_recognition()
        blob = (aligned_bgr.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        input_name = self._rec_session.get_inputs()[0].name
        return self._rec_session.run(None, {input_name: blob})[0][0]

    def embed_aligned_with_flip(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        ArcFace embedding averaged with its horizontally-flipped twin.
        Cancels per-sample left/right asymmetry noise — standard TTA.
        """
        e1 = self.embed_aligned(aligned_bgr)
        e2 = self.embed_aligned(cv2.flip(aligned_bgr, 1))
        return (e1 + e2) * 0.5


def compute_face_quality(face_data: dict) -> float:
    """
    Quality score in [0, 1] for a detected face. Combines:
      • detection confidence
      • face area as a fraction of the full image (bigger = sharper)
      • frontality from 5-pt landmark symmetry (eyes level, nose centered)

    Higher = more trustworthy embedding. The matcher uses this both
    to drop very low-quality candidates and to bias the final blend
    toward attribute matching when the structural signal is weak.
    """
    if not face_data:
        return 0.0

    det = max(0.0, min(1.0, face_data.get("det_score", 0.0)))

    bbox = face_data["bbox"]
    face_w = max(1.0, bbox[2] - bbox[0])
    face_h = max(1.0, bbox[3] - bbox[1])
    face_area = face_w * face_h
    img_area = max(1.0, face_data["img_h"] * face_data["img_w"])
    # Saturate at 30 % image area — anything bigger is plenty.
    size_score = min(1.0, (face_area / img_area) / 0.30)

    lmk = face_data["landmarks_5"]
    eye_l, eye_r, nose, mouth_l, mouth_r = lmk[0], lmk[1], lmk[2], lmk[3], lmk[4]
    eye_dy = abs(eye_l[1] - eye_r[1])
    eye_dx = abs(eye_l[0] - eye_r[0]) + 1e-6
    # Eye-line tilt: 0 → perfectly level, 1 → 45 °
    tilt = min(1.0, eye_dy / eye_dx)
    # Nose centering between eyes
    eye_cx = (eye_l[0] + eye_r[0]) * 0.5
    nose_offset = abs(nose[0] - eye_cx) / eye_dx
    nose_offset = min(1.0, nose_offset)
    frontality = max(0.0, 1.0 - 0.5 * tilt - 0.5 * nose_offset)

    return float(0.45 * det + 0.30 * size_score + 0.25 * frontality)


# ─── Similarity Transform Helper ─────────────────────────────

def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate 2D similarity transform (scale + rotation + translation)
    from source points to destination points.

    Returns 2×3 affine matrix for cv2.warpAffine.
    """
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.zeros((num * dim, 2 * dim))
    b = np.zeros(num * dim)

    for i in range(num):
        A[i * 2, 0] = src_demean[i, 0]
        A[i * 2, 1] = -src_demean[i, 1]
        A[i * 2, 2] = 1
        A[i * 2, 3] = 0
        b[i * 2] = dst_demean[i, 0]

        A[i * 2 + 1, 0] = src_demean[i, 1]
        A[i * 2 + 1, 1] = src_demean[i, 0]
        A[i * 2 + 1, 2] = 0
        A[i * 2 + 1, 3] = 1
        b[i * 2 + 1] = dst_demean[i, 1]

    # Solve least squares
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    sr = result[0]  # scale * cos(theta)
    sc = result[1]  # scale * sin(theta)
    tx = result[2]
    ty = result[3]

    # Build 2×3 affine matrix
    M = np.array([
        [sr, -sc, tx + dst_mean[0] - sr * src_mean[0] + sc * src_mean[1]],
        [sc,  sr, ty + dst_mean[1] - sc * src_mean[0] - sr * src_mean[1]],
    ], dtype=np.float64)

    return M


# ─── Model Download Helper ───────────────────────────────────

def ensure_models(model_dir: str | Path = None) -> Path:
    """
    Check if buffalo_l ONNX models exist. If not, download them.
    Returns the model directory path.
    """
    model_dir = Path(model_dir or _DEFAULT_MODEL_DIR)
    det_path = model_dir / _DET_MODEL_NAME
    rec_path = model_dir / _REC_MODEL_NAME

    if det_path.exists() and rec_path.exists():
        return model_dir

    print(f"  [onnx_face] Models not found in {model_dir}")
    print(f"  [onnx_face] Downloading buffalo_l model pack (~330 MB)...")

    model_dir.mkdir(parents=True, exist_ok=True)

    import requests

    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    zip_path = model_dir.parent / "buffalo_l.zip"

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  [onnx_face] Downloading: {pct}%", end="", flush=True)

        print()

        # Extract only the ONNX files we need
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename.endswith(".onnx"):
                    target = model_dir / basename
                    if not target.exists():
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        print(f"  [onnx_face] Extracted: {basename}")

        zip_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"\n  [onnx_face] Download failed: {e}")
        print(f"  [onnx_face] Please manually download buffalo_l.zip from:")
        print(f"    {url}")
        print(f"  [onnx_face] Extract .onnx files to: {model_dir}")
        raise

    if not det_path.exists() or not rec_path.exists():
        raise FileNotFoundError(
            f"Models still missing after download. "
            f"Need: {_DET_MODEL_NAME}, {_REC_MODEL_NAME} in {model_dir}"
        )

    return model_dir


# ─── Quick Test ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ONNX Face Processor Test ===\n")

    ensure_models()
    proc = OnnxFaceProcessor()
    proc.load()

    # Test with generated faces
    test_dir = Path(__file__).parent.parent / "data" / "generated_faces"
    for img_path in sorted(test_dir.glob("*.png")):
        emb = proc.get_face_embedding(str(img_path))
        if emb is not None:
            print(f"  {img_path.name}: embedding shape={emb.shape}, norm={np.linalg.norm(emb):.2f}")
        else:
            print(f"  {img_path.name}: no face detected")
