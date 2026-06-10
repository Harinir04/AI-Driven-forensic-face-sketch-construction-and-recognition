"""
AdaFace ONNX wrapper — second face-recognition backbone for the v5
matcher's score-level fusion.

AdaFace (CVPR 2022, mk-minchul/AdaFace) uses a quality-adaptive
margin loss, which makes it noticeably more robust than ArcFace on
low-quality / synthetic faces — exactly the regime our SD/FLUX
generated queries land in. It pairs well with ArcFace because the
two models make uncorrelated mistakes; averaging their cosine
similarities reliably outperforms either alone.

Behavior:
  • First call to ``load()`` checks for the ONNX file at
    ``config.ADAFACE_MODEL_PATH``; if missing, it downloads from
    ``config.ADAFACE_DOWNLOAD_URL`` (UniFace release asset).
  • If the download fails, ``AdaFaceUnavailable`` is raised so the
    matcher can fall back to ArcFace-only with a single log line.
  • Input alignment uses the standard 5-point ArcFace template, so
    the matcher reuses the already-aligned 112×112 crop produced
    by ``OnnxFaceProcessor``. No double-warping.

Preprocessing matches the official AdaFace inference:
    (BGR / 255 - 0.5) / 0.5  ==  (BGR - 127.5) / 127.5
which happens to be identical to ArcFace's normalization, so we
can reuse the same aligned blob.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class AdaFaceUnavailable(Exception):
    """Raised when the AdaFace ONNX model cannot be loaded.

    The matcher catches this and continues with ArcFace-only —
    no surfacing to the user, just a log line. AdaFace is an
    accuracy boost, not a hard requirement.
    """


class AdaFaceProcessor:
    """Pure-ONNX AdaFace IR50 inference. 512-d output, no torch."""

    def __init__(self, model_path: str | Path = None):
        self.model_path = Path(model_path or config.ADAFACE_MODEL_PATH)
        self._session = None
        self._input_name = None
        self._output_idx = 0  # which output is the embedding

    def is_available(self) -> bool:
        """Cheap pre-check: model file exists and ORT can probably load it."""
        return self.model_path.exists() and self.model_path.stat().st_size > 1024 * 1024

    def load(self):
        """Load the ONNX session, downloading the model if needed."""
        if self._session is not None:
            return

        if not self.model_path.exists():
            self._download_model()

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise AdaFaceUnavailable(f"onnxruntime not installed: {e}") from e

        try:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
        except Exception as e:
            raise AdaFaceUnavailable(f"Could not load AdaFace ONNX: {e}") from e

    def _download_model(self):
        """Download the AdaFace ONNX from UniFace release assets."""
        url = config.ADAFACE_DOWNLOAD_URL
        if not url:
            raise AdaFaceUnavailable("No ADAFACE_DOWNLOAD_URL configured.")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import requests
        except ImportError as e:
            raise AdaFaceUnavailable(f"requests not installed: {e}") from e

        print(f"  [adaface] Downloading AdaFace IR50 model from {url} ...")
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(self.model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  [adaface] {pct}%", end="", flush=True)
            print()
        except Exception as e:
            # Don't leave a half-written file lying around.
            try:
                self.model_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise AdaFaceUnavailable(f"AdaFace download failed: {e}") from e

        if not self.model_path.exists() or self.model_path.stat().st_size < 1024 * 1024:
            try:
                self.model_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise AdaFaceUnavailable("AdaFace download produced an invalid file.")

    def unload(self):
        self._session = None

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Run AdaFace on a pre-aligned 112×112 BGR face crop.

        Returns a raw (un-normalized) 512-d embedding. Matcher does the
        L2 normalization once, after averaging across all augmentations.
        """
        if self._session is None:
            self.load()

        # Same normalization as ArcFace — ((x/255)-0.5)/0.5 == (x-127.5)/127.5
        blob = (aligned_bgr.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1,3,112,112]

        outputs = self._session.run(None, {self._input_name: blob})

        # Some AdaFace ONNX exports return (embedding, norm); some return
        # just embedding. Pick the 512-d vector either way.
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[-1] == 512:
                return arr[0].astype(np.float32)
            if arr.ndim == 1 and arr.shape[0] == 512:
                return arr.astype(np.float32)

        # Fallback: take the largest output and assume it's the embedding.
        # (Should never trigger for the UniFace export, which is single-output.)
        arr = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        if arr.size != 512:
            raise AdaFaceUnavailable(
                f"Unexpected AdaFace output shape: {[np.asarray(o).shape for o in outputs]}"
            )
        return arr

    def embed_aligned_with_flip(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embedding averaged with its horizontally-flipped twin (TTA)."""
        import cv2
        e1 = self.embed_aligned(aligned_bgr)
        e2 = self.embed_aligned(cv2.flip(aligned_bgr, 1))
        return (e1 + e2) * 0.5
