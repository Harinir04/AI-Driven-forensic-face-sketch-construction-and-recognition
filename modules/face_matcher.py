"""
Face Matcher v5 — Layered, fail-soft forensic matching.

Pipeline (every layer degrades gracefully — no layer can crash the route):

    L0  query face quality gate
        - run detection on every candidate image (4 generated faces, or 1
          uploaded sketch/photo). Drop candidates with no face. Compute a
          per-candidate quality score (det confidence + face area +
          landmark frontality).

    L1  multi-image, multi-model embedding
        - for each surviving candidate produce an aligned 112×112 crop
        - ArcFace embedding + horizontally-flipped twin (TTA)
        - AdaFace embedding + flipped twin (when available)
        - average across all candidates and flips per model
        - L2-normalize the resulting query vector for each model

    L2  cosine similarity over per-model L2-normalized DB matrix
        - vectorized numpy dot product. (FAISS deferred until DB grows.)

    L3  quality-aware score-level fusion
        - structural = w_arc·s_arc + w_ada·s_ada
        - on poor query quality the AdaFace weight is bumped (its training
          objective is quality-adaptive). On AdaFace-unavailable the weight
          is renormalized to ArcFace = 1.0 silently.

    L4  hard gates
        - gender: if both witness and DB have explicit, opposite values → drop
        - age:    |Δ| > AGE_HARD_GATE_DELTA → drop
        - ethnicity: distant groups (e.g., south_asian vs european) → drop
        Entries with missing labels survive every gate.

    L5  attribute soft scoring
        - delegated to ``match_attribute.score_attributes``

    L6  confidence tiering
        - final = α·structural + β·attribute, where (α, β) adapts to query
          quality and number of attributes the witness gave.
        - tier := HIGH (≥ TIER_HIGH) / MEDIUM / LOW / NO_MATCH

The public ``match`` method always returns a structured dict with a
``status`` field — never ``None``, never raises out of the route. The
flask handler can render an explicit reason ("db empty", "no face in
query", "low quality") instead of an empty list with no explanation.
"""

from __future__ import annotations

import sys
import time
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from modules.onnx_face import (
    OnnxFaceProcessor, ensure_models, compute_face_quality,
)
from modules import match_attribute as ma

# AdaFace is optional. Importing the module is cheap; loading the
# session lazily lets us catch download failures.
try:
    from modules.adaface_onnx import AdaFaceProcessor, AdaFaceUnavailable
    _adaface_import_ok = True
except Exception:
    _adaface_import_ok = False
    AdaFaceProcessor = None
    class AdaFaceUnavailable(Exception):
        pass


# ─── Constants ────────────────────────────────────────────────

_CACHE_PATH = config.MODELS_DIR / "db_embeddings_v5.pkl"
_TIER_NO_MATCH = "NO_MATCH"
_TIER_LOW = "LOW"
_TIER_MEDIUM = "MEDIUM"
_TIER_HIGH = "HIGH"


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def _to_bgr(image_input) -> np.ndarray | None:
    """Normalize any supported input into a BGR numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        return img
    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(image_input, np.ndarray):
        return image_input
    return None


# ─── FaceMatcher ──────────────────────────────────────────────

class FaceMatcher:
    """Forensic face matcher v5. Stateless after load."""

    def __init__(self):
        self.arcface: OnnxFaceProcessor | None = None
        self.adaface: AdaFaceProcessor | None = None
        self._adaface_disabled = False  # tripped by failed load

        # DB index: parallel arrays so np.matmul stays simple.
        self.db_entries: list[dict] = []      # metadata per entry
        self._arc_matrix: np.ndarray | None = None  # (N, 512) L2-normalized
        self._ada_matrix: np.ndarray | None = None  # (N, 512) L2-normalized or None

    # ─── Model lifecycle ─────────────────────────────────────

    def load_models(self):
        """Load ArcFace (always) and AdaFace (best-effort)."""
        if self.arcface is None:
            print("  [matcher] Loading ArcFace (RetinaFace + w600k_r50)...")
            try:
                ensure_models()
            except Exception as e:
                # Detection model might still be present locally even if
                # ensure_models() failed (e.g. offline). Proceed and let
                # the session raise if truly missing.
                print(f"  [matcher] ensure_models warning: {e}")
            self.arcface = OnnxFaceProcessor()
            t = time.time()
            self.arcface.load()
            print(f"  [matcher] ArcFace ready ({time.time() - t:.1f}s)")

        if (
            config.USE_ADAFACE
            and _adaface_import_ok
            and self.adaface is None
            and not self._adaface_disabled
        ):
            print("  [matcher] Loading AdaFace (second backbone)...")
            try:
                ada = AdaFaceProcessor()
                t = time.time()
                ada.load()
                self.adaface = ada
                print(f"  [matcher] AdaFace ready ({time.time() - t:.1f}s)")
            except AdaFaceUnavailable as e:
                self._adaface_disabled = True
                print(f"  [matcher] AdaFace unavailable, running ArcFace-only ({e})")
            except Exception as e:
                self._adaface_disabled = True
                print(f"  [matcher] AdaFace load error, running ArcFace-only ({e})")

    def unload_models(self):
        if self.arcface:
            self.arcface.unload(); self.arcface = None
        if self.adaface:
            self.adaface.unload(); self.adaface = None
        import gc; gc.collect()

    # ─── Public legacy alias ─────────────────────────────────

    def load_model(self):
        """Back-compat alias — old code calls this."""
        self.load_models()

    def get_embedding(self, image_input) -> np.ndarray | None:
        """Back-compat: return ArcFace embedding only."""
        self.load_models()
        if self.arcface is None:
            return None
        return self.arcface.get_face_embedding(image_input)

    # ─── Database building ───────────────────────────────────

    def _load_db_attributes(self) -> dict:
        """Load attribute_json + name from SQLite, keyed by image stem."""
        try:
            from extensions import get_db
            conn = get_db()
            rows = conn.execute(
                "SELECT name, face_image_path, attributes_json "
                "FROM criminals WHERE face_image_path IS NOT NULL"
            ).fetchall()
            conn.close()
            out: dict[str, dict] = {}
            for r in rows:
                attrs = {}
                if r["attributes_json"]:
                    try:
                        attrs = json.loads(r["attributes_json"])
                    except Exception:
                        attrs = {}
                out[Path(r["face_image_path"]).stem] = {
                    "name": r["name"], "attributes": attrs,
                }
            return out
        except Exception as e:
            print(f"  [matcher] SQLite attribute load failed: {e}")
            return {}

    def build_database(self, db_dir: str | Path = None) -> int:
        """Embed every criminal photo. Returns number of entries indexed."""
        db_dir = Path(db_dir or config.CRIMINAL_DB_DIR)
        images = sorted(
            list(db_dir.glob("*.jpg"))
            + list(db_dir.glob("*.jpeg"))
            + list(db_dir.glob("*.png"))
        )
        if not images:
            print(f"  [matcher] No criminal images in {db_dir}.")
            self.db_entries = []
            self._arc_matrix = None
            self._ada_matrix = None
            return 0

        self.load_models()
        db_attrs = self._load_db_attributes()

        entries: list[dict] = []
        arc_vecs: list[np.ndarray] = []
        ada_vecs: list[np.ndarray] = []
        ada_ok = self.adaface is not None

        print(f"  [matcher] Building DB from {len(images)} images "
              f"(adaface={'on' if ada_ok else 'off'})...")
        t0 = time.time()

        ok = 0; fail = 0
        for path in images:
            try:
                fd = self.arcface.get_face_data_full(str(path))
            except Exception as e:
                print(f"    [matcher] {path.name}: detect error ({e})")
                fail += 1
                continue

            if fd is None:
                print(f"    [matcher] {path.name}: no face detected")
                fail += 1
                continue

            arc_emb = _l2_normalize(fd["embedding"].astype(np.float32))
            ada_emb = None
            if ada_ok:
                try:
                    ada_emb = _l2_normalize(
                        self.adaface.embed_aligned(fd["aligned"]).astype(np.float32)
                    )
                except Exception as e:
                    # First failure → disable for the rest of the build.
                    print(f"  [matcher] AdaFace runtime error, disabling: {e}")
                    self._adaface_disabled = True
                    self.adaface = None
                    ada_ok = False
                    ada_emb = None

            stem = path.stem
            attrs = {}
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        attrs = json.load(f)
                except Exception:
                    attrs = {}
            elif stem in db_attrs:
                attrs = db_attrs[stem]["attributes"] or {}

            name = db_attrs[stem]["name"] if stem in db_attrs else stem

            entries.append({
                "name": name,
                "image_path": str(path),
                "attributes": attrs,
                "landmarks_5": fd["landmarks_5"].tolist(),
                "img_h": fd["img_h"], "img_w": fd["img_w"],
                "det_score": fd["det_score"],
            })
            arc_vecs.append(arc_emb)
            if ada_emb is not None:
                ada_vecs.append(ada_emb)
            ok += 1

        if not entries:
            print(f"  [matcher] DB build produced 0 entries ({fail} failed).")
            self.db_entries = []
            self._arc_matrix = None
            self._ada_matrix = None
            return 0

        self.db_entries = entries
        self._arc_matrix = np.stack(arc_vecs, axis=0).astype(np.float32)
        if ada_vecs and len(ada_vecs) == len(entries):
            self._ada_matrix = np.stack(ada_vecs, axis=0).astype(np.float32)
        else:
            self._ada_matrix = None

        elapsed = time.time() - t0
        with_attrs = sum(1 for e in entries if e["attributes"])
        print(f"  [matcher] DB built: {ok} ok, {fail} failed, "
              f"{with_attrs} with attrs, ada={'yes' if self._ada_matrix is not None else 'no'} "
              f"({elapsed:.1f}s)")

        self._save_cache()
        return ok

    def _save_cache(self):
        try:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            blob = {
                "version": 5,
                "entries": self.db_entries,
                "arc_matrix": self._arc_matrix,
                "ada_matrix": self._ada_matrix,
            }
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump(blob, f)
            print(f"  [matcher] Cache saved → {_CACHE_PATH.name}")
        except Exception as e:
            print(f"  [matcher] Cache save failed: {e}")

    def load_database(self) -> bool:
        """Load cached DB, rebuild if missing/corrupt. Always returns boolean."""
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, "rb") as f:
                    blob = pickle.load(f)
                if isinstance(blob, dict) and blob.get("version") == 5:
                    self.db_entries = blob["entries"]
                    self._arc_matrix = blob.get("arc_matrix")
                    self._ada_matrix = blob.get("ada_matrix")
                    print(f"  [matcher] Loaded cache: {len(self.db_entries)} entries "
                          f"(ada={'yes' if self._ada_matrix is not None else 'no'})")
                    return len(self.db_entries) > 0
            except Exception as e:
                print(f"  [matcher] Cache corrupt ({e}), rebuilding...")

        # Cache miss → build from images on disk.
        image_count = sum(
            1 for _ in (
                list(config.CRIMINAL_DB_DIR.glob("*.jpg"))
                + list(config.CRIMINAL_DB_DIR.glob("*.jpeg"))
                + list(config.CRIMINAL_DB_DIR.glob("*.png"))
            )
        )
        if image_count == 0:
            print(f"  [matcher] DB empty: no images in {config.CRIMINAL_DB_DIR}")
            return False

        return self.build_database() > 0

    # ─── Query embedding (multi-image + flip TTA) ────────────

    def _embed_query(
        self, query_inputs: list, allow_flip: bool,
    ) -> dict:
        """
        Build the per-model query embedding from one or more candidate
        images. Returns a dict with averaged + L2-normalized vectors,
        the aggregate quality, and a status code.

        status:
          "ok"           — at least one face detected, embedding usable
          "no_face"      — no face in any candidate
          "low_quality"  — face(s) detected but every candidate below QUALITY_FLOOR
        """
        self.load_models()

        arc_pool: list[np.ndarray] = []
        ada_pool: list[np.ndarray] = []
        qualities: list[float] = []
        face_count = 0

        for inp in query_inputs:
            bgr = _to_bgr(inp)
            if bgr is None:
                continue
            try:
                fd = self.arcface.get_face_data_full(bgr)
            except Exception as e:
                print(f"  [matcher] query detect error: {e}")
                continue
            if fd is None:
                continue
            face_count += 1

            q = compute_face_quality(fd)
            qualities.append(q)

            aligned = fd["aligned"]
            try:
                arc = self.arcface.embed_aligned(aligned)
                arc_pool.append(arc)
                if allow_flip and config.USE_FLIP_TTA:
                    arc_pool.append(self.arcface.embed_aligned(cv2.flip(aligned, 1)))
            except Exception as e:
                print(f"  [matcher] ArcFace embed error: {e}")

            if self.adaface is not None and not self._adaface_disabled:
                try:
                    ada = self.adaface.embed_aligned(aligned)
                    ada_pool.append(ada)
                    if allow_flip and config.USE_FLIP_TTA:
                        ada_pool.append(self.adaface.embed_aligned(cv2.flip(aligned, 1)))
                except Exception as e:
                    print(f"  [matcher] AdaFace embed error, disabling: {e}")
                    self._adaface_disabled = True
                    self.adaface = None

        if face_count == 0:
            return {"status": "no_face", "arc": None, "ada": None, "quality": 0.0}

        # Mean quality across detected candidates.
        quality = float(np.mean(qualities)) if qualities else 0.0

        # Keep usable but low-quality query — caller will cap tier.
        arc_vec = None
        if arc_pool:
            arc_vec = _l2_normalize(np.mean(np.stack(arc_pool, axis=0), axis=0)).astype(np.float32)

        ada_vec = None
        if ada_pool:
            ada_vec = _l2_normalize(np.mean(np.stack(ada_pool, axis=0), axis=0)).astype(np.float32)

        status = "ok" if quality >= config.QUALITY_FLOOR else "low_quality"
        return {"status": status, "arc": arc_vec, "ada": ada_vec, "quality": quality}

    # ─── Hard gates ──────────────────────────────────────────

    def _passes_hard_gates(self, query_attrs: dict, db_attrs: dict) -> tuple[bool, list[str]]:
        """Return (kept, list_of_passed_gate_names). On any reject → (False, _)."""
        passed: list[str] = []
        if not query_attrs:
            return True, passed

        # Gender — only filter when both sides have explicit values.
        q_g = (query_attrs.get("gender") or "").strip().lower() or None
        d_g = (db_attrs.get("gender") or "").strip().lower() or None
        if q_g and d_g and q_g != d_g:
            return False, passed
        if q_g:
            passed.append("gender")

        # Age
        q_age = query_attrs.get("age")
        d_age = db_attrs.get("age")
        if q_age and d_age:
            if not ma.age_gate(q_age, d_age, config.AGE_HARD_GATE_DELTA):
                return False, passed
            passed.append("age")

        # Ethnicity
        q_eth = query_attrs.get("ethnicity")
        d_eth = db_attrs.get("ethnicity")
        if q_eth and d_eth:
            rel = ma.ethnicity_compatible(str(q_eth), str(d_eth))
            if rel == "reject":
                return False, passed
            passed.append(f"ethnicity:{rel}")

        return True, passed

    # ─── Score-level fusion + tiering ────────────────────────

    @staticmethod
    def _tier(score: float) -> str:
        if score >= config.TIER_HIGH:    return _TIER_HIGH
        if score >= config.TIER_MEDIUM:  return _TIER_MEDIUM
        if score >= config.TIER_LOW:     return _TIER_LOW
        return _TIER_NO_MATCH

    def _structural_score(
        self, q_arc: np.ndarray | None, q_ada: np.ndarray | None,
        quality: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Return (per-entry structural score 0–100, info dict with weights).
        Either model may be missing — weights re-normalize to sum to 1.
        """
        N = len(self.db_entries)
        if N == 0:
            return np.zeros(0, dtype=np.float32), {"w_arc": 1.0, "w_ada": 0.0}

        arc_sims = None
        ada_sims = None

        if q_arc is not None and self._arc_matrix is not None:
            arc_sims = self._arc_matrix @ q_arc.astype(np.float32)
            arc_sims = np.clip(arc_sims, 0.0, 1.0)

        if q_ada is not None and self._ada_matrix is not None and len(self._ada_matrix) == N:
            ada_sims = self._ada_matrix @ q_ada.astype(np.float32)
            ada_sims = np.clip(ada_sims, 0.0, 1.0)

        # Resolve weights — if AdaFace absent, ArcFace gets full weight.
        w_arc = config.MODEL_WEIGHT_ARCFACE
        w_ada = config.MODEL_WEIGHT_ADAFACE if ada_sims is not None else 0.0
        # Quality-aware shift: lower quality → trust AdaFace more.
        if ada_sims is not None and quality < 0.55:
            w_arc, w_ada = 0.45, 0.55
        total = w_arc + w_ada
        if total <= 0:
            return np.zeros(N, dtype=np.float32), {"w_arc": 0.0, "w_ada": 0.0}
        w_arc, w_ada = w_arc / total, w_ada / total

        if arc_sims is None:
            fused = ada_sims if ada_sims is not None else np.zeros(N, dtype=np.float32)
        elif ada_sims is None:
            fused = arc_sims
        else:
            fused = w_arc * arc_sims + w_ada * ada_sims

        return (fused * 100.0).astype(np.float32), {
            "w_arc": float(w_arc), "w_ada": float(w_ada),
            "have_arc": arc_sims is not None,
            "have_ada": ada_sims is not None,
        }

    # ─── Public match() ──────────────────────────────────────

    def match(
        self,
        query_images,
        query_attributes: dict | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Match a witness-generated face (or sketch/photo) against the DB.

        Args:
            query_images: a single path / PIL / numpy image, OR a list of
                them. When multiple are passed, embeddings are averaged
                across all of them — much more stable than relying on
                one diffusion sample.
            query_attributes: parsed witness-attribute dict, or None.
            top_k: max results returned. Defaults to config.TOP_K_MATCHES.

        Returns:
            {
              "status":         "ok" | "db_empty" | "no_face" | "low_quality"
                                 | "attribute_only",
              "results":        list[dict]  (always present; may be empty),
              "query_quality":  float in [0,1],
              "tier_counts":    {"HIGH":, "MEDIUM":, "LOW":},
              "fusion":         {"w_arc":, "w_ada":, "have_arc":, "have_ada":},
              "filtered":       int,   # entries dropped by hard gates
              "elapsed":        float,
            }
        """
        t0 = time.time()
        top_k = top_k or config.TOP_K_MATCHES
        query_attributes = query_attributes or {}

        # Normalize input shape
        if not isinstance(query_images, list):
            query_inputs = [query_images]
        else:
            query_inputs = list(query_images) or []

        if not query_inputs:
            return self._empty("no_query", t0)

        # Lazy DB load
        if not self.db_entries:
            self.load_database()
        if not self.db_entries:
            return self._empty("db_empty", t0)

        # Sketches → no horizontal flip TTA (asymmetric strokes confuse it).
        # Heuristic: if the input is a PIL or path containing 'sketch' or 'lineart'.
        allow_flip = True
        for q in query_inputs:
            try:
                p = str(q).lower()
                if "sketch" in p or "lineart" in p:
                    allow_flip = False
                    break
            except Exception:
                pass

        # L0+L1 — embed query
        try:
            q = self._embed_query(query_inputs, allow_flip=allow_flip)
        except Exception as e:
            traceback.print_exc()
            print(f"  [matcher] query embedding crashed, falling back to attribute-only: {e}")
            q = {"status": "no_face", "arc": None, "ada": None, "quality": 0.0}

        results = self._rank(q, query_attributes, top_k)
        elapsed = time.time() - t0

        tier_counts = {_TIER_HIGH: 0, _TIER_MEDIUM: 0, _TIER_LOW: 0}
        for r in results:
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

        # Decide overall status. If nothing scored above TIER_LOW we
        # surface "no_match" explicitly so the UI can show "no reliable
        # match" rather than a silent empty list.
        if q["status"] == "no_face":
            overall = "attribute_only" if query_attributes else "no_face"
        elif q["status"] == "low_quality":
            overall = "low_quality"
        else:
            overall = "ok"
        if not results:
            overall = "no_match" if overall == "ok" else overall

        print(
            f"  [matcher] status={overall} q_quality={q['quality']:.2f} "
            f"results={len(results)} (HIGH={tier_counts[_TIER_HIGH]}, "
            f"MEDIUM={tier_counts[_TIER_MEDIUM]}, LOW={tier_counts[_TIER_LOW]}) "
            f"in {elapsed:.2f}s"
        )

        return {
            "status": overall,
            "results": results,
            "query_quality": q["quality"],
            "tier_counts": tier_counts,
            "elapsed": elapsed,
        }

    # ─── Internal: ranking ───────────────────────────────────

    def _rank(self, q: dict, query_attributes: dict, top_k: int) -> list[dict]:
        """Apply gates, score, fuse, tier. Returns list of result dicts."""
        N = len(self.db_entries)
        if N == 0:
            return []

        # L2/L3 — structural per entry
        struct_scores, fusion_info = self._structural_score(
            q["arc"], q["ada"], q["quality"],
        )

        # L4 — gates
        eligible: list[int] = []
        gates_per_entry: list[list[str]] = [[] for _ in range(N)]
        filtered = 0
        for i, entry in enumerate(self.db_entries):
            ok, passed = self._passes_hard_gates(query_attributes, entry["attributes"] or {})
            if not ok:
                filtered += 1
                continue
            gates_per_entry[i] = passed
            eligible.append(i)

        if not eligible:
            return []

        # Adaptive blend weights:
        #   • more witness attributes → trust attributes more
        #   • lower query quality     → trust attributes more (cap at 0.8)
        n_attrs = sum(1 for k, v in (query_attributes or {}).items()
                      if v not in (None, "", [], {}))
        alpha = config.STRUCTURAL_BASE_WEIGHT
        beta = config.ATTRIBUTE_BASE_WEIGHT
        if n_attrs >= 6:
            alpha, beta = 0.40, 0.60
        if q["quality"] < config.QUALITY_FLOOR:
            alpha, beta = 0.20, 0.80
        if q["arc"] is None and q["ada"] is None:
            # No face at all — attribute-only ranking.
            alpha, beta = 0.0, 1.0

        results: list[dict] = []
        for i in eligible:
            entry = self.db_entries[i]
            structural = float(struct_scores[i]) if struct_scores.size else 0.0

            # L5 — attributes
            try:
                attr_score, matched, mismatched = ma.score_attributes(
                    query_attributes, entry["attributes"] or {}
                )
            except Exception as e:
                print(f"  [matcher] attribute scoring error for {entry['name']}: {e}")
                attr_score, matched, mismatched = 50.0, [], []

            final = alpha * structural + beta * attr_score
            tier = self._tier(final)

            # Cap tier on low-quality queries — never claim HIGH on a
            # face we couldn't detect well.
            if q["status"] != "ok" and tier == _TIER_HIGH:
                tier = _TIER_MEDIUM

            results.append({
                "name": entry["name"],
                "final_score": round(final, 1),
                "structural_score": round(structural, 1),
                "attribute_score": round(attr_score, 1),
                "matched_attrs": matched,
                "mismatched_attrs": mismatched,
                "image_path": entry["image_path"],
                "tier": tier,
                "model_scores": {
                    "arcface_used": fusion_info.get("have_arc", False),
                    "adaface_used": fusion_info.get("have_ada", False),
                    "w_arc": fusion_info.get("w_arc", 1.0),
                    "w_ada": fusion_info.get("w_ada", 0.0),
                },
                "gates_passed": gates_per_entry[i],
            })

        # L6 — sort + filter
        results.sort(key=lambda r: r["final_score"], reverse=True)
        # Strict floor: anything that didn't reach TIER_LOW (50) is a
        # weak signal at best and surfacing it produces false positives
        # like "50yo query → 33yo DB entry @ 25 %". Drop them entirely
        # and let the caller render "no reliable match".
        kept = [r for r in results if r["tier"] != _TIER_NO_MATCH]
        return kept[:top_k]

    def _empty(self, status: str, t0: float) -> dict:
        return {
            "status": status,
            "results": [],
            "query_quality": 0.0,
            "tier_counts": {_TIER_HIGH: 0, _TIER_MEDIUM: 0, _TIER_LOW: 0},
            "elapsed": time.time() - t0,
        }


# ─── Quick smoke test ────────────────────────────────────────
if __name__ == "__main__":
    print("=== Face Matcher v5 smoke test ===")
    m = FaceMatcher()
    if m.load_database():
        print(f"DB: {len(m.db_entries)} entries.")
        sample = next(iter(config.GENERATED_DIR.glob("*.png")), None)
        if sample:
            res = m.match(str(sample))
            print(json.dumps({
                "status": res["status"],
                "query_quality": res["query_quality"],
                "top": res["results"][:2],
            }, indent=2, default=str))
    else:
        print("DB empty — add criminals via Admin panel and rebuild.")
