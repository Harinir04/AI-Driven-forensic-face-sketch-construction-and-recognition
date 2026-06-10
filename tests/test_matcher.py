"""
Sanity tests for face matcher v5.

Run:  python tests/test_matcher.py

These are not pytest-driven — they use plain assertions so the
suite runs in any environment without extra deps.

Tests that require real face images (identity self-match, swapped
identity) fall back to a clear SKIP message rather than failing
when the criminal_db is empty.
"""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from modules.face_matcher import FaceMatcher
from modules import match_attribute as ma


GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"


def _ok(msg):    print(f"{GREEN}PASS{RESET}  {msg}")
def _skip(msg):  print(f"{YELLOW}SKIP{RESET}  {msg}")
def _fail(msg):
    print(f"{RED}FAIL{RESET}  {msg}")
    raise AssertionError(msg)


# ──────────────────────────────────────────────────────────────
# Test 1 — empty DB returns structured "db_empty" status (no exception).
# ──────────────────────────────────────────────────────────────
def test_empty_db_status():
    """Empty DB → status='db_empty', no exception. Cache must not leak in."""
    from modules import face_matcher as fm

    m = FaceMatcher()
    with tempfile.TemporaryDirectory() as tmp:
        # Redirect both the criminal_db dir AND the cache path so a real
        # cache from a prior run can't masquerade as an empty DB.
        orig_cdb = config.CRIMINAL_DB_DIR
        orig_cache = fm._CACHE_PATH
        config.CRIMINAL_DB_DIR = Path(tmp)
        fm._CACHE_PATH = Path(tmp) / "_no_cache.pkl"

        try:
            tiny = Path(tmp) / "_no_image.txt"
            tiny.write_text("not an image")
            res = m.match(str(tiny))
            assert isinstance(res, dict), "match() must return a dict"
            assert res["status"] == "db_empty", \
                f"expected db_empty, got {res['status']}"
            assert res["results"] == []
        finally:
            config.CRIMINAL_DB_DIR = orig_cdb
            fm._CACHE_PATH = orig_cache

    _ok("empty DB returns structured status (no exception)")


# ──────────────────────────────────────────────────────────────
# Test 2 — AdaFace missing falls through gracefully.
# ──────────────────────────────────────────────────────────────
def test_adaface_missing_fallback():
    m = FaceMatcher()
    m._adaface_disabled = True   # simulate failed load
    m.adaface = None
    m.load_models()
    assert m.arcface is not None, "ArcFace must still load when AdaFace is unavailable"
    assert m.adaface is None
    _ok("AdaFace unavailable -> ArcFace-only loads cleanly")


# ──────────────────────────────────────────────────────────────
# Test 3 — score_attributes never raises on weird inputs.
# ──────────────────────────────────────────────────────────────
def test_attribute_scoring_robust():
    score, _, _ = ma.score_attributes({}, {})
    assert 0 <= score <= 100

    score, _, _ = ma.score_attributes(
        {"gender": "male", "marks": [{"type": "mole", "location": "left cheek"}]},
        {"gender": "male", "marks": [{"type": "mole", "location": "left cheekbone"}]},
    )
    assert score > 60, f"Same gender + same-zone mole should score high, got {score}"

    # Cross-zone mole must NOT match.
    score, _, _ = ma.score_attributes(
        {"gender": "male", "marks": [{"type": "mole", "location": "left cheek"}]},
        {"gender": "male", "marks": [{"type": "mole", "location": "forehead"}]},
    )
    # Marks weight 30 / total ~45 → no-mark-match drops total significantly.
    assert score < 80, f"Cross-zone mole should not score as full match, got {score}"

    # Garbage values don't crash.
    score, _, _ = ma.score_attributes(
        {"age": "old man", "gender": 42, "marks": "unknown"},
        {"age": 30, "gender": "male", "marks": None},
    )
    assert 0 <= score <= 100
    _ok("attribute scorer is robust to malformed inputs and zone-aware on marks")


# ──────────────────────────────────────────────────────────────
# Test 4 — ethnicity gate behavior.
# ──────────────────────────────────────────────────────────────
def test_ethnicity_gate():
    assert ma.ethnicity_compatible("South Indian Tamil", "Tamil") == "ok"
    assert ma.ethnicity_compatible("South Indian", "Punjabi") == "ok"        # both south_asian
    assert ma.ethnicity_compatible("South Indian", "European") == "reject"
    assert ma.ethnicity_compatible("Arab", "South Indian") == "soft"          # adjacent
    assert ma.ethnicity_compatible(None, "European") == "soft"                # missing side
    _ok("ethnicity gate groups + adjacency logic correct")


# ──────────────────────────────────────────────────────────────
# Test 5 — mark zone classifier.
# ──────────────────────────────────────────────────────────────
def test_zone_classifier():
    assert ma.classify_mark_zone("below left eye") == "left_eye"
    assert ma.classify_mark_zone("LEFT cheekbone") == "left_cheek"
    assert ma.classify_mark_zone("upper lip area") == "lip"
    assert ma.classify_mark_zone("middle of forehead") == "forehead"
    assert ma.classify_mark_zone("tip of chin") == "chin"
    assert ma.classify_mark_zone("") is None
    _ok("mark zone classifier handles common phrasings")


# ──────────────────────────────────────────────────────────────
# Test 6 — identity self-match (requires at least one DB image).
# ──────────────────────────────────────────────────────────────
def test_identity_self_match():
    images = sorted(
        list(config.CRIMINAL_DB_DIR.glob("*.jpg"))
        + list(config.CRIMINAL_DB_DIR.glob("*.jpeg"))
        + list(config.CRIMINAL_DB_DIR.glob("*.png"))
    )
    if not images:
        _skip("identity self-match: no images in criminal_db")
        return

    m = FaceMatcher()
    if not m.load_database():
        _skip("identity self-match: DB build produced no entries")
        return

    target = images[0]
    res = m.match(str(target))
    assert res["status"] in ("ok", "low_quality")
    assert res["results"], "expected at least one result on a real DB photo query"
    top = res["results"][0]
    # Exact-image self-match should score very high structurally.
    assert top["structural_score"] >= 85, (
        f"self-match structural score too low: {top['structural_score']} "
        f"for {target.name} → top {top['name']}"
    )
    _ok(f"identity self-match: {top['name']} structural={top['structural_score']:.1f}")


# ──────────────────────────────────────────────────────────────
def main():
    print("=== matcher v5 sanity tests ===\n")
    fns = [
        test_empty_db_status,
        test_adaface_missing_fallback,
        test_attribute_scoring_robust,
        test_ethnicity_gate,
        test_zone_classifier,
        test_identity_self_match,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
        except AssertionError:
            failures += 1
        except Exception as e:
            failures += 1
            print(f"{RED}ERROR{RESET}  {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
    print()
    if failures == 0:
        print(f"{GREEN}All checks passed.{RESET}")
    else:
        print(f"{RED}{failures} failure(s).{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
