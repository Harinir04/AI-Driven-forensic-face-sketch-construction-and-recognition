"""
Attribute scoring for face matcher v5.

Pulled out of ``face_matcher.py`` and rebuilt so that:

  • Marks are matched on **face zone** (forehead / cheek / chin / …),
    not on raw location strings. Witness phrasing varies wildly
    ("below left eye", "left under-eye area", "left cheekbone")
    and the old fixed synonym table missed most of them.
  • Ethnicity has explicit compatibility groups so a ``South Indian``
    query never gets paired with a ``European`` DB entry.
  • Skin tone uses the existing 3-band fuzzy match (verbatim) but
    is callable independently.
  • Every comparison is wrapped so a malformed/unexpected attribute
    value never blows up the match call — worst case the attribute
    contributes 0 weight.

Public API:
    score_attributes(query, db_attrs) -> (score_0_100, matched, mismatched)
    age_gate(q_age, d_age, delta) -> bool
    ethnicity_compatible(q_eth, d_eth) -> "ok" | "soft" | "reject"
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple


# ─── Weights ──────────────────────────────────────────────────
# Tuned for forensic context: distinguishing marks dominate;
# gross demographics (gender, ethnicity, age, skin) are next;
# styling (hair, facial hair) is moderate; soft features
# (face shape, nose, lips, build) are light because witness
# descriptions of these are unreliable.

ATTR_WEIGHTS = {
    "gender":           15,
    "age":              8,
    "ethnicity":        12,
    "skin_tone":        10,
    "hair_color":       8,
    "hair_style":       4,
    "facial_hair":      9,
    "eye_color":        5,
    "eye_shape":        2,
    "eyebrows":         2,
    "forehead":         2,
    "face_shape":       3,
    "nose":             3,
    "lips":             2,
    "chin":             2,
    "build":            2,
    "accessories":      4,
    "religious_attire": 6,
    "marks":            30,
}


# ─── Skin tone (kept verbatim from v4) ────────────────────────

def fuzzy_skin_match(query: str, db: str) -> float:
    """3-band fuzzy match for skin tones. Returns score in [0, 1]."""
    def get_group(tone: str) -> int:
        t = (tone or "").lower()
        if any(w in t for w in ("very dark", "dark")):
            return 0
        if any(w in t for w in ("brown", "olive", "tan", "medium", "wheat")):
            return 1
        if any(w in t for w in ("fair", "light", "pale")):
            return 2
        return 1

    diff = abs(get_group(query) - get_group(db))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.1


# ─── Ethnicity compatibility ──────────────────────────────────
# Coarse groups. Within a group: full credit. Adjacent groups
# (geographically/genetically plausible confusion): soft credit.
# Otherwise: hard reject for this attribute (does not by itself
# eject the candidate; the matcher applies that gate separately
# when configured to).

_ETH_GROUPS = {
    "south_asian": [
        "south indian", "tamil", "telugu", "malayali", "kannada",
        "north indian", "punjabi", "bengali", "marathi", "gujarati",
        "indian", "pakistani", "bangladeshi", "sri lankan", "nepali",
        "kashmiri", "rajasthani",
    ],
    "east_asian":  ["chinese", "japanese", "korean", "mongolian", "vietnamese"],
    "se_asian":    ["thai", "filipino", "indonesian", "malay", "burmese"],
    "mena":        ["arab", "middle eastern", "persian", "iranian", "turkish",
                    "north african", "egyptian", "moroccan"],
    "european":    ["european", "white", "caucasian", "scandinavian", "slavic",
                    "german", "british", "french", "spanish", "italian"],
    "african":     ["african", "black", "sub-saharan", "ethiopian", "nigerian",
                    "kenyan", "somali"],
    "latin":       ["latino", "hispanic", "mexican", "brazilian", "colombian"],
}

# Adjacency: which groups are plausibly confusable.
_ETH_ADJACENT = {
    "south_asian": {"mena"},
    "east_asian":  {"se_asian"},
    "se_asian":    {"east_asian", "south_asian"},
    "mena":        {"south_asian", "european"},
    "european":    {"mena", "latin"},
    "african":     set(),
    "latin":       {"european"},
}


def _ethnicity_group(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    best_group = None
    best_len = 0
    for group, terms in _ETH_GROUPS.items():
        for term in terms:
            if term in t and len(term) > best_len:
                best_group = group
                best_len = len(term)
    return best_group


def ethnicity_compatible(query: str | None, db: str | None) -> str:
    """
    Returns:
        "ok"     — same group, full credit
        "soft"   — adjacent / one side unknown, half credit, no reject
        "reject" — distant groups, hard reject signal
    """
    if not query or not db:
        return "soft"

    q = _ethnicity_group(query)
    d = _ethnicity_group(db)
    if q is None or d is None:
        return "soft"
    if q == d:
        return "ok"
    if d in _ETH_ADJACENT.get(q, set()):
        return "soft"
    return "reject"


# ─── Age gate ─────────────────────────────────────────────────

def age_gate(q_age, d_age, delta: int) -> bool:
    """True if ages are close enough not to reject the candidate."""
    try:
        q = int(q_age)
        d = int(d_age)
    except (TypeError, ValueError):
        return True  # missing → don't reject
    return abs(q - d) <= delta


# ─── Mark zones (the big upgrade) ─────────────────────────────
# Both query and DB mark locations are bucketed into one of these
# coarse face zones via lexicon. Same-zone matches count; cross-zone
# is treated as a different identifying mark (no credit). This is
# vastly more robust than v4's small synonym-set table.

_MARK_ZONES: dict[str, list[str]] = {
    "forehead":   ["forehead", "brow ridge", "between eyebrows", "above eyebrow",
                   "above the eyebrow", "above eye"],
    "left_eye":   ["below left eye", "under left eye", "near left eye",
                   "left eye corner", "left under-eye", "left under eye",
                   "left eyelid", "left brow", "left eyebrow",
                   "beneath left eye"],
    "right_eye":  ["below right eye", "under right eye", "near right eye",
                   "right eye corner", "right under-eye", "right under eye",
                   "right eyelid", "right brow", "right eyebrow",
                   "beneath right eye"],
    "nose":       ["nose", "nostril", "bridge of nose", "tip of nose",
                   "nose bridge", "side of nose"],
    "left_cheek": ["left cheek", "left cheekbone", "left side of face",
                   "left jaw", "left jawline", "left face", "left side cheek"],
    "right_cheek":["right cheek", "right cheekbone", "right side of face",
                   "right jaw", "right jawline", "right face", "right side cheek"],
    "lip":        ["upper lip", "lower lip", "above lip", "above mouth",
                   "below mouth", "near mouth", "philtrum", "near lip", "lip"],
    "chin":       ["chin", "below chin", "under chin", "left chin",
                   "right chin", "left side of chin", "right side of chin",
                   "tip of chin"],
    "ear_left":   ["left ear", "near left ear", "left earlobe"],
    "ear_right":  ["right ear", "near right ear", "right earlobe"],
    "neck":       ["neck", "throat", "below jaw"],
    "head":       ["head", "scalp", "back of head", "top of head"],
}

# Pre-compile a lookup that maps every phrase → zone, longest first
# so "left cheekbone" wins over "cheek" when both could match.
_PHRASE_TO_ZONE: list[tuple[str, str]] = sorted(
    [(phrase.lower(), zone) for zone, phrases in _MARK_ZONES.items() for phrase in phrases],
    key=lambda p: -len(p[0]),
)


def classify_mark_zone(location: str) -> str | None:
    """Map a free-text mark location to one of the canonical zones."""
    if not location:
        return None
    text = re.sub(r"\s+", " ", location.lower().strip())
    for phrase, zone in _PHRASE_TO_ZONE:
        if phrase in text:
            return zone

    # Fallback: lone keywords.
    if "forehead" in text:           return "forehead"
    if "cheek" in text and "left" in text:  return "left_cheek"
    if "cheek" in text and "right" in text: return "right_cheek"
    if "cheek" in text:               return "left_cheek"
    if "chin"  in text:               return "chin"
    if "nose"  in text:               return "nose"
    if "lip"   in text or "mouth" in text: return "lip"
    if "eye"   in text and "left" in text:  return "left_eye"
    if "eye"   in text and "right" in text: return "right_eye"
    if "eye"   in text:               return "left_eye"
    if "ear"   in text and "left" in text:  return "ear_left"
    if "ear"   in text and "right" in text: return "ear_right"
    if "neck"  in text:               return "neck"
    return None


def _normalize_marks(marks) -> list[dict]:
    """Coerce marks into a list of {type, location, zone} dicts."""
    out = []
    if not marks:
        return out
    if isinstance(marks, dict):
        marks = [marks]
    if isinstance(marks, str):
        marks = [marks]
    for m in marks:
        if isinstance(m, dict):
            mtype = (m.get("type") or "").lower().strip()
            mloc = (m.get("location") or "").lower().strip()
        else:
            mtype = ""
            mloc = str(m).lower().strip()
        out.append({
            "type": mtype,
            "location": mloc,
            "zone": classify_mark_zone(mloc),
        })
    return out


def _mark_type_compatible(a: str, b: str) -> bool:
    """Lenient mark-type check. 'mole' ≈ 'beauty mark'; 'scar' ≈ 'cut'."""
    if not a or not b:
        return True  # one side unspecified → don't penalize
    if a == b or a in b or b in a:
        return True
    aliases = [
        {"mole", "beauty mark", "freckle", "spot", "birthmark"},
        {"scar", "cut", "wound mark", "gash"},
        {"tattoo", "ink"},
        {"pimple", "acne", "blemish"},
    ]
    for group in aliases:
        if a in group and b in group:
            return True
    return False


def score_marks(q_marks, d_marks) -> tuple[float, list[str], list[str]]:
    """
    Returns (fraction_in_[0,1], matched_strings, mismatched_strings).
    """
    q = _normalize_marks(q_marks)
    d = _normalize_marks(d_marks)

    matched: list[str] = []
    mismatched: list[str] = []

    if not q:
        # Witness described no marks → neutral.
        return 0.5, matched, mismatched
    if not d:
        # Witness described marks, DB has none → strong negative.
        mismatched.append("Marks: query has marks, DB has none")
        return 0.0, matched, mismatched

    # Greedy bipartite match: each query mark finds the best DB mark.
    used_db = set()
    hits = 0
    for qm in q:
        for j, dm in enumerate(d):
            if j in used_db:
                continue
            zone_match = (
                qm["zone"] is not None and dm["zone"] is not None
                and qm["zone"] == dm["zone"]
            )
            type_match = _mark_type_compatible(qm["type"], dm["type"])
            if zone_match and type_match:
                hits += 1
                used_db.add(j)
                matched.append(f"Mark: {dm['type'] or 'mark'} on {dm['location']}")
                break
        else:
            mismatched.append(
                f"Mark not found: {qm['type'] or 'mark'} on {qm['location']}"
            )

    score = hits / max(1, len(q))
    return float(score), matched, mismatched


# ─── String-similarity helpers ────────────────────────────────

def _word_overlap(a: str, b: str) -> float:
    """Jaccard over word tokens. Robust to ordering and adjectives."""
    aw = set(re.findall(r"\w+", (a or "").lower()))
    bw = set(re.findall(r"\w+", (b or "").lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _string_attr_score(q: str, d: str) -> float:
    """Soft score for free-text attributes like hair_color, nose, etc."""
    q = (q or "").strip().lower()
    d = (d or "").strip().lower()
    if not q or not d:
        return 0.5  # missing on one side → neutral
    if q == d:
        return 1.0
    if q in d or d in q:
        return 0.85
    overlap = _word_overlap(q, d)
    if overlap >= 0.5:
        return 0.7
    if overlap > 0:
        return 0.4
    return 0.0


def _list_attr_score(q, d) -> float:
    """Score for list-typed attributes (accessories, religious_attire)."""
    if not q:
        return 0.5
    if not d:
        return 0.0
    q_set = {str(x).lower().strip() for x in (q if isinstance(q, list) else [q])}
    d_set = {str(x).lower().strip() for x in (d if isinstance(d, list) else [d])}
    if not q_set or not d_set:
        return 0.5
    # Any overlap is a positive signal — accessories are noisy.
    inter = q_set & d_set
    if inter:
        return min(1.0, 0.5 + 0.5 * (len(inter) / len(q_set)))
    # Word-level partial match
    best = 0.0
    for qa in q_set:
        for da in d_set:
            best = max(best, _word_overlap(qa, da))
    return best * 0.6


# ─── Top-level scoring ────────────────────────────────────────

def score_attributes(query: dict, db_attrs: dict) -> tuple[float, list[str], list[str]]:
    """
    Compare witness-described attributes against DB metadata.

    Returns:
        (score_0_100, matched_descriptions, mismatched_descriptions)
    """
    if not query:
        return 50.0, [], []

    matched: list[str] = []
    mismatched: list[str] = []
    total_w = 0.0
    earned = 0.0

    db_attrs = db_attrs or {}

    for key, weight in ATTR_WEIGHTS.items():
        if key == "marks":
            continue  # handled below

        q_val = query.get(key)
        if q_val in (None, "", [], {}):
            continue

        d_val = db_attrs.get(key)
        total_w += weight

        if d_val in (None, "", [], {}):
            # DB unknown — half credit. (Better than penalizing entries
            # whose admin record is incomplete.)
            earned += weight * 0.5
            continue

        try:
            if key == "age":
                try:
                    q_age = int(q_val); d_age = int(d_val)
                    diff = abs(q_age - d_age)
                    if diff <= 3:
                        earned += weight; matched.append(f"Age ~{d_age} (query {q_age})")
                    elif diff <= 7:
                        earned += weight * 0.6
                        matched.append(f"Age ~{d_age} (close to {q_age})")
                    elif diff <= 12:
                        earned += weight * 0.3
                    else:
                        mismatched.append(f"Age: query {q_age} vs DB {d_age}")
                except (ValueError, TypeError):
                    earned += weight * 0.5

            elif key == "skin_tone":
                s = fuzzy_skin_match(str(q_val), str(d_val))
                earned += weight * s
                if s >= 0.7:
                    matched.append(f"Skin: {d_val}")
                elif s < 0.3:
                    mismatched.append(f"Skin: query '{q_val}' vs DB '{d_val}'")

            elif key == "ethnicity":
                rel = ethnicity_compatible(str(q_val), str(d_val))
                if rel == "ok":
                    earned += weight
                    matched.append(f"Ethnicity: {d_val}")
                elif rel == "soft":
                    earned += weight * 0.5
                else:
                    mismatched.append(f"Ethnicity: query '{q_val}' vs DB '{d_val}'")

            elif key == "gender":
                if str(q_val).lower().strip() == str(d_val).lower().strip():
                    earned += weight
                    matched.append(f"Gender: {d_val}")
                else:
                    mismatched.append(f"Gender: query '{q_val}' vs DB '{d_val}'")

            elif key in ("accessories", "religious_attire"):
                s = _list_attr_score(q_val, d_val)
                earned += weight * s
                if s >= 0.7:
                    matched.append(f"{key}: {d_val}")

            else:
                s = _string_attr_score(str(q_val), str(d_val))
                earned += weight * s
                if s >= 0.7:
                    matched.append(f"{key}: {d_val}")
                elif s < 0.2:
                    mismatched.append(f"{key}: query '{q_val}' vs DB '{d_val}'")
        except Exception:
            # Defensive: a single weird attribute value must not nuke
            # the whole match. Treat as half-credit and continue.
            earned += weight * 0.5

    # Marks
    q_marks = query.get("marks")
    if q_marks:
        mw = ATTR_WEIGHTS["marks"]
        total_w += mw
        try:
            mscore, mmatched, mmismatched = score_marks(q_marks, db_attrs.get("marks"))
        except Exception:
            mscore, mmatched, mmismatched = 0.5, [], []
        earned += mw * mscore
        matched.extend(mmatched)
        mismatched.extend(mmismatched)

    if total_w <= 0:
        return 50.0, matched, mismatched
    return float(earned / total_w * 100.0), matched, mismatched
