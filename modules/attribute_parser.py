"""
Module 2: Forensic Attribute Parser
Extracts precise facial attributes from witness text descriptions
and builds highly detailed prompts for accurate face generation.

Key design: every distinguishing mark (scar, mole, tattoo) is extracted
with its EXACT LOCATION on the face, and the prompt places heavy emphasis
on these features since they are the most identifying.
"""

import re
from typing import Optional


# ─── Attribute Vocabularies ───────────────────────────────────

GENDER_KEYWORDS = {
    "female": [
        r"\bfemale\b", r"\bwoman\b", r"\bwomen\b", r"\bgirl\b", r"\blady\b",
        r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bherself\b",
    ],
    "male": [
        r"\bmale\b", r"\bman\b", r"\bmen\b", r"\bboy\b", r"\bgentleman\b",
        r"\bhe\b", r"\bhis\b", r"\bhim\b", r"\bhimself\b",
    ],
}

SKIN_TONES = [
    "very dark brown", "dark brown", "medium brown", "light brown",
    "very dark", "dark", "brown", "wheatish", "olive", "tan",
    "medium", "fair", "light", "pale", "very fair", "dusky",
]

HAIR_COLORS = [
    "jet black", "black", "dark brown", "brown", "light brown",
    "auburn", "red", "ginger", "dirty blonde", "blonde", "blond",
    "grey", "gray", "white", "silver", "salt and pepper",
]

HAIR_STYLES = [
    "completely bald", "bald", "shaved head", "buzz cut", "crew cut",
    "very short", "short cropped", "short", "cropped",
    "medium length", "medium",
    "shoulder length", "long", "very long",
    "tightly curly", "curly", "wavy", "straight",
    "afro", "braided", "dreadlocks", "ponytail", "tied back",
    "receding hairline", "receding", "thinning", "parted to the side",
    "combed back", "messy", "spiky",
]

EYE_COLORS = [
    "black", "very dark brown", "dark brown", "brown", "light brown",
    "hazel", "amber", "green", "blue-green", "blue", "grey", "gray",
]

EYE_SHAPES = [
    "deep-set", "deep set", "sunken", "protruding", "bulging",
    "almond shaped", "almond-shaped", "round", "narrow", "wide-set",
    "close-set", "hooded", "droopy", "upturned", "downturned",
    "small", "large", "big",
]

EYEBROW_TYPES = [
    # Density/thickness — these are the reliable eyebrow descriptors.
    # "light" was removed because it over-triggers on unrelated uses
    # of the word (e.g. "light white hair", "light gold earrings").
    "thick bushy", "thick", "bushy", "heavy", "prominent",
    "thin", "pencil thin", "sparse",
    "arched", "straight", "curved",
    "unibrow", "connected",
]

FACIAL_HAIR = [
    "clean shaven", "clean-shaven", "no facial hair", "no beard",
    "light stubble", "stubble", "five o'clock shadow",
    "short trimmed beard", "short beard", "light beard",
    "full thick beard", "full beard", "thick beard", "long beard", "heavy beard",
    "thin pencil mustache", "thin mustache", "handlebar mustache",
    "thick bushy mustache", "thick mustache", "large mustache",
    "mustache", "moustache",
    "goatee", "van dyke", "soul patch",
    "long sideburns", "sideburns", "mutton chops",
]

FACE_SHAPES = [
    "round chubby", "round", "oval", "square angular", "square",
    "rectangular", "oblong", "heart shaped", "heart-shaped",
    "diamond", "triangular", "long narrow", "long", "wide", "narrow",
    "chubby", "gaunt", "angular", "chiseled",
]

NOSE_TYPES = [
    "very broad", "broad flat", "broad", "wide", "flat",
    "pointed sharp", "pointed", "sharp", "button", "upturned", "snub",
    "long straight", "long", "short",
    "crooked", "broken", "bent to the side", "bent",
    "aquiline", "roman", "hooked", "hawk-like",
    "bulbous", "large round", "large",
    "thin narrow", "thin", "narrow", "small",
]

LIP_TYPES = [
    "very thin lips", "thin lips", "narrow lips",
    "thick full lips", "thick lips", "full lips", "plump lips",
    "wide mouth", "small mouth", "large mouth",
    "uneven lips", "crooked smile",
]

CHIN_TYPES = [
    "prominent chin", "strong chin", "weak chin", "receding chin",
    "pointed chin", "square chin", "round chin", "double chin",
    "cleft chin", "dimpled chin",
]

FOREHEAD_TYPES = [
    "large forehead", "high forehead", "broad forehead",
    "small forehead", "narrow forehead", "low forehead",
    "prominent forehead", "flat forehead",
]

BUILD_TYPES = [
    "very slim", "slim", "thin", "skinny", "lean",
    "athletic", "fit", "muscular", "well-built",
    "medium build", "average build", "stocky",
    "heavy set", "heavy", "overweight", "large", "obese",
]

# Location words for marks — used to extract precise positions
FACE_LOCATIONS = [
    "left cheek", "right cheek", "left eye", "right eye",
    "below the left eye", "below the right eye", "below his left eye", "below her left eye",
    "below his right eye", "below her right eye",
    "above the left eye", "above the right eye", "above his left eye", "above her left eye",
    "above his right eye", "above her right eye",
    "under the left eye", "under the right eye", "under his left eye", "under her left eye",
    "under his right eye", "under her right eye",
    "near the left eye", "near the right eye",
    "left eyebrow", "right eyebrow",
    # Generic unspecified-side eyebrow — MUST come AFTER left/right
    # variants. The location-finder below sorts by length desc so
    # "left eyebrow" wins when both match a sentence.
    "eyebrow", "eyebrows", "brow",
    "left temple", "right temple",
    "left side of the face", "right side of the face",
    "left side of the chin", "right side of the chin",
    "left side of the nose", "right side of the nose",
    "left side of the forehead", "right side of the forehead",
    "left jaw", "right jaw", "jawline",
    "forehead", "chin", "nose", "upper lip", "lower lip",
    "left ear", "right ear", "neck", "throat",
    "bridge of the nose", "tip of the nose",
    "left nostril", "right nostril", "left side of the nostril",
    "corner of the mouth", "left corner of the mouth", "right corner of the mouth",
    "between the eyebrows", "center of the forehead",
]

# POST-PROCESSING MARKS — rendered via OpenCV after generation.
# These need precise pixel placement that text prompts can't achieve.
MARK_TYPES = [
    # Scars
    "deep scar", "long scar", "small scar", "vertical scar", "horizontal scar",
    "diagonal scar", "jagged scar", "surgical scar", "burn scar", "scar",
    # Moles
    "large mole", "small mole", "dark mole", "raised mole", "flat mole", "mole",
    # Other marks
    "birthmark", "port wine stain",
    "wart", "bump",
]

# PROMPT-SIDE ACCESSORIES — included in the AI generation prompt.
# The model generates these naturally as part of the face image.
ACCESSORY_TYPES = [
    "bindi", "red bindi", "black bindi", "maroon bindi",
    "nose ring", "nose stud", "gold nose ring", "silver nose ring",
    "earring", "ear ring", "hoop earring", "stud earring",
    "gold earring", "silver earring", "dangling earring",
    "nose piercing", "ear piercing", "lip piercing", "eyebrow piercing", "piercing",
    "tattoo",
    "spectacles", "glasses", "reading glasses", "sunglasses",
    "mangalsutra", "chain", "necklace", "pendant",
]

# RELIGIOUS / CULTURAL ATTIRE — headwear + body covering. Handled
# separately from accessories so we can push a strong prompt clause
# ("wearing X covering head and hair") and fire ethnicity negatives
# that block the model's default drift toward African features when
# the witness mentions "muslim woman in a burka" etc.
RELIGIOUS_ATTIRE = [
    "black burka", "burka", "burqa", "niqab", "hijab",
    "abaya", "chador", "headscarf", "head scarf",
    "dupatta", "ghoonghat", "veil",
    "turban", "sikh turban", "pagri", "dastar",
    "skullcap", "taqiyah", "kufi",
    "yarmulke", "kippah",
]

# Canonical rendering phrases for each attire keyword. These go into
# the positive prompt verbatim so the model has no room to interpret
# "burka" as "tribal clothing".
_ATTIRE_PROMPT = {
    "burka":   "wearing a full black burqa covering head and hair, only face visible",
    "burqa":   "wearing a full black burqa covering head and hair, only face visible",
    "black burka": "wearing a full black burqa covering head and hair, only face visible",
    "niqab":   "wearing a black niqab covering head and hair",
    "hijab":   "wearing a hijab headscarf covering all hair",
    "abaya":   "wearing an abaya with hijab covering hair",
    "chador":  "wearing a black chador covering head and hair",
    "headscarf": "wearing a headscarf covering hair",
    "head scarf": "wearing a headscarf covering hair",
    "dupatta": "wearing a dupatta over the head",
    "ghoonghat": "wearing a ghoonghat veil over the head",
    "veil":    "wearing a veil covering hair",
    "turban":  "wearing a traditional turban on the head",
    "sikh turban": "wearing a Sikh dastar turban",
    "pagri":   "wearing a traditional pagri turban",
    "dastar":  "wearing a Sikh dastar turban",
    "skullcap": "wearing a skullcap",
    "taqiyah": "wearing a taqiyah skullcap",
    "kufi":    "wearing a kufi cap",
    "yarmulke": "wearing a yarmulke",
    "kippah":  "wearing a kippah",
}


def parse_description(text: str) -> dict:
    """
    Extract structured facial attributes from a witness description.
    Focuses on precise extraction of every detail including exact
    location of distinguishing marks.
    """
    text_lower = f" {text.lower().strip()} "

    attrs = {
        "gender": _extract_gender(text_lower),
        "age": _extract_age(text_lower),
        "ethnicity": _extract_ethnicity(text_lower),
        "skin_tone": _extract_adjacent(text_lower, SKIN_TONES,
                                       after_words=["skin", "complexion", "tone", "colored", "coloured"]),
        "face_shape": _extract_from_list(text_lower, FACE_SHAPES, context_words=["face", "shaped"]),
        "forehead": _extract_from_list(text_lower, FOREHEAD_TYPES),
        "eyebrows": _extract_from_list(text_lower, EYEBROW_TYPES, context_words=["eyebrow", "brow"]),
        "eye_shape": _extract_adjacent(text_lower, EYE_SHAPES, after_words=["eye", "eyes"]),
        "eye_color": _extract_adjacent(text_lower, EYE_COLORS, after_words=["eye", "eyes"]),
        "nose": _extract_adjacent(text_lower, NOSE_TYPES, after_words=["nose"]),
        "lips": _extract_from_list(text_lower, LIP_TYPES),
        "chin": _extract_from_list(text_lower, CHIN_TYPES),
        "facial_hair": _extract_from_list(text_lower, FACIAL_HAIR),
        "hair_color": _extract_hair_color_detailed(text_lower),
        "hair_style": _extract_adjacent(text_lower, HAIR_STYLES, after_words=["hair", "head"]),
        "build": _extract_build(text_lower),
        "marks": _extract_marks_precise(text_lower),
        "accessories": _extract_accessories_with_sides(text_lower),
        "religious_attire": _extract_religious_attire(text_lower),
    }

    # Infer ethnicity from religious attire when the witness didn't give one.
    # "muslim woman in a burka" used to drop ethnicity entirely, and FLUX /
    # SD's dataset bias then produced African-looking faces. Attire is a
    # strong regional/cultural signal, so use it as a fallback anchor.
    if not attrs["ethnicity"]:
        attire = attrs["religious_attire"]
        if any(a in ("burka", "burqa", "black burka", "niqab", "hijab",
                     "abaya", "chador", "taqiyah", "kufi")
               for a in attire) or "muslim" in text_lower:
            attrs["ethnicity"] = "Muslim South Asian or Middle Eastern"
        elif any(a in ("turban", "sikh turban", "pagri", "dastar") for a in attire):
            attrs["ethnicity"] = "North Indian Punjabi Sikh"
        elif any(a in ("dupatta", "ghoonghat") for a in attire):
            attrs["ethnicity"] = "South Asian Indian"

    return attrs


def _extract_gender(text: str) -> Optional[str]:
    female_hits = sum(1 for pat in GENDER_KEYWORDS["female"] if re.search(pat, text))
    male_hits   = sum(1 for pat in GENDER_KEYWORDS["male"]   if re.search(pat, text))
    if male_hits > female_hits:
        return "male"
    if female_hits > male_hits:
        return "female"
    return None


def _extract_age(text: str) -> Optional[int]:
    # "around 30 to 35 years" → average
    range_match = re.search(r'(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:years?|yrs?|year\s*old)?', text)
    if range_match:
        return (int(range_match.group(1)) + int(range_match.group(2))) // 2

    # "30 years old" or "aged 30" or "around 30"
    age_match = re.search(r'(?:aged?|around|approximately|about|roughly)?\s*(\d{1,2})\s*(?:years?|yrs?|year\s*old)', text)
    if age_match:
        return int(age_match.group(1))

    # Standalone number near age context: "around 60," or "about 32,"
    standalone_match = re.search(r'(?:aged?|around|approximately|about|roughly)\s+(\d{1,2})\b', text)
    if standalone_match:
        return int(standalone_match.group(1))

    # Descriptive age
    age_map = {
        "teenager": 16, "teen": 16, "young": 25, "youth": 22,
        "middle aged": 45, "middle-aged": 45,
        "elderly": 65, "old man": 60, "old woman": 60, "senior": 65,
    }
    for desc, age in age_map.items():
        if desc in text:
            return age
    return None


def _extract_ethnicity(text: str) -> Optional[str]:
    """
    Extract ethnicity/race if mentioned (important for accurate face rendering).

    Order matters — we check the MOST SPECIFIC regional keywords first
    (e.g. "south indian" before "indian") so the resulting label carries
    maximum phenotype information into the prompt builder.
    """
    # Ordered list: (keyword, canonical_label). Longer / more specific first.
    ethnicity_pairs = [
        # ── South Indian (Dravidian) — distinctly darker complexion,
        #    rounder facial features; often confused with East Asian by
        #    generic SD/FLUX prompts, so we tag them explicitly.
        ("south indian",   "South Indian Dravidian"),
        ("dravidian",      "South Indian Dravidian"),
        ("tamil",          "South Indian Tamil"),
        ("telugu",          "South Indian Telugu"),
        ("malayali",       "South Indian Malayali Kerala"),
        ("keralite",       "South Indian Kerala"),
        ("kerala",         "South Indian Kerala"),
        ("kannadiga",      "South Indian Kannada"),
        # ── North Indian — lighter complexion, sharper features
        ("north indian",   "North Indian"),
        ("punjabi",        "North Indian Punjabi"),
        ("sikh",           "North Indian Punjabi Sikh"),
        ("kashmiri",       "North Indian Kashmiri"),
        ("marathi",        "Indian Marathi"),
        ("gujarati",       "Indian Gujarati"),
        ("bengali",        "Indian Bengali"),
        ("rajasthani",     "Indian Rajasthani"),
        # ── Broader groupings
        ("indian",         "South Asian Indian"),
        ("pakistani",      "South Asian Pakistani"),
        ("bangladeshi",    "South Asian Bangladeshi"),
        ("sri lankan",     "South Asian Sri Lankan"),
        ("south asian",    "South Asian"),
        # ── East Asian
        ("east asian",     "East Asian"),
        ("chinese",        "East Asian Chinese"),
        ("japanese",       "East Asian Japanese"),
        ("korean",         "East Asian Korean"),
        ("vietnamese",     "Southeast Asian Vietnamese"),
        ("filipino",       "Southeast Asian Filipino"),
        ("thai",           "Southeast Asian Thai"),
        ("asian",          "East Asian"),
        # ── Religious/cultural groupings (phenotype-by-attire fallback).
        #    These only fire if the witness explicitly names the religion;
        #    unrelated words like "sikh" already anchor to Punjabi above.
        ("muslim",         "Muslim South Asian or Middle Eastern"),
        ("islamic",        "Muslim South Asian or Middle Eastern"),
        # ── Other
        ("african american", "African American"),
        ("african",        "African"),
        ("caucasian",      "Caucasian"),
        ("hispanic",       "Hispanic Latino"),
        ("latino",         "Hispanic Latino"),
        ("latina",         "Hispanic Latino"),
        ("middle eastern", "Middle Eastern"),
        ("arab",           "Middle Eastern Arab"),
        ("persian",        "Middle Eastern Persian"),
    ]
    for keyword, value in ethnicity_pairs:
        if keyword not in text:
            continue
        idx = text.find(keyword)
        surrounding = text[max(0, idx - 30):idx + len(keyword) + 30]
        # Avoid false matches where the keyword refers to hair/clothing
        if any(w in surrounding for w in ["hair", "eye", "shirt", "dress"]):
            continue
        return value
    return None


def _extract_adjacent(text: str, options: list, after_words: list) -> Optional[str]:
    """Strict extraction: option must be directly adjacent to a context word."""
    sorted_options = sorted(options, key=len, reverse=True)
    for option in sorted_options:
        for ctx in after_words:
            # "[option] [0-1 words] [context]" e.g. "dark brown skin"
            pattern = rf'{re.escape(option)}\s+(?:\w+\s+){{0,1}}{re.escape(ctx)}'
            if re.search(pattern, text):
                return option
            # "[context] [0-1 words] [option]" e.g. "eyes are brown"
            pattern = rf'{re.escape(ctx)}\s+(?:\w+\s+){{0,1}}{re.escape(option)}'
            if re.search(pattern, text):
                return option
    return None


def _extract_from_list(text: str, options: list, context_words: list = None) -> Optional[str]:
    """Find matching option, optionally requiring a nearby context word."""
    sorted_options = sorted(options, key=len, reverse=True)
    for option in sorted_options:
        if option not in text:
            continue
        if context_words is None:
            return option
        start = 0
        while True:
            idx = text.find(option, start)
            if idx == -1:
                break
            surrounding = text[max(0, idx - 25):idx + len(option) + 25]
            if any(cw in surrounding for cw in context_words):
                return option
            start = idx + 1
    return None


def _extract_build(text: str) -> Optional[str]:
    """Extract build, avoiding false matches from mark descriptions like 'large scar'."""
    # First try compound phrases that are unambiguous
    for phrase in ["medium build", "average build", "heavy set", "well-built",
                   "slim build", "thin build", "athletic build", "stocky build",
                   "muscular build", "large build"]:
        if phrase in text:
            return phrase

    # Then check with context — require "build"/"built"/"body"/"physique" nearby
    for option in BUILD_TYPES:
        if option not in text:
            continue
        idx = text.find(option)
        surrounding = text[max(0, idx - 20):idx + len(option) + 20]
        if any(w in surrounding for w in ["build", "built", "body", "physique", "frame"]):
            return option

    # Special case: "very thin" / "very slim" etc. as standalone descriptions
    for pattern in [r'\bvery\s+(slim|thin|skinny|lean)\b',
                    r'\b(slim|skinny|lean|muscular|stocky|athletic)\s+(?:man|woman|male|female|person|guy)\b']:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()

    return None


def _extract_accessories(text: str) -> list[str]:
    """
    Extract accessories mentioned in the description.
    These go into the AI generation prompt (NOT post-processing).
    Returns list of accessory strings for prompt inclusion.
    """
    found = []
    # Sort by length descending so "gold nose ring" matches before "nose ring"
    sorted_types = sorted(ACCESSORY_TYPES, key=len, reverse=True)
    matched_spans = []

    for acc in sorted_types:
        idx = text.find(acc)
        if idx == -1:
            continue

        # Check this doesn't overlap with an already-matched longer string
        overlaps = False
        for start, end in matched_spans:
            if idx < end and idx + len(acc) > start:
                overlaps = True
                break
        if overlaps:
            continue

        found.append(acc)
        matched_spans.append((idx, idx + len(acc)))

    return found


def _extract_religious_attire(text: str) -> list[str]:
    """
    Extract head / body religious attire keywords. Longest match wins so
    "black burka" beats "burka".
    """
    found = []
    matched_spans = []
    for item in sorted(RELIGIOUS_ATTIRE, key=len, reverse=True):
        idx = text.find(item)
        if idx == -1:
            continue
        end = idx + len(item)
        if any(idx < me and end > ms for ms, me in matched_spans):
            continue
        found.append(item)
        matched_spans.append((idx, end))
    return found


def _extract_accessories_with_sides(text: str) -> list[str]:
    """
    Extract accessories and attach side info when the witness specifies it.
    E.g. "nose ring on her left nostril" → "nose ring on left nostril".

    Forensic-critical for accessories where laterality matters:
      - nose ring / nose stud / nose piercing  ← left vs right nostril
      - earring                                 ← left vs right vs both ears
      - eyebrow piercing                        ← left vs right
    """
    base = _extract_accessories(text)
    result = []
    for acc in base:
        sided = acc

        # Nose accessories — check for nostril side
        if any(w in acc for w in ("nose ring", "nose stud", "nose piercing")):
            if re.search(r'\bleft\s+nostril\b', text):
                sided = f"{acc} on left nostril"
            elif re.search(r'\bright\s+nostril\b', text):
                sided = f"{acc} on right nostril"

        # Ear accessories — check for which ear / both
        elif "earring" in acc or acc in ("ear piercing",):
            if re.search(r'\bboth\s+ears\b', text) or re.search(r'\bearrings\b', text):
                sided = f"{acc}s on both ears"
            elif re.search(r'\bleft\s+ear\b', text):
                sided = f"{acc} on left ear"
            elif re.search(r'\bright\s+ear\b', text):
                sided = f"{acc} on right ear"

        # Eyebrow piercing
        elif "eyebrow piercing" in acc:
            if re.search(r'\bleft\s+(?:eyebrow|brow)\b', text):
                sided = f"{acc} on left eyebrow"
            elif re.search(r'\bright\s+(?:eyebrow|brow)\b', text):
                sided = f"{acc} on right eyebrow"

        result.append(sided)
    return result


# ─── Hair: nuanced colour extraction ─────────────────────────
#
# Plain _extract_adjacent picks the FIRST matching colour word near
# "hair", which loses information when the witness says things like
# "predominantly black with grey streaks" — it returns just "grey"
# and the generator produces a grey-haired person instead.
#
# This extractor preserves the primary/secondary colour relationship.

_HAIR_COLOUR_WORDS = sorted(HAIR_COLORS, key=len, reverse=True)
_PRIMARY_HAIR_RE = re.compile(
    r'\b(?:predominantly|mostly|primarily|mainly|largely)\s+'
    r'(' + '|'.join(re.escape(c) for c in _HAIR_COLOUR_WORDS) + r')'
    r'(?:\s+hair)?', re.IGNORECASE,
)
_STREAK_HAIR_RE = re.compile(
    r'\b(?:streaks|patches|highlights|strands|hints?)\s+'
    r'(?:and\s+patches\s+)?'
    r'of\s+(?:light\s+|dark\s+)?'
    r'(' + '|'.join(re.escape(c) for c in _HAIR_COLOUR_WORDS) + r')',
    re.IGNORECASE,
)


def _extract_hair_color_detailed(text: str) -> Optional[str]:
    """
    Handle compound hair descriptions like "black with grey streaks".
    Falls back to the simple adjacent extractor when no primary/streak
    pattern matches.
    """
    primary = None
    streak = None

    m = _PRIMARY_HAIR_RE.search(text)
    if m:
        primary = m.group(1).lower()

    m = _STREAK_HAIR_RE.search(text)
    if m:
        streak = m.group(1).lower()

    if primary and streak and primary != streak:
        return f"{primary} with {streak} streaks"
    if primary:
        return primary
    # Plain adjacent fallback
    return _extract_adjacent(text, HAIR_COLORS, after_words=["hair"])


def _extract_marks_precise(text: str) -> list:
    """
    Extract distinguishing marks with PRECISE location.
    This is the most critical extraction for forensic accuracy.

    Returns list of dicts: [{type, location, description}, ...]
    """
    marks = []

    # Split text into sentences for isolated mark extraction
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for mark_type in MARK_TYPES:
            if mark_type not in sentence:
                continue

            idx = sentence.find(mark_type)

            # Find the CLOSEST location to this mark in the sentence.
            # This prevents cross-contamination when multiple marks share
            # a sentence (e.g. "scar below left eye and mole on right cheek").
            #
            # Two-pass approach to handle overlapping / nested locations:
            #   Pass 1 — collect all matches (loc, start, end), with
            #            longest locations tried first, so a specific
            #            match like "left eyebrow" consumes its span
            #            and the bare "eyebrow" inside it gets skipped.
            #   Pass 2 — pick the candidate closest to the mark, with
            #            ties broken by longer (more specific) location.
            candidates = []
            consumed_spans = []
            # Sort by length descending so "below the left eye" beats
            # "left eye" which beats "eye", etc.
            sorted_locs = sorted(FACE_LOCATIONS, key=len, reverse=True)
            for loc in sorted_locs:
                start_pos = 0
                while True:
                    loc_idx = sentence.find(loc, start_pos)
                    if loc_idx == -1:
                        break
                    loc_end = loc_idx + len(loc)
                    # Skip if covered by a longer location already matched
                    covered = any(loc_idx < ce and loc_end > cs
                                  for cs, ce in consumed_spans)
                    if not covered:
                        # Skip if this location is inside an accessory phrase
                        surrounding = sentence[
                            max(0, loc_idx - 5):loc_end + 10
                        ]
                        is_accessory_part = any(
                            acc in surrounding
                            for acc in ACCESSORY_TYPES
                            if loc in acc
                        )
                        if not is_accessory_part:
                            candidates.append((loc, loc_idx, loc_end))
                            consumed_spans.append((loc_idx, loc_end))
                    start_pos = loc_idx + 1

            best_location = None
            best_distance = float("inf")
            for loc, loc_idx, _ in candidates:
                distance = abs(loc_idx - idx)
                if distance < best_distance or (
                    distance == best_distance
                    and best_location is not None
                    and len(loc) > len(best_location)
                ):
                    best_location = loc
                    best_distance = distance

            # Clean up location — remove pronouns
            if best_location:
                best_location = re.sub(r'\b(his|her|their|the)\b\s*', '', best_location).strip()

            # Extract descriptors ONLY from nearby context (within 25 chars)
            descriptors = []
            nearby = sentence[max(0, idx - 25):idx + len(mark_type) + 25]
            for desc in ["small", "large", "big", "tiny", "deep", "long", "short",
                         "thin", "thick", "faint", "prominent", "visible", "noticeable",
                         "old", "fresh", "healed", "raised", "flat", "dark", "light",
                         "vertical", "horizontal", "diagonal", "curved", "jagged", "clearly"]:
                if desc in nearby and desc not in mark_type:
                    descriptors.append(desc)

            marks.append({
                "type": mark_type,
                "location": best_location,
                "descriptors": descriptors,
                "raw": sentence.strip(),
            })

    # Deduplicate — keep the MOST SPECIFIC match for each location.
    # e.g. if "scar" and "deep scar" both match at "left cheek", keep "deep scar".
    # If "nose ring" and "gold nose ring" both match, keep "gold nose ring".
    deduped = []
    for m in marks:
        is_redundant = False
        for i, existing in enumerate(deduped):
            # Same or overlapping location
            same_loc = (m.get("location") == existing.get("location")
                        or not m.get("location") or not existing.get("location"))
            # One type contains the other (e.g. "scar" in "deep scar")
            type_overlap = (m["type"] in existing["type"] or existing["type"] in m["type"])

            if same_loc and type_overlap:
                # Keep the more specific (longer) type
                if len(m["type"]) > len(existing["type"]):
                    deduped[i] = m  # Replace with more specific
                is_redundant = True
                break
        if not is_redundant:
            deduped.append(m)

    return deduped


# ─── Prompt Builder ───────────────────────────────────────────
#
# Prompt strategy ported from HiTS (Choi et al. 2025, IEEE Access):
#   - Split attributes into INTRINSIC (race, gender) and MUTABLE
#     (skin/hair/eye colors, facial hair, marks). HiTS measured that
#     intrinsic attrs align >80% with CLIP embeddings but mutable attrs
#     only ~30-50%, which is why one monolithic prompt produces mixed or
#     washed-out colors.
#   - Emit the CUFS reference format that HiTS trained against:
#     "A photo of [race] [gender] with [hair color] hair, [skin color]
#      skin, and [eye color] eyes."
#   - Repeat each mutable color attribute at two positions in the prompt
#     so the CLIP text encoder gets a strong signal for each.
#   - Build a color-biased NEGATIVE prompt that explicitly excludes the
#     competing colors for skin/hair/eyes — fights the dataset bias that
#     makes SDXL default to "average" appearance.
#
# Drop-in: this still returns (positive, negative) strings, so the API
# layer and UI preview don't need to change.

_SKIN_COMPETITORS = {
    "very dark brown": ["light skin", "fair skin", "pale skin", "white skin"],
    "dark brown":      ["light skin", "fair skin", "pale skin", "white skin"],
    "dark":            ["light skin", "fair skin", "pale skin", "white skin"],
    "very dark":       ["light skin", "fair skin", "pale skin", "white skin"],
    "dusky":           ["light skin", "pale skin"],
    "medium brown":    ["pale skin", "very dark skin", "very fair skin"],
    "brown":           ["pale skin", "very dark skin"],
    "wheatish":        ["pale skin", "very dark skin"],
    "olive":           ["pale skin", "very dark skin"],
    "tan":             ["pale skin", "very dark skin"],
    "medium":          ["pale skin", "very dark skin"],
    "light brown":     ["pale skin", "very dark skin"],
    "fair":            ["dark skin", "tan skin", "brown skin"],
    "light":           ["dark skin", "tan skin", "brown skin"],
    "pale":            ["dark skin", "tan skin", "brown skin"],
    "very fair":       ["dark skin", "tan skin", "brown skin"],
}

_HAIR_COMPETITORS = {
    "jet black":  ["blonde hair", "red hair", "ginger hair", "grey hair", "white hair"],
    "black":      ["blonde hair", "red hair", "ginger hair", "grey hair", "white hair"],
    "dark brown": ["blonde hair", "red hair", "grey hair", "white hair"],
    "brown":      ["blonde hair", "red hair"],
    "light brown":["black hair", "blonde hair", "red hair"],
    "auburn":     ["blonde hair", "black hair", "grey hair"],
    "red":        ["blonde hair", "black hair", "brown hair"],
    "ginger":     ["blonde hair", "black hair", "brown hair"],
    "blonde":     ["black hair", "dark hair", "brown hair"],
    "blond":      ["black hair", "dark hair", "brown hair"],
    "dirty blonde":["black hair", "dark hair", "red hair"],
    "grey":       ["black hair", "brown hair", "blonde hair"],
    "gray":       ["black hair", "brown hair", "blonde hair"],
    "white":      ["black hair", "brown hair", "blonde hair"],
    "silver":     ["black hair", "brown hair"],
    "salt and pepper": ["blonde hair", "red hair"],
}

_EYE_COMPETITORS = {
    "black":           ["blue eyes", "green eyes", "hazel eyes"],
    "very dark brown": ["blue eyes", "green eyes", "hazel eyes"],
    "dark brown":      ["blue eyes", "green eyes"],
    "brown":           ["blue eyes", "green eyes"],
    "light brown":     ["black eyes", "blue eyes"],
    "hazel":           ["blue eyes", "black eyes"],
    "amber":           ["blue eyes", "grey eyes"],
    "green":           ["brown eyes", "black eyes", "blue eyes"],
    "blue-green":      ["brown eyes", "black eyes"],
    "blue":            ["brown eyes", "black eyes"],
    "grey":            ["brown eyes", "green eyes"],
    "gray":            ["brown eyes", "green eyes"],
}


def _race_gender_phrase(attrs: dict) -> str:
    """HiTS intrinsic attribute phrase: '[race] [gender]' with age optional."""
    race = attrs.get("ethnicity") or ""
    gender = attrs.get("gender") or "person"
    age = attrs.get("age")

    parts = []
    if age:
        parts.append(f"{age} year old")
    if race:
        parts.append(race)
    parts.append(gender)
    return " ".join(parts)


# ── Ethnicity phenotype hints ────────────────────────────────
#
# SD 1.5 / FLUX are biased toward "average" East-Asian or European
# faces even when the prompt says "Indian". We combat that by
# appending an explicit phenotype descriptor + excluding competing
# ethnicities in the negative prompt.

_ETH_PHENOTYPE = {
    "South Indian Dravidian": "distinctly South Indian Dravidian features, "
                              "dark brown Tamil/Telugu complexion, "
                              "rounded facial structure, deep-set dark eyes",
    "South Indian Tamil":     "distinctly Tamil South Indian features, "
                              "dark brown complexion, broader nose, full lips",
    "South Indian Telugu":    "distinctly Telugu South Indian features, "
                              "dark brown complexion, rounded face",
    "South Indian Kerala":    "distinctly Malayali Kerala features, "
                              "dark brown complexion, South Indian face",
    "South Indian Malayali Kerala": "distinctly Malayali Kerala features, "
                                    "dark brown complexion, South Indian face",
    "South Indian Kannada":   "distinctly Kannadiga South Indian features, "
                              "dark brown complexion",
    "North Indian":           "distinctly North Indian features, wheatish to "
                              "light brown complexion, sharper nose",
    "North Indian Punjabi":   "distinctly Punjabi North Indian features, "
                              "wheatish complexion, broader build",
    "North Indian Punjabi Sikh": "distinctly Punjabi Sikh features, "
                                 "wheatish complexion",
    "North Indian Kashmiri":  "distinctly Kashmiri features, fair complexion, "
                              "sharp nose, light brown to hazel eyes",
    "Indian Marathi":         "distinctly Marathi Indian features",
    "Indian Gujarati":        "distinctly Gujarati Indian features",
    "Indian Bengali":         "distinctly Bengali Indian features",
    "Indian Rajasthani":      "distinctly Rajasthani Indian features",
    "South Asian Indian":     "distinctly South Asian Indian features, "
                              "brown complexion, subcontinental face",
    "South Asian Pakistani":  "distinctly Pakistani features, "
                              "brown South Asian complexion",
    "South Asian Bangladeshi":"distinctly Bangladeshi features, "
                              "brown South Asian complexion",
    "South Asian Sri Lankan": "distinctly Sri Lankan Sinhalese/Tamil features, "
                              "dark brown complexion",
    "South Asian":            "distinctly South Asian features, brown complexion",
    "East Asian Chinese":     "distinctly Han Chinese East Asian features",
    "East Asian Japanese":    "distinctly Japanese East Asian features",
    "East Asian Korean":      "distinctly Korean East Asian features",
    "East Asian":             "distinctly East Asian features",
    "Southeast Asian Vietnamese": "distinctly Vietnamese Southeast Asian features",
    "Southeast Asian Filipino":   "distinctly Filipino Southeast Asian features",
    "Southeast Asian Thai":       "distinctly Thai Southeast Asian features",
    "African":                "distinctly African features, dark brown to black skin",
    "African American":       "distinctly African American features",
    "Caucasian":              "distinctly Caucasian European features, fair skin",
    "Hispanic Latino":        "distinctly Hispanic Latino features",
    "Middle Eastern":         "distinctly Middle Eastern features",
    "Middle Eastern Arab":    "distinctly Arab Middle Eastern features",
    "Middle Eastern Persian": "distinctly Persian Middle Eastern features",
    "Muslim South Asian or Middle Eastern":
        "distinctly South Asian or Middle Eastern Muslim features, "
        "wheatish to light brown complexion, dark hair, "
        "neither African nor East Asian ancestry",
}

# For each canonical ethnicity, list of COMPETING ethnic features that
# the diffusion model should NOT produce. Pushes the model away from its
# dataset-default "average" face.
_ETH_NEGATIVE = {
    "South Indian Dravidian": ["east asian face", "chinese features",
                               "japanese features", "korean features",
                               "european face", "caucasian features",
                               "fair pale skin", "light skinned",
                               "north indian features"],
    "South Indian Tamil":     ["east asian face", "chinese features",
                               "european face", "caucasian features",
                               "fair pale skin", "north indian features"],
    "South Indian Telugu":    ["east asian face", "chinese features",
                               "european face", "fair pale skin"],
    "South Indian Kerala":    ["east asian face", "chinese features",
                               "european face", "fair pale skin"],
    "South Indian Malayali Kerala": ["east asian face", "chinese features",
                                     "european face", "fair pale skin"],
    "South Indian Kannada":   ["east asian face", "chinese features",
                               "european face", "fair pale skin"],
    "North Indian":           ["east asian face", "chinese features",
                               "african features", "caucasian features"],
    "North Indian Punjabi":   ["east asian face", "chinese features"],
    "North Indian Punjabi Sikh": ["east asian face", "chinese features"],
    "North Indian Kashmiri":  ["east asian face", "chinese features",
                               "african features"],
    "Indian Marathi":         ["east asian face", "chinese features"],
    "Indian Gujarati":        ["east asian face", "chinese features"],
    "Indian Bengali":         ["east asian face", "chinese features"],
    "Indian Rajasthani":      ["east asian face", "chinese features"],
    "South Asian Indian":     ["east asian face", "chinese features",
                               "japanese features", "korean features",
                               "caucasian features", "european face"],
    "South Asian Pakistani":  ["east asian face", "chinese features",
                               "european face"],
    "South Asian Bangladeshi":["east asian face", "chinese features"],
    "South Asian Sri Lankan": ["east asian face", "chinese features"],
    "South Asian":            ["east asian face", "chinese features",
                               "caucasian features"],
    "East Asian":             ["south asian features", "indian features",
                               "african features", "caucasian features"],
    "East Asian Chinese":     ["south asian features", "indian features",
                               "african features"],
    "East Asian Japanese":    ["south asian features", "indian features",
                               "african features"],
    "East Asian Korean":      ["south asian features", "indian features",
                               "african features"],
    "African":                ["east asian face", "caucasian features",
                               "south asian features"],
    "African American":       ["east asian face", "caucasian features"],
    "Caucasian":              ["east asian face", "south asian features",
                               "african features"],
    "Hispanic Latino":        ["east asian face"],
    "Middle Eastern":         ["east asian face", "south asian features"],
    "Middle Eastern Arab":    ["east asian face", "south asian features"],
    "Middle Eastern Persian": ["east asian face", "south asian features"],
    "Muslim South Asian or Middle Eastern": [
        "african features", "dark african skin", "south african features",
        "sub-saharan features", "east asian face", "chinese features",
        "european face",
    ],
}


def _mark_phrases(marks: list) -> list:
    """Build anatomically precise mark descriptions — highest-value attribute."""
    phrases = []
    for m in marks:
        desc_words = " ".join(m.get("descriptors", []))
        mark_type = m.get("type", "")
        loc = m.get("location", "")

        if loc:
            loc_clean = loc.replace("below ", "below the ").replace("above ", "above the ")
            if not loc_clean.startswith(("below", "above", "under", "near", "on")):
                loc_clean = f"on the {loc_clean}"
            phrase = f"a {desc_words} {mark_type} {loc_clean}"
        else:
            phrase = f"a {desc_words} {mark_type}"
        phrases.append(" ".join(phrase.split()))  # collapse whitespace
    return phrases


def build_prompt(attrs: dict) -> tuple[str, str]:
    """
    Build a hierarchical HiTS-style prompt for forensic face generation.

    Structure of the positive prompt:
      1. CUFS-format anchor sentence (race, gender, hair, skin, eyes)
      2. Marks line — highest weight, placed early
      3. Secondary features (face shape, nose, lips, facial hair, build)
      4. Color REPETITION line — re-emphasizes skin/hair/eye color
      5. Style anchors (mugshot framing, lighting, background)

    The negative prompt includes COLOR-BIASED terms that exclude the
    competitors of each specified color, which fights the SDXL/FLUX
    dataset bias toward "average" appearance.
    """
    gender   = attrs.get("gender")
    ethnicity = attrs.get("ethnicity") or ""
    skin     = attrs.get("skin_tone") or ""
    hair_c   = attrs.get("hair_color") or ""
    hair_s   = attrs.get("hair_style") or ""
    eye_c    = attrs.get("eye_color") or ""
    marks    = attrs.get("marks", []) or []

    # ── 1. CUFS anchor sentence (HiTS Section III.B) ──
    # "A photo of [race] [gender] with [hair] hair, [skin] skin, and [eye] eyes."
    intrinsic = _race_gender_phrase(attrs)
    cufs_parts = []
    if hair_c:
        # Compound form like "black with grey streaks" → insert "hair"
        # after the primary colour: "black hair with grey streaks".
        if " with " in hair_c:
            primary, modifier = hair_c.split(" with ", 1)
            cufs_parts.append(f"{primary} hair with {modifier}")
        else:
            cufs_parts.append(f"{hair_c} hair")
    if skin:
        cufs_parts.append(f"{skin} skin")
    if eye_c:
        cufs_parts.append(f"{eye_c} eyes")
    cufs_tail = ", ".join(cufs_parts[:-1]) + (", and " + cufs_parts[-1] if len(cufs_parts) > 1 else cufs_parts[0]) if cufs_parts else ""
    cufs_sentence = f"A photo of a {intrinsic}" + (f" with {cufs_tail}" if cufs_tail else "") + "."

    # Phenotype reinforcement: SD 1.5 and FLUX tend to drift toward
    # generic East-Asian or European faces even with ethnicity named.
    # Adding an explicit phenotype sentence fights that drift.
    phenotype_sentence = ""
    if ethnicity and ethnicity in _ETH_PHENOTYPE:
        phenotype_sentence = _ETH_PHENOTYPE[ethnicity] + "."

    # ── 2. Religious/cultural attire — strongest-signal clothing cue,
    #    emitted BEFORE accessories so the model fixes hair/head
    #    covering first and doesn't invent unrelated clothing.
    attire = attrs.get("religious_attire", []) or []
    attire_line = ""
    if attire:
        attire_phrases = []
        for item in attire:
            phrase = _ATTIRE_PROMPT.get(item, f"wearing a {item}")
            if phrase not in attire_phrases:
                attire_phrases.append(phrase)
        attire_line = ", ".join(attire_phrases) + "."

    # ── 3. Accessories line — these are generated by the AI model ──
    accessories = attrs.get("accessories", []) or []
    accessory_line = ""
    if accessories:
        accessory_line = "Wearing: " + ", ".join(accessories) + "."

    # ── 3. Secondary features ──
    secondary = []
    if hair_s:
        secondary.append(f"{hair_s} hairstyle")
    if attrs.get("face_shape"):
        secondary.append(f"{attrs['face_shape']} face")
    if attrs.get("forehead"):
        secondary.append(attrs["forehead"])
    if attrs.get("eyebrows"):
        secondary.append(f"{attrs['eyebrows']} eyebrows")
    if attrs.get("eye_shape"):
        secondary.append(f"{attrs['eye_shape']} eyes")
    if attrs.get("nose"):
        secondary.append(f"{attrs['nose']} nose")
    if attrs.get("lips"):
        secondary.append(attrs["lips"])
    if attrs.get("chin"):
        secondary.append(attrs["chin"])
    if attrs.get("facial_hair"):
        secondary.append(attrs["facial_hair"])
    if attrs.get("build"):
        secondary.append(f"{attrs['build']} build")
    secondary_line = ""
    if secondary:
        secondary_line = "Features: " + ", ".join(secondary) + "."

    # ── Assemble ──
    # NOTE: Scars/moles are NOT in the prompt — rendered in post-processing.
    # Only accessories (bindi, earrings, nose ring) go in the prompt.
    # Style/quality tokens are appended by face_generator._fit_to_clip()
    # which uses the actual CLIP tokenizer to fill exactly 75 tokens.

    positive = " ".join(filter(None, [
        cufs_sentence,
        phenotype_sentence,
        attire_line,
        accessory_line,
        secondary_line,
    ]))

    # ── Color-biased negative prompt ──
    color_negatives = []
    if skin and skin.lower() in _SKIN_COMPETITORS:
        color_negatives.extend(_SKIN_COMPETITORS[skin.lower()])
    if hair_c and hair_c.lower() in _HAIR_COMPETITORS:
        color_negatives.extend(_HAIR_COMPETITORS[hair_c.lower()])
    if eye_c and eye_c.lower() in _EYE_COMPETITORS:
        color_negatives.extend(_EYE_COMPETITORS[eye_c.lower()])

    # Ethnicity-biased negatives: block competing ethnic face types
    if ethnicity and ethnicity in _ETH_NEGATIVE:
        color_negatives.extend(_ETH_NEGATIVE[ethnicity])

    base_negative = (
        "cartoon, anime, illustration, painting, drawing, sketch, 3D render, CGI, "
        "blurry, out of focus, low quality, deformed, distorted, disfigured, "
        "bad anatomy, wrong proportions, extra limbs, watermark, text, signature, "
        "logo, multiple people, group, side view, profile, back of head, "
        "smiling, laughing, open mouth, hat, sunglasses, mask, makeup, jewelry"
    )
    negative = base_negative
    if color_negatives:
        # Dedup while preserving order
        seen = set()
        unique = [c for c in color_negatives if not (c in seen or seen.add(c))]
        negative = ", ".join(unique) + ", " + base_negative

    return positive, negative


def format_attributes_display(attrs: dict) -> str:
    """Format attributes as a readable display string for the UI."""
    lines = []
    display_map = {
        "gender": "Gender",
        "age": "Age",
        "ethnicity": "Ethnicity",
        "skin_tone": "Skin Tone",
        "face_shape": "Face Shape",
        "forehead": "Forehead",
        "eyebrows": "Eyebrows",
        "eye_shape": "Eye Shape",
        "eye_color": "Eye Color",
        "nose": "Nose",
        "lips": "Lips",
        "chin": "Chin",
        "facial_hair": "Facial Hair",
        "hair_color": "Hair Color",
        "hair_style": "Hair Style",
        "build": "Build",
    }

    for key, label in display_map.items():
        value = attrs.get(key)
        if value is not None:
            if key == "age":
                lines.append(f"  {label}: ~{value} years")
            else:
                lines.append(f"  {label}: {value}")

    marks = attrs.get("marks", [])
    if marks:
        lines.append("  --- Distinguishing Marks ---")
        for m in marks:
            desc = ""
            if m["descriptors"]:
                desc = " ".join(m["descriptors"]) + " "
            loc = f" on {m['location']}" if m["location"] else ""
            lines.append(f"    * {desc}{m['type']}{loc}")

    accessories = attrs.get("accessories", [])
    if accessories:
        lines.append("  --- Accessories (AI-generated) ---")
        for acc in accessories:
            lines.append(f"    * {acc}")

    attire = attrs.get("religious_attire", [])
    if attire:
        lines.append("  --- Religious / Cultural Attire ---")
        for item in attire:
            lines.append(f"    * {item}")

    if not lines:
        return "  No attributes detected"

    return "\n".join(lines)


# ─── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Module 2: Forensic Attribute Parser ===\n")

    test_descriptions = [
        "The suspect was a male, around 30 to 35 years old. He had dark brown skin, short black hair, and brown eyes. He had a thick mustache and a small scar near his left eyebrow. He was of medium build.",
        "She appeared to be a young woman, maybe 25, with fair skin and long brown wavy hair. She had green eyes and a mole on her right cheek. She had thin lips and an oval face.",
        "An elderly man, around 60, bald, with a full grey beard. He had a broad nose and dark skin. Very thin build. Deep-set eyes. Large scar on his left cheek.",
        "Indian male, about 32, dark complexion, short black hair. He had thick eyebrows, deep set brown eyes, a broad nose, and a thick black mustache. There was a visible vertical scar below his left eye. Small mole on the right side of his chin.",
        "Young woman around 20, fair skin with freckles. Red curly hair, green eyes, small button nose. She had a piercing on her left nostril and a tiny scar on her upper lip.",
    ]

    for i, desc in enumerate(test_descriptions, 1):
        print(f"{'='*60}")
        print(f"TEST {i}")
        print(f"{'='*60}")
        print(f"Input: {desc}\n")
        attrs = parse_description(desc)
        print(f"Extracted Attributes:")
        print(format_attributes_display(attrs))
        pos, neg = build_prompt(attrs)
        print(f"\nGenerated Prompt:")
        print(f"  {pos[:200]}...")
        print()
