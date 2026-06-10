"""
Forensic Combined System — Configuration
Merges System A (AI pipeline) + System B (Flask/DB) settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CRIMINAL_DB_DIR = DATA_DIR / "criminal_db"
GENERATED_DIR   = DATA_DIR / "generated_faces"
UPLOADS_DIR     = DATA_DIR / "uploads"
MODELS_DIR      = BASE_DIR / "models"
DATABASE_PATH   = DATA_DIR / "forensic.db"

# Create dirs on import
for d in [CRIMINAL_DB_DIR, GENERATED_DIR, UPLOADS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Flask ──────────────────────────────────────────────────────
SECRET_KEY              = os.getenv("SECRET_KEY", "change-this-in-production-please")
SESSION_LIFETIME_HOURS  = 8
MAX_UPLOAD_MB           = 16

# ── AI APIs ────────────────────────────────────────────────────
DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL    = "nova-2"
DEEPGRAM_LANGUAGE = "en"
TEST_AUDIO_DIR    = DATA_DIR / "test_audio"

# Cloudflare Workers AI — free tier (10,000 neurons/day) hosts
# FLUX.1-schnell. Get both values at https://dash.cloudflare.com
# (Account ID is on the dashboard; API token under My Profile →
# API Tokens with "Workers AI" read permission).
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN",  "")

# Legacy HuggingFace token — kept so existing .env files don't break.
# Not used by the default api pipeline anymore.
HF_API_TOKEN     = os.getenv("HF_API_TOKEN", "")

# ── Face Generation ────────────────────────────────────────────
# GENERATION_MODE supports exactly two values:
#   "local" → SD 1.5 Realistic Vision + LCM-LoRA on this machine.
#             ControlNet sketch-to-photo included. ~2.5 GB one-time
#             download, then fully offline. Required for sketch uploads.
#   "api"   → FLUX.1-schnell via Cloudflare Workers AI (free tier,
#             10,000 neurons/day ≈ 100–200 images). Best quality for
#             ethnicity accuracy. Sketch uploads not supported — use
#             local mode for sketches.
GENERATION_MODE = os.getenv("GENERATION_MODE", "local")

# API mode has two free providers. Flip USE_CLOUDFLARE in .env:
#
#   USE_CLOUDFLARE=true  → Cloudflare Workers AI FLUX-schnell
#                          Fast (~2-4s/image), 10k neurons/day quota,
#                          requires CLOUDFLARE_ACCOUNT_ID + TOKEN.
#
#   USE_CLOUDFLARE=false → Pollinations.ai (image.pollinations.ai)
#                          Free forever, no auth, FLUX-dev under the
#                          hood (noticeably higher quality than schnell
#                          for forensic ethnicity accuracy), but slower
#                          (~10-70s/image — first call for a new prompt
#                          is slowest, cached calls return instantly).
USE_CLOUDFLARE = os.getenv("USE_CLOUDFLARE", "true").strip().lower() in ("1", "true", "yes", "on")

# Cloudflare FLUX-schnell model config
CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CLOUDFLARE_API_BASE    = "https://api.cloudflare.com/client/v4/accounts"
CLOUDFLARE_FLUX_STEPS  = 4   # schnell-optimal, max 8 on Cloudflare

# Pollinations.ai config
# NOTE: On the free /prompt/ endpoint the ?model= parameter is
# effectively ignored — flux, zimage, turbo all resolve to the same
# default backend. We still pass it for forward compatibility.
POLLINATIONS_BASE   = "https://image.pollinations.ai/prompt"
POLLINATIONS_MODEL  = os.getenv("POLLINATIONS_MODEL", "flux")
POLLINATIONS_TIMEOUT = 180   # seconds — first call can take ~70s

# Local SD model
SD_MODEL_ID  = "SG161222/Realistic_Vision_V5.1_noVAE"

# ControlNet model for sketch-to-photo
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_lineart"

# Generation parameters
# Local SD 1.5 path uses LCM-LoRA: 4 steps, CFG 1.5 (fast CPU inference).
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE      = 1.5
IMAGE_SIZE          = 512
NUM_IMAGES          = 4
MAX_NUM_IMAGES      = 4






# ── Face Matching v5 ──────────────────────────────────────────
# Layered, fail-soft pipeline:
#   L0  query face quality gate
#   L1  multi-image (4 candidates) + multi-model ensemble (ArcFace + AdaFace)
#         + horizontal-flip TTA, all averaged into one query vector per model
#   L2  cosine similarity over per-model L2-normalized DB matrix
#   L3  quality-aware score-level fusion of model similarities
#   L4  hard gates (gender from SQLite only, age, ethnicity)
#   L5  attribute soft scoring (skin/hair/marks with landmark zones)
#   L6  confidence tiering (HIGH/MEDIUM/LOW/NO_MATCH)
#
# Every stage degrades gracefully — never throws.
FACE_MODEL_DIR           = Path.home() / ".insightface" / "models" / "buffalo_l"
TOP_K_MATCHES            = 5

# Second model: AdaFace (quality-adaptive). Disabled by default because
# there is no canonical CDN-hosted ONNX export — users who want the
# accuracy boost should:
#   1. Convert the official PyTorch checkpoint themselves
#      (https://github.com/mk-minchul/AdaFace), OR grab a community
#      ONNX (e.g. PINTO_model_zoo / 290_AdaFace).
#   2. Place it at ADAFACE_MODEL_PATH (or set ADAFACE_DOWNLOAD_URL
#      to a direct .onnx URL the matcher can fetch on first run).
#   3. Set USE_ADAFACE=true in .env.
# When unavailable the matcher silently runs ArcFace-only with multi-
# image averaging + flip TTA, which already gives most of the gain.
ADAFACE_MODEL_PATH       = Path.home() / ".insightface" / "models" / "adaface_ir50.onnx"
ADAFACE_DOWNLOAD_URL     = os.getenv("ADAFACE_DOWNLOAD_URL", "")
USE_ADAFACE              = os.getenv("USE_ADAFACE", "false").strip().lower() in ("1", "true", "yes", "on")
USE_FLIP_TTA             = True   # horizontal flip averaging on query side
USE_MULTI_IMAGE_QUERY    = True   # average across all generated candidates

# Score-level fusion weights. Must sum to ~1.0. Re-normalized at runtime
# when AdaFace is unavailable.
MODEL_WEIGHT_ARCFACE     = 0.55
MODEL_WEIGHT_ADAFACE     = 0.45

# Final-score blend: structural × attribute. α (structural) is reduced
# automatically when query quality is low or witness gave many attributes.
STRUCTURAL_BASE_WEIGHT   = 0.45
ATTRIBUTE_BASE_WEIGHT    = 0.55

# Hard gates
# Tightened to 15: the previous 18 still let a 50yo query keep a 33yo
# DB entry (diff 17). 15 reflects what witnesses can plausibly misjudge
# while still rejecting two-decade gaps that produce false positives.
AGE_HARD_GATE_DELTA      = 15    # |query_age - db_age| > this → reject

# Quality (0–1) below this floor flips the matcher into low-quality mode:
# attribute weight bumped, results capped at MEDIUM tier.
QUALITY_FLOOR            = 0.35

# Confidence tiers — final score thresholds
TIER_HIGH                = 78.0
TIER_MEDIUM              = 65.0
TIER_LOW                 = 50.0  # below this → NO_MATCH (filtered out)

# ── Backwards-compat aliases (kept for one release) ──────────
# Older code paths read these names directly. New code uses the
# tier constants above.
MATCH_STRUCTURAL_WEIGHT     = STRUCTURAL_BASE_WEIGHT
MATCH_ATTRIBUTE_WEIGHT      = ATTRIBUTE_BASE_WEIGHT
MIN_MATCH_SCORE             = TIER_LOW
MATCH_CONFIDENCE_THRESHOLD  = TIER_MEDIUM

# ── Allowed upload extensions ──────────────────────────────────
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm"}
