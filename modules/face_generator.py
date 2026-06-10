"""
Module 3: Forensic Face Generation
------------------------------------
Two modes only, selected via config.GENERATION_MODE:

  "local" → SD 1.5 Realistic Vision + LCM-LoRA on this machine.
            ControlNet lineart sketch-to-photo included. ~2.5 GB one-time
            download. Fully offline after first run. Best reliability.

  "api"   → FLUX.1-schnell via HuggingFace Inference API. Requires a
            paid provider (fal-ai recommended, ~$0.003/image) linked
            to HF_API_TOKEN. Sketch uploads are not supported.

Heavy torch/diffusers imports are lazy: api mode never imports them.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_sd_pipe         = None
_controlnet_pipe = None


# ═══════════════════════════════════════════════════════════════
# SFW guardrails — applied to every generation path (local + api).
# Forensic mugshots must never contain nudity. We rely on strong
# prompt-side steering (positive clothing hint + explicit negatives)
# because enabling the diffusers safety_checker model would add a
# ~400 MB download and sometimes false-positive on dark skin tones.
# ═══════════════════════════════════════════════════════════════

_CLOTHING_HINT = "wearing gray shirt, clothed."

_SFW_NEGATIVE = (
    "nude, naked, nudity, nsfw, topless, shirtless, bare chest, "
    "bare shoulders, exposed skin below neck, underwear, lingerie, "
    "bikini, cleavage, sexual, suggestive, "
)

# Style/quality tokens added to fill remaining CLIP budget (75 tokens).
# Ordered by importance — added until budget is full.
_STYLE_FILL = [
    "front-facing mugshot,",
    "neutral expression,",
    "gray background,",
    "realistic photo,",
    "sharp focus,",
    "detailed skin texture,",
    "85mm lens,",
    "DSLR,",
    "studio lighting,",
    "direct eye contact,",
    "one person,",
    "high resolution.",
]

_clip_tokenizer = None


def _get_clip_tokenizer():
    """Lazy-load CLIP tokenizer for token counting."""
    global _clip_tokenizer
    if _clip_tokenizer is None:
        try:
            from transformers import CLIPTokenizer
            _clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        except Exception:
            _clip_tokenizer = "unavailable"
    return _clip_tokenizer


def _count_tokens(text: str) -> int:
    """Count CLIP tokens. Falls back to word-based estimate."""
    tok = _get_clip_tokenizer()
    if tok and tok != "unavailable":
        return len(tok.encode(text)) - 2  # subtract start/end tokens
    return int(len(text.split()) * 1.3)


def _apply_safety(positive: str, negative: str) -> tuple[str, str]:
    """
    Build final prompt that fills exactly 75 CLIP tokens:
    1. Start with core identity prompt from attribute_parser
    2. Append clothing hint (SFW)
    3. Fill remaining tokens with style/quality descriptors
    """
    pos = positive.strip()

    # Add clothing hint
    if "clothed" not in pos.lower():
        pos = f"{pos} {_CLOTHING_HINT}"

    # Fill remaining budget with style tokens
    for phrase in _STYLE_FILL:
        candidate = f"{pos} {phrase}"
        if _count_tokens(candidate) <= 75:
            pos = candidate
        else:
            break

    token_count = _count_tokens(pos)
    print(f"  [prompt] {token_count}/75 CLIP tokens used")

    neg = (negative or "").strip()
    if "nude" not in neg.lower():
        neg = _SFW_NEGATIVE + neg
    return pos, neg


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _describe_exc(e: Exception) -> str:
    """Build a non-empty description even when HfHubHTTPError has no message."""
    msg = str(e).strip() or repr(e)
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            body = resp.text[:200] if hasattr(resp, "text") else ""
            msg = f"HTTP {resp.status_code} {msg} {body}".strip()
        except Exception:
            pass
    return f"{type(e).__name__}: {msg}"


def _to_pencil_sketch(img: Image.Image) -> Image.Image:
    """
    Convert a realistic face photo into a pencil-drawing style using the
    classic color-dodge technique (gray + inverted-blurred-gray dodge).
    Used to render one of the N variants as a forensic-artist sketch.
    """
    import numpy as np
    import cv2

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    # Color dodge: out = gray * 255 / (255 - blur), clipped
    denom = 255 - blur
    denom[denom == 0] = 1
    dodge = cv2.divide(gray, denom, scale=256.0)
    dodge = np.clip(dodge, 0, 255).astype(np.uint8)
    # Slight contrast lift so lines read as pencil strokes, not photocopy
    dodge = cv2.addWeighted(dodge, 1.15, dodge, 0, -15)
    dodge = np.clip(dodge, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(dodge, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _placeholder(i: int) -> Image.Image:
    from PIL import ImageDraw
    img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    draw.text(
        (config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2),
        f"Face #{i + 1}\nGeneration failed",
        fill=(100, 100, 100),
        anchor="mm",
    )
    save_path = config.GENERATED_DIR / f"generated_{i + 1}.png"
    img.save(save_path)
    return img


# ═══════════════════════════════════════════════════════════════
# LOCAL MODE: SD 1.5 Realistic Vision + LCM-LoRA (text-to-photo)
# ═══════════════════════════════════════════════════════════════

def _load_sd_pipeline():
    """Load SD 1.5 + LCM-LoRA. ~2.5 GB first run, cached thereafter."""
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler

    print("  [local] Loading SD 1.5 Realistic Vision + LCM-LoRA...")
    print("  (first run downloads ~2.5 GB, subsequent runs use the cache)")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing(1)
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    _sd_pipe = pipe
    print(f"  [local] SD pipeline ready on {device}.")
    return pipe


def generate_from_text(
    positive_prompt: str,
    negative_prompt: str,
    num_images: int = 4,
) -> list[Image.Image]:
    """Text-to-photo via local SD 1.5 + LCM. 4 diverse candidates."""
    import torch

    pipe = _load_sd_pipeline()

    base_positive, base_negative = _apply_safety(
        positive_prompt,
        (negative_prompt or "").strip() or (
            "cartoon, anime, blurry, deformed, multiple people, "
            "watermark, text, side view, eyes closed, smiling"
        ),
    )

    # Consecutive seeds for identity consistency
    base_seed = 7331
    lighting_hints = [
        "soft even studio lighting",
        "clinical flash lighting",
        "warm tungsten lighting",
        "neutral daylight balanced lighting",
    ]

    images = []
    for i in range(num_images):
        variant = f"{base_positive} {lighting_hints[i % len(lighting_hints)]}."
        print(f"  [local] Generating face {i + 1}/{num_images}...")
        t = time.time()

        result = pipe(
            prompt=variant,
            negative_prompt=base_negative,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            generator=torch.Generator().manual_seed(base_seed + i),
        )
        img = result.images[0]
        img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        images.append(img)

        save_path = config.GENERATED_DIR / f"generated_{i + 1}.png"
        img.save(save_path)
        print(f"    saved: {save_path.name} ({time.time() - t:.1f}s)")

    return images


# ═══════════════════════════════════════════════════════════════
# LOCAL MODE: ControlNet lineart sketch-to-photo
# ═══════════════════════════════════════════════════════════════

def _load_controlnet_pipeline():
    global _controlnet_pipe
    if _controlnet_pipe is not None:
        return _controlnet_pipe

    import torch
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        LCMScheduler,
    )

    print("  [local] Loading ControlNet lineart + SD 1.5...")
    print("  (first run downloads ~1.5 GB ControlNet + reuses SD 1.5 cache)")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_MODEL_ID, torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.SD_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # LCM-LoRA here too — same speedup for ControlNet
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing(1)
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    _controlnet_pipe = pipe
    print(f"  [local] ControlNet pipeline ready on {device}.")
    return pipe


def _preprocess_sketch(sketch_image: Image.Image) -> Image.Image:
    """Clean a sketch into crisp B&W lineart for ControlNet."""
    import numpy as np
    import cv2

    gray = np.array(sketch_image.convert("L"))
    if gray.mean() < 110:
        gray = 255 - gray  # auto-invert scanned negatives

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 25, 10,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    lineart_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(lineart_rgb)


def generate_from_sketch(
    sketch_image: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    num_images: int = 4,
    controlnet_conditioning_scale: float = 0.9,
) -> list[Image.Image]:
    """Sketch-to-photo via local ControlNet lineart + SD 1.5 + LCM-LoRA."""
    import torch

    pipe = _load_controlnet_pipeline()
    sketch_processed = _preprocess_sketch(sketch_image)
    sketch_resized = sketch_processed.resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS
    )

    base_positive, base_negative = _apply_safety(
        positive_prompt,
        (negative_prompt or "").strip() or (
            "cartoon, anime, blurry, deformed, multiple people, "
            "watermark, text, side view, eyes closed, smiling"
        ),
    )

    base_seed = 7331
    lighting_hints = [
        "soft even studio lighting",
        "clinical flash lighting",
        "warm tungsten lighting",
        "neutral daylight balanced lighting",
    ]

    images = []
    for i in range(num_images):
        variant = f"{base_positive} {lighting_hints[i % len(lighting_hints)]}."
        print(f"  [local] Sketch-to-photo {i + 1}/{num_images}...")
        t = time.time()

        result = pipe(
            prompt=variant,
            negative_prompt=base_negative,
            image=sketch_resized,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            generator=torch.Generator().manual_seed(base_seed + i),
        )
        img = result.images[0]
        img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        images.append(img)

        save_path = config.GENERATED_DIR / f"generated_{i + 1}.png"
        img.save(save_path)
        print(f"    saved: {save_path.name} ({time.time() - t:.1f}s)")

    return images


# ═══════════════════════════════════════════════════════════════
# API MODE: FLUX.1-schnell via Cloudflare Workers AI (free tier)
# ═══════════════════════════════════════════════════════════════
#
# Why Cloudflare over HuggingFace paid providers:
#   - Free tier: 10,000 neurons/day ≈ 100–200 images at no cost
#   - FLUX-schnell handles ethnicity prompts far better than SD 1.5
#     (critical for forensic accuracy — "South Indian Tamil" should
#     not produce East-Asian faces)
#   - No per-image charge, no payment-required failures
#
# FLUX-schnell notes:
#   - CFG is 0 (guidance is baked in) so it IGNORES negative prompts.
#     All ethnicity/quality steering must go into the POSITIVE prompt.
#   - 4 steps is the designed optimum; 8 max on Cloudflare.
#   - Native resolution 1024×1024 — we downsize to config.IMAGE_SIZE
#     after generation so the rest of the pipeline is unchanged.

def _cloudflare_flux_request(
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
) -> "Image.Image":
    """
    One FLUX.1-schnell call via Cloudflare Workers AI.

    Returns a PIL.Image. Raises a clear RuntimeError on auth/quota
    failures so the UI can surface the problem instead of silently
    returning a placeholder.
    """
    import base64
    import io
    import requests

    account_id = config.CLOUDFLARE_ACCOUNT_ID
    token      = config.CLOUDFLARE_API_TOKEN

    if not account_id or not token:
        raise RuntimeError(
            "Cloudflare credentials are not set in .env.\n"
            "Add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "(free at https://dash.cloudflare.com — Account ID on the "
            "dashboard, API token under My Profile → API Tokens with "
            "Workers AI read permission), or switch GENERATION_MODE=local."
        )

    # Fail fast on the most common setup mistake: pasting an email or
    # API token into the ACCOUNT_ID field. Real IDs are 32 hex chars.
    if "@" in account_id or len(account_id) < 20:
        raise RuntimeError(
            f"CLOUDFLARE_ACCOUNT_ID looks invalid: {account_id!r}\n"
            f"Expected: 32-character hex string (e.g. a1b2c3d4e5f6...).\n"
            f"Find yours at https://dash.cloudflare.com — it's on the "
            f"right sidebar of the dashboard, labelled 'Account ID'. "
            f"Do NOT paste your email or API token here."
        )

    url = (
        f"{config.CLOUDFLARE_API_BASE}/{account_id}"
        f"/ai/run/{config.CLOUDFLARE_IMAGE_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    body = {
        "prompt": prompt,
        "steps":  config.CLOUDFLARE_FLUX_STEPS,
        "seed":   seed,
        "width":  width,
        "height": height,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
        except requests.RequestException as e:
            if attempt < 2:
                print(f"    Cloudflare network error (retry {attempt+1}/3): {e}")
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Cloudflare request failed: {e}") from e

        if resp.status_code == 401 or resp.status_code == 403:
            raise RuntimeError(
                f"Cloudflare API returned {resp.status_code}. Check "
                f"CLOUDFLARE_API_TOKEN in .env — it needs Workers AI read "
                f"permission on the correct account."
            )

        if resp.status_code == 429:
            # Daily quota exhausted OR per-minute rate limit
            if attempt < 2:
                print(f"    Cloudflare rate limit, back-off {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(
                "Cloudflare Workers AI free-tier quota exhausted "
                "(10,000 neurons/day). Resets at UTC midnight. Switch "
                "to GENERATION_MODE=local to continue today."
            )

        if resp.status_code >= 500:
            if attempt < 2:
                print(f"    Cloudflare {resp.status_code}, retrying...")
                time.sleep(3)
                continue
            raise RuntimeError(
                f"Cloudflare server error {resp.status_code}: {resp.text[:200]}"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Cloudflare HTTP {resp.status_code}: {resp.text[:300]}"
            )

        # FLUX on Cloudflare returns JSON: { "result": { "image": "<base64>" }, ... }
        # SDXL variants return raw image bytes, so we handle both.
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            payload = resp.json()
            if not payload.get("success", True):
                errs = payload.get("errors", [])
                raise RuntimeError(f"Cloudflare API error: {errs}")
            img_b64 = payload.get("result", {}).get("image")
            if not img_b64:
                raise RuntimeError(
                    f"Cloudflare returned no image: {str(payload)[:300]}"
                )
            img_bytes = base64.b64decode(img_b64)
        else:
            img_bytes = resp.content

        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise RuntimeError("Cloudflare API failed after 3 attempts")


def _pollinations_request(
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
) -> "Image.Image":
    """
    One image from Pollinations.ai free legacy endpoint.

    GET https://image.pollinations.ai/prompt/<url-encoded-prompt>
        ?model=flux&width=1024&height=1024&seed=42&nologo=true

    No auth required. Runs FLUX-dev under the hood — noticeably
    higher face-quality and ethnicity fidelity than Cloudflare's
    FLUX-schnell, but 10-70s per cold generation.
    """
    import io
    import urllib.parse
    import requests

    url = (
        f"{config.POLLINATIONS_BASE}/"
        f"{urllib.parse.quote(prompt)}"
        f"?model={config.POLLINATIONS_MODEL}"
        f"&width={width}&height={height}"
        f"&seed={seed}&nologo=true"
    )

    # Pollinations free tier limits each IP to 1 concurrent request.
    # With serialized callers (max_workers=1 in generate_from_api) the
    # 429 path should rarely fire, but we keep a robust backoff anyway
    # in case another process on the same LAN is also hitting the API.
    max_attempts = 5
    backoff_schedule = [3, 8, 15, 30]  # seconds between attempts

    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, timeout=config.POLLINATIONS_TIMEOUT)
        except requests.RequestException as e:
            if attempt < max_attempts - 1:
                delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                print(f"    Pollinations network error "
                      f"(retry {attempt + 1}/{max_attempts} in {delay}s): {e}")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Pollinations request failed: {e}") from e

        if resp.status_code == 200 and resp.headers.get(
            "content-type", ""
        ).startswith("image/"):
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        # Transient errors: queue-full, server overload, bad gateway
        preview = resp.text[:200] if resp.text else ""
        if resp.status_code in (429, 502, 503, 504):
            if attempt < max_attempts - 1:
                delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                print(f"    Pollinations {resp.status_code} "
                      f"(retry {attempt + 1}/{max_attempts} in {delay}s): "
                      f"{preview[:160]}")
                time.sleep(delay)
                continue

        raise RuntimeError(
            f"Pollinations HTTP {resp.status_code}: {preview}"
        )

    raise RuntimeError(f"Pollinations API failed after {max_attempts} attempts")


def generate_from_api(
    positive_prompt: str,
    negative_prompt: str,
    num_images: int = 4,
) -> list[Image.Image]:
    """
    4 candidates via the selected free API provider, in parallel.
    Provider is chosen by USE_CLOUDFLARE in .env:
      true  → Cloudflare Workers AI FLUX-schnell (~2-4s/image, quota-limited)
      false → Pollinations.ai FLUX-dev (~10-70s/image, no quota, better quality)

    FLUX in both providers runs at CFG 0 and ignores negative prompts,
    so all ethnicity/quality steering lives in the positive prompt
    (the phenotype sentence from attribute_parser._ETH_PHENOTYPE).
    """
    # FLUX has a ~512-token text encoder budget vs CLIP's 75, so we
    # skip the SFW token-trim pass and just bake the clothing hint +
    # a forensic-mugshot style anchor into the positive prompt directly.
    base_prompt = positive_prompt.strip()
    if "clothed" not in base_prompt.lower():
        base_prompt = f"{base_prompt} {_CLOTHING_HINT}"
    base_prompt = (
        f"{base_prompt} front-facing forensic mugshot, neutral expression, "
        f"even gray studio background, realistic photograph, "
        f"sharp focus, detailed skin texture, direct eye contact, "
        f"one person only, high resolution, photorealistic."
    )

    base_seed = 7331
    lighting_hints = [
        "soft even studio lighting",
        "clinical flash lighting",
        "warm tungsten lighting",
        "neutral daylight balanced lighting",
    ]

    n = min(num_images, config.MAX_NUM_IMAGES)
    use_cf = config.USE_CLOUDFLARE
    provider = "Cloudflare FLUX-schnell" if use_cf else "Pollinations FLUX-dev"

    # Concurrency has to match the provider's per-IP limit:
    #   Cloudflare: no per-concurrent limit on free tier → fire all in parallel
    #   Pollinations free tier: hard limit of 1 queued request per IP
    #     (otherwise you get HTTP 429 "Queue full for IP"). Must serialize.
    max_workers = n if use_cf else 1
    print(f"  [api] Generating {n} candidates via {provider} "
          f"({'parallel' if max_workers > 1 else 'serialized'})...")

    def _one(i: int) -> tuple[int, Image.Image]:
        variant = f"{base_prompt} {lighting_hints[i % len(lighting_hints)]}."
        t = time.time()
        print(f"  [api] candidate {i + 1}/{n} — requesting...")
        if use_cf:
            img = _cloudflare_flux_request(variant, seed=base_seed + i)
        else:
            img = _pollinations_request(variant, seed=base_seed + i)
        img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        save_path = config.GENERATED_DIR / f"generated_{i + 1}.png"
        img.save(save_path)
        print(f"    saved: {save_path.name} ({time.time() - t:.1f}s)")
        return i, img

    results: list[Image.Image | None] = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, img in pool.map(_one, range(n)):
            results[i] = img
    return [img for img in results if img is not None]


# ═══════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════

def generate(
    positive_prompt: str,
    negative_prompt: str,
    num_images: int = 4,
    sketch_image: Image.Image = None,
    marks: list = None,
) -> list[Image.Image]:
    """
    Two-stage face generation:

    Stage 1 — Generate base face via AI model.
      Accessories (bindi, earrings, nose ring) are in the prompt.
      Scars/moles are NOT in the prompt.

    Stage 2 — Post-process: render scars/moles at precise pixel
      locations using face landmark detection + OpenCV compositing.

    Modes:
      local  → SD 1.5 + LCM-LoRA (+ ControlNet if sketch provided)
      api    → FLUX.1-schnell via HuggingFace (text-only, no sketch)
    """
    mode = (config.GENERATION_MODE or "local").lower()
    num_images = min(num_images, config.MAX_NUM_IMAGES)
    print(f"  [generate] Mode: {mode} (from config.GENERATION_MODE={config.GENERATION_MODE!r})")

    # ── Stage 1: AI generation ──
    if mode == "local":
        if sketch_image is not None:
            base_images = generate_from_sketch(
                sketch_image, positive_prompt, negative_prompt, num_images
            )
        else:
            base_images = generate_from_text(positive_prompt, negative_prompt, num_images)
    elif mode == "api":
        if sketch_image is not None:
            raise RuntimeError(
                "Sketch-to-photo is only supported in GENERATION_MODE=local. "
                "FLUX.1-schnell via the HuggingFace API does not accept a "
                "sketch reference. Either switch to local mode in .env, or "
                "remove the sketch upload and generate from attributes only."
            )
        base_images = generate_from_api(positive_prompt, negative_prompt, num_images)
    else:
        raise RuntimeError(
            f"Unknown GENERATION_MODE={mode!r}. Valid options: 'local', 'api'."
        )

    # ── Stage 2: Post-process scars/moles onto generated faces ──
    if marks:
        from modules.face_enhancer import add_marks_to_face

        print(f"  [stage2] Adding {len(marks)} marks to {len(base_images)} faces...")
        enhanced_images = []
        for i, img in enumerate(base_images):
            enhanced = add_marks_to_face(img, marks)
            enhanced_images.append(enhanced)

            # Re-save the enhanced image over the base
            save_path = config.GENERATED_DIR / f"generated_{i + 1}.png"
            enhanced.save(save_path)
            print(f"    marks applied: {save_path.name}")

        base_images = enhanced_images

    # ── Stage 3: Render one pencil-sketch variant ──
    # When the officer asks for 2 or 4 variants, the last slot is replaced
    # with a forensic-artist pencil drawing of variant #1. A single-image
    # run stays as a photo so ArcFace matching never loses its input.
    if len(base_images) >= 2:
        sketch_idx = len(base_images) - 1
        print(f"  [sketch] Rendering pencil sketch into slot "
              f"{sketch_idx + 1}/{len(base_images)}...")
        sketch_img = _to_pencil_sketch(base_images[0])
        sketch_path = config.GENERATED_DIR / f"generated_{sketch_idx + 1}.png"
        sketch_img.save(sketch_path)
        base_images[sketch_idx] = sketch_img

    return base_images


def unload_pipelines():
    """Free RAM by dropping local pipelines."""
    global _sd_pipe, _controlnet_pipe
    import gc
    _sd_pipe = None
    _controlnet_pipe = None
    gc.collect()
    print("  Pipelines unloaded.")
