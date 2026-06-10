"""
Module 1: Speech-to-Text using Deepgram Nova-2 REST API
Converts multilingual witness audio into English text.
Zero RAM usage - all processing happens in the cloud.

Uses direct REST API (not SDK) for maximum compatibility.
"""

import os
import sys
import mimetypes
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Deepgram REST endpoint
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe a witness audio file using Deepgram Nova-2 REST API.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, ogg, etc.)
        language: Language code ("en", "hi", "ta", "te") or None for auto-detect

    Returns:
        dict with keys: transcript, confidence, language
    """
    api_key = config.DEEPGRAM_API_KEY

    if not api_key or api_key == "your_deepgram_api_key_here":
        return _fallback_no_key(audio_path)

    # Check if file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return _fallback_no_key(audio_path)

    # Build query parameters
    params = {
        "model": config.DEEPGRAM_MODEL,
        "smart_format": "true",
        "punctuate": "true",
    }

    if language:
        params["language"] = language
    else:
        params["detect_language"] = "true"
        params["language"] = config.DEEPGRAM_LANGUAGE

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(audio_file))
    if not mime_type:
        mime_type = "audio/wav"

    # Send audio to Deepgram
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mime_type,
    }

    try:
        with open(audio_file, "rb") as f:
            response = requests.post(
                DEEPGRAM_API_URL,
                headers=headers,
                params=params,
                data=f.read(),
                timeout=60,
            )

        if response.status_code != 200:
            print(f"  Deepgram API error: {response.status_code} - {response.text[:200]}")
            return {
                "transcript": f"[Deepgram error: {response.status_code}]",
                "confidence": 0.0,
                "language": "error",
            }

        data = response.json()
        channel = data["results"]["channels"][0]
        alternative = channel["alternatives"][0]

        detected_lang = channel.get("detected_language", "unknown")
        confidence = alternative.get("confidence", 0.0)

        return {
            "transcript": alternative["transcript"],
            "confidence": round(confidence * 100, 1),
            "language": detected_lang,
        }

    except requests.exceptions.ConnectionError:
        print("  No internet connection. Using fallback.")
        return _fallback_no_key(audio_path)
    except Exception as e:
        print(f"  Transcription error: {e}")
        return {
            "transcript": f"[Error: {e}]",
            "confidence": 0.0,
            "language": "error",
        }


def _fallback_no_key(audio_path: str) -> dict:
    """
    Fallback when no Deepgram API key is set.
    Returns a sample transcript for testing the pipeline.
    """
    print("[WARNING] No Deepgram API key set. Using sample transcript for testing.")
    print("  Set DEEPGRAM_API_KEY in .env to enable real transcription.")

    sample = (
        "The suspect was a male, around 30 to 35 years old. "
        "He had dark brown skin, short black hair, and brown eyes. "
        "He had a thick mustache and a small scar near his left eyebrow. "
        "He was of medium build."
    )
    return {
        "transcript": sample,
        "confidence": 0.0,
        "language": "demo-mode",
    }


# ─── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Module 1: Speech-to-Text (Deepgram) ===\n")

    # Test with a sample file or fallback
    test_files = list(Path(config.TEST_AUDIO_DIR).glob("*"))
    if test_files:
        result = transcribe_audio(str(test_files[0]))
    else:
        print("No audio files found in data/test_audio/")
        print("Testing with fallback (no API key demo mode)...\n")
        result = _fallback_no_key("")

    print(f"Language:   {result['language']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Transcript: {result['transcript']}")
