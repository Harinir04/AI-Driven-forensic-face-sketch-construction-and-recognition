"""
repair.py — one-shot cleanup for stale state in a Forensic AI install.

Run from the project root:

    python repair.py

What it does (and only this):
  • Deletes every cached face-embedding pickle (models/db_embeddings_*.pkl)
    and any FAISS index file in models/. These caches contain criminal
    names + embeddings from the developer's machine and can leak into a
    fresh client install, producing "ghost" matches with broken
    thumbnails (the photos were correctly excluded, but the cache was
    not in older builds of _make_client_zip.py).
  • Removes any orphan generated-face / upload artefacts left behind by
    interrupted runs.
  • Leaves data/forensic.db and data/criminal_db/ alone — those hold
    the user's own data.

The matcher will rebuild its index from whatever criminal photos exist
in data/criminal_db/ the next time the app runs and a match is requested
(or when an admin clicks "Rebuild Face Database").
"""

from __future__ import annotations

import sys
from pathlib import Path


GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def _delete(path: Path) -> bool:
    """Delete a file if it exists. Returns True if something was deleted."""
    try:
        if path.is_file():
            path.unlink()
            return True
    except Exception as e:
        print(f"  {RED}!{RESET} could not delete {path}: {e}")
    return False


def main() -> int:
    root = Path(__file__).resolve().parent
    print(f"Repairing install at: {root}\n")

    deleted = 0
    inspected = 0

    # ── 1. Embedding caches and any future FAISS index ──────────────
    print("Scanning models/ for stale caches...")
    models_dir = root / "models"
    if models_dir.is_dir():
        patterns = ("db_embeddings_*.pkl", "*.faiss", "face_index*")
        for pat in patterns:
            for p in models_dir.glob(pat):
                inspected += 1
                if _delete(p):
                    print(f"  {GREEN}-{RESET} removed {p.relative_to(root)}")
                    deleted += 1
    else:
        print(f"  {YELLOW}~{RESET} models/ does not exist yet (will be created on first run)")

    # ── 2. Orphan runtime artefacts ─────────────────────────────────
    print("\nScanning data/ for orphan runtime artefacts...")
    artefact_dirs = [
        ("data/generated_faces", ("*.png", "*.jpg", "*.jpeg")),
        ("data/uploads",         ("*.png", "*.jpg", "*.jpeg", "*.webm", "*.wav", "*.mp3", "*.m4a", "*.ogg")),
        ("data/test_audio",      ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.webm")),
    ]
    for rel_dir, globs in artefact_dirs:
        d = root / rel_dir
        if not d.is_dir():
            continue
        for g in globs:
            for p in d.glob(g):
                inspected += 1
                if _delete(p):
                    print(f"  {GREEN}-{RESET} removed {p.relative_to(root)}")
                    deleted += 1

    # ── 3. Sidecar JSON metadata stranded without an image ──────────
    print("\nScanning data/criminal_db/ for orphan sidecar JSON...")
    cdb = root / "data" / "criminal_db"
    if cdb.is_dir():
        for j in cdb.glob("*.json"):
            inspected += 1
            stem = j.stem
            has_image = any(
                (cdb / f"{stem}{ext}").is_file()
                for ext in (".jpg", ".jpeg", ".png")
            )
            if not has_image:
                if _delete(j):
                    print(f"  {GREEN}-{RESET} removed orphan {j.relative_to(root)}")
                    deleted += 1

    # ── Summary ─────────────────────────────────────────────────────
    print()
    if deleted == 0:
        print(f"{GREEN}OK{RESET}  Nothing to repair — install already clean "
              f"({inspected} files inspected).")
    else:
        print(f"{GREEN}OK{RESET}  Repair complete. "
              f"{deleted} stale file(s) removed of {inspected} inspected.")
        print()
        print("Next steps:")
        print("  1. Start the app:        python app.py   (or run.bat)")
        print("  2. Log in as Admin and add criminal photos via the Admin panel.")
        print("  3. Click 'Rebuild Face Database' so the matcher indexes them.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
