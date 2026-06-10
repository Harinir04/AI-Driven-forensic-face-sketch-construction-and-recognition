"""
Forensic AI System — Setup & Compatibility Checker
Run this on any machine to verify everything works.

Usage:
    python setup_check.py          # Check & fix everything
    python setup_check.py --install # Also install missing packages
"""

import importlib
import os
import platform
import sys
from pathlib import Path


def main():
    print()
    print("=" * 60)
    print("  Forensic AI System — Setup Checker")
    print("=" * 60)
    print()

    install_mode = "--install" in sys.argv
    all_ok = True

    # ── 1. Python version ──
    v = sys.version_info
    py_ver = f"{v.major}.{v.minor}.{v.micro}"
    print(f"  Python version:  {py_ver}")
    print(f"  Platform:        {platform.system()} {platform.machine()}")
    print(f"  Executable:      {sys.executable}")
    print()

    if v < (3, 10):
        print(f"  [FAIL] Python 3.10+ required. You have {py_ver}")
        print(f"         Download from: https://www.python.org/downloads/")
        return False

    print(f"  [OK] Python {py_ver} is supported (3.10+)")
    print()

    # ── 2. Check packages ──
    print("  Checking packages:")
    packages = {
        "flask":            "flask",
        "flask_login":      "flask-login",
        "flask_bcrypt":     "flask-bcrypt",
        "dotenv":           "python-dotenv",
        "requests":         "requests",
        "numpy":            "numpy",
        "cv2":              "opencv-python-headless",
        "PIL":              "Pillow",
        "onnxruntime":      "onnxruntime",
        "huggingface_hub":  "huggingface_hub",
    }

    # These are only needed for GENERATION_MODE=local
    local_packages = {
        "torch":            "torch",
        "diffusers":        "diffusers",
        "transformers":     "transformers",
        "accelerate":       "accelerate",
        "safetensors":      "safetensors",
        "peft":             "peft",
    }

    missing_core = []
    missing_local = []

    for import_name, pip_name in packages.items():
        try:
            m = importlib.import_module(import_name)
            ver = getattr(m, "__version__", "ok")
            print(f"    [OK]   {pip_name:28s} {ver}")
        except ImportError:
            print(f"    [MISS] {pip_name:28s} not installed")
            missing_core.append(pip_name)
            all_ok = False

    print()
    print("  Local generation packages (needed for GENERATION_MODE=local):")
    for import_name, pip_name in local_packages.items():
        try:
            m = importlib.import_module(import_name)
            ver = getattr(m, "__version__", "ok")
            print(f"    [OK]   {pip_name:28s} {ver}")
        except ImportError:
            print(f"    [MISS] {pip_name:28s} not installed")
            missing_local.append(pip_name)

    print()

    # ── 3. Check .env ──
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print("  [OK] .env file exists")
        # Check for placeholder values
        content = env_path.read_text()
        if "change-me" in content or "change-this" in content:
            print("  [WARN] SECRET_KEY is still the default — change it for production")
    else:
        print("  [MISS] .env file not found — copy .env.example to .env and add your keys")
        all_ok = False

    # ── 4. Check face detection models ──
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    det_model = model_dir / "det_10g.onnx"
    rec_model = model_dir / "w600k_r50.onnx"

    if det_model.exists() and rec_model.exists():
        det_mb = det_model.stat().st_size / (1024 * 1024)
        rec_mb = rec_model.stat().st_size / (1024 * 1024)
        print(f"  [OK] Face detection model:   {det_mb:.0f} MB")
        print(f"  [OK] Face recognition model: {rec_mb:.0f} MB")
    else:
        print(f"  [INFO] Face models not yet downloaded")
        print(f"         They will auto-download on first use (~330 MB)")
        print(f"         Location: {model_dir}")

    # ── 5. Check database ──
    db_path = Path(__file__).parent / "data" / "forensic.db"
    if db_path.exists():
        db_mb = db_path.stat().st_size / 1024
        print(f"  [OK] Database exists ({db_mb:.0f} KB)")
    else:
        print(f"  [INFO] Database will be created on first run")

    # ── 6. Check criminal DB images ──
    criminal_dir = Path(__file__).parent / "data" / "criminal_db"
    if criminal_dir.exists():
        images = list(criminal_dir.glob("*.jpg")) + list(criminal_dir.glob("*.jpeg")) + list(criminal_dir.glob("*.png"))
        if images:
            print(f"  [OK] Criminal database: {len(images)} face images")
        else:
            print(f"  [INFO] Criminal database is empty — add photos via Admin panel")
    print()

    # ── 7. Test module imports ──
    print("  Testing module imports:")
    modules_ok = True
    for mod_name in ["modules.onnx_face", "modules.face_matcher",
                     "modules.face_enhancer", "modules.attribute_parser",
                     "modules.face_generator", "modules.speech_to_text"]:
        try:
            importlib.import_module(mod_name)
            short = mod_name.split(".")[-1]
            print(f"    [OK] {short}")
        except Exception as e:
            short = mod_name.split(".")[-1]
            print(f"    [FAIL] {short}: {e}")
            modules_ok = False
            all_ok = False
    print()

    # ── 8. Install missing packages if requested ──
    if install_mode and (missing_core or missing_local):
        print("  Installing missing packages...")
        import subprocess
        to_install = missing_core + missing_local
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + to_install,
            capture_output=False
        )
        if result.returncode == 0:
            print(f"\n  [OK] Installed: {', '.join(to_install)}")
            print(f"  Run this script again to verify.")
        else:
            print(f"\n  [ERROR] Some packages failed to install.")
        return result.returncode == 0

    # ── Summary ──
    print("=" * 60)
    if all_ok and modules_ok:
        print("  ALL CHECKS PASSED")
        print()
        print("  To start the system:")
        if platform.system() == "Windows":
            print("    run.bat")
            print("    OR: python app.py")
        else:
            print("    python app.py")
        print()
        print("  Open: http://127.0.0.1:5000")
        print("  Login: superadmin / Admin@123")
    else:
        print("  SOME CHECKS FAILED")
        if missing_core:
            print(f"\n  Missing core packages: {', '.join(missing_core)}")
            print(f"  Fix: pip install {' '.join(missing_core)}")
        if missing_local:
            print(f"\n  Missing local-mode packages: {', '.join(missing_local)}")
            print(f"  Fix: pip install {' '.join(missing_local)}")
            print(f"  (Only needed if GENERATION_MODE=local)")
        if not modules_ok:
            print(f"\n  Some modules failed to import — check errors above")
        print(f"\n  Or run: python setup_check.py --install")

    print("=" * 60)
    print()
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
