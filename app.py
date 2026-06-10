"""
Forensic Combined System — Main Flask Application
Combines System A (AI pipeline) + System B (Flask/Role-based UI)
"""
import json
import os
import threading
import traceback
import uuid
from datetime import timedelta, datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory, abort,
)
from flask_login import (
    login_user, logout_user, login_required, current_user, UserMixin,
)
from werkzeug.utils import secure_filename

import config
from extensions import bcrypt, login_manager, init_db, get_db

# ── Dependency check (runs once at startup) ──────────────────
def _check_dependencies():
    import numpy as np
    print(f"  [OK] numpy {np.__version__}")

    try:
        import onnxruntime
        print(f"  [OK] onnxruntime {onnxruntime.__version__}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  ERROR: onnxruntime not available: {e}")
        print(f"  Fix:  pip install onnxruntime")
        print(f"{'='*60}\n")

    try:
        from modules.onnx_face import OnnxFaceProcessor
        print(f"  [OK] onnx_face processor (pure ONNX, no insightface needed)")
    except Exception as e:
        print(f"  [WARN] onnx_face import issue: {e}")

_check_dependencies()

# ── AI modules from System A ───────────────────────────────────
from modules.speech_to_text import transcribe_audio
from modules.attribute_parser import parse_description, build_prompt, format_attributes_display
from modules.face_generator import generate
from modules.face_matcher import FaceMatcher

# ══════════════════════════════════════════════════════════════
# App setup
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.permanent_session_lifetime = timedelta(hours=config.SESSION_LIFETIME_HOURS)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

bcrypt.init_app(app)
login_manager.init_app(app)

# Singleton face matcher — loaded once
_matcher = None

def get_matcher():
    global _matcher
    if _matcher is None:
        _matcher = FaceMatcher()
        _matcher.load_database()
    return _matcher


# ══════════════════════════════════════════════════════════════
# User model for Flask-Login
# ══════════════════════════════════════════════════════════════

class User(UserMixin):
    def __init__(self, row):
        self.id          = row["id"]
        self.username    = row["username"]
        self.role        = row["role"]
        self.full_name   = row["full_name"]
        self._active     = bool(row["is_active"])
        self.force_pw    = bool(row["force_password_change"])

    # Override UserMixin's is_active property to use our DB value
    @property
    def is_active(self):
        return self._active

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return User(row) if row else None


# ══════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))
            if current_user.role not in roles:
                abort(403)
            if current_user.force_pw and request.endpoint != "change_password":
                flash("Please change your password before continuing.", "warning")
                return redirect(url_for("change_password"))
            return f(*args, **kwargs)
        return wrapped
    return decorator


def audit(action, target_type=None, target_id=None, details=None):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO audit_log (user_id, action, target_type, target_id, details, ip_address) "
            "VALUES (?,?,?,?,?,?)",
            (
                current_user.id if current_user.is_authenticated else None,
                action, target_type, target_id,
                json.dumps(details) if details else None,
                request.remote_addr,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_IMAGE_EXTENSIONS

def allowed_audio(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_AUDIO_EXTENSIONS


# ══════════════════════════════════════════════════════════════
# Serve generated/criminal images (protected)
# ══════════════════════════════════════════════════════════════

@app.route("/data/generated/<filename>")
@login_required
def serve_generated(filename):
    return send_from_directory(str(config.GENERATED_DIR), secure_filename(filename))

@app.route("/data/criminal/<filename>")
@login_required
def serve_criminal(filename):
    return send_from_directory(str(config.CRIMINAL_DB_DIR), secure_filename(filename))


# ══════════════════════════════════════════════════════════════
# Public routes
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════
# Auth
# ══════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return _redirect_by_role(current_user.role)

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if row and row["is_active"] and bcrypt.check_password_hash(row["password_hash"], password):
            user = User(row)
            login_user(user, remember=False)
            session.permanent = True

            # Update last_login
            conn = get_db()
            conn.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (user.id,))
            conn.commit()
            conn.close()

            audit("login")

            if user.force_pw:
                flash("Please change your default password.", "warning")
                return redirect(url_for("change_password"))

            return _redirect_by_role(user.role)

        flash("Invalid username or password.", "danger")

    return render_template("auth/login.html")


@app.route("/logout")
@login_required
def logout():
    audit("logout")
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        new_pw  = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        if len(new_pw) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif new_pw != confirm:
            flash("Passwords do not match.", "danger")
        else:
            hashed = bcrypt.generate_password_hash(new_pw).decode("utf-8")
            conn = get_db()
            conn.execute(
                "UPDATE users SET password_hash=?, force_password_change=0 WHERE id=?",
                (hashed, current_user.id),
            )
            conn.commit()
            conn.close()
            audit("change_password")
            flash("Password changed successfully.", "success")
            return _redirect_by_role(current_user.role)

    return render_template("auth/change_password.html")


def _redirect_by_role(role):
    if role == "superadmin":
        return redirect(url_for("superadmin_dashboard"))
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("police_dashboard"))


# ══════════════════════════════════════════════════════════════
# Superadmin routes
# ══════════════════════════════════════════════════════════════

@app.route("/superadmin/dashboard")
@login_required
@role_required("superadmin")
def superadmin_dashboard():
    conn = get_db()
    stats = {
        "admins":    conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0],
        "police":    conn.execute("SELECT COUNT(*) FROM users WHERE role='police'").fetchone()[0],
        "criminals": conn.execute("SELECT COUNT(*) FROM criminals").fetchone()[0],
        "cases":     conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
    }
    admins = conn.execute(
        "SELECT * FROM users WHERE role='admin' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("superadmin/dashboard.html", stats=stats, admins=admins)


@app.route("/superadmin/admins/create", methods=["POST"])
@login_required
@role_required("superadmin")
def superadmin_create_admin():
    username  = request.form.get("username", "").strip()
    full_name = request.form.get("full_name", "").strip()
    email     = request.form.get("email", "").strip()
    password  = request.form.get("password", "")

    if not all([username, full_name, password]):
        flash("All fields are required.", "danger")
        return redirect(url_for("superadmin_dashboard"))

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, email, force_password_change, created_by) "
            "VALUES (?,?,?,?,?,1,?)",
            (username, hashed, "admin", full_name, email or None, current_user.id),
        )
        conn.commit()
        conn.close()
        audit("create_admin", "user", details={"username": username})
        flash(f"Admin '{username}' created successfully.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")

    return redirect(url_for("superadmin_dashboard"))


@app.route("/superadmin/admins/<int:uid>/toggle", methods=["POST"])
@login_required
@role_required("superadmin")
def superadmin_toggle_admin(uid):
    conn = get_db()
    row = conn.execute("SELECT is_active FROM users WHERE id=? AND role='admin'", (uid,)).fetchone()
    if row:
        new_state = 0 if row["is_active"] else 1
        conn.execute("UPDATE users SET is_active=? WHERE id=?", (new_state, uid))
        conn.commit()
        audit("toggle_user", "user", uid)
    conn.close()
    return redirect(url_for("superadmin_dashboard"))


@app.route("/superadmin/audit")
@login_required
@role_required("superadmin")
def superadmin_audit():
    conn = get_db()
    logs = conn.execute(
        "SELECT al.*, u.username FROM audit_log al "
        "LEFT JOIN users u ON al.user_id=u.id "
        "ORDER BY al.logged_at DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return render_template("superadmin/audit.html", logs=logs)


# ══════════════════════════════════════════════════════════════
# Admin routes
# ══════════════════════════════════════════════════════════════

@app.route("/admin/dashboard")
@login_required
@role_required("admin", "superadmin")
def admin_dashboard():
    conn = get_db()
    stats = {
        "police":    conn.execute("SELECT COUNT(*) FROM users WHERE role='police'").fetchone()[0],
        "criminals": conn.execute("SELECT COUNT(*) FROM criminals").fetchone()[0],
        "cases":     conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
        "wanted":    conn.execute("SELECT COUNT(*) FROM criminals WHERE status='wanted'").fetchone()[0],
    }
    conn.close()
    return render_template("admin/dashboard.html", stats=stats)


# ── Police management ──────────────────────────────────────────

@app.route("/admin/police")
@login_required
@role_required("admin", "superadmin")
def admin_police():
    conn = get_db()
    officers = conn.execute(
        "SELECT * FROM users WHERE role='police' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("admin/view_police.html", officers=officers)


@app.route("/admin/police/add", methods=["GET", "POST"])
@login_required
@role_required("admin", "superadmin")
def admin_add_police():
    if request.method == "POST":
        f = request.form
        password = f.get("password", "")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return render_template("admin/add_police.html")

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password_hash, role, full_name, email, mobile, "
                "police_id, station_name, location, rank, force_password_change, created_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,1,?)",
                (
                    f.get("username"), hashed, "police", f.get("name"),
                    f.get("email") or None, f.get("mobile") or None,
                    f.get("police_id") or None, f.get("station") or None,
                    f.get("location") or None, f.get("rank") or None,
                    current_user.id,
                ),
            )
            conn.commit()
            conn.close()
            audit("create_police", "user", details={"username": f.get("username")})
            flash("Police officer added successfully.", "success")
            return redirect(url_for("admin_police"))
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("admin/add_police.html")


@app.route("/admin/police/<int:uid>/toggle", methods=["POST"])
@login_required
@role_required("admin", "superadmin")
def admin_toggle_police(uid):
    conn = get_db()
    row = conn.execute("SELECT is_active FROM users WHERE id=? AND role='police'", (uid,)).fetchone()
    if row:
        conn.execute("UPDATE users SET is_active=? WHERE id=?", (0 if row["is_active"] else 1, uid))
        conn.commit()
        audit("toggle_police", "user", uid)
    conn.close()
    return redirect(url_for("admin_police"))


# ── Criminal database management ───────────────────────────────

@app.route("/admin/criminals")
@login_required
@role_required("admin", "superadmin")
def admin_criminals():
    conn = get_db()
    criminals = conn.execute(
        "SELECT * FROM criminals ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return render_template("admin/criminals.html", criminals=criminals)


@app.route("/admin/criminals/add", methods=["GET", "POST"])
@login_required
@role_required("admin", "superadmin")
def admin_add_criminal():
    if request.method == "POST":
        f = request.form
        face_image_path = None

        # Handle photo upload
        if "photo" in request.files and request.files["photo"].filename:
            photo = request.files["photo"]
            if not allowed_image(photo.filename):
                flash("Only JPG/PNG images allowed.", "danger")
                return render_template("admin/add_criminal.html")
            fname = f"{uuid.uuid4().hex}{Path(photo.filename).suffix}"
            save_path = config.CRIMINAL_DB_DIR / fname
            photo.save(str(save_path))
            face_image_path = fname

        # Build attributes JSON for face matcher
        attrs = {
            "gender":     f.get("gender", ""),
            "age":        int(f.get("age") or 0),
            "skin_tone":  f.get("skin_tone", ""),
            "hair_color": f.get("hair_color", ""),
            "marks":      [],
        }
        raw_marks = f.get("marks_json", "[]")
        try:
            attrs["marks"] = json.loads(raw_marks)
        except Exception:
            attrs["marks"] = []

        # Write sidecar JSON so face_matcher.build_database() picks up attributes
        if face_image_path:
            sidecar_path = (config.CRIMINAL_DB_DIR / face_image_path).with_suffix(".json")
            with open(sidecar_path, "w") as sf:
                json.dump(attrs, sf)

        code = f"CR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO criminals (criminal_code, name, age, gender, skin_tone, crime_type, "
                "crime_year, no_of_crimes, last_known_location, status, description, "
                "face_image_path, attributes_json, added_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    code, f.get("name"), f.get("age") or None, f.get("gender") or None,
                    f.get("skin_tone") or None, f.get("crime_type") or None,
                    f.get("crime_year") or None, f.get("no_of_crimes") or 0,
                    f.get("location") or None, f.get("status", "wanted"),
                    f.get("description") or None, face_image_path,
                    json.dumps(attrs), current_user.id,
                ),
            )
            conn.commit()
            conn.close()
            audit("add_criminal", "criminal", details={"name": f.get("name")})
            flash(f"Criminal record added. Code: {code}", "success")
            return redirect(url_for("admin_criminals"))
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("admin/add_criminal.html")


@app.route("/admin/criminals/<int:cid>")
@login_required
@role_required("admin", "superadmin")
def admin_criminal_detail(cid):
    conn = get_db()
    criminal = conn.execute("SELECT * FROM criminals WHERE id=?", (cid,)).fetchone()
    conn.close()
    if not criminal:
        abort(404)
    return render_template("admin/criminal_detail.html", criminal=criminal)


@app.route("/admin/criminals/upload-csv", methods=["POST"])
@login_required
@role_required("admin", "superadmin")
def admin_upload_csv():
    """Bulk upload criminals from CSV."""
    if "file" not in request.files or not request.files["file"].filename:
        flash("No file selected.", "danger")
        return redirect(url_for("admin_criminals"))

    import csv, io
    file = request.files["file"]
    content = file.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(content))

    conn = get_db()
    added, skipped = 0, 0
    for row in reader:
        name = row.get("name", "").strip()
        if not name:
            skipped += 1
            continue
        code = f"CR-CSV-{uuid.uuid4().hex[:6].upper()}"
        try:
            conn.execute(
                "INSERT INTO criminals (criminal_code, name, age, gender, crime_type, "
                "crime_year, no_of_crimes, description, status, added_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    code, name, row.get("age") or None, row.get("gender") or None,
                    row.get("crime_type") or None, row.get("crime_year") or None,
                    row.get("no_of_crimes") or 0, row.get("description") or None,
                    "wanted", current_user.id,
                ),
            )
            added += 1
        except Exception:
            skipped += 1
    conn.commit()
    conn.close()
    audit("upload_csv", "criminal", details={"added": added, "skipped": skipped})
    flash(f"CSV upload complete: {added} added, {skipped} skipped.", "success")
    return redirect(url_for("admin_criminals"))


@app.route("/admin/db/rebuild", methods=["POST"])
@login_required
@role_required("admin", "superadmin")
def admin_rebuild_db():
    """Rebuild InsightFace embedding index for all criminal photos."""
    try:
        matcher = get_matcher()
        matcher.build_database()
        audit("rebuild_face_db")
        return jsonify({"status": "ok", "message": "Face database rebuilt successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# Police routes
# ══════════════════════════════════════════════════════════════

@app.route("/police/dashboard")
@login_required
@role_required("police")
def police_dashboard():
    conn = get_db()
    my_cases = conn.execute(
        "SELECT * FROM cases WHERE officer_id=? ORDER BY created_at DESC LIMIT 10",
        (current_user.id,),
    ).fetchall()
    total_cases = conn.execute(
        "SELECT COUNT(*) FROM cases WHERE officer_id=?", (current_user.id,)
    ).fetchone()[0]
    conn.close()
    return render_template("police/dashboard.html", my_cases=my_cases, total_cases=total_cases)


@app.route("/police/cases")
@login_required
@role_required("police")
def police_cases():
    conn = get_db()
    cases = conn.execute(
        "SELECT * FROM cases WHERE officer_id=? ORDER BY created_at DESC",
        (current_user.id,),
    ).fetchall()
    conn.close()
    return render_template("police/cases.html", cases=cases)


@app.route("/police/cases/new", methods=["GET"])
@login_required
@role_required("police")
def police_new_case():
    """4-step AI pipeline page — the core forensic interface."""
    return render_template("police/new_case.html")


@app.route("/police/cases/<int:cid>")
@login_required
@role_required("police")
def police_case_detail(cid):
    conn = get_db()
    case = conn.execute(
        "SELECT * FROM cases WHERE id=? AND officer_id=?",
        (cid, current_user.id),
    ).fetchone()
    if not case:
        abort(404)
    runs = conn.execute(
        "SELECT * FROM investigation_runs WHERE case_id=? ORDER BY run_at DESC",
        (cid,),
    ).fetchall()
    conn.close()
    return render_template("police/case_detail.html", case=case, runs=runs)


# ══════════════════════════════════════════════════════════════
# AI Pipeline API routes (JSON)
# ══════════════════════════════════════════════════════════════

@app.route("/api/transcribe", methods=["POST"])
@login_required
@role_required("police")
def api_transcribe():
    """Step 1: Upload audio → transcript."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    if not allowed_audio(audio_file.filename):
        return jsonify({"error": "Unsupported audio format"}), 400

    fname = f"{uuid.uuid4().hex}{Path(audio_file.filename).suffix}"
    tmp_path = config.UPLOADS_DIR / fname
    audio_file.save(str(tmp_path))

    try:
        result = transcribe_audio(str(tmp_path))
        return jsonify({
            "transcript":  result.get("transcript", ""),
            "confidence":  result.get("confidence", 0),
            "language":    result.get("language", "en"),
            "audio_path":  fname,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parse", methods=["POST"])
@login_required
@role_required("police")
def api_parse():
    """Step 2: Text description → parsed attributes + prompts."""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No description text"}), 400

    try:
        attrs = parse_description(text)
        positive, negative = build_prompt(attrs)
        display = format_attributes_display(attrs)
        return jsonify({
            "attributes":      attrs,
            "display":         display,
            "positive_prompt": positive,
            "negative_prompt": negative,
            # Marks for Stage 2 post-processing (scars, moles)
            "marks":           attrs.get("marks", []),
            # Accessories included in prompt (bindi, earrings, etc.)
            "accessories":     attrs.get("accessories", []),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_gen_status = {
    "running":  False,
    "done":     False,
    "error":    None,
    "num":      0,
    "mode":     "text",
}
_gen_lock = threading.Lock()


def _run_generation_job(positive, negative, num, sketch_image, marks=None):
    """Background worker. Stage 1: AI generation, Stage 2: mark rendering."""
    global _gen_status
    try:
        mark_count = len(marks) if marks else 0
        print(f"  [gen-job] num_images={num}  sketch={sketch_image is not None}  "
              f"marks_received={mark_count}")
        if mark_count:
            for m in marks:
                print(f"    mark: {m.get('type', '?')} @ "
                      f"{m.get('location') or '(no location)'}")
        generate(positive, negative, num_images=num,
                 sketch_image=sketch_image, marks=marks)
        with _gen_lock:
            _gen_status["done"]    = True
            _gen_status["running"] = False
    except Exception as e:
        with _gen_lock:
            _gen_status["error"]   = str(e)
            _gen_status["done"]    = True
            _gen_status["running"] = False


@app.route("/api/generate", methods=["POST"])
@login_required
@role_required("police")
def api_generate():
    """Step 3: Start generation in a background thread. Returns immediately."""
    global _gen_status

    with _gen_lock:
        if _gen_status["running"]:
            return jsonify({"error": "A generation is already in progress."}), 409

    positive = request.form.get("positive_prompt", "")
    negative = request.form.get("negative_prompt", "")
    num      = min(int(request.form.get("num_images", 4)), config.MAX_NUM_IMAGES)

    # Parse marks JSON for Stage 2 post-processing (scars, moles)
    marks_json = request.form.get("marks_json", "[]")
    try:
        marks = json.loads(marks_json) if marks_json else []
    except (json.JSONDecodeError, TypeError):
        marks = []

    if not positive:
        return jsonify({"error": "No prompt provided"}), 400

    sketch_image = None
    if "sketch" in request.files and request.files["sketch"].filename:
        from PIL import Image as PILImage
        sketch_file = request.files["sketch"]
        if allowed_image(sketch_file.filename):
            sketch_image = PILImage.open(sketch_file).convert("RGB")

    # Clear previous run's images so polling starts fresh
    for f in config.GENERATED_DIR.glob("generated_*.png"):
        try:
            f.unlink()
        except Exception:
            pass

    mode_label = "text"
    if sketch_image is not None:
        gen_mode = (config.GENERATION_MODE or "api").lower()
        mode_label = "sketch+api" if gen_mode == "api" else "sketch+controlnet"

    with _gen_lock:
        _gen_status = {
            "running": True,
            "done":    False,
            "error":   None,
            "num":     num,
            "mode":    mode_label,
        }

    thread = threading.Thread(
        target=_run_generation_job,
        args=(positive, negative, num, sketch_image, marks),
        daemon=True,
    )
    thread.start()

    return jsonify({"started": True, "num_images": num, "mode": _gen_status["mode"],
                    "marks_count": len(marks)})


@app.route("/api/generate-status", methods=["GET"])
@login_required
@role_required("police")
def api_generate_status():
    """Poll generation progress. Returns URLs of images saved so far."""
    with _gen_lock:
        status = dict(_gen_status)

    # Parallel API mode finishes images out of order, so collect every
    # slot that exists on disk rather than stopping at the first gap.
    # Append mtime to bust the browser cache when Stage 3 rewrites a
    # slot (the photo → pencil-sketch variant at the last index).
    images = []
    for i in range(1, status.get("num", 4) + 1):
        fname = f"generated_{i}.png"
        fpath = config.GENERATED_DIR / fname
        if fpath.exists():
            mtime = int(fpath.stat().st_mtime)
            images.append(url_for("serve_generated", filename=fname) + f"?t={mtime}")

    return jsonify({
        "running": status["running"],
        "done":    status["done"],
        "error":   status["error"],
        "mode":    status["mode"],
        "count":   len(images),
        "total":   status["num"],
        "images":  images,
    })


@app.route("/api/match", methods=["POST"])
@login_required
@role_required("police")
def api_match():
    """Step 4: Match selected generated face against criminal DB."""
    data = request.get_json()
    image_filename = data.get("image_filename", "")
    attributes     = data.get("attributes")

    if not image_filename:
        return jsonify({"error": "No image filename"}), 400

    image_path = config.GENERATED_DIR / secure_filename(image_filename)
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404

    # Multi-image query: matcher v5 averages embeddings across all 4
    # generated candidates for a much more stable identity vector
    # than a single diffusion sample. The selected image still drives
    # the UI; this is purely for the matching computation.
    query_inputs: list = [str(image_path)]
    try:
        if config.USE_MULTI_IMAGE_QUERY and image_path.parent == config.GENERATED_DIR:
            siblings = sorted(config.GENERATED_DIR.glob("generated_*.png"))
            extras = [str(p) for p in siblings if p.resolve() != image_path.resolve()]
            query_inputs.extend(extras)
    except Exception:
        # Worst case we keep the single selected image — never break match.
        query_inputs = [str(image_path)]

    try:
        matcher = get_matcher()
        match_response = matcher.match(
            query_inputs, query_attributes=attributes, top_k=config.TOP_K_MATCHES
        )
        # Backward-compatible: legacy callers expect a list. The new
        # response wraps that list with status/quality/tier metadata.
        results = match_response.get("results", []) if isinstance(match_response, dict) else match_response

        # Enrich results with DB criminal info
        enriched = []
        conn = get_db()
        for r in results:
            criminal_name = r.get("name", "")
            db_criminal = conn.execute(
                "SELECT * FROM criminals WHERE criminal_code=? OR name=?",
                (criminal_name, criminal_name),
            ).fetchone()
            entry = dict(r)
            if db_criminal:
                entry["db_record"] = {
                    "id":           db_criminal["id"],
                    "criminal_code": db_criminal["criminal_code"],
                    "name":         db_criminal["name"],
                    "age":          db_criminal["age"],
                    "crime_type":   db_criminal["crime_type"],
                    "crime_year":   db_criminal["crime_year"],
                    "no_of_crimes": db_criminal["no_of_crimes"],
                    "last_known_location": db_criminal["last_known_location"],
                    "status":       db_criminal["status"],
                    "description":  db_criminal["description"],
                    "face_url":     url_for("serve_criminal", filename=db_criminal["face_image_path"])
                                    if db_criminal["face_image_path"] else None,
                }
            enriched.append(entry)
        conn.close()

        # Surface matcher v5 metadata so the UI can render an explicit
        # state ("strong match" / "possible" / "no reliable match") instead
        # of guessing from scores. Old callers that only read .matches
        # keep working.
        payload: dict = {"matches": enriched}
        if isinstance(match_response, dict):
            payload["status"] = match_response.get("status", "ok")
            payload["query_quality"] = match_response.get("query_quality", 0.0)
            payload["tier_counts"] = match_response.get("tier_counts", {})
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e)
        return jsonify({"error": err_msg}), 500


@app.route("/api/save-run", methods=["POST"])
@login_required
@role_required("police")
def api_save_run():
    """Save a completed pipeline run to a case (creates case if needed)."""
    data = request.get_json()

    # Create or get case
    case_number = f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO cases (case_number, officer_id, incident_location, status) VALUES (?,?,?,?)",
        (case_number, current_user.id, data.get("incident_location", ""), "open"),
    )
    case_id = cur.lastrowid

    conn.execute(
        "INSERT INTO investigation_runs "
        "(case_id, officer_id, audio_file_path, transcript, stt_language, stt_confidence, "
        "description_text, parsed_attributes, positive_prompt, negative_prompt, "
        "generated_image_paths, generation_mode, had_sketch_input, "
        "selected_image_path, match_results, top_match_name, top_match_score) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            case_id, current_user.id,
            data.get("audio_path"), data.get("transcript"),
            data.get("stt_language"), data.get("stt_confidence"),
            data.get("description_text"),
            json.dumps(data.get("attributes", {})),
            data.get("positive_prompt"), data.get("negative_prompt"),
            json.dumps(data.get("generated_images", [])),
            data.get("generation_mode", "text"),
            1 if data.get("had_sketch") else 0,
            data.get("selected_image"),
            json.dumps(data.get("match_results", [])),
            data.get("top_match_name"), data.get("top_match_score"),
        ),
    )
    conn.commit()
    conn.close()

    audit("run_pipeline", "case", case_id, {
        "case_number": case_number,
        "top_match":   data.get("top_match_name"),
        "score":       data.get("top_match_score"),
    })

    return jsonify({"status": "ok", "case_id": case_id, "case_number": case_number})


# ══════════════════════════════════════════════════════════════
# Error handlers
# ══════════════════════════════════════════════════════════════

@app.errorhandler(403)
def forbidden(e):
    return render_template("errors/403.html"), 403

@app.errorhandler(404)
def not_found(e):
    return render_template("errors/404.html"), 404


# ══════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    # Create default superadmin with proper bcrypt hash on first run
    from extensions import bcrypt as _bcrypt, get_db as _get_db
    conn = _get_db()
    row = conn.execute("SELECT * FROM users WHERE username='superadmin'").fetchone()
    if row and row["password_hash"].startswith("$2b$12$placeholder"):
        real_hash = _bcrypt.generate_password_hash("Admin@123").decode("utf-8")
        conn.execute("UPDATE users SET password_hash=? WHERE username='superadmin'", (real_hash,))
        conn.commit()
        print("[STARTUP] Superadmin created — username: superadmin, password: Admin@123")
        print("[STARTUP] CHANGE THIS PASSWORD IMMEDIATELY after first login.")
    conn.close()

    print(f"[STARTUP] Generation mode: {config.GENERATION_MODE}")
    app.run(host="127.0.0.1", port=5000, debug=True)
