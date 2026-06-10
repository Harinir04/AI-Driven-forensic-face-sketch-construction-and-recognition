-- Forensic Combined System — SQLite Schema

-- ── Users (all roles in one table) ────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    username              TEXT NOT NULL UNIQUE,
    password_hash         TEXT NOT NULL,
    role                  TEXT NOT NULL CHECK(role IN ('superadmin','admin','police')),
    full_name             TEXT NOT NULL,
    email                 TEXT UNIQUE,
    mobile                TEXT,
    police_id             TEXT UNIQUE,
    station_name          TEXT,
    location              TEXT,
    rank                  TEXT,
    is_active             INTEGER NOT NULL DEFAULT 1,
    force_password_change INTEGER NOT NULL DEFAULT 0,
    created_by            INTEGER REFERENCES users(id),
    created_at            TEXT DEFAULT (datetime('now')),
    last_login            TEXT
);

-- ── Criminal Records ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS criminals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    criminal_code       TEXT UNIQUE,
    name                TEXT NOT NULL,
    age                 INTEGER,
    gender              TEXT,
    skin_tone           TEXT,
    crime_type          TEXT,
    crime_year          INTEGER,
    no_of_crimes        INTEGER DEFAULT 0,
    last_known_location TEXT,
    status              TEXT DEFAULT 'wanted' CHECK(status IN ('wanted','arrested','released','deceased')),
    description         TEXT,
    face_image_path     TEXT,
    attributes_json     TEXT,
    added_by            INTEGER REFERENCES users(id),
    added_at            TEXT DEFAULT (datetime('now')),
    updated_at          TEXT
);

-- ── Investigation Cases ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_number       TEXT UNIQUE,
    officer_id        INTEGER NOT NULL REFERENCES users(id),
    incident_date     TEXT,
    incident_location TEXT,
    status            TEXT DEFAULT 'open' CHECK(status IN ('open','closed','pending')),
    created_at        TEXT DEFAULT (datetime('now')),
    updated_at        TEXT
);

-- ── Investigation Pipeline Runs ────────────────────────────────
CREATE TABLE IF NOT EXISTS investigation_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id              INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    officer_id           INTEGER NOT NULL REFERENCES users(id),

    -- Step 1: STT
    audio_file_path      TEXT,
    transcript           TEXT,
    stt_language         TEXT,
    stt_confidence       REAL,

    -- Step 2: Attribute parser
    description_text     TEXT,
    parsed_attributes    TEXT,
    positive_prompt      TEXT,
    negative_prompt      TEXT,

    -- Step 3: Face generation
    generated_image_paths TEXT,
    generation_mode       TEXT,
    had_sketch_input      INTEGER DEFAULT 0,

    -- Step 4: Matching
    selected_image_path  TEXT,
    match_results        TEXT,
    top_match_name       TEXT,
    top_match_score      REAL,

    run_at               TEXT DEFAULT (datetime('now'))
);

-- ── Audit Log ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    action      TEXT NOT NULL,
    target_type TEXT,
    target_id   INTEGER,
    details     TEXT,
    ip_address  TEXT,
    logged_at   TEXT DEFAULT (datetime('now'))
);

-- ── Default superadmin (password: Admin@123) ───────────────────
-- Hash generated with bcrypt rounds=12
-- Change password on first login!
INSERT OR IGNORE INTO users (username, password_hash, role, full_name, email, force_password_change)
VALUES (
    'superadmin',
    '$2b$12$placeholder_replace_on_first_run',
    'superadmin',
    'System Administrator',
    'admin@forensic.local',
    1
);
