"""
Flask extensions and SQLite database layer.
All extensions initialized here to avoid circular imports.
"""
import sqlite3
from pathlib import Path
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

bcrypt       = Bcrypt()
login_manager = LoginManager()
login_manager.login_view       = "auth.login"
login_manager.login_message    = "Please log in to access this page."
login_manager.login_message_category = "warning"


# ── SQLite helpers ─────────────────────────────────────────────

def get_db():
    """Return a fresh SQLite connection. Caller must close it."""
    import config
    conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    """Create all tables if they don't exist yet."""
    schema_path = Path(__file__).parent / "database" / "schema.sql"
    conn = get_db()
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print("[DB] Schema initialised.")
