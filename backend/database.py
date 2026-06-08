# backend/database.py
# Database connection and reusable CRUD operations for emotion_history.
# Attempts to use MySQL by default. If MySQL is unavailable or credentials fail,
# it gracefully falls back to a local SQLite database (emontic_ai.db).

import os
import sqlite3
import logging
from datetime import datetime
from contextlib import closing
from mysql.connector import pooling, Error as MySQLError

logger = logging.getLogger("emontic_ai")

# ── Connection Management ────────────────────────────────────────────────────
_pool = None
_use_sqlite = False
SQLITE_DB_PATH = "emontic_ai.db"


def _get_db_config():
    """Read DB config from environment at call time."""
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "user": os.environ.get("DB_USER", "root"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "emontic_ai_db"),
    }


def init_db():
    """
    Initialize the database connection. Tries MySQL first. 
    If it fails, falls back to SQLite. Creates tables if needed.
    Called once during FastAPI startup.
    """
    global _pool, _use_sqlite
    cfg = _get_db_config()

    try:
        # Try MySQL
        _pool = pooling.MySQLConnectionPool(
            pool_name="emontic_pool",
            pool_size=5,
            pool_reset_session=True,
            host=cfg["host"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
            autocommit=True,
        )
        
        # Test connection & create MySQL table
        with closing(_pool.get_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                create_table_mysql = """
                CREATE TABLE IF NOT EXISTS emotion_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_name VARCHAR(100) NOT NULL,
                    image_path VARCHAR(255),
                    predicted_emotion VARCHAR(50) NOT NULL,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_person_name (person_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
                cursor.execute(create_table_mysql)
                
        logger.info(f"Connected to MySQL (host={cfg['host']}, db={cfg['database']})")
        _use_sqlite = False
        return

    except MySQLError as e:
        logger.warning(f"MySQL connection failed: {e}")
        logger.warning("Falling back to SQLite database (emontic_ai.db)...")
        _use_sqlite = True
        _pool = None

    # Fallback to SQLite
    if _use_sqlite:
        try:
            with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
                with closing(conn.cursor()) as cursor:
                    create_table_sqlite = """
                    CREATE TABLE IF NOT EXISTS emotion_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT NOT NULL,
                        image_path TEXT,
                        predicted_emotion TEXT NOT NULL,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                    cursor.execute(create_table_sqlite)
                    # Create index for SQLite
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_person_name ON emotion_history(person_name);")
                    conn.commit()
            logger.info("SQLite database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite fallback: {e}")
            raise


# ── CRUD Operations ──────────────────────────────────────────────────────────

def save_prediction(person_name: str, image_path: str, emotion: str, confidence: float):
    try:
        if _use_sqlite:
            sql = """
            INSERT INTO emotion_history (person_name, image_path, predicted_emotion, confidence)
            VALUES (?, ?, ?, ?)
            """
            with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute(sql, (person_name, image_path, emotion, confidence))
                    conn.commit()
        else:
            sql = """
            INSERT INTO emotion_history (person_name, image_path, predicted_emotion, confidence)
            VALUES (%s, %s, %s, %s)
            """
            with closing(_pool.get_connection()) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute(sql, (person_name, image_path, emotion, confidence))
                    
        logger.info(f"Saved prediction: {person_name} → {emotion} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        raise


def get_unique_names() -> list[str]:
    sql = "SELECT DISTINCT person_name FROM emotion_history ORDER BY person_name ASC"
    try:
        if _use_sqlite:
            with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute(sql)
                    return [row[0] for row in cursor.fetchall()]
        else:
            with closing(_pool.get_connection()) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute(sql)
                    return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch unique names: {e}")
        raise


def get_history_by_name(name: str) -> list[dict]:
    sql = """
    SELECT id, person_name, image_path, predicted_emotion, confidence, created_at
    FROM emotion_history
    WHERE person_name = ? 
    ORDER BY created_at DESC
    """ if _use_sqlite else """
    SELECT id, person_name, image_path, predicted_emotion, confidence, created_at
    FROM emotion_history
    WHERE person_name = %s
    ORDER BY created_at DESC
    """
    
    rows = []
    try:
        if _use_sqlite:
            with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row  # To return dict-like rows
                with closing(conn.cursor()) as cursor:
                    cursor.execute(sql, (name,))
                    rows = [dict(row) for row in cursor.fetchall()]
        else:
            with closing(_pool.get_connection()) as conn:
                with closing(conn.cursor(dictionary=True)) as cursor:
                    cursor.execute(sql, (name,))
                    rows = cursor.fetchall()
                    
        # Serialize datetime to ISO strings
        for row in rows:
            if isinstance(row.get("created_at"), datetime):
                row["created_at"] = row["created_at"].isoformat()
            elif isinstance(row.get("created_at"), str):
                pass # SQLite returns strings

        return rows
    except Exception as e:
        logger.error(f"Failed to fetch history for '{name}': {e}")
        raise


def delete_prediction(record_id: int) -> str | None:
    """
    Delete a single prediction record by ID.
    Returns the image_path of the deleted record (for file cleanup), or None.
    """
    try:
        image_path = None
        if _use_sqlite:
            with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute("SELECT image_path FROM emotion_history WHERE id = ?", (record_id,))
                    row = cursor.fetchone()
                    if row:
                        image_path = row[0]
                        cursor.execute("DELETE FROM emotion_history WHERE id = ?", (record_id,))
                        conn.commit()
                    else:
                        return None
        else:
            with closing(_pool.get_connection()) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute("SELECT image_path FROM emotion_history WHERE id = %s", (record_id,))
                    row = cursor.fetchone()
                    if row:
                        image_path = row[0]
                        cursor.execute("DELETE FROM emotion_history WHERE id = %s", (record_id,))

        logger.info(f"Deleted prediction record #{record_id}")
        return image_path
    except Exception as e:
        logger.error(f"Failed to delete prediction #{record_id}: {e}")
        raise