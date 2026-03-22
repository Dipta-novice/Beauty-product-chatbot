import sqlite3
import os
from typing import List, Dict

# SQLite database file (local, no storage limits)
DB_PATH = "chat_history.db"

def get_connection() -> sqlite3.Connection:
    """Get SQLite connection with optimized settings."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Optimize for chat app (high writes, good reads)
    conn.execute("PRAGMA journal_mode=WAL")      # Better concurrency
    conn.execute("PRAGMA synchronous=NORMAL")    # Faster writes  
    conn.execute("PRAGMA cache_size=10000")      # Larger cache
    conn.execute("PRAGMA temp_store=MEMORY")     # Temp tables in RAM
    return conn

def init_db(conn: sqlite3.Connection):
    """Initialize database tables (separate CREATE INDEX)."""
    # ✅ FIXED: Separate statements for SQLite
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            message     TEXT NOT NULL,
            escalated   INTEGER DEFAULT 0,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_session_time 
        ON conversations (session_id, created_at DESC);
    """)
    conn.commit()

def save_message(conn: sqlite3.Connection, session_id: str, role: str, message: str, escalated: bool = False):
    """Save message atomically."""
    sql = """
        INSERT INTO conversations (session_id, role, message, escalated)
        VALUES (?, ?, ?, ?)
    """
    try:
        with conn:
            conn.execute(sql, (session_id, role, message, 1 if escalated else 0))
    except sqlite3.Error as e:
        st.error(f"DB save error: {e}")
        raise

def fetch_history(conn: sqlite3.Connection, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Fetch recent chat history."""
    sql = """
        SELECT role, message 
        FROM conversations
        WHERE session_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    """
    try:
        cursor = conn.execute(sql, (session_id, limit))
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    except sqlite3.Error as e:
        st.error(f"DB fetch error: {e}")
        return []

# Convenience functions (auto-connection)
def save_message_simple(session_id: str, role: str, message: str, escalated: bool = False):
    """Save message (auto-connects/closes)."""
    conn = get_connection()
    try:
        save_message(conn, session_id, role, message, escalated)
    finally:
        conn.close()

def fetch_history_simple(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Fetch history (auto-connects/closes)."""
    conn = get_connection()
    try:
        return fetch_history(conn, session_id, limit)
    finally:
        conn.close()

def init_db_simple():
    """Initialize DB (auto-connects/closes)."""
    conn = get_connection()
    try:
        init_db(conn)
    finally:
        conn.close()
