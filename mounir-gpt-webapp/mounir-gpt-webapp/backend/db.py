import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "chat.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )
    """)
    conn.commit()
    conn.close()

def upsert_conversation(conversation_id: str, title: str | None = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conversation_id, title or "New chat", datetime.utcnow().isoformat())
        )
    elif title:
        cur.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
    conn.commit()
    conn.close()

def insert_message(conversation_id: str, role: str, content: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
