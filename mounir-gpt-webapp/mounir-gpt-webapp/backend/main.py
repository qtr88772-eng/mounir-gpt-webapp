import os
import uuid
import shutil
import sqlite3
import mimetypes
import base64
import re
from datetime import datetime
from typing import List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

# ================= ENV =================
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "MOUNIR GPT-5.2")

# Default (cheap/stable) model
MODEL_DEFAULT = os.getenv("MODEL", "openai/gpt-4o-mini")
# Strong mode (only when enabled from UI)
MODEL_STRONG = os.getenv("MODEL_STRONG", "openai/gpt-5.2-chat-latest")

# Vision model for images
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Cost control
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Prompt safety
PROMPT_CHAR_BUDGET = int(os.getenv("PROMPT_CHAR_BUDGET", "2200"))
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "8"))

# ================= APP =================
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DB =================
DB_PATH = os.path.join(os.path.dirname(__file__), "chat.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cols = [r["name"] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
    return col in cols

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            model TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()

    if not _has_column(conn, "messages", "model"):
        cur.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        conn.commit()

    conn.close()

init_db()

def _sanitize_title(text: str) -> str:
    if not text:
        return "New chat"
    t = text.strip()
    t = re.sub(r"\[images:.*?\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    if len(t) > 42:
        t = t[:42].rstrip() + "…"
    return t or "New chat"

def upsert_conversation(conv_id: str, title_hint: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO conversations (id, title, created_at) VALUES (?,?,?)",
        (conv_id, "New chat", datetime.utcnow().isoformat()),
    )
    if title_hint:
        title = _sanitize_title(title_hint)
        cur.execute(
            "UPDATE conversations SET title=? WHERE id=? AND (title IS NULL OR title='New chat')",
            (title, conv_id),
        )
    conn.commit()
    conn.close()

def ensure_conversation_title(conv_id: str):
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if not row:
        conn.close()
        return
    if row["title"] and row["title"] != "New chat":
        conn.close()
        return
    first_user = cur.execute(
        "SELECT content FROM messages WHERE conversation_id=? AND role='user' ORDER BY id ASC LIMIT 1",
        (conv_id,),
    ).fetchone()
    if first_user:
        new_title = _sanitize_title(first_user["content"])
        cur.execute("UPDATE conversations SET title=? WHERE id=?", (new_title, conv_id))
        conn.commit()
    conn.close()

def add_message(conv_id: str, role: str, content: str, model: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content, model, created_at) VALUES (?,?,?,?,?)",
        (conv_id, role, content, model, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

def get_history(conv_id: str, limit: int = 8) -> List[dict]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT role, content, model FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (conv_id, limit),
    ).fetchall()
    conn.close()
    return list(reversed([{ "role": r["role"], "content": r["content"], "model": r["model"] } for r in rows]))

# ================= OpenAI (OpenRouter) =================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[ChatMessage]
    strong: Optional[bool] = False  # ✅ toggle from UI

SYSTEM_PROMPT = (
    f"You are {APP_NAME}.\n"
    "Rules:\n"
    "- If asked who made you OR (مين صنعك؟) reply exactly: طورني منير\n"
    "- Reply in Arabic/Egyptian when user uses Arabic.\n"
    "- Keep continuity from recent messages.\n"
)

# ================= Prompt helpers =================
def _extract_affordable_tokens(err: str) -> Optional[int]:
    m = re.search(r"can only afford\s+(\d+)", err, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract_prompt_limit(err: str) -> bool:
    return "Prompt tokens limit exceeded" in err

def _trim_messages_by_chars(messages: List[dict], budget: int) -> List[dict]:
    if budget < 800:
        budget = 800
    sys = messages[0:1] if messages and messages[0].get("role") == "system" else []
    rest = messages[1:] if sys else messages[:]
    keep = []
    total = 0
    for msg in reversed(rest):
        c = msg.get("content") or ""
        add = len(c) + 20
        if keep and total + add > budget:
            break
        if not keep and add > budget:
            keep.append({"role": msg.get("role","user"), "content": c[-budget:]})
            total = budget
            break
        keep.append({"role": msg.get("role"), "content": c})
        total += add
    trimmed = sys + list(reversed(keep))
    if sys and len(trimmed) == 1 and rest:
        trimmed.append({"role": rest[-1].get("role"), "content": rest[-1].get("content","")})
    return trimmed

def _call_once(model: str, messages: List[dict], max_tokens: int) -> Tuple[bool, str]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            stream=False,
        )
        return True, (r.choices[0].message.content or "").strip()
    except Exception as e:
        return False, str(e)

def _call_with_retry(model: str, messages: List[dict], max_tokens: int) -> Tuple[bool, str]:
    ok, out = _call_once(model, messages, max_tokens)
    if ok:
        return True, out

    err = out

    afford = _extract_affordable_tokens(err)
    if afford:
        retry_tokens = max(64, afford - 50)
        ok2, out2 = _call_once(model, messages, retry_tokens)
        if ok2:
            return True, out2
        err = out2

    if _extract_prompt_limit(err):
        shrunk = _trim_messages_by_chars(messages, min(PROMPT_CHAR_BUDGET, 1600))
        ok3, out3 = _call_once(model, shrunk, max_tokens)
        if ok3:
            return True, out3

        sys = messages[0:1] if messages and messages[0].get("role") == "system" else []
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        minimal = sys + ([last_user] if last_user else [])
        ok4, out4 = _call_once(model, minimal, max_tokens)
        if ok4:
            return True, out4
        err = out4

    return False, err

def build_context_messages(history_rows: List[dict], new_user_content: str) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history_rows or []:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": new_user_content})
    return _trim_messages_by_chars(msgs, PROMPT_CHAR_BUDGET)

# ================= Files =================
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def safe_name(name: str) -> str:
    return os.path.basename(name).replace("\\", "_").replace("/", "_")

def file_path(name: str) -> str:
    return os.path.join(UPLOAD_DIR, safe_name(name))

def detect_kind(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    if mt == "application/pdf":
        return "pdf"
    if mt and mt.startswith("image/"):
        return "image"
    return "file"

def to_data_uri(file_bytes: bytes, filename: str) -> str:
    kind = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return f"data:{kind};base64,{base64.b64encode(file_bytes).decode()}"

# ================= Chat (text) =================
@app.post("/chat")
def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())

    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "messages required", "conversation_id": conv_id})

    last = req.messages[-1]
    if last.role != "user":
        return JSONResponse(status_code=400, content={"error": "last must be user", "conversation_id": conv_id})

    upsert_conversation(conv_id, last.content)

    history = get_history(conv_id, limit=HISTORY_MAX_MESSAGES)

    add_message(conv_id, "user", last.content, model=None)
    ensure_conversation_title(conv_id)

    model_used = MODEL_STRONG if (req.strong is True) else MODEL_DEFAULT
    msgs = build_context_messages(history, last.content)

    ok, reply = _call_with_retry(model_used, msgs, MAX_TOKENS)
    if ok:
        if reply:
            add_message(conv_id, "assistant", reply, model=model_used)
        return {"conversation_id": conv_id, "reply": reply, "model": model_used}

    if "Prompt tokens limit exceeded" in reply:
        hint = "الرسائل طويلة على الحد المسموح. افتح شات جديد أو امسح جزء من الشات."
        return JSONResponse(status_code=402, content={"error": hint, "raw": reply, "conversation_id": conv_id, "model": model_used})

    return JSONResponse(status_code=500, content={"error": reply, "conversation_id": conv_id, "model": model_used})

# ================= Chat (vision) =================
@app.post("/chat/vision")
async def chat_vision(
    prompt: str = Form(""),
    conversation_id: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
):
    conv_id = conversation_id or str(uuid.uuid4())
    upsert_conversation(conv_id, prompt or "صور")

    user_text = prompt.strip() if prompt.strip() else "اشرحلي الصور دي."
    history = get_history(conv_id, limit=max(6, HISTORY_MAX_MESSAGES // 2))

    image_parts = []
    saved = []
    for img in images:
        name = safe_name(img.filename or f"image_{uuid.uuid4().hex[:6]}.png")
        path = file_path(name)
        with open(path, "wb") as f:
            shutil.copyfileobj(img.file, f)
        saved.append(name)
        with open(path, "rb") as f:
            image_parts.append({"type": "image_url", "image_url": {"url": to_data_uri(f.read(), name)}})

    add_message(conv_id, "user", f"{user_text}\n\n[images: {', '.join(saved)}]", model=None)
    ensure_conversation_title(conv_id)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs = _trim_messages_by_chars(msgs, min(PROMPT_CHAR_BUDGET, 1600))

    msgs.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}] + image_parts
    })

    ok, reply = _call_with_retry(VISION_MODEL, msgs, MAX_TOKENS)
    if ok:
        if reply:
            add_message(conv_id, "assistant", reply, model=VISION_MODEL)
        return {"conversation_id": conv_id, "reply": reply, "model": VISION_MODEL}

    if "Prompt tokens limit exceeded" in reply:
        hint = "محتوى الصور/الشات كبير على الحد المسموح. افتح شات جديد وجرب تاني."
        return JSONResponse(status_code=402, content={"error": hint, "raw": reply, "conversation_id": conv_id, "model": VISION_MODEL})

    return JSONResponse(status_code=500, content={"error": reply, "conversation_id": conv_id, "model": VISION_MODEL})

# ================= Conversations =================
@app.get("/conversations")
def list_conversations():
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/conversations/{cid}")
def get_conversation(cid: str):
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT role, content, model FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (cid,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.delete("/conversations/{cid}")
def delete_conversation(cid: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    cur.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    return {"deleted": cid}

class RenameConversationRequest(BaseModel):
    title: str

@app.post("/conversations/{cid}/title")
def rename_conversation(cid: str, req: RenameConversationRequest):
    title = _sanitize_title(req.title)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
    conn.commit()
    conn.close()
    return {"id": cid, "title": title}

# ================= Upload & Download =================
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = safe_name(f.filename)
        path = file_path(name)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved.append({"name": name, "kind": detect_kind(name)})
    return {"files": saved}

@app.get("/files")
def list_files():
    items = []
    for name in os.listdir(UPLOAD_DIR):
        p = file_path(name)
        if os.path.isfile(p):
            items.append({
                "name": name,
                "kind": detect_kind(name),
                "size": os.path.getsize(p),
                "url": f"/download/{name}"
            })
    return items

@app.get("/download/{name}")
def download(name: str):
    p = file_path(name)
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(p, filename=safe_name(name))

# ================= Health =================
@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "default_model": MODEL_DEFAULT,
        "strong_model": MODEL_STRONG,
        "vision_model": VISION_MODEL,
        "max_tokens": MAX_TOKENS,
        "prompt_char_budget": PROMPT_CHAR_BUDGET,
        "history_max_messages": HISTORY_MAX_MESSAGES,
        "base_url": BASE_URL,
    }

# ================= Frontend =================
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def index():
    p = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"message": "Frontend not found"}
