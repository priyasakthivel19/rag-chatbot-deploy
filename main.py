
import os, io, re, time, hmac, hashlib, secrets, requests, jwt
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
 
load_dotenv()
E = os.getenv
INDEX_NAME = E("PINECONE_INDEX", "rag-chatbot")
CHAT_MODEL = E("CHAT_MODEL", "openai/gpt-oss-120b")
EMB_MODEL, DIM = "gemini-embedding-001", 768
SECRET = E("JWT_SECRET", "change-me")
MIN_SCORE = float(E("MIN_SCORE", "0.5"))  # lower = use documents more often
app = FastAPI()
 
@app.exception_handler(Exception)
async def show_error(request, exc):
    print("SERVER ERROR:", repr(exc))
    return JSONResponse({"detail": f"{type(exc).__name__}: {str(exc)[:250]}"}, status_code=500)
_index = None
 
def index():
    """Connect to Pinecone; create the index the first time."""
    global _index
    if _index is None:
        pc = Pinecone(api_key=E("PINECONE_API_KEY", ""))
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(name=INDEX_NAME, dimension=DIM, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region=E("PINECONE_REGION", "us-east-1")))
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        _index = pc.Index(INDEX_NAME)
    return _index
 
def embed(texts, task):
    out = []
    for i in range(0, len(texts), 100):
        reqs = [{"model": f"models/{EMB_MODEL}", "content": {"parts": [{"text": t}]},
                 "taskType": task, "outputDimensionality": DIM} for t in texts[i:i + 100]]
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{EMB_MODEL}:batchEmbedContents",
                              headers={"x-goog-api-key": E("GEMINI_API_KEY", "")}, json={"requests": reqs}, timeout=60)
        except requests.RequestException as e:
            raise HTTPException(502, f"Cannot reach Gemini ({type(e).__name__}). Check your internet/DNS.")
        if r.status_code != 200:
            raise HTTPException(502, f"Gemini embedding error {r.status_code}: {r.text[:200]}")
        emb = r.json().get("embeddings")
        if not emb:
            raise HTTPException(502, f"Gemini returned no embeddings: {r.text[:200]}")
        out += [e["values"] for e in emb]
    return out
 
def parse(name, data):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if ext == "docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    if ext in ("xlsx", "xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        lines = []
        for ws in wb:
            rows = ws.iter_rows(values_only=True)
            head = next(rows, None)
            if not head:
                continue
            head = [str(h) if h is not None else f"col{i + 1}" for i, h in enumerate(head)]
            for r in rows:
                cells = [f"{h}: {v}" for h, v in zip(head, r) if v is not None]
                if cells:
                    lines.append(f"[{ws.title}] " + "; ".join(cells))
        return "\n".join(lines)
    return data.decode("utf-8", "ignore")  # txt, md, csv
 
def chunk(text, size=350, ov=60):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size - ov) if w[i:i + size]]
 
# ---------- accounts (stored in Pinecone, namespace "users", so they survive restarts) ----------
class Cred(BaseModel):
    username: str
    password: str
 
def hpw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200000).hex()
 
def token(u):
    return jwt.encode({"sub": u, "exp": int(time.time()) + 7 * 86400}, SECRET, "HS256")
 
def find_user(u):
    return index().fetch(ids=["user:" + u], namespace="users").vectors.get("user:" + u)
 
def user(authorization: str = Header("")):
    try:
        return jwt.decode(authorization.replace("Bearer ", ""), SECRET, ["HS256"])["sub"]
    except Exception:
        raise HTTPException(401, "Please log in again")
 
@app.post("/api/register")
def register(c: Cred):
    u = c.username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.@+-]{3,60}", u) or len(c.password) < 6:
        raise HTTPException(400, "Enter a valid username or email, and a password of at least 6 characters.")
    if find_user(u):
        raise HTTPException(409, "That username is taken")
    salt = secrets.token_hex(16)
    index().upsert(vectors=[{"id": "user:" + u, "values": [1.0] + [0.0] * (DIM - 1),
                             "metadata": {"salt": salt, "pw": hpw(c.password, salt)}}], namespace="users")
    return {"token": token(u), "username": u}
 
@app.post("/api/login")
def login(c: Cred):
    u = c.username.strip().lower()
    rec = find_user(u)
    if not rec or not hmac.compare_digest(rec.metadata["pw"], hpw(c.password, rec.metadata["salt"])):
        raise HTTPException(401, "Wrong username or password")
    return {"token": token(u), "username": u}
 
# ---------- documents + chat (each user has a private Pinecone namespace) ----------
def ns(u):
    return "u_" + hashlib.sha1(u.encode()).hexdigest()[:20]
 
@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...), u: str = Depends(user)):
    total = 0
    for f in files:
        data = f.file.read()
        if len(data) > 15_000_000:
            raise HTTPException(413, f"{f.filename} is over 15 MB")
        chunks = chunk(parse(f.filename, data))
        if not chunks:
            continue
        base = hashlib.md5(f.filename.encode()).hexdigest()[:10]
        items = [{"id": f"{base}-{i}", "values": v, "metadata": {"text": c, "source": f.filename}}
                 for i, (c, v) in enumerate(zip(chunks, embed(chunks, "RETRIEVAL_DOCUMENT")))]
        for i in range(0, len(items), 100):
            index().upsert(vectors=items[i:i + 100], namespace=ns(u))
        total += len(items)
    return {"chunks": total}
 
@app.get("/api/status")
def status(u: str = Depends(user)):
    s = index().describe_index_stats().namespaces.get(ns(u))
    return {"username": u, "chunks": s.vector_count if s else 0}
 
@app.delete("/api/clear")
def clear(u: str = Depends(user)):
    try:
        index().delete(delete_all=True, namespace=ns(u))
    except Exception:
        pass
    return {"ok": True}
 
class Chat(BaseModel):
    message: str
    history: list[dict] = []
 
@app.post("/api/chat")
def chat(c: Chat, u: str = Depends(user)):
    q = embed([c.message], "RETRIEVAL_QUERY")[0]
    hits = [m for m in index().query(vector=q, top_k=4, include_metadata=True, namespace=ns(u)).matches
            if m.score >= MIN_SCORE]
    ctx = "\n\n".join(f"[{m.metadata['source']}] {m.metadata['text']}" for m in hits)
    system = "You are a helpful assistant. " + (
        f"Use this context from the user's documents when relevant:\n{ctx}" if ctx
        else "No document matched the question, so answer from general knowledge.")
    hist = [{"role": h["role"], "content": str(h.get("content", ""))}
            for h in c.history[-6:] if h.get("role") in ("user", "assistant")]
    try:
        r = Groq(api_key=E("GROQ_API_KEY", "")).chat.completions.create(
            model=CHAT_MODEL, temperature=0.3,
            messages=[{"role": "system", "content": system}] + hist + [{"role": "user", "content": c.message}])
    except Exception as e:
        raise HTTPException(502, f"Groq error: {e}")
    return {"answer": r.choices[0].message.content,
            "source": "documents" if hits else "general knowledge",
            "refs": sorted({m.metadata["source"] for m in hits})}
 
@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8").read()
 