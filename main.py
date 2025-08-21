#!/usr/bin/env python3
"""
FastAPI AI microservice (zero-cost defaults):
- Generation: Ollama (local), optional OpenAI fallback (not required here)
- Retrieval: SentenceTransformers embeddings + FAISS
- Endpoints: /health, /classify, /rag, /reload-index, /__routes__ (debug)

Run:
  uvicorn main:app --reload
"""

import os
import glob
import json
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")  # relative, robust

# ---------------------------
# Globals (lazy-initialized)
# ---------------------------
_EMBED_MODEL: Optional[SentenceTransformer] = None
_INDEX: Optional[faiss.IndexFlatIP] = None
_CORPUS: List[str] = []
_METAS: List[Dict[str, str]] = []
_READY: bool = False


def get_embed_model() -> SentenceTransformer:
    """Load the embedding model once (lazy)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


# ---------------------------
# Data ingestion & chunking
# ---------------------------
def read_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (path, content) for all .txt files with non-empty content."""
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    items: List[Tuple[str, str]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    items.append((p, content))
        except Exception:
            # Skip unreadable files rather than crash
            continue
    return items


def chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """Naive word-based chunking with safe step."""
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, size - overlap)  # never zero/negative
    chunks: List[str] = []
    for i in range(0, len(tokens), step):
        c = tokens[i:i + size]
        if not c:
            break
        chunks.append(" ".join(c))
    return chunks


# ---------------------------
# Index build & search
# ---------------------------
def build_index(data_folder: str) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict[str, str]]]:
    """Build FAISS index and return (index, corpus, metas)."""
    files = read_files(data_folder)
    if not files:
        raise RuntimeError(
            f"No .txt files with content found in {data_folder}. "
            f"Add a few files (e.g., 01_rag_concepts.txt) and restart."
        )

    corpus: List[str] = []
    metas: List[Dict[str, str]] = []
    for path, content in files:
        for ch in chunk(content):
            corpus.append(ch)
            metas.append({"source": os.path.basename(path)})

    if not corpus:
        raise RuntimeError(
            "No chunks produced from input files. "
            "Consider smaller CHUNK_SIZE or add more content."
        )

    model = get_embed_model()
    embs = model.encode(corpus, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    # Validate (N, D)
    if embs.ndim != 2 or embs.shape[0] == 0 or embs.shape[1] == 0:
        raise RuntimeError(f"Invalid embedding shape {embs.shape}. Check corpus/data.")

    dim = int(embs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, corpus, metas


def search(query: str, top_k: int = 4) -> List[Dict[str, object]]:
    """Return list of {text, score, source} hits."""
    if not _READY or _INDEX is None:
        raise RuntimeError("Index not ready.")
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = _INDEX.search(q_emb, top_k)
    results: List[Dict[str, object]] = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        results.append({
            "text": _CORPUS[i],
            "score": float(s),
            "source": _METAS[i]["source"],
        })
    return results


def ollama_generate(prompt: str) -> str:
    """Call local Ollama for generation."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


# ---------------------------
# FastAPI app & models
# ---------------------------
app = FastAPI(title="AI Microservice", version="0.2.0")


class ClassifyRequest(BaseModel):
    text: str


class RAGRequest(BaseModel):
    question: Optional[str] = None
    top_k: Optional[int] = 4

    # Accept 'query' as an alias for 'question' (more forgiving API).
    @field_validator("question", mode="before")
    @classmethod
    def accept_query_alias(cls, v, info):
        if v is not None:
            return v
        raw = getattr(info, "data", None)
        if isinstance(raw, dict):
            q = raw.get("query")
            if q:
                return q
        return v


@app.on_event("startup")
def on_startup():
    """Build the index on startup; app still serves even if empty, but shows ready=false."""
    global _INDEX, _CORPUS, _METAS, _READY
    try:
        t0 = time.time()
        _INDEX, _CORPUS, _METAS = build_index(DATA_DIR)
        _READY = True
        print(f"[startup] Indexed {len(_CORPUS)} chunks from {DATA_DIR} in {time.time()-t0:.2f}s")
    except Exception as e:
        _READY = False
        print(f"[startup] WARNING: index not built: {e}")


@app.get("/", include_in_schema=False)
def root():
    """Send browsers to interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "ready": _READY, "chunks": len(_CORPUS)}


@app.get("/__routes__", include_in_schema=False)
def list_routes():
    return [r.path for r in app.routes if isinstance(r, APIRoute)]


@app.post("/reload-index", tags=["system"])
def reload_index():
    """Rebuild index after adding/removing files in ./data without restarting."""
    global _INDEX, _CORPUS, _METAS, _READY
    try:
        t0 = time.time()
        _INDEX, _CORPUS, _METAS = build_index(DATA_DIR)
        _READY = True
        return {"status": "ok", "ready": True, "chunks": len(_CORPUS), "took_sec": round(time.time()-t0, 2)}
    except Exception as e:
        _READY = False
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify")
def classify(req: ClassifyRequest):
    prompt = (
        "Classify the following feedback into one of [bug, feature_request, praise, question]. "
        "Return JSON only with keys: label, rationale.\n\n"
        f"Input: {req.text}\nOutput:"
    )
    out = ollama_generate(prompt)
    # Best-effort JSON parse; otherwise return raw
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


@app.post("/rag")
def rag(req: RAGRequest):
    if not _READY:
        raise HTTPException(status_code=400, detail="Index not ready. Add .txt files to /data and /reload-index.")
    if not req.question:
        raise HTTPException(status_code=422, detail="Provide 'question' or 'query'.")
    top_k = max(1, int(req.top_k or 4))

    hits = search(req.question, top_k)
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    context = "\n\n".join(h["text"] for h in hits)
    prompt = (
        "You are a careful assistant. Use only this context to answer. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )
    answer = ollama_generate(prompt)
    return {"answer": answer, "chunks": hits}
