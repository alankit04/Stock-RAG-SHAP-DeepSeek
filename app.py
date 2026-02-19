import io
import os
import base64
import pickle
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required inside Docker
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import requests as http_requests
import shap
from PIL import Image
from sentence_transformers import SentenceTransformer

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1")
DATA_DIR     = os.environ.get("DATA_DIR",     ".")           # override in Docker


def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ── Shared state ──────────────────────────────────────────────────────────────
embedder     = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection   = chroma_client.get_or_create_collection("deepseek_rag")


# ── Startup: build index automatically ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_index()
    yield


app = FastAPI(
    title="Stock RAG SHAP API",
    version="1.0.0",
    description="RAG + SHAP explainability over stock data, powered by DeepSeek via Ollama.",
    lifespan=lifespan,
)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _build_index() -> None:
    """Index CSV summary + SHAP OCR text into ChromaDB."""
    global collection, chroma_client

    csv_df   = pd.read_csv(_path("X_test_subset.csv"))
    csv_text = (
        "CSV Summary:\n"
        + csv_df.describe().to_string()
        + "\n\nSample Rows:\n"
        + csv_df.head(5).to_string()
    )

    image     = Image.open(_path("Shap Plot.png"))
    shap_text = "SHAP Plot (OCR):\n" + pytesseract.image_to_string(image)

    # recreate collection cleanly
    try:
        chroma_client.delete_collection("deepseek_rag")
    except Exception:
        pass
    chroma_client = chromadb.Client()
    collection    = chroma_client.get_or_create_collection("deepseek_rag")

    docs       = [csv_text, shap_text]
    embeddings = embedder.encode(docs).tolist()
    collection.add(
        documents=docs,
        metadatas=[{"source": "csv"}, {"source": "shap_image"}],
        ids=["doc1", "doc2"],
        embeddings=embeddings,
    )


def _generate_shap_png() -> io.BytesIO:
    """Compute fresh SHAP summary plot and return as PNG bytes."""
    with open(_path("xgb_model_subset.pkl"), "rb") as f:
        model = pickle.load(f)
    X_test       = pd.read_csv(_path("X_test_subset.csv"))
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def _call_ollama(prompt: str) -> str:
    """Call Ollama HTTP API (works both locally and inside Docker compose)."""
    resp = http_requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Request models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — returns 200 if API is up."""
    return {"status": "ok"}


@app.post("/index", tags=["ops"])
def reindex():
    """Re-index CSV + SHAP data into ChromaDB (call after fresh data arrives)."""
    try:
        _build_index()
        return {"status": "index rebuilt"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/rag", tags=["rag"])
def rag_query(req: QueryRequest):
    """
    Query the RAG pipeline.
    Retrieves relevant context from ChromaDB, then sends it to DeepSeek via Ollama.
    """
    try:
        q_emb   = embedder.encode([req.question]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        context = "\n\n".join(results["documents"][0])
        prompt  = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer  = _call_ollama(prompt)
        return {"question": req.question, "answer": answer}
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running?",
        )
    except http_requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="DeepSeek response timed out (>120s).")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/shap/image", tags=["shap"])
def shap_image():
    """Return the SHAP summary plot as a PNG image (computed fresh each call)."""
    try:
        return StreamingResponse(_generate_shap_png(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/shap/base64", tags=["shap"])
def shap_base64():
    """Return the SHAP summary plot as a base64-encoded PNG string (handy for frontends)."""
    try:
        buf     = _generate_shap_png()
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": encoded}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
