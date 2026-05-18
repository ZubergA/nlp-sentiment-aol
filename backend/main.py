import re, json, os
from pathlib import Path
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# NLTK SETUP  (download otomatis jika belum ada)
# ─────────────────────────────────────────────
def ensure_nltk_data():
    resources = {
        "corpora/stopwords":    "stopwords",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

ensure_nltk_data()

stemmer   = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────
# PREPROCESSING  (identik dengan train_model.py)
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = re.sub(r"<[^>]+>", " ", text)
    text   = re.sub(r"[^a-zA-Z\s]", " ", text)
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="IMDB Sentiment Analysis API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = Path(__file__).parent
pipeline = None
report   = {}

@app.on_event("startup")
def load_model():
    global pipeline, report
    model_path   = BASE_DIR / "model" / "sentiment_pipeline.joblib"
    metrics_path = BASE_DIR / "model" / "metrics.json"
    if model_path.exists():
        pipeline = joblib.load(model_path)
    if metrics_path.exists():
        with open(metrics_path) as f:
            report = json.load(f)

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment:        str
    confidence:       float
    positive_prob:    float
    negative_prob:    float
    word_count:       int
    processed_tokens: int

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":    "IMDB Sentiment Analysis API",
        "status":     "running",
        "best_model": report.get("best_model", "unknown"),
    }

@app.get("/metrics")
def get_metrics():
    return report

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not pipeline:
        raise HTTPException(503, "Model not loaded. Jalankan train_model.py terlebih dahulu.")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text tidak boleh kosong.")
    cleaned  = preprocess(text)
    proba    = pipeline.predict_proba([cleaned])[0]
    neg_prob, pos_prob = float(proba[0]), float(proba[1])
    return PredictResponse(
        sentiment        = "positive" if pos_prob >= 0.5 else "negative",
        confidence       = round(max(pos_prob, neg_prob), 4),
        positive_prob    = round(pos_prob, 4),
        negative_prob    = round(neg_prob, 4),
        word_count       = len(text.split()),
        processed_tokens = len(cleaned.split()),
    )

@app.post("/predict/batch")
def predict_batch(texts: list[str]):
    if not pipeline:
        raise HTTPException(503, "Model not loaded.")
    results = []
    for text in texts[:20]:
        cleaned = preprocess(text)
        proba   = pipeline.predict_proba([cleaned])[0]
        results.append({
            "text":       text[:100] + "..." if len(text) > 100 else text,
            "sentiment":  "positive" if proba[1] >= 0.5 else "negative",
            "confidence": round(float(max(proba)), 4),
        })
    return results