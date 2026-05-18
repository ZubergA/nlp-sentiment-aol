import re
import json
import os
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="IMDB Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Preprocessing (mirrors training, no NLTK needed) ---
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','was','are','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall','can',
    'not','no','nor','so','yet','both','either','neither','as','if','then',
    'than','this','that','these','those','i','me','my','we','our','you','your',
    'he','his','she','her','it','its','they','their','them','what','which','who',
    'whom','when','where','why','how','all','each','every','few','more','most',
    'other','some','such','very','just','also','about','up','out','into','over',
    'after','before','between','through','during','again','further','once','now'
} #note: Change this shit later

def simple_stem(word: str) -> str:
    suffixes = ['ingness','ingly','ational','tional','enci','anci','izer',
                'ising','izing','ation','ness','ment','ful','less','ing','ion',
                'ed','er','ly','al','ic','ous','ive','es','s']
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[:-len(s)]
    return word

def preprocess(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [simple_stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# --- Load model & metrics at startup ---
BASE_DIR = Path(__file__).parent
pipeline = None
metrics = {}

@app.on_event("startup")
def load_model():
    global pipeline, metrics
    model_path = BASE_DIR / "model" / "sentiment_pipeline.joblib"
    metrics_path = BASE_DIR / "model" / "metrics.json"
    if model_path.exists():
        pipeline = joblib.load(model_path)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

# --- Schemas ---
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    positive_prob: float
    negative_prob: float
    word_count: int
    processed_tokens: int

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "IMDB Sentiment Analysis API", "status": "running"}

@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    cleaned = preprocess(text)
    proba = pipeline.predict_proba([cleaned])[0]
    neg_prob, pos_prob = float(proba[0]), float(proba[1])
    sentiment = "positive" if pos_prob >= 0.5 else "negative"
    confidence = max(pos_prob, neg_prob)
    
    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        positive_prob=round(pos_prob, 4),
        negative_prob=round(neg_prob, 4),
        word_count=len(text.split()),
        processed_tokens=len(cleaned.split()),
    )

@app.post("/predict/batch")
def predict_batch(texts: list[str]):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for text in texts[:20]:  # max 20
        cleaned = preprocess(text)
        proba = pipeline.predict_proba([cleaned])[0]
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": "positive" if proba[1] >= 0.5 else "negative",
            "confidence": round(float(max(proba)), 4),
        })
    return results
