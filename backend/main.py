import re, json, os
from pathlib import Path
from typing import Optional
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
def ensure_nltk_data():
    for path, name in [("corpora/stopwords", "stopwords"), ("tokenizers/punkt_tab", "punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

ensure_nltk_data()
stemmer   = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────
# SPACY NER SETUP
# ─────────────────────────────────────────────
nlp_ner = None
ENTITY_TYPES = {"PERSON", "WORK_OF_ART", "ORG"}
ENTITY_LABELS = {
    "PERSON":      "Person",
    "WORK_OF_ART": "Film/Title",
    "ORG":         "Organization",
}

def load_spacy():
    global nlp_ner
    try:
        import spacy
        nlp_ner = spacy.load("en_core_web_md")
        print("spaCy NER loaded successfully (en_core_web_md)")
    except Exception as e:
        print(f"spaCy not available: {e} — NER disabled")
        nlp_ner = None

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = re.sub(r"<[^>]+>", " ", text)
    text   = re.sub(r"[^a-zA-Z\s]", " ", text)
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# NER HELPERS
# ─────────────────────────────────────────────
def is_valid_entity(text: str) -> bool:
    """Filter entitas yang tidak valid / terlalu noise."""
    text = text.strip()
    if len(text) < 3:
        return False
    # Harus ada minimal 1 huruf kapital (proper noun)
    if text == text.lower():
        return False
    # Filter kata tunggal yang umum
    noise = {'the', 'a', 'an', 'it', 'he', 'she', 'they', 'this', 'that'}
    if text.lower() in noise:
        return False
    return True

def get_surrounding_sentences(text: str, ent_start: int, ent_end: int) -> str:
    """Ambil kalimat yang mengandung entitas + kalimat sebelum/sesudahnya."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    char_count = 0
    target_idx = 0
    sent_positions = []

    for i, sent in enumerate(sentences):
        sent_positions.append((char_count, char_count + len(sent)))
        if char_count <= ent_start <= char_count + len(sent):
            target_idx = i
        char_count += len(sent) + 1

    # Ambil kalimat target + 1 sebelum dan 1 sesudah untuk konteks lebih baik
    start_idx = max(0, target_idx - 1)
    end_idx   = min(len(sentences), target_idx + 2)
    return " ".join(sentences[start_idx:end_idx]).strip()

def extract_entities(text: str, pipeline) -> list:
    """Ekstrak entitas + sentimen per entitas."""
    if nlp_ner is None:
        return []

    doc  = nlp_ner(text)
    seen = set()
    entities = []

    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        if not is_valid_entity(ent.text):
            continue
        key = ent.text.lower()
        if key in seen:
            continue
        seen.add(key)

        # Konteks kalimat untuk entity-level sentiment
        context = get_surrounding_sentences(text, ent.start_char, ent.end_char)
        cleaned = preprocess(context)

        if cleaned.strip():
            proba      = pipeline.predict_proba([cleaned])[0]
            neg_p, pos_p = float(proba[0]), float(proba[1])
            sentiment  = "positive" if pos_p >= 0.5 else "negative"
            confidence = round(max(pos_p, neg_p), 4)
        else:
            sentiment  = "neutral"
            confidence = 0.5

        entities.append({
            "text":          ent.text,
            "label":         ent.label_,
            "label_display": ENTITY_LABELS.get(ent.label_, ent.label_),
            "sentiment":     sentiment,
            "confidence":    confidence,
            "context":       context.strip(),
        })

    return entities

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="IMDB Sentiment Analysis API", version="3.1.0")
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
    load_spacy()

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class EntityResult(BaseModel):
    text:          str
    label:         str
    label_display: str
    sentiment:     str
    confidence:    float
    context:       str

class PredictResponse(BaseModel):
    sentiment:        str
    confidence:       float
    positive_prob:    float
    negative_prob:    float
    word_count:       int
    processed_tokens: int
    entities:         list[EntityResult]
    ner_available:    bool

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":       "IMDB Sentiment Analysis API",
        "status":        "running",
        "best_model":    report.get("best_model", "unknown"),
        "ner_available": nlp_ner is not None,
        "ner_model":     "en_core_web_md",
    }

@app.get("/metrics")
def get_metrics():
    return report

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not pipeline:
        raise HTTPException(503, "Model not loaded.")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text tidak boleh kosong.")

    cleaned  = preprocess(text)
    proba    = pipeline.predict_proba([cleaned])[0]
    neg_prob, pos_prob = float(proba[0]), float(proba[1])
    entities = extract_entities(text, pipeline)

    return PredictResponse(
        sentiment        = "positive" if pos_prob >= 0.5 else "negative",
        confidence       = round(max(pos_prob, neg_prob), 4),
        positive_prob    = round(pos_prob, 4),
        negative_prob    = round(neg_prob, 4),
        word_count       = len(text.split()),
        processed_tokens = len(cleaned.split()),
        entities         = entities,
        ner_available    = nlp_ner is not None,
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