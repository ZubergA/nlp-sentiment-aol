import re, json, os, time, joblib
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


DATASET_PATH  = "IMDB_Dataset.csv"
MODEL_DIR     = "model"
SCORE_WEIGHTS = {"accuracy": 0.3, "f1_macro": 0.4, "auc_roc": 0.3}


def ensure_nltk_data():
    resources = {
        "corpora/stopwords":    "stopwords",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK '{name}'...")
            nltk.download(name, quiet=True)

#preprocess
stemmer   = PorterStemmer()
STOPWORDS = None  # diinisialisasi setelah NLTK data ready

def init_stopwords():
    global STOPWORDS
    STOPWORDS = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    # 1. Hapus HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Hapus karakter non-huruf
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # 3. Lowercase
    text = text.lower()
    # 4. Tokenisasi pakai NLTK word_tokenize
    tokens = word_tokenize(text)
    # 5. Filter stopwords + stemming pakai PorterStemmer NLTK
    tokens = [
        stemmer.stem(w)
        for w in tokens
        if w not in STOPWORDS and len(w) > 2
    ]
    return " ".join(tokens)


def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2)),
            ("clf",   LogisticRegression(C=5, max_iter=1000, solver="lbfgs", random_state=42)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2)),
            ("clf",   CalibratedClassifierCV(LinearSVC(C=1, max_iter=2000, random_state=42), cv=3)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True, min_df=3)),
            ("clf",   RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ]),
    }


def weighted_score(acc, f1, auc):
    return (SCORE_WEIGHTS["accuracy"] * acc +
            SCORE_WEIGHTS["f1_macro"] * f1  +
            SCORE_WEIGHTS["auc_roc"]  * auc)

def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)
    ws  = weighted_score(acc, f1, auc)
    return {
        "accuracy":round(acc, 4),
        "f1_macro":round(f1, 4),
        "auc_roc":round(auc, 4),
        "weighted_score":round(ws, 4),
        "f1_positive":round(report["1"]["f1-score"], 4),
        "f1_negative":round(report["0"]["f1-score"], 4),
        "precision_positive":round(report["1"]["precision"], 4),
        "recall_positive":round(report["1"]["recall"], 4),
        "precision_negative":round(report["0"]["precision"], 4),
        "recall_negative":round(report["0"]["recall"], 4),
        "confusion_matrix":cm,
    }


SEP = "─" * 65
def hdr(text): print(f"\n{SEP}\n  {text}\n{SEP}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    hdr("0. NLTK SETUP")
    ensure_nltk_data()
    init_stopwords()
    print(f"  PorterStemmer       : aktif")
    print(f"  Stopwords (NLTK)    : {len(STOPWORDS)} kata")
    print(f"  Tokenizer           : word_tokenize (NLTK punkt)")

    # 1. Load dataset
    hdr("1. LOADING DATASET")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}\n"
            "Pastikan IMDB_Dataset.csv ada di folder backend/"
        )
    df = pd.read_csv(DATASET_PATH)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"  Total   : {len(df):,}")
    print(f"  Positive: {df['label'].sum():,}  |  Negative: {(df['label']==0).sum():,}")

    # 2. Preprocess
    hdr("2. PREPROCESSING")
    print("  HTML strip → word_tokenize → remove stopwords → PorterStemmer")
    t0 = time.time()
    df["clean"] = df["review"].apply(preprocess)
    dur = time.time() - t0
    avg_tokens = df["clean"].str.split().str.len().mean()
    print(f"  Done in {dur:.1f}s")
    print(f"  Avg tokens per review (after preprocessing): {avg_tokens:.0f}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"  Train : {len(X_train):,}  |  Test: {len(X_test):,}")

    # 4. Train all 4 models
    hdr("3. TRAINING 4 MODELS")
    print(f"  Score = {SCORE_WEIGHTS['accuracy']}×Acc"
          f" + {SCORE_WEIGHTS['f1_macro']}×F1"
          f" + {SCORE_WEIGHTS['auc_roc']}×AUC\n")

    all_results = {}
    for name, pipeline in get_models().items():
        print(f"  ▶ {name}...", end="", flush=True)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        duration = time.time() - t0
        metrics  = evaluate(pipeline, X_test, y_test)
        metrics["training_time_seconds"] = round(duration, 1)
        all_results[name] = {"pipeline": pipeline, "metrics": metrics}
        print(f"  {duration:.1f}s  →  score={metrics['weighted_score']:.4f}")

    # 5. Pilih model terbaik
    hdr("4. MODEL COMPARISON")
    best_name = max(all_results, key=lambda n: all_results[n]["metrics"]["weighted_score"])

    print(f"\n  {'Model':<28} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Score':>7}")
    print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for name, r in all_results.items():
        m= r["metrics"]
        tag = " ★" if name == best_name else ""
        print(f"  {name+tag:<30} {m['accuracy']:>7.4f} {m['f1_macro']:>7.4f}"
              f" {m['auc_roc']:>7.4f} {m['weighted_score']:>7.4f}")

    print(f"\n  Best model: {best_name}")

    # 6. Simpan model terbaik
    hdr("5. SAVING")
    best_pipeline = all_results[best_name]["pipeline"]
    best_metrics  = all_results[best_name]["metrics"]

    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "sentiment_pipeline.joblib"))
    print(f"  Pipeline - model/sentiment_pipeline.joblib")

    report = {
        "best_model":best_name,
        "score_weights": SCORE_WEIGHTS,
        "preprocessing": {
            "stemmer":"PorterStemmer (NLTK)",
            "stopwords_library":"NLTK English stopwords",
            "stopwords_count":len(STOPWORDS),
            "tokenizer":"word_tokenize (NLTK punkt)",
        },
        "dataset": {
            "total_samples":len(df),
            "positive_samples":int(df["label"].sum()),
            "negative_samples":int((df["label"]==0).sum()),
            "train_samples":len(X_train),
            "test_samples":len(X_test),
        },
        "best_metrics": best_metrics,
        "all_models":   {n: r["metrics"] for n, r in all_results.items()},
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report  - model/metrics.json")

    # 7. Summary
    hdr("6. DONE")
    print(f"Best Model: {best_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {best_metrics['f1_macro']:.4f}")
    print(f"AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print(f"Weighted Score: {best_metrics['weighted_score']:.4f}")
    print(f"\n  Model siap untuk deploy.\n")


if __name__ == "__main__":
    main()