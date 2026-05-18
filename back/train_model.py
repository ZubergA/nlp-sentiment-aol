

import re
import json
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

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
}

def simple_stem(word):
    suffixes = ['ingness','ingly','ational','tional','enci','anci','izer',
                'ising','izing','ation','ness','ment','ful','less','ing','ion',
                'ed','er','ly','al','ic','ous','ive','es','s']
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[:-len(s)]
    return word

def preprocess(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [simple_stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

if __name__ == "__main__":
    print("Loading IMDB_Dataset.csv ...")
    df = pd.read_csv("IMDB_Dataset.csv")
    df['label'] = (df['sentiment'] == 'positive').astype(int)

    print("Preprocessing 50,000 reviews")
    df['clean'] = df['review'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf', LogisticRegression(C=5, max_iter=1000, solver='lbfgs'))
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/sentiment_pipeline.joblib")

    metrics = {
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "f1_positive": round(report['1']['f1-score'], 4),
        "f1_negative": round(report['0']['f1-score'], 4),
        "precision_positive": round(report['1']['precision'], 4),
        "recall_positive": round(report['1']['recall'], 4),
        "precision_negative": round(report['0']['precision'], 4),
        "recall_negative": round(report['0']['recall'], 4),
        "total_samples": len(df),
        "positive_samples": int(df['label'].sum()),
        "negative_samples": int((df['label'] == 0).sum()),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to model/sentiment_pipeline.joblib")
    print("Metrics saved to model/metrics.json")
