FastAPI backend for IMDB Sentiment Analysis.
Kelompok 13 - NLP

# CineScope — IMDB Sentiment Analysis

**Kelompok 13 · NLP LE01 Semester 4**

TF-IDF + Logistic Regression sentiment classifier trained on 50,000 IMDB reviews.  
**Accuracy: 90.9% · AUC-ROC: 0.9682**

---

## Project Structure

```
sentiment-project/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── train_model.py           # Training script
│   ├── requirements.txt
│   ├── render.yaml              # Render.com config
│   └── model/
│       ├── sentiment_pipeline.joblib   # Trained model (generated)
│       └── metrics.json                # Eval metrics (generated)
└── frontend/
    ├── src/app/
    │   ├── page.tsx             # Main UI
    │   ├── layout.tsx
    │   └── globals.css
    ├── src/lib/api.ts           # API client
    ├── package.json
    ├── .env.example
    └── ...
```

---

## Step 1 — Train the Model (run once locally)

```bash
cd backend
pip install -r requirements.txt

# Copy the dataset here
cp /path/to/IMDB_Dataset.csv .

python train_model.py
# → Creates model/sentiment_pipeline.joblib
# → Creates model/metrics.json
```

---

## Step 2 — Deploy Backend to Render.com (Free)

1. Push the `backend/` folder to a **GitHub repo**  
   _(include the `model/` folder — commit the .joblib file)_

2. Go to [render.com](https://render.com) → **New → Web Service**

3. Connect your GitHub repo

4. Render auto-detects `render.yaml` — just click **Deploy**

5. Copy your service URL, e.g.:  
   `https://imdb-sentiment-api.onrender.com`

> ⚠️ Free tier sleeps after 15 min inactivity. First request may take ~30s to wake up.

---

## Step 3 — Deploy Frontend to Vercel (Free)

1. Push the `frontend/` folder to a **GitHub repo**

2. Go to [vercel.com](https://vercel.com) → **New Project** → Import repo

3. Add environment variable:

   ```
   NEXT_PUBLIC_API_URL = https://your-backend.onrender.com
   ```

4. Click **Deploy** — Vercel handles the rest

5. Your app will be live at `https://your-project.vercel.app` 🎉

---

## Local Development

**Backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
# → Docs at http://localhost:8000/docs
```

**Frontend:**

```bash
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local → NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
# → http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint         | Description               |
| ------ | ---------------- | ------------------------- |
| GET    | `/`              | Health check              |
| GET    | `/metrics`       | Model performance metrics |
| POST   | `/predict`       | Predict single review     |
| POST   | `/predict/batch` | Predict up to 20 reviews  |

**POST /predict — Request:**

```json
{ "text": "This movie was absolutely incredible!" }
```

**POST /predict — Response:**

```json
{
  "sentiment": "positive",
  "confidence": 0.9743,
  "positive_prob": 0.9743,
  "negative_prob": 0.0257,
  "word_count": 6,
  "processed_tokens": 3
}
```

---

## Model Details

| Component        | Config                                                 |
| ---------------- | ------------------------------------------------------ |
| Vectorizer       | TF-IDF, 50k features, unigrams + bigrams, sublinear TF |
| Classifier       | Logistic Regression, C=5, L2, lbfgs solver             |
| Preprocessing    | HTML strip, lowercase, stemming, stopword removal      |
| Train/Test split | 80% / 20%, stratified                                  |
| Dataset          | 50,000 IMDB reviews, balanced (25k pos / 25k neg)      |

---

## Troubleshooting

**CORS error in browser** → Make sure `NEXT_PUBLIC_API_URL` has no trailing slash

**Model not found on Render** → Ensure `model/` folder is committed to git (remove it from `.gitignore`)

**Slow first response** → Free Render tier cold-starts in ~30s. Normal behaviour.

**`joblib` version mismatch** → Train model on same Python version as Render uses (3.11)
