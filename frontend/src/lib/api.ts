const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface PredictResponse {
  sentiment: 'positive' | 'negative'
  confidence: number
  positive_prob: number
  negative_prob: number
  word_count: number
  processed_tokens: number
}

// Struktur flat (metrics.json lama / single model)
interface FlatMetrics {
  accuracy: number
  auc_roc: number
  f1_positive: number
  f1_negative: number
  precision_positive: number
  recall_positive: number
  precision_negative: number
  recall_negative: number
  total_samples: number
  positive_samples: number
  negative_samples: number
}

// Struktur nested (metrics.json baru / multi-model)
interface NestedMetrics {
  best_model: string
  score_weights: Record<string, number>
  dataset: {
    total_samples: number
    positive_samples: number
    negative_samples: number
    train_samples: number
    test_samples: number
  }
  best_metrics: {
    accuracy: number
    auc_roc: number
    f1_macro: number
    f1_positive: number
    f1_negative: number
    precision_positive: number
    recall_positive: number
    precision_negative: number
    recall_negative: number
    weighted_score: number
  }
  all_models: Record<string, {
    accuracy: number
    auc_roc: number
    weighted_score: number
    f1_macro: number
    f1_positive: number
    f1_negative: number
  }>
}

// Interface yang digunakan frontend — selalu flat & konsisten
export interface ModelMetrics {
  best_model: string
  accuracy: number
  auc_roc: number
  f1_positive: number
  f1_negative: number
  f1_macro: number
  precision_positive: number
  recall_positive: number
  precision_negative: number
  recall_negative: number
  weighted_score: number
  total_samples: number
  positive_samples: number
  negative_samples: number
  all_models: Record<string, { accuracy: number; auc_roc: number; weighted_score: number; f1_macro: number }> | null
}

// Normalisasi: handle kedua format lama & baru
function normalizeMetrics(raw: FlatMetrics | NestedMetrics): ModelMetrics {
  // Format baru (ada best_metrics)
  if ('best_metrics' in raw && raw.best_metrics) {
    const r = raw as NestedMetrics
    return {
      best_model:         r.best_model || 'Unknown',
      accuracy:           r.best_metrics.accuracy,
      auc_roc:            r.best_metrics.auc_roc,
      f1_positive:        r.best_metrics.f1_positive,
      f1_negative:        r.best_metrics.f1_negative,
      f1_macro:           r.best_metrics.f1_macro ?? r.best_metrics.f1_positive,
      precision_positive: r.best_metrics.precision_positive,
      recall_positive:    r.best_metrics.recall_positive,
      precision_negative: r.best_metrics.precision_negative,
      recall_negative:    r.best_metrics.recall_negative,
      weighted_score:     r.best_metrics.weighted_score ?? r.best_metrics.accuracy,
      total_samples:      r.dataset.total_samples,
      positive_samples:   r.dataset.positive_samples,
      negative_samples:   r.dataset.negative_samples,
      all_models:         r.all_models ?? null,
    }
  }

  // Format lama (flat)
  const r = raw as FlatMetrics
  return {
    best_model:         'Logistic Regression',
    accuracy:           r.accuracy,
    auc_roc:            r.auc_roc,
    f1_positive:        r.f1_positive,
    f1_negative:        r.f1_negative,
    f1_macro:           r.f1_positive,
    precision_positive: r.precision_positive,
    recall_positive:    r.recall_positive,
    precision_negative: r.precision_negative,
    recall_negative:    r.recall_negative,
    weighted_score:     r.accuracy,
    total_samples:      r.total_samples,
    positive_samples:   r.positive_samples,
    negative_samples:   r.negative_samples,
    all_models:         null,
  }
}

export async function predictSentiment(text: string): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}

export async function getMetrics(): Promise<ModelMetrics> {
  const res = await fetch(`${API_URL}/metrics`)
  if (!res.ok) throw new Error('Failed to fetch metrics')
  const raw = await res.json()
  return normalizeMetrics(raw)
}