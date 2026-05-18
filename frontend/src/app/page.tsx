'use client'

import { useState, useEffect } from 'react'
import { predictSentiment, getMetrics, PredictResponse, ModelMetrics } from '@/lib/api'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'

// ── Confidence Bar ──────────────────────────────────────────
function ConfidenceBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-mono">
        <span style={{ color }}>{label}</span>
        <span className="text-[var(--text)]">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-[var(--surface2)] overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${value * 100}%`, background: color }} />
      </div>
    </div>
  )
}

// ── Result Card ─────────────────────────────────────────────
function ResultCard({ result }: { result: PredictResponse }) {
  const isPositive = result.sentiment === 'positive'
  const accent = isPositive ? 'var(--green)' : 'var(--red)'
  return (
    <div className="rounded-xl border p-5 space-y-4 transition-all duration-500"
      style={{ borderColor: accent, background: 'var(--surface)' }}>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold"
          style={{ background: `${accent}22`, color: accent }}>
          {isPositive ? '★' : '✕'}
        </div>
        <div>
          <p className="text-xs font-mono text-[var(--text-dim)] uppercase tracking-widest">Verdict</p>
          <p className="font-display text-2xl font-bold capitalize" style={{ color: accent }}>
            {result.sentiment}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-xs font-mono text-[var(--text-dim)]">Confidence</p>
          <p className="font-mono text-2xl font-bold text-[var(--text)]">
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
      <div className="space-y-3 pt-1">
        <ConfidenceBar label="POSITIVE" value={result.positive_prob} color="var(--green)" />
        <ConfidenceBar label="NEGATIVE" value={result.negative_prob} color="var(--red)" />
      </div>
      <div className="grid grid-cols-2 gap-3 pt-1">
        {[
          { label: 'Word Count', value: result.word_count },
          { label: 'Tokens (processed)', value: result.processed_tokens },
        ].map(s => (
          <div key={s.label} className="rounded-lg bg-[var(--surface2)] p-3">
            <p className="text-xs font-mono text-[var(--text-dim)]">{s.label}</p>
            <p className="font-mono text-xl font-semibold text-[var(--text)] mt-0.5">{s.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Metrics Panel ───────────────────────────────────────────
function MetricsPanel({ metrics }: { metrics: ModelMetrics }) {
  const pieData = [
    { name: 'Positive', value: metrics.positive_samples },
    { name: 'Negative', value: metrics.negative_samples },
  ]
  const barData = [
    { name: 'Accuracy', value: +(metrics.accuracy * 100).toFixed(1) },
    { name: 'AUC-ROC',  value: +(metrics.auc_roc * 100).toFixed(1) },
    { name: 'F1 (+)',   value: +(metrics.f1_positive * 100).toFixed(1) },
    { name: 'F1 (−)',   value: +(metrics.f1_negative * 100).toFixed(1) },
  ]
  const COLORS = ['var(--green)', 'var(--red)']

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="font-display text-lg font-bold text-[var(--text)]">Model Performance</h2>
      </div>

      {/* Best model badge */}
      <div className="rounded-lg px-3 py-2 flex items-center gap-2"
        style={{ background: 'rgba(201,168,76,0.1)', border: '1px solid var(--gold-dim)' }}>
        <span style={{ color: 'var(--gold)' }}>★</span>
        <span className="text-xs font-mono" style={{ color: 'var(--gold)' }}>
          Best: {metrics.best_model}
        </span>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'Accuracy', value: `${(metrics.accuracy * 100).toFixed(1)}%`, highlight: true },
          { label: 'AUC-ROC',  value: metrics.auc_roc.toFixed(4) },
          { label: 'F1 Positive', value: metrics.f1_positive.toFixed(4) },
          { label: 'F1 Negative', value: metrics.f1_negative.toFixed(4) },
        ].map(m => (
          <div key={m.label} className="rounded-lg p-3"
            style={{
              background: m.highlight ? 'rgba(201,168,76,0.1)' : 'var(--surface2)',
              border: m.highlight ? '1px solid var(--gold-dim)' : '1px solid transparent'
            }}>
            <p className="text-xs font-mono" style={{ color: m.highlight ? 'var(--gold)' : 'var(--text-dim)' }}>
              {m.label}
            </p>
            <p className="font-mono text-xl font-bold" style={{ color: m.highlight ? 'var(--gold)' : 'var(--text)' }}>
              {m.value}
            </p>
          </div>
        ))}
      </div>

      {/* Bar chart */}
      <div>
        <p className="text-xs font-mono text-[var(--text-dim)] mb-2 uppercase tracking-wider">Metrics Overview</p>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={barData} barSize={28}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
            <YAxis domain={[85, 100]} tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, fontFamily: 'DM Mono', fontSize: 12 }}
              formatter={(v: number) => [`${v}%`, '']} />
            <Bar dataKey="value" fill="var(--gold)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Model comparison table jika ada data multi-model */}
      {metrics.all_models && Object.keys(metrics.all_models).length > 1 && (
        <div>
          <p className="text-xs font-mono text-[var(--text-dim)] mb-2 uppercase tracking-wider">Model Comparison</p>
          <div className="space-y-1">
            {Object.entries(metrics.all_models)
              .sort((a, b) => b[1].weighted_score - a[1].weighted_score)
              .map(([name, m]) => {
                const isBest = name === metrics.best_model
                return (
                  <div key={name} className="rounded-lg px-3 py-2 flex items-center justify-between text-xs font-mono"
                    style={{
                      background: isBest ? 'rgba(201,168,76,0.08)' : 'var(--surface2)',
                      border: isBest ? '1px solid var(--gold-dim)' : '1px solid transparent'
                    }}>
                    <span style={{ color: isBest ? 'var(--gold)' : 'var(--text-dim)' }}>
                      {isBest ? '★ ' : ''}{name}
                    </span>
                    <span style={{ color: isBest ? 'var(--gold)' : 'var(--text)' }}>
                      {(m.weighted_score * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Pie chart */}
      <div>
        <p className="text-xs font-mono text-[var(--text-dim)] mb-2 uppercase tracking-wider">Dataset Distribution</p>
        <div className="flex items-center gap-4">
          <ResponsiveContainer width={100} height={100}>
            <PieChart>
              <Pie data={pieData} innerRadius={28} outerRadius={46} dataKey="value" strokeWidth={0}>
                {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-2 flex-1">
            {pieData.map((d, i) => (
              <div key={d.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ background: COLORS[i] }} />
                  <span className="font-mono text-[var(--text-dim)] text-xs">{d.name}</span>
                </div>
                <span className="font-mono text-xs text-[var(--text)]">{d.value.toLocaleString()}</span>
              </div>
            ))}
            <div className="pt-1 border-t border-[var(--border)] flex justify-between text-xs font-mono">
              <span className="text-[var(--text-dim)]">Total</span>
              <span className="text-[var(--text)]">{metrics.total_samples.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Example reviews ─────────────────────────────────────────
const EXAMPLES = [
  "This film is an absolute masterpiece. The storytelling is breathtaking and the performances are unforgettable.",
  "Terrible movie. Bad acting, predictable plot, and a complete waste of two hours.",
  "It had its moments, but overall the pacing felt off and the ending was disappointing.",
  "One of the best films I've seen this year. Visually stunning and emotionally resonant.",
]

// ── Main Page ───────────────────────────────────────────────
export default function Home() {
  const [text, setText]       = useState('')
  const [result, setResult]   = useState<PredictResponse | null>(null)
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {})
  }, [])

  async function handlePredict() {
    if (!text.trim()) return
    setLoading(true); setError(''); setResult(null)
    try {
      setResult(await predictSentiment(text))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--bg)' }}>
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[var(--gold)] flex items-center justify-center text-[var(--bg)] font-bold text-sm">CS</div>
          <span className="font-display text-xl font-bold text-[var(--text)] tracking-tight">CineScope</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-dim)]">
          <span className="w-2 h-2 rounded-full bg-[var(--green)] animate-pulse inline-block" />
          {metrics ? metrics.best_model : 'Loading...'} · 50k reviews
        </div>
      </header>

      {/* Hero */}
      <div className="px-6 pt-10 pb-6 text-center border-b border-[var(--border)]">
        <p className="text-xs font-mono text-[var(--gold)] tracking-[0.3em] uppercase mb-3">NLP · Kelompok 13</p>
        <h1 className="font-display text-4xl md:text-5xl font-black text-[var(--text)] leading-tight mb-2">
          Sentiment<br /><span style={{ color: 'var(--gold)' }}>Analysis</span>
        </h1>
        <p className="text-sm text-[var(--text-dim)] max-w-md mx-auto font-mono">
          Trained on 50,000 IMDB reviews · Best model auto-selected
        </p>
      </div>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8 grid md:grid-cols-[1fr_320px] gap-6">
        {/* Left: Analyzer */}
        <div className="space-y-5">
          <div className="space-y-2">
            <label className="text-xs font-mono text-[var(--text-dim)] uppercase tracking-widest">Movie Review</label>
            <div className="relative rounded-xl overflow-hidden border border-[var(--border)] focus-within:border-[var(--gold-dim)] transition-colors">
              <textarea value={text} onChange={e => setText(e.target.value)} rows={7}
                placeholder="Paste a movie review here..."
                className="w-full bg-[var(--surface)] text-[var(--text)] p-4 text-sm font-sans resize-none outline-none placeholder:text-[var(--border)]" />
              {text && (
                <button onClick={() => { setText(''); setResult(null) }}
                  className="absolute top-3 right-3 text-[var(--text-dim)] hover:text-[var(--text)] text-xs font-mono">
                  clear
                </button>
              )}
            </div>
          </div>

          {/* Examples */}
          <div>
            <p className="text-xs font-mono text-[var(--text-dim)] mb-2 uppercase tracking-wider">Try an example</p>
            <div className="grid grid-cols-2 gap-2">
              {EXAMPLES.map((ex, i) => (
                <button key={i} onClick={() => { setText(ex); setResult(null) }}
                  className="text-left text-xs text-[var(--text-dim)] bg-[var(--surface)] hover:bg-[var(--surface2)] border border-[var(--border)] rounded-lg p-3 transition-colors font-mono">
                  {ex.slice(0, 70)}…
                </button>
              ))}
            </div>
          </div>

          <button onClick={handlePredict} disabled={!text.trim() || loading}
            className="w-full py-3.5 rounded-xl font-mono text-sm font-medium tracking-widest uppercase transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: 'var(--gold)', color: 'var(--bg)' }}>
            {loading ? '⏳  Analyzing...' : '▶  Analyze Sentiment'}
          </button>

          {error && (
            <div className="rounded-xl border border-[var(--red)] bg-[var(--surface)] p-4 text-sm font-mono text-[var(--red)]">
              ⚠ {error}
            </div>
          )}

          {result && <ResultCard result={result} />}
        </div>

        {/* Right: Metrics */}
        <aside className="space-y-5">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5">
            {metrics
              ? <MetricsPanel metrics={metrics} />
              : (
                <div className="space-y-3 animate-pulse">
                  <div className="h-4 w-32 rounded bg-[var(--surface2)]" />
                  <div className="grid grid-cols-2 gap-2">
                    {[...Array(4)].map((_, i) => <div key={i} className="h-16 rounded-lg bg-[var(--surface2)]" />)}
                  </div>
                </div>
              )}
          </div>

          {/* Model info */}
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 space-y-3">
            <h3 className="font-display text-base font-bold text-[var(--text)]">Model Info</h3>
            {[
              ['Architecture', metrics?.best_model ?? '—'],
              ['Vectorizer', '50,000 features, bigrams'],
              ['Preprocessing', 'NLTK stopwords + PorterStemmer'],
              ['Train / Test', '80% / 20%'],
              ['Selection Metric', '0.3×Acc + 0.4×F1 + 0.3×AUC'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2 text-xs font-mono border-b border-[var(--surface2)] pb-2 last:border-0">
                <span className="text-[var(--text-dim)]">{k}</span>
                <span className="text-[var(--text)] text-right">{v}</span>
              </div>
            ))}
          </div>
        </aside>
      </main>

      <footer className="border-t border-[var(--border)] py-4 text-center text-xs font-mono text-[var(--text-dim)]">
        Kelompok 13 · NLP LE01 Semester 4 · IMDB Sentiment Analysis
      </footer>
    </div>
  )
}