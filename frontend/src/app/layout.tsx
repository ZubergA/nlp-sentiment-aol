import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CineVibes Sentiment Analysis',
  description: 'NLP powered movie review sentiment analysis using TF-IDF + Logistic Regression trained on 50,000 IMDB reviews.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
