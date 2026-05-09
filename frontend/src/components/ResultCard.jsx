// src/components/ResultCard.jsx
import { motion } from 'framer-motion';
import EmotionBadge from './EmotionBadge';
import ConfidenceBar from './ConfidenceBar';

const COLORS = {
  Happy: 'var(--neon-happy)', Sad: 'var(--neon-sad)',
  Angry: 'var(--neon-angry)', Neutral: 'var(--neon-neutral)',
  Disgust: 'var(--neon-disgust)', Fear: 'var(--neon-fear)',
  Surprise: 'var(--neon-surprise)', Uncertain: 'var(--neon-uncertain)',
};

export default function ResultCard({ result }) {
  const { emotion, confidence, all_probs } = result;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-card"
      style={{ padding: 32, marginTop: 24 }}
    >
      <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
        Detection Result
      </p>

      <div className="flex justify-center mb-6">
        <EmotionBadge emotion={emotion} />
      </div>

      <ConfidenceBar confidence={confidence} emotion={emotion} />

      {all_probs && (
        <div className="mt-7">
          <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
            All Probabilities
          </p>
          {Object.entries(all_probs).map(([label, prob], i) => (
            <div key={label} className="mb-2.5">
              <div className="flex justify-between mb-1">
                <span className="text-sm" style={{ color: COLORS[label] || 'var(--text-secondary)' }}>{label}</span>
                <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{Math.round(prob * 100)}%</span>
              </div>
              <div className="prob-track">
                <motion.div
                  className="prob-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(prob * 100)}%` }}
                  transition={{ duration: 0.7, ease: 'easeOut', delay: 0.1 + i * 0.08 }}
                  style={{ background: COLORS[label] || 'var(--accent-violet)' }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {result.latency_ms && (
        <p className="mt-5 text-xs text-center" style={{ color: 'var(--text-muted)' }}>
          Inference: {result.latency_ms}ms
        </p>
      )}
    </motion.div>
  );
}
