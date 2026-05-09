// src/components/ConfidenceBar.jsx
import { motion } from 'framer-motion';

export default function ConfidenceBar({ confidence }) {
  const pct = Math.round(confidence * 100);
  return (
    <div className="mt-5">
      <div className="flex justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Confidence</span>
        <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{pct}%</span>
      </div>
      <div className="confidence-track">
        <motion.div
          className="confidence-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: 'easeOut', delay: 0.25 }}
        />
      </div>
    </div>
  );
}
