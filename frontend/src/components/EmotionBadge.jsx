// src/components/EmotionBadge.jsx
import { motion } from 'framer-motion';

const CFG = {
  Happy:     { color: 'var(--neon-happy)',     glow: 'var(--neon-happy-glow)',     emoji: '😄' },
  Sad:       { color: 'var(--neon-sad)',       glow: 'var(--neon-sad-glow)',       emoji: '😢' },
  Angry:     { color: 'var(--neon-angry)',     glow: 'var(--neon-angry-glow)',     emoji: '😠' },
  Neutral:   { color: 'var(--neon-neutral)',   glow: 'var(--neon-neutral-glow)',   emoji: '😐' },
  Disgust:   { color: 'var(--neon-disgust)',   glow: 'var(--neon-disgust-glow)',   emoji: '🤢' },
  Fear:      { color: 'var(--neon-fear)',      glow: 'var(--neon-fear-glow)',      emoji: '😨' },
  Surprise:  { color: 'var(--neon-surprise)',  glow: 'var(--neon-surprise-glow)',  emoji: '😲' },
  Uncertain: { color: 'var(--neon-uncertain)', glow: 'var(--neon-uncertain-glow)', emoji: '🤔' },
};

export default function EmotionBadge({ emotion }) {
  const c = CFG[emotion] || CFG.Uncertain;
  return (
    <motion.div
      initial={{ scale: 0.6, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="emotion-badge"
      style={{ border: `1.5px solid ${c.color}`, background: c.glow, boxShadow: `0 0 28px ${c.glow}`, color: c.color }}
    >
      <span style={{ fontSize: 30 }}>{c.emoji}</span>
      {emotion}
    </motion.div>
  );
}
