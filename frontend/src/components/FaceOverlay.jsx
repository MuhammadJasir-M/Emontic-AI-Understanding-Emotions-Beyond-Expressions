// src/components/FaceOverlay.jsx
// Draws a neon bounding box over the uploaded image where the face was detected.

import { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function FaceOverlay({ imageSrc, bbox, imageSize, emotion }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [loaded, setLoaded] = useState(false);

  const COLORS = {
    Happy: '#facc15', Sad: '#60a5fa', Angry: '#f87171',
    Neutral: '#a3e635', Uncertain: '#94a3b8',
  };
  const color = COLORS[emotion] || COLORS.Uncertain;

  useEffect(() => {
    if (!loaded || !canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = container.querySelector('img');
    if (!img) return;

    const displayW = img.clientWidth;
    const displayH = img.clientHeight;
    canvas.width = displayW;
    canvas.height = displayH;

    const scaleX = displayW / imageSize.width;
    const scaleY = displayH / imageSize.height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, displayW, displayH);

    const x = bbox.x * scaleX;
    const y = bbox.y * scaleY;
    const w = bbox.w * scaleX;
    const h = bbox.h * scaleY;

    // Glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 16;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(x, y, w, h);

    // Corner accents
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
    ctx.lineWidth = 3;
    const corner = 12;
    // Top-left
    ctx.beginPath(); ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x, y + h - corner); ctx.lineTo(x, y + h); ctx.lineTo(x + corner, y + h); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x + w - corner, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - corner); ctx.stroke();
  }, [loaded, bbox, imageSize, color]);

  if (!imageSrc || !bbox || !imageSize) return null;

  return (
    <motion.div
      ref={containerRef}
      className="face-overlay-container"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      style={{ margin: '0 auto', maxWidth: '100%' }}
    >
      <img src={imageSrc} alt="Face detection result" onLoad={() => setLoaded(true)} style={{ maxHeight: 320, maxWidth: '100%', objectFit: 'contain', borderRadius: 'var(--radius-md)' }} />
      <canvas ref={canvasRef} />
    </motion.div>
  );
}
