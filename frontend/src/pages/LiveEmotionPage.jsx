// src/pages/LiveEmotionPage.jsx
// Real-time webcam emotion recognition page.
// Captures frames, sends to /api/live-predict, displays results live.

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Video,
  VideoOff,
  Camera,
  CameraOff,
  AlertCircle,
  Zap,
  RefreshCw,
} from "lucide-react";
import { useLiveEmotion } from "../hooks/useLiveEmotion";
import EmotionBadge from "../components/EmotionBadge";

export default function LiveEmotionPage() {
  const {
    videoRef,
    canvasRef,
    status,
    emotion,
    confidence,
    allProbs,
    bbox,
    imageSize,
    latency,
    isPredicting,
    startCamera,
    stopCamera,
    startDetection,
    stopDetection,
  } = useLiveEmotion();

  const [isDetecting, setIsDetecting] = useState(false);
  const overlayCanvasRef = useRef(null);

  // ── Draw bounding box overlay on the video ──────────────────────────
  useEffect(() => {
    if (!overlayCanvasRef.current || !videoRef.current) return;
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!bbox || !imageSize || !emotion) return;

    const scaleX = canvas.width / imageSize.width;
    const scaleY = canvas.height / imageSize.height;

    const rawX = bbox.x * scaleX;
    const rawY = bbox.y * scaleY;
    const rawW = bbox.w * scaleX;
    const rawH = bbox.h * scaleY;

    // Flip the bounding box horizontally to match the mirrored video
    const x = canvas.width - (rawX + rawW);
    const y = rawY;
    const w = rawW;
    const h = rawH;

    // Emotion colors
    const COLORS = {
      Happy: "#facc15",
      Sad: "#60a5fa",
      Angry: "#f87171",
      Neutral: "#a3e635",
      Disgust: "#c084fc",
      Fear: "#fb923c",
      Surprise: "#2dd4bf",
      Uncertain: "#94a3b8",
    };
    const color = COLORS[emotion] || COLORS.Uncertain;

    // Glow box
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
    const corner = 14;
    ctx.beginPath();
    ctx.moveTo(x, y + corner);
    ctx.lineTo(x, y);
    ctx.lineTo(x + corner, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + w - corner, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + corner);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y + h - corner);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + corner, y + h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + w - corner, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - corner);
    ctx.stroke();

    // Emotion label above box
    ctx.font = "bold 14px Rajdhani, sans-serif";
    ctx.fillStyle = color;
    ctx.shadowColor = "rgba(0,0,0,0.8)";
    ctx.shadowBlur = 4;
    const label = `${emotion} ${Math.round((confidence || 0) * 100)}%`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(x, y - 24, textWidth + 12, 22);
    ctx.fillStyle = color;
    ctx.shadowBlur = 0;
    ctx.fillText(label, x + 6, y - 8);
  }, [bbox, imageSize, emotion, confidence, videoRef]);

  const handleToggleDetection = useCallback(() => {
    if (isDetecting) {
      stopDetection();
      setIsDetecting(false);
    } else {
      startDetection();
      setIsDetecting(true);
    }
  }, [isDetecting, startDetection, stopDetection]);

  const handleStopCamera = useCallback(() => {
    setIsDetecting(false);
    stopCamera();
  }, [stopCamera]);

  // Probability bars for sidebar
  const probEntries = useMemo(() => {
    return Object.entries(allProbs).sort((a, b) => b[1] - a[1]);
  }, [allProbs]);

  const EMOTION_COLORS = {
    Happy: "var(--neon-happy)",
    Sad: "var(--neon-sad)",
    Angry: "var(--neon-angry)",
    Neutral: "var(--neon-neutral)",
    Disgust: "var(--neon-disgust)",
    Fear: "var(--neon-fear)",
    Surprise: "var(--neon-surprise)",
    Uncertain: "var(--neon-uncertain)",
  };

  return (
    <main className="relative z-10 w-full max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-12 pt-28 pb-32 min-h-[85vh]">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-10"
      >
        <div
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{
            background: "rgba(6, 182, 212, 0.12)",
            border: "1px solid rgba(6, 182, 212, 0.25)",
          }}
        >
          <Video size={14} color="var(--accent-cyan)" />
          <span
            className="text-xs font-bold tracking-widest uppercase"
            style={{ color: "var(--accent-cyan)" }}
          >
            Real-Time Analysis
          </span>
        </div>
        <h1
          className="text-3xl md:text-5xl font-extrabold mb-4"
          style={{
            background:
              "linear-gradient(135deg, #ffffff 10%, #22d3ee 50%, #a855f7 90%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Live Emotion Recognition
        </h1>
        <p
          className="text-base md:text-lg max-w-xl mx-auto"
          style={{ color: "var(--text-secondary)" }}
        >
          Use your webcam to detect and classify emotions in real time.
        </p>
      </motion.div>

      {/* Camera States */}
      <AnimatePresence mode="wait">
        {status === "idle" && (
          <motion.div
            key="idle"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex flex-col items-center gap-6 py-16"
          >
            <div
              className="flex items-center justify-center w-32 h-32 rounded-full"
              style={{
                background: "rgba(6, 182, 212, 0.1)",
                border: "2px solid rgba(6, 182, 212, 0.2)",
              }}
            >
              <Camera size={56} style={{ color: "var(--accent-cyan)" }} />
            </div>
            <p
              className="text-lg font-semibold"
              style={{ color: "var(--text-secondary)" }}
            >
              Ready to start live emotion detection
            </p>
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              Your webcam will be used to capture and analyze facial
              expressions.
            </p>
            <button
              onClick={startCamera}
              className="btn-primary flex items-center gap-2 px-8 py-3"
            >
              <Video size={18} />
              Start Camera
            </button>
          </motion.div>
        )}

        {status === "requesting" && (
          <motion.div
            key="requesting"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-4 py-20"
          >
            <div className="spinner" style={{ width: 40, height: 40 }} />
            <p
              className="text-lg font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              Requesting camera access...
            </p>
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              Please allow camera permission in your browser.
            </p>
          </motion.div>
        )}

        {status === "denied" && (
          <motion.div
            key="denied"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-4 py-20"
          >
            <AlertCircle size={56} style={{ color: "#f87171" }} />
            <p className="text-lg font-bold" style={{ color: "#f87171" }}>
              Camera Access Denied
            </p>
            <p
              className="text-sm max-w-md text-center"
              style={{ color: "var(--text-muted)" }}
            >
              Emontic AI needs camera permission for live emotion detection.
              Please allow camera access in your browser settings and try again.
            </p>
            <button
              onClick={startCamera}
              className="btn-primary flex items-center gap-2 mt-4"
            >
              <RefreshCw size={16} />
              Try Again
            </button>
          </motion.div>
        )}

        {status === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-4 py-20"
          >
            <AlertCircle size={56} style={{ color: "#f87171" }} />
            <p className="text-lg font-bold" style={{ color: "#f87171" }}>
              Camera Error
            </p>
            <p
              className="text-sm max-w-md text-center"
              style={{ color: "var(--text-muted)" }}
            >
              Could not access your camera. Make sure no other application is
              using it and try again.
            </p>
            <button
              onClick={startCamera}
              className="btn-primary flex items-center gap-2 mt-4"
            >
              <RefreshCw size={16} />
              Retry
            </button>
          </motion.div>
        )}

        {status === "active" && (
          <motion.div
            key="active"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col lg:flex-row justify-center items-start gap-8 max-w-5xl mx-auto relative"
            style={{ perspective: 1200 }}
          >
            {/* Background Glow */}
            <div
              className="absolute -inset-4 bg-gradient-to-r from-violet-600/20 to-cyan-600/20 blur-3xl z-0 rounded-full opacity-50 pointer-events-none"
            />

            {/* Video Feed */}
            <motion.div 
              className="w-full lg:w-3/5 max-w-[600px] relative z-10"
              whileHover={{ rotateX: 1, rotateY: -1, scale: 1.005 }}
              transition={{ duration: 0.4 }}
            >
              <div className="live-container glass-card overflow-hidden p-6 sm:p-8 shadow-2xl relative">
                <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-cyan-500/5 pointer-events-none" />
                <div className="relative w-full rounded-xl overflow-hidden bg-black shadow-inner">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-auto rounded-lg"
                    style={{
                      transform: "scaleX(-1)",
                      background: "#000",
                    }}
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  <canvas
                    ref={overlayCanvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  />

                  {/* Live indicator */}
                  {isDetecting && (
                    <div
                      className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1 rounded-full"
                      style={{
                        background: "rgba(0,0,0,0.7)",
                        border: "1px solid rgba(239, 68, 68, 0.5)",
                      }}
                    >
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          background: "#ef4444",
                          animation: "pulse-glow 1.5s infinite",
                        }}
                      />
                      <span
                        className="text-xs font-bold uppercase"
                        style={{ color: "#ef4444" }}
                      >
                        Live
                      </span>
                    </div>
                  )}

                  {/* Processing indicator */}
                  {isPredicting && (
                    <div
                      className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1 rounded-full"
                      style={{
                        background: "rgba(0,0,0,0.7)",
                        border: "1px solid rgba(168, 85, 247, 0.5)",
                      }}
                    >
                      <div
                        className="spinner"
                        style={{ width: 12, height: 12, borderWidth: 1.5 }}
                      />
                      <span
                        className="text-xs"
                        style={{ color: "var(--accent-violet-light)" }}
                      >
                        Analyzing
                      </span>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="flex flex-wrap items-center justify-center gap-4 mt-6">
                  <button
                    onClick={handleToggleDetection}
                    className="btn-primary flex items-center gap-2 px-6 py-2.5"
                    style={{
                      background: isDetecting
                        ? "linear-gradient(135deg, #ef4444, #dc2626)"
                        : undefined,
                    }}
                  >
                    {isDetecting ? (
                      <>
                        <CameraOff size={16} />
                        Stop Detection
                      </>
                    ) : (
                      <>
                        <Zap size={16} />
                        Start Detection
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleStopCamera}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
                    style={{
                      background: "rgba(255,255,255,0.05)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <VideoOff size={16} />
                    Stop Camera
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Sidebar — Live Results */}
            <motion.div 
              className="w-full lg:w-2/5 max-w-[400px] flex flex-col gap-6 relative z-10"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              {/* Current Emotion */}
              <motion.div
                className="glass-card p-6"
                animate={{
                  borderColor: emotion
                    ? "rgba(168, 85, 247, 0.3)"
                    : "rgba(255,255,255,0.08)",
                }}
              >
                <p
                  className="text-xs font-semibold uppercase tracking-widest mb-4"
                  style={{ color: "var(--text-muted)" }}
                >
                  Detected Emotion
                </p>

                {emotion ? (
                  <motion.div
                    key={emotion}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  >
                    <EmotionBadge emotion={emotion} />
                    <div className="mt-4">
                      <div className="flex justify-between mb-1.5">
                        <span
                          className="text-xs font-medium"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Confidence
                        </span>
                        <span
                          className="text-sm font-bold"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {Math.round((confidence || 0) * 100)}%
                        </span>
                      </div>
                      <div className="confidence-track">
                        <motion.div
                          className="confidence-fill"
                          animate={{
                            width: `${Math.round((confidence || 0) * 100)}%`,
                          }}
                          transition={{ duration: 0.4 }}
                        />
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <p
                    className="text-sm py-4 text-center"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {isDetecting
                      ? "Looking for a face..."
                      : "Start detection to begin"}
                  </p>
                )}

                {latency > 0 && (
                  <p
                    className="text-xs text-center mt-4"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Inference: {latency}ms
                  </p>
                )}
              </motion.div>

              {/* Probability Distribution */}
              {probEntries.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-6"
                >
                  <p
                    className="text-xs font-semibold uppercase tracking-widest mb-4"
                    style={{ color: "var(--text-muted)" }}
                  >
                    All Probabilities
                  </p>
                  {probEntries.map(([label, prob]) => (
                    <div key={label} className="mb-2.5">
                      <div className="flex justify-between mb-1">
                        <span
                          className="text-xs"
                          style={{
                            color:
                              EMOTION_COLORS[label] || "var(--text-secondary)",
                          }}
                        >
                          {label}
                        </span>
                        <span
                          className="text-xs font-medium"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {Math.round(prob * 100)}%
                        </span>
                      </div>
                      <div className="prob-track">
                        <motion.div
                          className="prob-fill"
                          animate={{ width: `${Math.round(prob * 100)}%` }}
                          transition={{ duration: 0.4 }}
                          style={{
                            background:
                              EMOTION_COLORS[label] || "var(--accent-violet)",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </main>
  );
}
