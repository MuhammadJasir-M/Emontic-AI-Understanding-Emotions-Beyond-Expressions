// src/App.jsx
import { Toaster } from "react-hot-toast";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Brain, Shield, Clock } from "lucide-react";
import { useState } from "react";
import { useEmotionDetect } from "./hooks/useEmotionDetect";
import Navbar from "./components/Navbar";
import UploadZone from "./components/UploadZone";
import ResultCard from "./components/ResultCard";
import FaceOverlay from "./components/FaceOverlay";
import "./styles/globals.css";

const FEATURES = [
  {
    icon: Brain,
    label: "EfficientNetB0",
    desc: "ImageNet pretrained backbone",
  },
  {
    icon: Shield,
    label: "7 Emotions",
    desc: "Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise",
  },
  { icon: Clock, label: "Sub-second", desc: "TTA-enhanced inference" },
];

export default function App() {
  const { result, isLoading, detect, reset } = useEmotionDetect();
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleFile = (file) => {
    setUploadedFile(file);
    detect(file);
  };

  const handleReset = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    setUploadedFile(null);
    reset();
  };

  const previewUrl = uploadedFile ? URL.createObjectURL(uploadedFile) : null;

  return (
    <div className="min-h-screen relative">
      <Toaster position="top-right" toastOptions={{ duration: 4000 }} />
      <Navbar />

      {/* Ambient glow & Floating Particles */}
      <div className="aurora-bg">
        <div className="aurora-blob-1" />
        <div className="aurora-blob-2" />
        {[...Array(25)].map((_, i) => {
          const isCircle = i % 2 === 0;
          const isSquare = i % 4 === 1;
          const size = Math.random() * 30 + 10;
          return (
            <motion.div
              key={i}
              className="absolute pointer-events-none"
              style={{
                width: size,
                height: size,
                borderRadius: isCircle ? "50%" : isSquare ? "20%" : "0%",
                background:
                  i % 3 === 0
                    ? "rgba(124, 58, 237, 0.3)"
                    : i % 3 === 1
                      ? "rgba(6, 182, 212, 0.3)"
                      : "rgba(168, 85, 247, 0.3)",
                boxShadow:
                  i % 3 === 0
                    ? "0 0 20px 5px rgba(124, 58, 237, 0.2)"
                    : i % 3 === 1
                      ? "0 0 20px 5px rgba(6, 182, 212, 0.2)"
                      : "0 0 20px 5px rgba(168, 85, 247, 0.2)",
                filter: "blur(2px)",
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, Math.random() * 400 - 200, 0],
                x: [0, Math.random() * 400 - 200, 0],
                scale: [1, Math.random() * 0.5 + 0.8, 1],
                rotate: [0, Math.random() * 360, 0],
              }}
              transition={{
                duration: Math.random() * 10 + 8,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          );
        })}
      </div>

      <main className="relative z-10 w-full max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-12 2xl:px-20 pt-24 pb-32 grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 2xl:gap-24 items-start min-h-[85vh] 2xl:min-h-[80vh]">
        {/* Left Side: Hero */}
        <motion.div
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          className="lg:col-span-5 2xl:col-span-5 text-left h-fit"
        >
          <motion.div
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-8 shadow-lg"
            style={{
              background: "rgba(124,58,237,0.12)",
              border: "1px solid rgba(168,85,247,0.25)",
              boxShadow: "0 0 20px rgba(124,58,237,0.2)",
            }}
          >
            <Zap
              size={14}
              color="var(--accent-violet-light)"
              className="pulse-glow"
            />
            <span
              className="text-xs font-bold tracking-widest uppercase"
              style={{ color: "var(--accent-violet-light)" }}
            >
              Emontic AI
            </span>
          </motion.div>

          <h1
            className="font-extrabold leading-tight mb-6"
            style={{
              fontSize: "clamp(40px, 5vw, 64px)",
              background:
                "linear-gradient(135deg, #ffffff 10%, #a855f7 50%, #22d3ee 90%)",
              backgroundSize: "200% 200%",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: "-0.02em",
              animation: "gradient-shift 8s ease infinite",
            }}
          >
            Decode human emotion with
            <br />
            <span
              style={{
                fontFamily: "Orbitron, Inter, sans-serif",
                fontWeight: 700,
                fontStyle: "italic",
                letterSpacing: "0.05em",
                paddingRight: "0.1em",
                display: "inline-block",
                background:
                  "linear-gradient(180deg, #ffffff 55%, #dbeafe 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                textShadow:
                  "0 0 12px rgba(151, 176, 228, 0.6), 0 2px 20px rgba(50,100,255,0.4)",
              }}
            >
              EMONTIC
            </span>
            <span
              style={{
                fontFamily: "Orbitron, Inter, sans-serif",
                fontWeight: 700,
                fontStyle: "italic",
                letterSpacing: "0.05em",
                paddingRight: "0.1em",
                display: "inline-block",
                marginLeft: "12px",
                background:
                  "linear-gradient(90deg, #00f0ff 0%, #8a2be2 50%, #ff00ff 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                textShadow:
                  "-2px 0 10px rgba(0, 240, 255, 0.7), 0 0 18px rgba(138, 43, 226, 0.65), 2px 0 12px rgba(255, 0, 255, 0.7) `",
              }}
            >
              AI
            </span>
          </h1>

          <p
            className="text-lg 2xl:text-xl leading-relaxed max-w-lg mb-12"
            style={{ color: "var(--text-secondary)" }}
          >
            Upload a portrait to witness our neural network instantly analyze
            micro-expressions and classify facial emotions with high precision.
          </p>

          {/* Feature chips */}
          <div
            className="flex flex-col gap-4 max-w-lg"
            style={{ perspective: 1000 }}
          >
            {FEATURES.map(({ icon: Icon, label, desc }, i) => (
              <motion.div
                initial={{ opacity: 0, y: 20, rotateX: 15 }}
                animate={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{
                  duration: 0.8,
                  delay: 0.3 + i * 0.1,
                  ease: [0.16, 1, 0.3, 1],
                }}
                key={label}
                whileHover={{ scale: 1.02, rotateX: 2, rotateY: -2, z: 10 }}
                className="flex items-center gap-5 px-6 py-5 rounded-[20px] glass-card transition-all cursor-default"
              >
                <div
                  className="flex items-center justify-center w-14 h-14 rounded-[14px]"
                  style={{
                    background: "rgba(6, 182, 212, 0.1)",
                    border: "1px solid rgba(6, 182, 212, 0.2)",
                  }}
                >
                  <Icon size={24} style={{ color: "var(--accent-cyan)" }} />
                </div>
                <div className="text-left">
                  <p
                    className="text-base 2xl:text-lg font-bold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {label}
                  </p>
                  <p
                    className="text-sm 2xl:text-base mt-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {desc}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Right Side: Interactive */}
        <motion.div
          initial={{ opacity: 0, scale: 0.92, x: 40 }}
          animate={{ opacity: 1, scale: 1, x: 0 }}
          transition={{ duration: 1.2, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="lg:col-span-7 2xl:col-span-7 flex flex-col justify-center gap-6 relative min-h-[500px] lg:min-h-[600px] 2xl:min-h-[700px] w-full max-w-3xl mx-auto"
          style={{ perspective: 1200 }}
        >
          <div
            className="absolute -inset-4 bg-gradient-to-r from-violet-600/20 to-cyan-600/20 blur-3xl z-0 rounded-full opacity-50 pointer-events-none"
            style={{ zIndex: 0 }}
          />

          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="upload"
                initial={{ opacity: 0, rotateX: 10, y: 30 }}
                animate={{ opacity: 1, rotateX: 0, y: 0 }}
                exit={{ opacity: 0, rotateX: -10, y: -30, scale: 0.95 }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                className="glass-card p-10 sm:p-12 2xl:p-16 shadow-2xl relative overflow-hidden"
                whileHover={{ rotateX: 1, rotateY: -1, scale: 1.005 }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-cyan-500/5 pointer-events-none" />
                <div className="relative z-10">
                  <UploadZone
                    onFile={handleFile}
                    isLoading={isLoading}
                    onReset={handleReset}
                    hasResult={!!result}
                  />
                  {isLoading && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-center justify-center gap-3 mt-8 p-4 rounded-xl"
                      style={{ background: "rgba(124,58,237,0.1)" }}
                    >
                      <div className="spinner" />
                      <span
                        className="text-sm font-medium"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Analyzing expression...
                      </span>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, rotateX: 10, y: 30 }}
                animate={{ opacity: 1, rotateX: 0, y: 0 }}
                transition={{
                  duration: 0.8,
                  staggerChildren: 0.1,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className="flex flex-col gap-8 w-full"
              >
                {previewUrl && result.bbox && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                    whileHover={{ rotateX: 1, rotateY: -1, scale: 1.005 }}
                    className="flex justify-center glass-card p-8 overflow-hidden shadow-2xl"
                  >
                    <FaceOverlay
                      imageSrc={previewUrl}
                      bbox={result.bbox}
                      imageSize={result.image_size}
                      emotion={result.emotion}
                    />
                  </motion.div>
                )}

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  whileHover={{ rotateX: 1, rotateY: -1, scale: 1.005 }}
                  className="shadow-2xl rounded-[28px]"
                >
                  <ResultCard result={result} />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4, duration: 0.5 }}
                  className="flex justify-center mt-4"
                >
                  <button
                    onClick={handleReset}
                    className="btn-primary flex items-center gap-2 group px-8 py-3 text-sm 2xl:text-base"
                  >
                    <Zap
                      size={18}
                      className="group-hover:rotate-12 transition-transform"
                    />
                    Try Another Photo
                  </button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>

      {/* How it works / Description Section */}
      <section className="relative z-10 w-full max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-12 2xl:px-20 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          className="glass-card p-12 md:p-20 2xl:p-24 relative shadow-2xl transform-gpu"
        >
          <div className="absolute top-[-10%] right-[-5%] w-96 h-96 bg-violet-500/20 blur-[100px] rounded-full pointer-events-none" />
          <div className="absolute bottom-[-10%] left-[-5%] w-96 h-96 bg-cyan-500/20 blur-[100px] rounded-full pointer-events-none" />

          <div className="relative z-10 max-w-4xl 2xl:max-w-5xl mx-auto text-center">
            <h2
              className="text-3xl md:text-5xl 2xl:text-6xl font-extrabold mb-8 tracking-tight"
              style={{ color: "var(--text-primary)" }}
            >
              Understanding Emotions{" "}
              <span style={{ color: "var(--accent-violet-light)" }}>
                Beyond Expressions
              </span>
            </h2>
            <p
              className="text-base md:text-xl 2xl:text-2xl leading-relaxed mb-16"
              style={{ color: "var(--text-secondary)" }}
            >
              Emontic AI is built on a highly optimized deep learning pipeline
              designed to detect, align, and classify human emotion in
              real-time. By leveraging a robust <strong>EfficientNetB0</strong>{" "}
              backbone trained across diverse datasets like AffectNet and
              RAF-DB, the model understands nuanced micro-expressions that go
              beyond simple smiles or frowns.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 2xl:gap-12 text-left">
              <motion.div
                whileHover={{ y: -8 }}
                transition={{ ease: "easeOut", duration: 0.3 }}
                className="p-10 rounded-[32px] transition-colors"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <div
                  className="flex items-center justify-center w-14 h-14 2xl:w-16 2xl:h-16 rounded-xl mb-6"
                  style={{
                    background: "rgba(168, 85, 247, 0.1)",
                    border: "1px solid rgba(168, 85, 247, 0.2)",
                  }}
                >
                  <span className="text-2xl 2xl:text-3xl">👁️</span>
                </div>
                <h3
                  className="font-bold mb-4 text-lg 2xl:text-xl"
                  style={{ color: "var(--text-primary)" }}
                >
                  1. Face Detection
                </h3>
                <p
                  className="text-sm 2xl:text-base leading-relaxed"
                  style={{ color: "var(--text-muted)" }}
                >
                  Using advanced face detectors, the system rapidly scans the
                  image to locate human faces with high confidence, even in
                  complex lighting or varied angles.
                </p>
              </motion.div>

              <motion.div
                whileHover={{ y: -8 }}
                transition={{ ease: "easeOut", duration: 0.3 }}
                className="p-10 rounded-[32px] transition-colors"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <div
                  className="flex items-center justify-center w-14 h-14 2xl:w-16 2xl:h-16 rounded-xl mb-6"
                  style={{
                    background: "rgba(6, 182, 212, 0.1)",
                    border: "1px solid rgba(6, 182, 212, 0.2)",
                  }}
                >
                  <span className="text-2xl 2xl:text-3xl">📐</span>
                </div>
                <h3
                  className="font-bold mb-4 text-lg 2xl:text-xl"
                  style={{ color: "var(--text-primary)" }}
                >
                  2. Alignment
                </h3>
                <p
                  className="text-sm 2xl:text-base leading-relaxed"
                  style={{ color: "var(--text-muted)" }}
                >
                  MediaPipe processes the detected face, aligning the geometric
                  features to ensure the neural network analyzes a perfectly
                  normalized and centered crop.
                </p>
              </motion.div>

              <motion.div
                whileHover={{ y: -8 }}
                transition={{ ease: "easeOut", duration: 0.3 }}
                className="p-10 rounded-[32px] transition-colors"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <div
                  className="flex items-center justify-center w-14 h-14 2xl:w-16 2xl:h-16 rounded-xl mb-6"
                  style={{
                    background: "rgba(124, 58, 237, 0.1)",
                    border: "1px solid rgba(124, 58, 237, 0.2)",
                  }}
                >
                  <span className="text-2xl 2xl:text-3xl">🧠</span>
                </div>
                <h3
                  className="font-bold mb-4 text-lg 2xl:text-xl"
                  style={{ color: "var(--text-primary)" }}
                >
                  3. Classification
                </h3>
                <p
                  className="text-sm 2xl:text-base leading-relaxed"
                  style={{ color: "var(--text-muted)" }}
                >
                  Our fine-tuned classification model analyzes the facial
                  topography and outputs a confidence distribution across 7
                  distinct emotional states instantly.
                </p>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <p
        className="text-center pb-8 text-xs relative z-10"
        style={{ color: "var(--text-muted)" }}
      >
        Built by Emontic AI · EfficientNetB0 · RetinaFace · MediaPipe · FastAPI
      </p>
    </div>
  );
}
