// src/pages/HomePage.jsx
// Image emotion analysis page — extracted from original App.jsx.
// Adds a required "Person Name" input before upload.

import { motion, AnimatePresence } from "framer-motion";
import { Zap, Brain, Shield, Clock, User } from "lucide-react";
import { useState } from "react";
import { useEmotionDetect } from "../hooks/useEmotionDetect";
import UploadZone from "../components/UploadZone";
import ResultCard from "../components/ResultCard";
import FaceOverlay from "../components/FaceOverlay";

const FEATURES = [
  {
    icon: Brain,
    label: "EfficientNetV2S",
    desc: "ImageNet pretrained backbone with self-attention",
  },
  {
    icon: Shield,
    label: "7 Emotions",
    desc: "Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise",
  },
  { icon: Clock, label: "Sub-second", desc: "TTA-enhanced inference" },
];

export default function AnalyzePage() {
  const { result, isLoading, detect, reset } = useEmotionDetect();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [personName, setPersonName] = useState("");
  const [nameError, setNameError] = useState("");
  const [fileError, setFileError] = useState("");

  const handleFile = (file) => {
    setUploadedFile(file);
    setFileError("");
  };

  const handleAnalyze = () => {
    const trimmed = personName.trim();
    if (!trimmed) {
      setNameError("Please enter a name before analyzing.");
      return;
    }
    if (!uploadedFile) {
      setFileError("Please upload a portrait image first.");
      return;
    }
    setNameError("");
    setFileError("");
    detect(uploadedFile, trimmed);
  };

  const handleReset = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    setUploadedFile(null);
    setPersonName("");
    setNameError("");
    setFileError("");
    reset();
  };

  const previewUrl = uploadedFile ? URL.createObjectURL(uploadedFile) : null;

  return (
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
              background: "linear-gradient(180deg, #ffffff 55%, #dbeafe 100%)",
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
                "0 0 12px rgba(6, 137, 253, 0.6), 0 0 24px rgba(135, 11, 137, 0.52)",
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
                {/* Person Name Input */}
                <div className="mb-8">
                  <label
                    htmlFor="person-name"
                    className="flex items-center gap-2 text-sm font-semibold uppercase tracking-widest mb-3"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    <User size={16} />
                    Who is this person?
                  </label>
                  <input
                    id="person-name"
                    type="text"
                    placeholder="Enter person's name..."
                    value={personName}
                    onChange={(e) => {
                      setPersonName(e.target.value);
                      if (e.target.value.trim()) setNameError("");
                    }}
                    disabled={isLoading}
                    className="name-input"
                    autoComplete="off"
                  />
                  {nameError && (
                    <motion.p
                      initial={{ opacity: 0, y: -5 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-xs mt-2 font-medium"
                      style={{ color: "#f87171" }}
                    >
                      {nameError}
                    </motion.p>
                  )}
                </div>

                <UploadZone
                  onFile={handleFile}
                  isLoading={isLoading}
                  onReset={handleReset}
                  hasResult={!!result}
                />
                {fileError && (
                  <motion.p
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-xs mt-4 text-center font-medium"
                    style={{ color: "#f87171" }}
                  >
                    {fileError}
                  </motion.p>
                )}

                <div className="mt-8 flex justify-center">
                  <button
                    className="btn-primary w-full max-w-sm py-4 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleAnalyze}
                    disabled={isLoading || !uploadedFile}
                  >
                    {isLoading ? "Analyzing..." : "Analyze Emotion"}
                  </button>
                </div>

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
              {/* Person name badge */}
              {result.person_name && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-2 justify-center"
                >
                  <span
                    className="inline-flex items-center gap-2 px-5 py-2 rounded-full text-sm font-bold"
                    style={{
                      background: "rgba(124,58,237,0.15)",
                      border: "1px solid rgba(168,85,247,0.3)",
                      color: "var(--accent-violet-light)",
                    }}
                  >
                    <User size={14} />
                    {result.person_name}
                  </span>
                </motion.div>
              )}

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
  );
}
