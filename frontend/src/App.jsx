// src/App.jsx
// Root layout component with routing.
// Preserves the aurora background, toaster, and navbar across all pages.

import { Toaster } from "react-hot-toast";
import { Routes, Route } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "./components/Navbar";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import PersonHistoryPage from "./pages/PersonHistoryPage";
import LiveEmotionPage from "./pages/LiveEmotionPage";
import "./styles/globals.css";

export default function App() {
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

      {/* Page Routes */}
      <Routes>
        <Route path="/" element={<AnalyzePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:name" element={<PersonHistoryPage />} />
        <Route path="/live" element={<LiveEmotionPage />} />
      </Routes>

      {/* Footer */}
      <p
        className="text-center pb-8 text-xs relative z-10"
        style={{ color: "var(--text-muted)" }}
      >
        Built by Muhammad Jasir M · EfficientNetV2S · Haar Cascades · FastAPI
      </p>
    </div>
  );
}
