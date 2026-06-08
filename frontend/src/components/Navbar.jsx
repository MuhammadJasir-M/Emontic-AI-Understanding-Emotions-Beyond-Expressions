// src/components/Navbar.jsx
// Glassmorphism top navigation with route links and mobile menu.

import { useState } from "react";
import { NavLink } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Github, Camera, History, Video, Menu, X } from "lucide-react";

const NAV_LINKS = [
  { to: "/", label: "Image Analysis", icon: Camera },
  { to: "/history", label: "History", icon: History },
  { to: "/live", label: "Live Emotion", icon: Video },
];

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <motion.nav
      className="navbar"
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div
        className="flex items-center justify-between w-full max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-12 2xl:px-20 py-3"
        style={{ gap: 16 }}
      >
        {/* Brand */}
        <div
          className="flex items-center gap-3"
          style={{ flexShrink: 0, minWidth: "fit-content" }}
        >
          <div
            className="flex items-center justify-center"
            style={{ width: 45, height: 45, flex: "0 0 40px" }}
          >
            <img
              src="/Emontic%20AI.png"
              alt="Emontic AI"
              style={{ width: 45, height: 45 }}
            />
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "nowrap",
            }}
          >
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                whiteSpace: "nowrap",
                flexShrink: 0,
                minWidth: "max-content",
              }}
            >
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
                    "0 0 12px rgba(100,150,255,0.6), 0 2px 20px rgba(50,100,255,0.4)",
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
                  marginRight: 4,
                  background:
                    "linear-gradient(90deg, #00f0ff 0%, #8a2be2 50%, #ff00ff 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  textShadow:
                    "0 0 12px rgba(0, 240, 255, 0.6), 0 0 24px rgba(255, 0, 255, 0.6)",
                }}
              >
                AI
              </span>
            </div>

            <span
              className="text-xs font-bold px-2.5 py-0.5 rounded-full hidden sm:inline-flex"
              style={{
                background: "linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(34, 211, 238, 0.15) 100%)",
                color: "var(--text-primary)",
                border: "1px solid rgba(34, 211, 238, 0.3)",
                flexShrink: 0,
                alignItems: "center",
                justifyContent: "center",
                textShadow: "0 0 10px rgba(34, 211, 238, 0.5)",
              }}
            >
              v2.5 ArcFace SOTA
            </span>
          </div>
        </div>

        {/* Desktop Nav Links */}
        <div className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`
              }
            >
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </div>

        {/* Right side: GitHub + Mobile toggle */}
        <div className="flex items-center gap-3">
          <motion.a
            href="https://github.com/MuhammadJasir-M/Emontic-AI-Understanding-Emotions-Beyond-Expressions.git"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-all duration-200 relative overflow-hidden"
            style={{
              borderRadius: "9999px",
              background:
                "linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(34, 211, 238, 0.15) 100%)",
              backdropFilter: "blur(12px)",
              color: "white",
              boxShadow:
                "0 8px 32px rgba(124, 58, 237, 0.2), inset 0 1px 0 rgba(168, 85, 247, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.15)",
              border: "1px solid rgba(168, 85, 247, 0.4)",
              position: "relative",
            }}
            whileHover={{
              scale: 1.05,
              background:
                "linear-gradient(135deg, rgba(124, 58, 237, 0.3) 0%, rgba(34, 211, 238, 0.25) 100%)",
              boxShadow:
                "0 12px 40px rgba(124, 58, 237, 0.3), inset 0 1px 0 rgba(168, 85, 247, 0.4), inset 0 -1px 0 rgba(0, 0, 0, 0.2)",
            }}
            whileTap={{ scale: 0.98 }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "50%",
                background:
                  "linear-gradient(180deg, rgba(168, 85, 247, 0.2) 0%, transparent 100%)",
                pointerEvents: "none",
                borderRadius: "9999px",
              }}
            />
            <div
              style={{
                position: "relative",
                zIndex: 1,
                display: "flex",
                alignItems: "center",
                gap: "0.375rem",
              }}
            >
              <Github size={14} />
              <span className="hidden sm:inline">Source</span>
            </div>
          </motion.a>

          {/* Mobile hamburger */}
          <button
            className="md:hidden flex items-center justify-center w-9 h-9 rounded-lg transition-colors"
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "var(--text-secondary)",
            }}
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="md:hidden overflow-hidden"
            style={{
              borderTop: "1px solid var(--border-subtle)",
              background: "rgba(10, 10, 15, 0.95)",
            }}
          >
            <div className="flex flex-col py-2 px-4">
              {NAV_LINKS.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === "/"}
                  className={({ isActive }) =>
                    `nav-link-mobile ${isActive ? "nav-link-mobile-active" : ""}`
                  }
                  onClick={() => setMobileOpen(false)}
                >
                  <Icon size={16} />
                  {label}
                </NavLink>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
