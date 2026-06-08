// src/pages/HistoryPage.jsx
// Displays all unique person names that have prediction history.
// Clicking a name navigates to /history/:name for that person's records.

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { History, User, ArrowRight, Search, AlertCircle } from "lucide-react";
import api from "../utils/api";

export default function HistoryPage() {
  const [names, setNames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    fetchNames();
  }, []);

  const fetchNames = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { data } = await api.get("/history/names");
      setNames(data.names || []);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        (err?.code === "ERR_NETWORK"
          ? "Cannot connect to the server."
          : "Failed to load prediction history.");
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredNames = names.filter((name) =>
    name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <main className="relative z-10 w-full max-w-[1200px] mx-auto px-4 sm:px-6 lg:px-12 pt-28 pb-32 min-h-[85vh]">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-12"
      >
        <div
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{
            background: "rgba(124,58,237,0.12)",
            border: "1px solid rgba(168,85,247,0.25)",
          }}
        >
          <History size={14} color="var(--accent-violet-light)" />
          <span
            className="text-xs font-bold tracking-widest uppercase"
            style={{ color: "var(--accent-violet-light)" }}
          >
            Prediction History
          </span>
        </div>
        <h1
          className="text-3xl md:text-5xl font-extrabold mb-4"
          style={{
            background:
              "linear-gradient(135deg, #ffffff 10%, #a855f7 50%, #22d3ee 90%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Analyzed Persons
        </h1>
        <p
          className="text-base md:text-lg max-w-xl mx-auto"
          style={{ color: "var(--text-secondary)" }}
        >
          Browse all individuals whose emotions have been analyzed by Emontic AI.
        </p>
      </motion.div>

      {/* Search Bar */}
      {!isLoading && !error && names.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="max-w-md mx-auto mb-10"
        >
          <div className="relative">
            <Search
              size={18}
              className="absolute left-4 top-1/2 -translate-y-1/2"
              style={{ color: "var(--text-muted)" }}
            />
            <input
              type="text"
              placeholder="Search by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="name-input"
              style={{ paddingLeft: 44 }}
            />
          </div>
        </motion.div>
      )}

      {/* Content */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="skeleton-card" />
          ))}
        </div>
      ) : error ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4 py-20"
        >
          <AlertCircle size={48} style={{ color: "#f87171" }} />
          <p className="text-lg font-medium" style={{ color: "#f87171" }}>
            {error}
          </p>
          <button onClick={fetchNames} className="btn-primary mt-4">
            Try Again
          </button>
        </motion.div>
      ) : filteredNames.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center gap-4 py-20"
        >
          <div
            className="flex items-center justify-center w-24 h-24 rounded-full mb-4"
            style={{
              background: "rgba(124,58,237,0.1)",
              border: "1px solid rgba(168,85,247,0.2)",
            }}
          >
            <History size={40} style={{ color: "var(--text-muted)" }} />
          </div>
          <p
            className="text-lg font-semibold"
            style={{ color: "var(--text-secondary)" }}
          >
            {searchQuery
              ? `No results for "${searchQuery}"`
              : "No prediction history yet"}
          </p>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            {searchQuery
              ? "Try a different search term."
              : "Start by analyzing a photo on the Image Analysis page!"}
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {filteredNames.map((name, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: i * 0.06 }}
              whileHover={{ scale: 1.03, y: -4 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate(`/history/${encodeURIComponent(name)}`)}
              className="history-card group cursor-pointer"
            >
              <div className="flex items-center gap-4">
                <div
                  className="flex items-center justify-center w-14 h-14 rounded-full flex-shrink-0"
                  style={{
                    background: "rgba(168, 85, 247, 0.15)",
                    border: "1px solid rgba(168, 85, 247, 0.3)",
                  }}
                >
                  <User
                    size={24}
                    style={{ color: "var(--accent-violet-light)" }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className="text-lg font-bold truncate"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {name}
                  </p>
                  <p
                    className="text-xs mt-0.5"
                    style={{ color: "var(--text-muted)" }}
                  >
                    View prediction history →
                  </p>
                </div>
                <ArrowRight
                  size={20}
                  className="flex-shrink-0 transition-transform group-hover:translate-x-1"
                  style={{ color: "var(--text-muted)" }}
                />
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}
    </main>
  );
}
