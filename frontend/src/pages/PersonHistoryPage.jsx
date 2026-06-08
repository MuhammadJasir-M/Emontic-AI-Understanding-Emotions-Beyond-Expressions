// src/pages/PersonHistoryPage.jsx
// Shows all prediction records for a specific person.
// Reads :name from URL params and fetches from /api/history/:name.

import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  User,
  Calendar,
  AlertCircle,
  Clock,
  Trash2,
} from "lucide-react";
import toast from "react-hot-toast";
import api from "../utils/api";
import { API_URL } from "../utils/api";
import EmotionBadge from "../components/EmotionBadge";
import ConfidenceBar from "../components/ConfidenceBar";

// Derive the backend base URL from API_URL (strip /api suffix)
const BACKEND_URL = API_URL.replace(/\/api\/?$/, "");

export default function PersonHistoryPage() {
  const { name } = useParams();
  const navigate = useNavigate();
  const [records, setRecords] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, [name]);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { data } = await api.get(`/history/${encodeURIComponent(name)}`);
      setRecords(data.records || []);
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

  const handleDelete = async (recordId) => {
    if (!window.confirm("Are you sure you want to delete this prediction?")) return;

    const toastId = toast.loading("Deleting...", {
      style: {
        background: "#16161f",
        color: "#f1f5f9",
        border: "1px solid rgba(248, 113, 113, 0.3)",
        borderRadius: "12px",
        fontSize: "14px",
      },
    });

    try {
      await api.delete(`/history/record/${recordId}`);
      setRecords((prev) => prev.filter((r) => r.id !== recordId));
      toast.success("Prediction deleted.", {
        id: toastId,
        duration: 3000,
        style: {
          background: "#16161f",
          color: "#a3e635",
          border: "1px solid rgba(163, 230, 53, 0.3)",
          borderRadius: "12px",
          fontSize: "14px",
        },
      });
    } catch (err) {
      const msg = err?.response?.data?.detail || "Failed to delete prediction.";
      toast.error(msg, {
        id: toastId,
        duration: 4000,
        style: {
          background: "#16161f",
          color: "#f87171",
          border: "1px solid rgba(248, 113, 113, 0.3)",
          borderRadius: "12px",
          fontSize: "14px",
        },
      });
    }
  };

  const formatDate = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoString;
    }
  };

  return (
    <main className="relative z-10 w-full max-w-[1200px] mx-auto px-4 sm:px-6 lg:px-12 pt-28 pb-32 min-h-[85vh]">
      {/* Back button + Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="mb-10"
      >
        <button
          onClick={() => navigate("/history")}
          className="flex items-center gap-2 mb-6 text-sm font-medium transition-colors hover:opacity-80"
          style={{ color: "var(--accent-violet-light)" }}
        >
          <ArrowLeft size={16} />
          Back to History
        </button>

        <div className="flex items-center gap-4 mb-4">
          <div
            className="flex items-center justify-center w-16 h-16 rounded-full flex-shrink-0"
            style={{
              background: "rgba(168, 85, 247, 0.15)",
              border: "1px solid rgba(168, 85, 247, 0.3)",
            }}
          >
            <User size={28} style={{ color: "var(--accent-violet-light)" }} />
          </div>
          <div>
            <h1
              className="text-3xl md:text-4xl font-extrabold"
              style={{
                background:
                  "linear-gradient(135deg, #ffffff 10%, #a855f7 50%, #22d3ee 90%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              {decodeURIComponent(name)}
            </h1>
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
              {isLoading
                ? "Loading..."
                : `${records.length} prediction${records.length !== 1 ? "s" : ""} found`}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Content */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton-card" style={{ height: 280 }} />
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
          <button onClick={fetchHistory} className="btn-primary mt-4">
            Try Again
          </button>
        </motion.div>
      ) : records.length === 0 ? (
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
            <Clock size={40} style={{ color: "var(--text-muted)" }} />
          </div>
          <p
            className="text-lg font-semibold"
            style={{ color: "var(--text-secondary)" }}
          >
            No prediction history found for this person.
          </p>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Upload a photo of {decodeURIComponent(name)} to start building their
            emotion profile.
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
        <AnimatePresence>
          {records.map((record, i) => (
            <motion.div
              key={record.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: -10 }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="history-record glass-card overflow-hidden relative group"
            >
              {/* Delete button */}
              <motion.button
                onClick={() => handleDelete(record.id)}
                className="absolute top-3 right-3 z-10 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: "50%",
                  background: "rgba(248, 113, 113, 0.15)",
                  border: "1px solid rgba(248, 113, 113, 0.3)",
                  color: "#f87171",
                  cursor: "pointer",
                }}
                whileHover={{ scale: 1.1, background: "rgba(248, 113, 113, 0.3)" }}
                whileTap={{ scale: 0.9 }}
                title="Delete this prediction"
              >
                <Trash2 size={14} />
              </motion.button>

              {/* Image */}
              {record.image_path && (
                <div className="relative w-full" style={{ maxHeight: 220, overflow: "hidden" }}>
                  <img
                    src={`${BACKEND_URL}/uploads/${record.image_path}`}
                    alt={`${record.person_name} - ${record.predicted_emotion}`}
                    className="w-full object-cover"
                    style={{
                      maxHeight: 220,
                      borderRadius: "var(--radius-lg) var(--radius-lg) 0 0",
                    }}
                    onError={(e) => {
                      e.target.style.display = "none";
                    }}
                  />
                </div>
              )}

              {/* Details */}
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <EmotionBadge emotion={record.predicted_emotion} />
                </div>

                <ConfidenceBar confidence={record.confidence || 0} />

                <div
                  className="flex items-center gap-2 mt-4 text-xs"
                  style={{ color: "var(--text-muted)" }}
                >
                  <Calendar size={12} />
                  {formatDate(record.created_at)}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        </motion.div>
      )}
    </main>
  );
}
