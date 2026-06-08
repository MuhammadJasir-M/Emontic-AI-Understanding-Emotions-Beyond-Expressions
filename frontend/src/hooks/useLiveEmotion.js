// src/hooks/useLiveEmotion.js
// Custom hook for webcam stream management + live emotion prediction.
// Uses adaptive fire-when-ready capture loop for near real-time updates.

import { useState, useRef, useCallback, useEffect } from "react";
import api from "../utils/api";

// Minimum gap between captures to avoid overwhelming the backend
const MIN_CAPTURE_GAP_MS = 150;

export function useLiveEmotion() {
  const [status, setStatus] = useState("idle"); // idle | requesting | active | denied | error
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [allProbs, setAllProbs] = useState({});
  const [bbox, setBbox] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [latency, setLatency] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const activeRef = useRef(false);   // Whether the capture loop is active
  const timeoutRef = useRef(null);   // setTimeout ID for the next capture
  const mountedRef = useRef(true);
  const isPredictingRef = useRef(false);

  // ── Attach stream when video element is ready ───────────────────────
  useEffect(() => {
    let pollInterval;
    if (status === "active" && streamRef.current) {
      // The video element may be delayed in mounting due to AnimatePresence exit animations.
      // We poll briefly until videoRef.current is populated.
      let attempts = 0;
      const attachStream = () => {
        if (videoRef.current) {
          if (videoRef.current.srcObject !== streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
            videoRef.current.onloadedmetadata = () => {
              videoRef.current
                .play()
                .catch((e) => console.error("Play error:", e));
            };
          }
          clearInterval(pollInterval);
        } else if (attempts > 40) {
          // Give up after 2 seconds
          clearInterval(pollInterval);
        }
        attempts++;
      };

      pollInterval = setInterval(attachStream, 50);
      attachStream();
    }
    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [status]);

  // ── Start webcam stream ─────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          aspectRatio: { ideal: 1 },
          width: { ideal: 640 },
          height: { ideal: 640 },
        },
        audio: false,
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      streamRef.current = stream;
      setStatus("active");
    } catch (err) {
      console.error("Camera access denied:", err);
      if (mountedRef.current) {
        setStatus(err.name === "NotAllowedError" ? "denied" : "error");
      }
    }
  }, []);

  // ── Stop webcam stream ──────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    activeRef.current = false;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    isPredictingRef.current = false;
    setStatus("idle");
    setEmotion(null);
    setConfidence(0);
    setAllProbs({});
    setBbox(null);
    setImageSize(null);
    setIsPredicting(false);
  }, []);

  // ── Capture one frame and send to backend ────────────────────────────
  // Adaptive loop: fires next capture as soon as the previous one completes,
  // with a minimum gap of MIN_CAPTURE_GAP_MS to avoid overwhelming the server.
  const captureAndPredict = useCallback(async () => {
    if (!activeRef.current || !mountedRef.current) return;
    if (isPredictingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      // Video element may not be ready yet — retry shortly
      if (activeRef.current) {
        timeoutRef.current = setTimeout(captureAndPredict, 100);
      }
      return;
    }

    // Guard: video must be playing and have valid dimensions
    if (
      video.readyState < 2 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      if (activeRef.current) {
        timeoutRef.current = setTimeout(captureAndPredict, 100);
      }
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    // Convert frame to base64 JPEG (1.0 quality eliminates lossy compression
    // artifacts that would otherwise blur critical micro-expressions)
    const base64 = canvas.toDataURL("image/jpeg", 1.0);

    isPredictingRef.current = true;
    setIsPredicting(true);

    try {
      const { data } = await api.post("/live-predict", { image: base64 });

      if (mountedRef.current) {
        if (data.emotion) {
          setEmotion(data.emotion);
          setConfidence(data.confidence);
          setAllProbs(data.all_probs || {});
          setBbox(data.bbox);
          setImageSize(data.image_size);
          setLatency(data.latency_ms || 0);
        } else {
          // No face detected — clear old result
          setEmotion(null);
          setConfidence(0);
          setBbox(null);
        }
      }
    } catch (err) {
      console.error("Live prediction error:", err);
    } finally {
      if (mountedRef.current) {
        isPredictingRef.current = false;
        setIsPredicting(false);
      }
      // Schedule next capture immediately after this one completes
      if (activeRef.current && mountedRef.current) {
        timeoutRef.current = setTimeout(captureAndPredict, MIN_CAPTURE_GAP_MS);
      }
    }
  }, []); // No deps — only uses refs internally, so this is always stable

  // ── Start frame capture loop ─────────────────────────────────────────
  const startDetection = useCallback(() => {
    if (activeRef.current) return; // Already running
    activeRef.current = true;
    captureAndPredict(); // Fire immediately
  }, [captureAndPredict]);

  // ── Stop frame capture loop ──────────────────────────────────────────
  const stopDetection = useCallback(() => {
    activeRef.current = false;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    isPredictingRef.current = false;
    setIsPredicting(false);
    setEmotion(null);
    setConfidence(0);
    setBbox(null);
  }, []);

  // ── Cleanup on unmount ──────────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      activeRef.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (streamRef.current)
        streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    // Refs (attach to DOM elements)
    videoRef,
    canvasRef,

    // State
    status,
    emotion,
    confidence,
    allProbs,
    bbox,
    imageSize,
    latency,
    isPredicting,

    // Actions
    startCamera,
    stopCamera,
    startDetection,
    stopDetection,
  };
}
