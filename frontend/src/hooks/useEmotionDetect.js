// src/hooks/useEmotionDetect.js
// Custom hook that manages the upload → predict → result lifecycle.

import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import api from '../utils/api';

const TOAST_STYLE = {
  background: '#16161f',
  color: '#f1f5f9',
  border: '1px solid rgba(168, 85, 247, 0.3)',
  borderRadius: '12px',
  fontSize: '14px',
};

const ERROR_TOAST_STYLE = {
  ...TOAST_STYLE,
  color: '#f87171',
  border: '1px solid rgba(248, 113, 113, 0.3)',
};

export function useEmotionDetect() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const detect = useCallback(async (file) => {
    setResult(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    const toastId = toast.loading('Analyzing emotion...', { style: TOAST_STYLE });

    try {
      const { data } = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setResult(data);

      const emoji =
        data.emotion === 'Happy' ? '😄' :
        data.emotion === 'Sad' ? '😢' :
        data.emotion === 'Angry' ? '😠' :
        data.emotion === 'Neutral' ? '😐' : '🤔';

      toast.success(
        `${emoji} Detected: ${data.emotion} (${Math.round(data.confidence * 100)}%)`,
        {
          id: toastId,
          duration: 4000,
          style: TOAST_STYLE,
        }
      );
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        (err?.code === 'ERR_NETWORK'
          ? 'Cannot connect to the server. Is the backend running?'
          : 'Something went wrong. Please try again.');

      toast.error(message, {
        id: toastId,
        duration: 5000,
        style: ERROR_TOAST_STYLE,
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
  }, []);

  return { result, isLoading, detect, reset };
}
