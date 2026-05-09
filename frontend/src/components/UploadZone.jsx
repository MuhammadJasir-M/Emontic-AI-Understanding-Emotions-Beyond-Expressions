// src/components/UploadZone.jsx
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, ImageIcon, RotateCcw } from 'lucide-react';
import { useState, useCallback } from 'react';

const MAX_SIZE = 5 * 1024 * 1024;
const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/webp': [] };

export default function UploadZone({ onFile, isLoading, onReset, hasResult }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    const file = accepted[0];
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    onFile(file);
  }, [onFile, preview]);

  const handleReset = useCallback((e) => {
    e.stopPropagation();
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    onReset?.();
  }, [preview, onReset]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: MAX_SIZE, maxFiles: 1, disabled: isLoading,
  });

  const zoneClasses = [
    'upload-zone',
    isDragActive && 'drag-active',
    preview && 'has-preview',
    isLoading && 'is-loading',
  ].filter(Boolean).join(' ');

  return (
    <div className="relative">
      <motion.div
        {...getRootProps()}
        className={zoneClasses}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <input {...getInputProps()} id="upload-input" />
        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div key="preview" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center">
              <img src={preview} alt="Uploaded preview" className="rounded-xl shadow-2xl" style={{ maxHeight: 380, maxWidth: '100%', objectFit: 'contain' }} />
              {!isLoading && <p className="mt-4 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Click or drop another image to replace</p>}
            </motion.div>
          ) : (
            <motion.div key="placeholder" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center py-6">
              <div className="flex items-center justify-center mb-6" style={{ width: 88, height: 88, borderRadius: '50%', background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.1))', border: '1px solid rgba(168,85,247,0.3)', boxShadow: isDragActive ? '0 0 40px rgba(168,85,247,0.4)' : 'inset 0 0 20px rgba(124,58,237,0.1)' }}>
                {isDragActive ? <Upload size={36} color="var(--accent-violet-light)" /> : <ImageIcon size={36} color="var(--text-secondary)" />}
              </div>
              <p className="text-xl font-bold mb-2 tracking-tight" style={{ color: 'var(--text-primary)' }}>{isDragActive ? 'Drop it right here' : 'Upload a portrait'}</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Drag and drop or click to browse</p>
              <div className="flex gap-3 mt-8">
                {['😄', '😢', '😠', '😐', '😲'].map((e) => <span key={e} className="text-2xl" style={{ opacity: 0.3, filter: 'grayscale(50%)' }}>{e}</span>)}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
      {preview && !isLoading && hasResult && (
        <motion.button onClick={handleReset} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} className="absolute -top-3 -right-3 flex items-center justify-center" style={{ width: 36, height: 36, borderRadius: '50%', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', cursor: 'pointer', zIndex: 10, color: 'var(--text-secondary)' }} title="Clear and upload new image">
          <RotateCcw size={14} />
        </motion.button>
      )}
    </div>
  );
}
