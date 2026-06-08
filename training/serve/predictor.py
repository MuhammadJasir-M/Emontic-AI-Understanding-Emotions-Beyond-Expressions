# training/serve/predictor.py
import os
import json
import numpy as np
import keras

from typing import Optional
from training.core.config import get_input_shape, get_calibration_json
from training.eval.calibrate import _softmax_with_temperature

# Import custom architectures to prevent load_model crashes
from training.core.model import SpatialPositionalEmbedding, AttentionPooling
from training.core.losses import MacroF1, SparseFocalLoss

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


class Predictor:
    def __init__(self, arch: str, backend: str = "keras",
                 checkpoint: Optional[str] = None):
        self.arch        = arch
        self.backend     = backend.lower()
        self.input_shape = get_input_shape(arch) 
        self.temperature = 1.0
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: str):
        if self.backend == "keras":
            # Safely inject the architectural blueprints during initialization
            self.model = keras.models.load_model(
                checkpoint, 
                custom_objects={
                    "SpatialPositionalEmbedding": SpatialPositionalEmbedding,
                    "AttentionPooling": AttentionPooling,
                    "MacroF1": MacroF1,
                    "SparseFocalLoss": SparseFocalLoss
                },
                compile=False
            )
        elif self.backend == "onnx":
            if ort is None:
                raise RuntimeError("onnxruntime not installed in this execution context")
            sess_options  = ort.SessionOptions()
            self.session  = ort.InferenceSession(checkpoint, sess_options)
        else:
            raise ValueError("Unsupported backend framework")

        cal_path = get_calibration_json(self.arch)
        if os.path.isfile(cal_path):
            with open(cal_path, "r") as f:
                data = json.load(f)
            self.temperature = float(data.get("temperature", 1.0))

    def _preprocess(self, image_np: np.ndarray) -> np.ndarray:
        """
        Accepts HxWx3 uint8 or float32 image matrix profiles.
        Resizes to design dimensions via BICUBIC operations to preserve micro-expressions.
        Keeps baseline pixels bounded within [0.0, 255.0].
        """
        img = image_np.astype("float32")
        
        target_h, target_w = self.input_shape[:2]
        if cv2 is not None:
            # OPTIMIZATION: Upgraded to INTER_CUBIC to match the production training data pipeline
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            import tensorflow as tf
            # Fallback path also upgraded to BICUBIC alignment
            img = tf.image.resize(img, [target_h, target_w], method=tf.image.ResizeMethod.BICUBIC).numpy()
            
        img = np.clip(img, 0.0, 255.0)
        return np.expand_dims(img, axis=0)

    def predict(self, image_np: np.ndarray) -> dict:
        x = self._preprocess(image_np)

        if self.backend == "keras":
            probs = self.model.predict(x, verbose=0).astype(np.float32)
        else:
            inp_name = self.session.get_inputs()[0].name
            probs    = self.session.run(None, {inp_name: x.astype(np.float32)})[0]
            probs    = probs.astype(np.float32)

        if self.temperature != 1.0:
            eps    = 1e-7
            logits = np.log(np.clip(probs, eps, 1.0 - eps))
            probs  = _softmax_with_temperature(logits, self.temperature)

        pred = int(np.argmax(probs, axis=-1)[0])
        return {
            "pred":  pred,
            "prob":  float(probs[0][pred]),
            "probs": probs[0].astype(np.float32).tolist(),
        }