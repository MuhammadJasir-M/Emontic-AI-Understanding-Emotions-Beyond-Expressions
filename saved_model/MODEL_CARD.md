# Emontic AI v2.5 (Production) — Model Card

## Model Overview

| Field | Value |
|---|---|
| **Model name** | Emontic AI v2.5 |
| **Architecture** | EfficientNetV2-S + Position-Aware Attention + ArcFace Margin |
| **Task** | 7-class Facial Emotion Recognition (FER) |
| **Input** | RGB face image, 224×224 |
| **Output** | Hyperspherical Cosine Similarity vector mapping + explicit Ambiguity Gate |
| **Framework** | TensorFlow / Keras (training), ONNX Runtime (production) |
| **License** | Project-internal — not for redistribution |

---

## Emotion Classes

| Index | Label | AffectNet ID | RAF-DB ID |
|---|---|---|---|
| 0 | Angry | 0 | 6 |
| 1 | Disgust | 2 | 3 |
| 2 | Fear | 3 | 2 |
| 3 | Happy | 4 | 4 |
| 4 | Neutral | 5 | 7 |
| 5 | Sad | 6 | 5 |
| 6 | Surprise | 7 | 1 |

> [!WARNING]
> **Dataset Domain Constraint:** Contempt (AffectNet class 1) is explicitly stripped during dataset tokenization and is entirely excluded from training and inference loops. The model contains no weights or output paths for contempt.

---

## Training Datasets & Strategies

### Stage 1: Pretraining (AffectNet)
- **Source**: Kaggle — mstjebashazida/affectnet
- **Size**: ~280,000 images
- **Strategy**: Frozen backbone → unfreezes top 60 layers.
- **Loss Optimization**: ArcFace Loss (margin=0.3, scale=30.0) combined with **0.1 Label Smoothing** to protect the network from AffectNet's estimated 20-30% label noise.
- **Class Balancing**: **Inverse class-weighting** baked natively into the `tf.data` pipeline.

### Stage 2: Fine-tuning (RAF-DB)
- **Source**: Kaggle — shuvoalok/raf-db-dataset
- **Size**: ~12,271 images
- **Strategy**: Unfreezes top 120 layers.
- **Loss Optimization**: Strict ArcFace Loss. Label smoothing is dropped to **0.0** to establish extremely sharp, confident decision boundaries on the high-quality crowdsourced labels.
- **Learning Rate**: Dropped to **2e-5** for precise target domain alignment.

---

## Validation Performance (RAF-DB Test Set)

The model was evaluated against the rigorous RAF-DB test split (3,008 images). The integration of Position-Aware Attention and ArcFace natively mitigates the extreme class imbalance, resulting in strong performance across the board.

| Metric | Score | Note |
|---|---|---|
| **Overall Accuracy** | **76.56%** | Exceptional for 7-class in-the-wild facial recognition. |
| **Weighted F1-Score** | **77.85%** | Accounts for class support distribution. |
| **Macro F1-Score** | **68.68%** | Unweighted average highlighting strong minority class capture. |

### Per-Class F1-Scores

| Emotion | F1-Score | Precision | Recall |
|---|---|---|---|
| **Happy** | 89.01% | 97.29% | 82.02% |
| **Surprise** | 78.71% | 75.63% | 82.06% |
| **Sad** | 75.57% | 75.41% | 75.73% |
| **Neutral** | 71.85% | 76.50% | 67.74% |
| **Angry** | 64.57% | 60.10% | 69.75% |
| **Fear** | 52.32% | 45.91% | 60.81% |
| **Disgust** | 48.69% | 35.90% | 75.62% |

> [!NOTE]
> Minority classes like Disgust and Fear typically score below 30% in standard baseline models due to severe dataset starvation (e.g., AffectNet has 70k+ Happy images but only ~3k Disgust). The combination of Inverse Class-Weighting and the ArcFace Margin has massively elevated these minority recall rates (Disgust Recall: **75.62%**).

---

## Architectural Enhancements

### 1. Position-Aware Multi-Head Self Attention
Instead of relying purely on GlobalAveragePooling, the network routes the final `7x7x1280` spatial grid through a bespoke Multi-Head Self Attention layer infused with **Spatial Positional Embeddings**. This allows the model to structurally understand where facial features (e.g., eyes vs. mouth) are located, dramatically boosting performance on complex emotions like Fear and Disgust.

### 2. ArcFace Dense Margin Head
Softmax saturation and overconfidence were previously solved via external Temperature Scaling. In v2.5, this is solved natively at the architectural level using **Additive Angular Margin Loss (ArcFace)**. The network is forced to separate classes on a hypersphere with a strict 0.3 radian margin, inherently calibrating the logits.

> Production Update (v2.5): The production inference pipeline strips training scale factors to evaluate the true, unwarped Cosine Similarity vector ($\cos\theta$) independently per class, enabling accurate multi-emotion tracking.

### 3. Multi-Stage Haar Cascades
The legacy RetinaFace and MediaPipe dependencies were entirely stripped out for extreme performance. Face detection is now handled by a multi-stage fallback pipeline using highly optimized Haar Cascades:
- Primary pass: `haarcascade_frontalface_alt2.xml`
- Fallback 1: `haarcascade_profileface.xml`
- Fallback 2: Flipped `haarcascade_profileface.xml` (for left profiles)

---

## ⟁ Calibration & Ambiguity Gates

| Parameter | Value | Description |
|---|---|---|
| `AMBIGUITY_THRESHOLD` | 0.15 | Minimum mathematical delta required between the top-1 and top-2 class cosine scores. If the margin is narrower, the expression triggers the `is_ambiguous` system flag. |

---

## Inference Pipeline

```text
Webcam/Image Upload
    │
    ▼
Multi-Stage Haar Cascade Face Detection (Frontal → Profiles)
    │
    ▼
Bicubic Resize → (224, 224)
    │
    ▼
EfficientNetV2S Backbone (includes native [0, 255] normalization)
    │
    ▼
Spatial Positional Attention → ArcFace Embeddings
    │
    ▼
Cosine Similarity Engine → [Ambiguity Threshold Gate]
    │
    ▼
Live View: 5-Frame Rolling Window Temporal Smoothing Average
    │
    ▼
Final Output: Uncoupled Geometric Metric Matches [0.0 - 1.0]
```

**Optional**: Test-Time Augmentation (TTA). Stacks the original, horizontally flipped, and luminosity-enhanced variations into a single `(3, 224, 224, 3)` tensor. The ONNX Runtime processes the batch in parallel, and the probabilities are averaged for a ~1-2% accuracy boost. TTA is disabled during Live Webcam inference to preserve sub-30ms latency.

---

## Deployment Configuration

| Format | Path | Notes |
|---|---|---|
| ONNX | `backend/models/emontic_ai.onnx` | 100% TensorFlow-free production deployment. Supports CUDA execution for <30ms latency. |
| DB | `backend/emontic_ai.db` | Dual MySQL / SQLite fallback system. |

---

[View Full Validation Report](eval_report_v2s.json)