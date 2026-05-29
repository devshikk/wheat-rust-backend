---
title: Wheat Rust Backend
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# Wheat Rust Detection — FastAPI Backend

**Author:** Vadde Vanshika  
**Part of:** NextGen Wheat Rust Intelligence — an AI-powered precision agriculture platform  
**Live API:** https://shiikk-wheat-rust-backend.hf.space  
**Deployed on:** Hugging Face Spaces (Docker)

This is the inference backend that serves the trained ML models via a REST API. It accepts wheat leaf images, runs the full detection pipeline, and returns a structured result including disease classification, severity estimation, a Grad-CAM explainability heatmap, and a treatment recommendation.

---

## API Reference

### `POST /predict`

Upload a wheat leaf image and receive a full diagnostic result.

**Request:** `multipart/form-data` with a `file` field (JPEG / PNG / WEBP, max 10 MB)

**Response:**
```json
{
  "status":         "Diseased",
  "rust_type":      "leaf_rust",
  "confidence":     0.9341,
  "severity_level": "Moderate",
  "severity_ratio": 0.3712,
  "gradcam_image":  "<base64-encoded PNG>",
  "recommendation": "Apply Propiconazole 25% EC @ 0.1% as foliar spray..."
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `"Healthy"` or `"Diseased"` | Stage-1 classification result |
| `rust_type` | `string` or `null` | `"leaf_rust"`, `"stem_rust"`, or `"stripe_rust"` |
| `confidence` | `float` | Stage-2 SVM prediction probability (0–1) |
| `severity_level` | `string` | `"Very Mild"` → `"Mild"` → `"Moderate"` → `"Moderately Severe"` → `"Severe"` |
| `severity_ratio` | `float` | Fraction of disease-active area that is rust-coloured (0–1) |
| `gradcam_image` | `string` | Base64-encoded PNG of the Grad-CAM activation overlay |
| `recommendation` | `string` | Actionable treatment advice specific to rust type and severity |

### `GET /market`

Returns current Indian mandi wheat prices, 30-day trend, historical price data, and a regional breakdown of 10 states.

### `GET /health`

Liveness probe. Returns `{"status": "ok"}` when the service is ready.

---

## Inference Pipeline

```
POST /predict
    |
    +-- Validate (image/* only)
    +-- Decode with OpenCV
    |
    +-- CDAE denoiser (cdae_denoiser.h5)
    |     Removes noise, normalises image quality
    |
    +-- EfficientNet-B3 feature extraction
    |     Produces a 1536-dimensional feature vector
    |
    +-- Stage-1 SVM (stage1_healthy_vs_diseased.pkl)
    |     Binary: Healthy vs. Diseased
    |     If Healthy -> return immediately
    |
    +-- Stage-2 SVM (stage2_rust_classifier.pkl)
    |     Multi-class: leaf_rust / stem_rust / stripe_rust
    |
    +-- Grad-CAM heatmap generation
    |     Highlights which image regions drove the prediction
    |
    +-- CIELAB severity estimation (Grad-CAM guided)
    |     Quantifies rust-coloured infected area in CIE L*a*b* space
    |
    +-- Treatment lookup (treatments.json)
          Returns rust-type and severity-specific advice
```

---

## Repository Structure

```
wheat-rust-backend/
├── main.py              # FastAPI application and route definitions
├── inference.py         # Model loading and prediction logic
├── severity.py          # CIELAB-based severity estimation
├── treatments.json      # Treatment recommendations (by rust type and severity)
├── models/
│   ├── cdae_denoiser.h5                  # Trained CDAE model
│   ├── stage1_healthy_vs_diseased.pkl    # Stage-1 binary SVM
│   └── stage2_rust_classifier.pkl       # Stage-2 multi-class SVM
├── Dockerfile           # Hugging Face Spaces deployment
├── requirements.txt     # Pinned Python dependencies
└── README.md
```

---

## Running Locally

**Prerequisites:** Python 3.11, the three model files in `models/`

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

API available at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.5 | REST API framework |
| `uvicorn[standard]` | 0.32.1 | ASGI server |
| `tensorflow` | 2.17.0 | CDAE and EfficientNet-B3 inference |
| `scikit-learn` | 1.5.2 | SVM classifiers |
| `opencv-python-headless` | 4.10.0.84 | Image decoding and Grad-CAM rendering |
| `numpy` | 1.26.4 | Numerical operations |
| `joblib` | 1.4.2 | Model deserialisation |
| `pillow` | 10.4.0 | Image utilities |
| `python-multipart` | 0.0.12 | Multipart file upload handling |
