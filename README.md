# Wheat Rust Detection — FastAPI Backend

> **Author:** Vadde Vanshika  
> **Part of:** NextGen Wheat Rust Intelligence — an AI-powered precision agriculture platform

This is the Python/FastAPI backend that serves the trained ML models via a REST API. It handles image uploads, runs the full inference pipeline, and returns structured results including disease classification, severity estimation, a Grad-CAM heatmap, and a treatment recommendation.

---

## API Endpoints

### `POST /predict`

Analyse a wheat leaf image through the complete ML pipeline.

**Request:** `multipart/form-data` with a single `file` field (JPEG / PNG / WEBP).

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
| `status` | `"Healthy" \| "Diseased"` | Stage-1 classification result |
| `rust_type` | `string \| null` | `"leaf_rust"`, `"stem_rust"`, or `"stripe_rust"` |
| `confidence` | `float` | Stage-2 SVM prediction probability (0–1) |
| `severity_level` | `string` | `"Very Mild"` → `"Severe"` |
| `severity_ratio` | `float` | Fraction of disease-active area that is rust-coloured (0–1) |
| `gradcam_image` | `string` | Base64-encoded PNG of the Grad-CAM activation overlay |
| `recommendation` | `string` | Actionable treatment advice |

---

### `GET /market`

Returns current Indian mandi wheat prices, 30-day trend, price history, and regional breakdown.

### `GET /health`

Liveness probe. Returns `{"status": "ok"}` when the service is running.

---

## Inference Pipeline

```
POST /predict
    │
    ├── Validate: must be image/*
    ├── Decode with OpenCV
    │
    ├── CDAE denoiser (cdae_denoiser.h5)
    │     Reduces noise, normalises image quality
    │
    ├── EfficientNet-B3 feature extraction
    │     Produces a 1536-dim feature vector
    │
    ├── Stage-1 SVM (stage1_healthy_vs_diseased.pkl)
    │     Healthy? → return immediately
    │
    ├── Stage-2 SVM (stage2_rust_classifier.pkl)
    │     Classify: leaf_rust / stem_rust / stripe_rust
    │
    ├── Grad-CAM heatmap generation
    │     Highlights activation regions on the input image
    │
    ├── CIELAB severity estimation (Grad-CAM guided)
    │     Quantifies rust-coloured infected area
    │
    └── Treatment lookup (treatments.json)
          Returns rust-type + severity specific advice
```

---

## Repository Structure

```
wheat-rust-backend/
├── main.py            # FastAPI app and route definitions
├── inference.py       # Model loading and prediction logic
├── severity.py        # CIELAB-based severity estimation
├── treatments.json    # Treatment recommendations by rust type and severity
├── models/            # Trained model files (copied from NextGen-WheatRustDet)
│   ├── cdae_denoiser.h5
│   ├── stage1_healthy_vs_diseased.pkl
│   └── stage2_rust_classifier.pkl
├── Procfile           # Render deployment startup command
├── requirements.txt   # Pinned Python dependencies
└── .gitignore
```

---

## Running Locally

### Prerequisites

- Python 3.9 – 3.11
- The three model files in `models/` (see [NextGen-WheatRustDet](../NextGen-WheatRustDet))

### Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Start the server

```bash
uvicorn main:app --reload
```

The API is now available at `http://127.0.0.1:8000`.

Interactive API docs: `http://127.0.0.1:8000/docs`

---

## Deploying to Render

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Set the following:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy**. Render will assign a URL like `https://wheat-rust-backend.onrender.com`.

> **Note:** The free tier spins down after 15 minutes of inactivity. The first request after a cold start takes approximately 30 seconds while models are reloaded into memory.

### Environment Variables (optional)

| Variable | Description | Default |
|---|---|---|
| `MODELS_DIR` | Absolute path to the models directory | `./models` (relative to `main.py`) |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `*` |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.5 | Web framework |
| `uvicorn[standard]` | 0.32.1 | ASGI server |
| `tensorflow` | 2.17.0 | CDAE and EfficientNet-B3 |
| `scikit-learn` | 1.5.2 | SVM classifiers |
| `opencv-python-headless` | 4.10.0.84 | Image decoding and Grad-CAM rendering |
| `numpy` | 1.26.4 | Numerical operations |
| `joblib` | 1.4.2 | Model serialisation |
| `pillow` | 10.4.0 | Image utilities |
| `python-multipart` | 0.0.12 | File upload handling |
