# NextGen Wheat Rust Intelligence

**Author:** Vadde Vanshika  
**Live Demo:** https://project-90vi5.vercel.app
**Backend API:** https://shiikk-wheat-rust-backend.hf.space

---

An end-to-end AI-powered platform for detecting and classifying wheat rust disease from leaf photographs. The system combines a Convolutional Denoising Autoencoder, EfficientNet-B3 deep feature extraction, Falcon-optimised Support Vector Machines, Grad-CAM explainability, and CIELAB-based severity estimation — all served through a React web interface with multilingual support (English, Telugu, Hindi).

---

## Project Structure

This project is split across three repositories:

| Repository | Purpose | README |
|---|---|---|
| [`NextGen-WheatRustDet`](https://github.com/devshikk/NextGen-WheatRustDet) | ML training pipeline — CDAE, feature extraction, SVM training, evaluation | [README](https://github.com/devshikk/NextGen-WheatRustDet#readme) |
| [`wheat-rust-backend`](https://github.com/devshikk/wheat-rust-backend) | FastAPI inference server — REST API, model serving, treatment recommendations | [README](https://github.com/devshikk/wheat-rust-backend#readme) |
| [`rust-guardian-ai-main`](https://github.com/devshikk/rust-guardian-ai-main) | React/TypeScript frontend — detection UI, market insights, multilingual support | [README](https://github.com/devshikk/rust-guardian-ai-main#readme) |

---

## System Architecture

```
User (Browser)
      │
      ▼
React Frontend (Vercel CDN)
      │  POST /predict   GET /market   GET /health
      ▼
FastAPI Backend (Render.com)
      │
      ├── CDAE Denoiser (TensorFlow/Keras)
      │     Normalises image quality
      │
      ├── EfficientNet-B3 Feature Extractor
      │     1536-dim deep feature vectors
      │
      ├── Stage-1 SVM — Healthy vs. Diseased
      │
      ├── Stage-2 SVM — Leaf / Stem / Stripe Rust
      │     (Falcon Optimisation Algorithm hyperparameters)
      │
      ├── Grad-CAM — Explainability heatmap
      │
      └── CIELAB Severity Estimation
            Returns: rust type, confidence, severity, heatmap, treatment
```

---

## Key Results

| Metric | Value |
|---|---|
| Overall Accuracy | **97%** |
| Macro F1-Score | **0.96** |
| Classes | Healthy, Leaf Rust, Stem Rust, Stripe Rust |
| Severity Levels | Very Mild → Mild → Moderate → Moderately Severe → Severe |

---

## Running the Full Stack Locally

### 1. Backend

```bash
cd wheat-rust-backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
# API available at http://127.0.0.1:8000
```

### 2. Frontend

```bash
cd rust-guardian-ai-main
npm install
npm run dev
# App available at http://localhost:8080
```

The frontend automatically connects to the backend at `http://127.0.0.1:8000`.

---

## Deployment

| Component | Platform | Notes |
|---|---|---|
| Frontend | Vercel | Set `VITE_API_URL` env var to the Render backend URL |
| Backend | Render.com | Free tier; models load from the `models/` directory |

See the individual repository READMEs for detailed deployment instructions.

---

## Technology Stack

**Machine Learning**
- TensorFlow / Keras — CDAE architecture
- EfficientNet-B3 (ImageNet pre-trained) — transfer learning feature extractor
- scikit-learn — SVM classifiers
- OpenCV — image preprocessing and Grad-CAM rendering
- CIE L\*a\*b\* colour space — disease severity quantification

**Backend**
- FastAPI — REST API framework
- Uvicorn — ASGI server
- Python 3.11

**Frontend**
- React 18 + TypeScript
- Vite 5
- Tailwind CSS + shadcn/ui + Radix UI
- TanStack Query, Recharts, React Router v6

---

## Features

- 📷 **Image upload** — drag-and-drop or file picker (JPEG / PNG / WEBP)
- 🔬 **Disease detection** — distinguishes 4 classes with 97% accuracy
- 📊 **Severity estimation** — CIELAB colour analysis guided by Grad-CAM
- 🔥 **Explainability** — Grad-CAM activation heatmap overlaid on the input image
- 💊 **Treatment advice** — specific fungicide and management recommendations per rust type and severity
- 📈 **Market insights** — Indian mandi wheat price data with regional breakdown
- 🌐 **Multilingual** — English, Telugu, Hindi
- 🧪 **Sample evaluation** — pre-loaded test images for demonstration

---

## Academic Context

This project was developed as a major project submission demonstrating the application of:
- Transfer learning for agricultural disease detection
- Nature-inspired metaheuristic optimisation (Falcon Optimisation Algorithm)
- Explainable AI (Grad-CAM) for agricultural decision support
- Full-stack deployment of deep learning systems in production
