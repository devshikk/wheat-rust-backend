"""
FastAPI backend for the NextGen Wheat Rust Detection Platform.

Endpoints:
  POST /predict  - Full ML pipeline: CDAE -> EfficientNet-B3 -> SVM ->
                   Grad-CAM -> CIELAB severity estimation -> treatment advice
  GET  /market   - Indian wheat mandi price data (regional + historical)
  GET  /health   - Liveness probe for deployment health checks
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import generate_gradcam, predict
from severity import estimate_severity

# Load treatment recommendations from JSON
_TREATMENTS_PATH = Path(__file__).parent / "treatments.json"
with open(_TREATMENTS_PATH, encoding="utf-8") as f:
    TREATMENTS: dict = json.load(f)

# Maps severity labels to treatment lookup keys
_SEV_KEY = {
    "Very Mild":         "very_mild",
    "Mild":              "mild",
    "Moderate":          "moderate",
    "Moderately Severe": "moderately_severe",
    "Severe":            "severe",
}

# Indian mandi wheat price data
_MARKET_DATA = {
    "current_price": 2800,
    "forecast_price": 2920,
    "trend_30d": "+4.2%",
    "price_history": [
        {"date": "Jan 15", "price": 2450},
        {"date": "Feb 1",  "price": 2520},
        {"date": "Feb 15", "price": 2480},
        {"date": "Mar 1",  "price": 2600},
        {"date": "Mar 15", "price": 2670},
        {"date": "Apr 1",  "price": 2750},
        {"date": "Apr 15", "price": 2800},
    ],
    "regional_prices": [
        {"region": "Punjab",         "price": 2850, "trend": "+3.2%", "up": True},
        {"region": "Haryana",        "price": 2780, "trend": "+2.8%", "up": True},
        {"region": "Uttar Pradesh",  "price": 2750, "trend": "+2.5%", "up": True},
        {"region": "Madhya Pradesh", "price": 2700, "trend": "+2.1%", "up": True},
        {"region": "Rajasthan",      "price": 2680, "trend": "+1.9%", "up": True},
        {"region": "Gujarat",        "price": 2720, "trend": "+2.3%", "up": True},
        {"region": "Telangana",      "price": 2650, "trend": "-0.5%", "up": False},
        {"region": "Andhra Pradesh", "price": 2630, "trend": "+1.2%", "up": True},
        {"region": "Karnataka",      "price": 2600, "trend": "+0.8%", "up": True},
        {"region": "Tamil Nadu",     "price": 2580, "trend": "-0.3%", "up": False},
    ],
}

app = FastAPI(
    title="Wheat Rust Detection API",
    description=(
        "NextGen WheatRustDet — CDAE denoising + EfficientNet-B3 feature extraction "
        "+ two-stage SVM classification + Grad-CAM explainability + CIELAB severity estimation."
    ),
    version="2.0.0",
)

# Allow requests from any origin (Vercel frontend + local development)
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_treatment(rust_type: str | None, severity_level: str) -> str:
    """Look up a treatment recommendation by rust type and severity."""
    if rust_type is None or rust_type == "healthy":
        return TREATMENTS["healthy"]["any"]
    sev_key  = _SEV_KEY.get(severity_level, "mild")
    rust_key = rust_type.lower().replace(" ", "_")
    return TREATMENTS.get(rust_key, {}).get(sev_key, "Consult your local agricultural officer.")


@app.get("/health")
def health():
    """Liveness probe — returns 200 OK when the service is ready."""
    return {"status": "ok"}


@app.get("/market")
def market():
    """Return current and historical Indian wheat mandi price data."""
    return _MARKET_DATA


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Analyse an uploaded wheat leaf image through the full ML pipeline.

    Returns disease status, rust type, confidence score, severity level,
    infected-area ratio, a base64-encoded Grad-CAM overlay, and a
    treatment recommendation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw       = await file.read()
    np_arr    = np.frombuffer(raw, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Run two-stage SVM classification
    cls_result = predict(image_rgb)

    # Generate Grad-CAM heatmap
    heatmap_array, gradcam_b64 = generate_gradcam(image_rgb)

    # Estimate severity using CIELAB colour analysis guided by Grad-CAM
    if cls_result["status"] == "Diseased":
        severity_level, severity_ratio = estimate_severity(image_rgb, heatmap_array)
    else:
        severity_level, severity_ratio = "Very Mild", 0.0

    treatment = _get_treatment(cls_result.get("rust_type"), severity_level)

    return {
        "status":         cls_result["status"],
        "rust_type":      cls_result.get("rust_type"),
        "confidence":     round(cls_result["confidence"], 4),
        "severity_level": severity_level,
        "severity_ratio": round(severity_ratio, 4),
        "gradcam_image":  gradcam_b64,
        "recommendation": treatment,
    }