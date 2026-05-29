"""
Inference engine for the Wheat Rust Detection backend.

Loads all trained models once at startup and exposes:
  - predict(image_rgb)          -> disease classification result dict
  - generate_gradcam(image_rgb) -> (heatmap_array, base64_png)

Model files are resolved relative to this file so the backend works
on any machine or deployment environment without path changes.
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# Model directory — can be overridden via environment variable
_DEFAULT_MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR = Path(os.getenv("MODELS_DIR", _DEFAULT_MODELS_DIR))

CDAE_PATH   = MODELS_DIR / "cdae_denoiser.h5"
STAGE1_PATH = MODELS_DIR / "stage1_healthy_vs_diseased.pkl"
STAGE2_PATH = MODELS_DIR / "stage2_rust_classifier.pkl"

# Load all models at import time (once per worker process)
print(f"Loading models from: {MODELS_DIR}")

print("  [1/4] CDAE denoiser ...")
cdae = tf.keras.models.load_model(str(CDAE_PATH), compile=False)

print("  [2/4] Stage-1 binary classifier ...")
stage1 = joblib.load(STAGE1_PATH)

print("  [3/4] Stage-2 rust classifier ...")
stage2 = joblib.load(STAGE2_PATH)

print("  [4/4] EfficientNet-B3 feature extractor ...")
feature_extractor = EfficientNetB3(weights="imagenet", include_top=False, pooling="avg")

# Grad-CAM sub-model: outputs conv layer activations and global pooled features
LAST_CONV_LAYER = "top_conv"
grad_cam_model = tf.keras.models.Model(
    inputs=feature_extractor.input,
    outputs=[
        feature_extractor.get_layer(LAST_CONV_LAYER).output,
        feature_extractor.output,
    ],
)

RUST_CLASSES = ["leaf_rust", "stem_rust", "stripe_rust"]
print("All models loaded successfully.\n")


def _prepare_input(image_rgb: np.ndarray) -> np.ndarray:
    """Resize, denoise with CDAE, and preprocess for EfficientNet-B3."""
    img = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
    denoised = cdae.predict(img[np.newaxis, ...], verbose=0)
    return preprocess_input(denoised * 255.0)


def _extract_features(img_input: np.ndarray) -> np.ndarray:
    """Return a 1536-dim EfficientNet-B3 feature vector."""
    return feature_extractor.predict(img_input, verbose=0).flatten()


def predict(image_rgb: np.ndarray) -> dict:
    """
    Run the two-stage SVM classification pipeline.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image (H x W x 3, uint8).

    Returns
    -------
    dict with keys: status, rust_type, confidence
    """
    img_input = _prepare_input(image_rgb)
    features  = _extract_features(img_input)

    # Stage 1: Healthy vs. Diseased
    stage1_pred = stage1.predict([features])[0]
    if stage1_pred == 0:
        return {"status": "Healthy", "rust_type": None, "confidence": 1.0}

    # Stage 2: Rust type classification
    stage2_pred = stage2.predict([features])[0]
    proba       = stage2.predict_proba([features])[0]
    confidence  = float(proba[stage2_pred])
    rust_type   = RUST_CLASSES[int(stage2_pred)]

    return {"status": "Diseased", "rust_type": rust_type, "confidence": confidence}


def generate_gradcam(image_rgb: np.ndarray):
    """
    Generate a Grad-CAM activation heatmap for the given image.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image (H x W x 3, uint8).

    Returns
    -------
    heatmap_array : np.ndarray
        Normalised heatmap (H x W), values in [0, 1].
    overlay_b64 : str
        Base64-encoded PNG of the JET-coloured heatmap blended over the input.
    """
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_norm    = img_resized.astype(np.float32) / 255.0
    denoised    = cdae.predict(img_norm[np.newaxis, ...], verbose=0)
    img_input   = preprocess_input(denoised * 255.0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_model(img_input)
        loss = tf.reduce_mean(predictions)

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    max_val  = tf.math.reduce_max(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (max_val + 1e-8)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match input image size
    heatmap_full  = cv2.resize(heatmap_np, (img_resized.shape[1], img_resized.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_full)

    # Apply JET colormap and blend with original image
    jet     = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.55, jet_rgb, 0.45, 0)

    # Encode as base64 PNG
    success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")

    overlay_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return heatmap_np, overlay_b64