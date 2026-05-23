"""
CIELAB-based severity estimator — matches the training pipeline in
NextGen-WheatRustDet/scripts/severity_estimation.py
"""
import cv2
import numpy as np


def estimate_severity(image_rgb: np.ndarray, gradcam_heatmap: np.ndarray = None):
    """
    Estimate infection severity using CIELAB color space.

    Args:
        image_rgb:       RGB uint8 image (H x W x 3)
        gradcam_heatmap: Optional Grad-CAM heatmap (H x W), values 0–1.
                         When None, the whole image is analysed (HSV fallback).

    Returns:
        (severity_level: str, severity_ratio: float)
    """
    if gradcam_heatmap is not None:
        # ── Grad-CAM guided CIELAB method (precise) ──────────────────────
        heatmap_resized = cv2.resize(
            gradcam_heatmap.astype(np.float32),
            (image_rgb.shape[1], image_rgb.shape[0])
        )
        disease_mask = (heatmap_resized > 0.3).astype(np.uint8)

        # Morphological close to smooth noisy mask
        kernel = np.ones((5, 5), np.uint8)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)

        if np.sum(disease_mask) == 0:
            return "Very Mild", 0.0

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        _, a_ch, b_ch = cv2.split(lab)

        rust_pixels = ((a_ch > 140) | (b_ch > 150)) & disease_mask.astype(bool)
        severity_ratio = float(np.sum(rust_pixels)) / float(np.sum(disease_mask))
    else:
        # ── HSV fallback (whole-image) ────────────────────────────────────
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([10, 80, 80])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        infected = np.sum(mask > 0)
        total = image_rgb.shape[0] * image_rgb.shape[1]
        severity_ratio = float(infected) / float(total)

    # ── Map ratio → severity label ────────────────────────────────────────
    if severity_ratio > 0.70:
        level = "Severe"
    elif severity_ratio > 0.50:
        level = "Moderately Severe"
    elif severity_ratio > 0.30:
        level = "Moderate"
    elif severity_ratio > 0.15:
        level = "Mild"
    else:
        level = "Very Mild"

    return level, severity_ratio