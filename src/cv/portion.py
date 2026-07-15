"""Portion estimation: food mask area -> S/M/L bucket -> grams.

Two estimates (spec section 4):
  1. Bucket: coverage = mask_area / plate_area (or image area) -> S/M/L ->
     grams = typical_serving_g * {S: 0.6, M: 1.0, L: 1.6}.
  2. Scale-refined (only when a plate is detected): assume a standard 26 cm
     plate, convert px^2 -> cm^2, grams = area_cm2 * density_g_per_cm2.

Monocular portion estimation is inherently approximate; the error is measured
honestly in reports/portion_validation.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import settings
from src.cv.segment import Circle

PLATE_DIAMETER_CM = 26.0
BUCKET_FACTORS = {"S": 0.6, "M": 1.0, "L": 1.6}


@dataclass
class PortionEstimate:
    """Portion estimate for a segmented food region.

    Attributes:
        bucket: "S" | "M" | "L".
        coverage: mask_area / reference_area, in [0, 1+].
        grams_bucket: Serving-based estimate (always available).
        grams_scaled: Plate-scale refined estimate, None without a plate.
        grams: Final estimate (scaled when available, else bucket).
        confidence: "high" with plate, "low" without.
    """

    bucket: str
    coverage: float
    grams_bucket: float
    grams_scaled: Optional[float]
    grams: float
    confidence: str


def _load_nutrition_entry(food_class: str) -> dict:
    with settings.nutrition_db_path.open(encoding="utf-8") as f:
        db = json.load(f)
    if food_class not in db:
        raise KeyError(f"Unknown food class: {food_class}")
    return db[food_class]


def estimate_portion(mask: np.ndarray, plate: Optional[Circle],
                     food_class: str) -> PortionEstimate:
    """Estimate grams of food from a binary mask.

    Args:
        mask: uint8 binary mask (255 = food).
        plate: Detected plate circle, or None (falls back to image area).
        food_class: One of the 25 class names (nutrition_db key).

    Returns:
        PortionEstimate with bucket + refined gram estimates.
    """
    entry = _load_nutrition_entry(food_class)
    typical = float(entry["typical_serving_g"])
    density = float(entry["density_g_per_cm2"])

    mask_area_px = float((mask > 0).sum())
    ref_area_px = plate.area if plate is not None else float(mask.size)
    coverage = mask_area_px / max(ref_area_px, 1.0)

    if coverage < 0.25:
        bucket = "S"
    elif coverage <= 0.5:
        bucket = "M"
    else:
        bucket = "L"
    grams_bucket = typical * BUCKET_FACTORS[bucket]

    grams_scaled: Optional[float] = None
    if plate is not None and plate.r > 0:
        cm_per_px = PLATE_DIAMETER_CM / (2.0 * plate.r)
        area_cm2 = mask_area_px * cm_per_px ** 2
        grams_scaled = area_cm2 * density

    # A degenerate (empty) mask makes the scaled estimate 0 g, which is
    # meaningless; fall back to the serving-based bucket estimate in that case.
    if grams_scaled is not None and grams_scaled > 0:
        grams = grams_scaled
    else:
        grams = grams_bucket
    confidence = "high" if (plate is not None and grams_scaled and grams_scaled > 0) else "low"
    return PortionEstimate(bucket=bucket, coverage=round(coverage, 3),
                           grams_bucket=round(grams_bucket, 1),
                           grams_scaled=None if grams_scaled is None else round(grams_scaled, 1),
                           grams=round(grams, 1), confidence=confidence)


if __name__ == "__main__":
    demo_mask = np.zeros((400, 400), np.uint8)
    demo_mask[100:300, 100:300] = 255
    print(estimate_portion(demo_mask, Circle(200, 200, 180), "pizza"))
