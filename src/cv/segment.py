"""Plate detection (Hough circles, contour fallback) and GrabCut food mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Circle:
    """Detected plate as a circle in pixel coordinates."""

    x: int
    y: int
    r: int

    @property
    def area(self) -> float:
        return float(np.pi * self.r ** 2)


def detect_plate(img: np.ndarray) -> Optional[Circle]:
    """Detect the plate as the dominant circle; fall back to largest contour.

    Args:
        img: BGR uint8 image.

    Returns:
        Circle or None if nothing plate-like is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    min_r, max_r = int(min(h, w) * 0.20), int(min(h, w) * 0.60)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min(h, w) // 2, param1=120, param2=60,
                               minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        x, y, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        return Circle(int(x), int(y), int(r))

    # Fallback: largest bright-ish contour approximated by its enclosing circle.
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.05 * h * w:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(biggest)
    if not (min_r <= r <= max_r * 1.2):
        return None
    return Circle(int(cx), int(cy), int(r))


def segment_food(img: np.ndarray, plate: Optional[Circle]) -> np.ndarray:
    """GrabCut food mask seeded by the plate ROI (or centre 70% without plate).

    Args:
        img: BGR uint8 image.
        plate: Detected plate circle, or None.

    Returns:
        Binary uint8 mask (255 = food), post-processed with morphological
        open+close and reduced to the largest connected component.
    """
    h, w = img.shape[:2]
    if plate is not None:
        r = int(plate.r * 0.9)  # plate bbox shrunk 10%
        x0, y0 = max(plate.x - r, 0), max(plate.y - r, 0)
        x1, y1 = min(plate.x + r, w - 1), min(plate.y + r, h - 1)
    else:
        mx, my = int(w * 0.15), int(h * 0.15)  # centre 70%
        x0, y0, x1, y1 = mx, my, w - mx, h - my
    rect = (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Degenerate rect or uniform image: return the rect itself as mask.
        out = np.zeros((h, w), np.uint8)
        out[y0:y1, x0:x1] = 255
        return out

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def masked_crop(img: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop the image to the mask bbox (with padding), background zeroed.

    Returns the original image unchanged if the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img
    h, w = img.shape[:2]
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h - 1)
    out = cv2.bitwise_and(img, img, mask=mask)
    return out[y0:y1 + 1, x0:x1 + 1]


if __name__ == "__main__":
    demo = np.full((400, 400, 3), 200, np.uint8)
    cv2.circle(demo, (200, 200), 150, (230, 230, 230), -1)
    cv2.circle(demo, (200, 200), 70, (0, 80, 200), -1)
    plate = detect_plate(demo)
    m = segment_food(demo, plate)
    print(f"plate={plate} mask_px={int((m > 0).sum())}")
