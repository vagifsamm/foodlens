"""CV layer tests with synthetic images (no dataset needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.cv.portion import estimate_portion
from src.cv.quality import check_quality
from src.cv.segment import Circle, detect_plate, masked_crop, segment_food


def make_plate_image(size: int = 480) -> np.ndarray:
    """Grey background, white plate, saturated food blob in the middle."""
    img = np.full((size, size, 3), 90, np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.4), (235, 235, 235), -1)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.4), (180, 180, 180), 4)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.18), (30, 70, 200), -1)
    noise = np.random.default_rng(42).integers(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


class TestQuality:
    def test_sharp_image_passes(self) -> None:
        report = check_quality(make_plate_image())
        assert report.ok, report.reasons_az

    def test_blurred_image_rejected(self) -> None:
        blurred = cv2.GaussianBlur(make_plate_image(), (51, 51), 20)
        report = check_quality(blurred)
        assert not report.ok
        assert any("bulanıq" in r for r in report.reasons_az)

    def test_dark_image_rejected(self) -> None:
        dark = (make_plate_image() * 0.1).astype(np.uint8)
        report = check_quality(dark)
        assert not report.ok
        assert any("qaranlıq" in r for r in report.reasons_az)

    def test_small_image_rejected(self) -> None:
        small = cv2.resize(make_plate_image(), (100, 100))
        report = check_quality(small)
        assert not report.ok
        assert any("kiçik" in r for r in report.reasons_az)

    def test_scores_populated(self) -> None:
        report = check_quality(make_plate_image())
        assert {"blur_score", "brightness", "min_side"} <= set(report.scores)


class TestSegment:
    def test_plate_detected_near_centre(self) -> None:
        img = make_plate_image()
        plate = detect_plate(img)
        assert plate is not None
        assert abs(plate.x - 240) < 60 and abs(plate.y - 240) < 60

    def test_segment_food_returns_binary_mask(self) -> None:
        img = make_plate_image()
        mask = segment_food(img, detect_plate(img))
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 255}
        assert (mask > 0).sum() > 500  # found something

    def test_segment_without_plate_falls_back(self) -> None:
        img = make_plate_image()
        mask = segment_food(img, None)
        assert mask.shape == img.shape[:2]

    def test_masked_crop_smaller_than_original(self) -> None:
        img = make_plate_image()
        mask = segment_food(img, detect_plate(img))
        crop = masked_crop(img, mask)
        assert crop.size <= img.size

    def test_masked_crop_empty_mask_returns_original(self) -> None:
        img = make_plate_image()
        crop = masked_crop(img, np.zeros(img.shape[:2], np.uint8))
        assert crop.shape == img.shape


class TestPortion:
    def make_mask(self, frac_of_plate: float, plate: Circle) -> np.ndarray:
        mask = np.zeros((480, 480), np.uint8)
        r = int(np.sqrt(frac_of_plate) * plate.r)
        cv2.circle(mask, (plate.x, plate.y), r, 255, -1)
        return mask

    def test_small_portion(self) -> None:
        plate = Circle(240, 240, 190)
        est = estimate_portion(self.make_mask(0.10, plate), plate, "pizza")
        assert est.bucket == "S"

    def test_medium_portion(self) -> None:
        plate = Circle(240, 240, 190)
        est = estimate_portion(self.make_mask(0.35, plate), plate, "pizza")
        assert est.bucket == "M"

    def test_large_portion(self) -> None:
        plate = Circle(240, 240, 190)
        est = estimate_portion(self.make_mask(0.70, plate), plate, "pizza")
        assert est.bucket == "L"

    def test_scaled_estimate_present_with_plate(self) -> None:
        plate = Circle(240, 240, 190)
        est = estimate_portion(self.make_mask(0.35, plate), plate, "steak")
        assert est.grams_scaled is not None
        assert est.confidence == "high"
        assert est.grams == est.grams_scaled

    def test_no_plate_low_confidence(self) -> None:
        mask = np.zeros((480, 480), np.uint8)
        mask[100:300, 100:300] = 255
        est = estimate_portion(mask, None, "pizza")
        assert est.confidence == "low"
        assert est.grams_scaled is None
        assert est.grams == est.grams_bucket

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(KeyError):
            estimate_portion(np.zeros((10, 10), np.uint8), None, "plov")
