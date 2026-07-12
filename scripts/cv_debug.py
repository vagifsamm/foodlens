"""Debug visualiser: original | plate detection | food mask | masked crop.

Usage:
    python scripts/cv_debug.py path/to/image.jpg [more images...]
    python scripts/cv_debug.py --from-dataset 3   # N random Food101 test images

Saves 4-panel figures to reports/cv_debug/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SEED, settings  # noqa: E402
from src.cv.segment import detect_plate, masked_crop, segment_food  # noqa: E402

OUT_DIR = settings.reports_dir / "cv_debug"


def debug_image(path: Path) -> Path:
    """Run plate detection + segmentation on one image, save 4-panel figure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    plate = detect_plate(img)
    mask = segment_food(img, plate)
    crop = masked_crop(img, mask)

    vis_plate = img.copy()
    if plate is not None:
        cv2.circle(vis_plate, (plate.x, plate.y), plate.r, (0, 255, 0), 3)
        cv2.circle(vis_plate, (plate.x, plate.y), 4, (0, 0, 255), -1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(vis_plate, cv2.COLOR_BGR2RGB),
         f"Plate: {'found' if plate else 'NOT found'}"),
        (mask, "GrabCut mask"),
        (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), "Masked crop"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_debug.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def sample_from_dataset(n: int) -> list[Path]:
    """Pick n random test images from splits.json."""
    with (settings.data_dir / "splits.json").open(encoding="utf-8") as f:
        splits = json.load(f)
    rng = random.Random(SEED)
    picks = rng.sample(splits["test"], n)
    root = settings.raw_dir / "food-101"
    return [root / rel for rel, _ in picks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--from-dataset", type=int, default=0)
    args = ap.parse_args()

    paths = list(args.images)
    if args.from_dataset:
        paths += sample_from_dataset(args.from_dataset)
    if not paths:
        ap.error("Give image paths or --from-dataset N")
    for p in paths:
        print(f"-> {debug_image(p)}")


if __name__ == "__main__":
    main()
