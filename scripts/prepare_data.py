"""Download Food101, filter to the 25-class subset, build train/val/test splits.

Outputs ``data/splits.json``:
    {"classes": [...], "train": [[rel_path, label], ...], "val": ..., "test": ...}

Val = 10% of official train, stratified, seed 42. Prints a class-distribution table.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASSES, SEED, settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("prepare_data")

FOOD101_DIR = settings.raw_dir / "food-101"


def download_food101() -> None:
    """Fetch and extract Food101 via torchvision (no-op if already present)."""
    from torchvision.datasets import Food101

    log.info("Ensuring Food101 is downloaded to %s ...", settings.raw_dir)
    Food101(root=str(settings.raw_dir), split="train", download=True)


def load_meta(split: str) -> dict[str, list[str]]:
    """Read Food101 meta json: class name -> list of image ids ('class/id')."""
    meta_path = FOOD101_DIR / "meta" / f"{split}.json"
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_splits() -> dict[str, object]:
    """Filter to CLASSES, carve stratified val from train, return splits dict."""
    train_meta = load_meta("train")
    test_meta = load_meta("test")

    missing = [c for c in CLASSES if c not in train_meta]
    if missing:
        raise ValueError(f"Classes missing from Food101: {missing}")

    rng = random.Random(SEED)
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []

    for label, cls in enumerate(CLASSES):
        ids = sorted(train_meta[cls])
        rng.shuffle(ids)
        n_val = int(len(ids) * 0.10)
        val += [(f"images/{i}.jpg", label) for i in ids[:n_val]]
        train += [(f"images/{i}.jpg", label) for i in ids[n_val:]]
        test += [(f"images/{i}.jpg", label) for i in sorted(test_meta[cls])]

    return {"classes": CLASSES, "train": train, "val": val, "test": test}


def print_distribution(splits: dict[str, object]) -> None:
    """Print a per-class count table for train/val/test."""
    counts: dict[int, list[int]] = {i: [0, 0, 0] for i in range(len(CLASSES))}
    for col, name in enumerate(("train", "val", "test")):
        for _, label in splits[name]:  # type: ignore[union-attr]
            counts[label][col] += 1
    print(f"\n{'class':<22}{'train':>7}{'val':>6}{'test':>6}")
    print("-" * 41)
    for i, cls in enumerate(CLASSES):
        tr, va, te = counts[i]
        print(f"{cls:<22}{tr:>7}{va:>6}{te:>6}")
    tot = [sum(c[j] for c in counts.values()) for j in range(3)]
    print("-" * 41)
    print(f"{'TOTAL':<22}{tot[0]:>7}{tot[1]:>6}{tot[2]:>6}")


def main() -> None:
    download_food101()
    splits = build_splits()
    out = settings.data_dir / "splits.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)
    log.info("Wrote %s", out)
    print_distribution(splits)


if __name__ == "__main__":
    main()
