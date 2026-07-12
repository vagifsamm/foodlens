"""Food101 25-class subset Dataset and DataLoader factories.

Reads ``data/splits.json`` produced by ``scripts/prepare_data.py``.
Transforms follow PROJECT_SPEC 3.1 exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import IMG_SIZE, SEED, settings

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FOOD101_DIR = settings.raw_dir / "food-101"


def train_transform() -> Callable:
    """Augmented training transform (spec 3.1)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> Callable:
    """Deterministic eval transform (spec 3.1)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FoodSubset(Dataset):
    """25-class Food101 subset backed by splits.json.

    Args:
        split: One of "train" | "val" | "test".
        transform: Torchvision transform; defaults to the split-appropriate one.
        limit: Optional cap on samples (for smoke tests), applied per split
            after a seeded shuffle so classes stay mixed.
    """

    def __init__(self, split: str, transform: Optional[Callable] = None,
                 limit: Optional[int] = None) -> None:
        splits_path = settings.data_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"{splits_path} not found - run scripts/prepare_data.py first")
        with splits_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.classes: list[str] = data["classes"]
        self.samples: list[tuple[str, int]] = [tuple(s) for s in data[split]]
        if limit is not None:
            g = torch.Generator().manual_seed(SEED)
            idx = torch.randperm(len(self.samples), generator=g)[:limit].tolist()
            self.samples = [self.samples[i] for i in idx]
        self.transform = transform or (
            train_transform() if split == "train" else eval_transform())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.samples[i]
        img = Image.open(FOOD101_DIR / Path(rel_path)).convert("RGB")
        return self.transform(img), label


def get_dataloaders(batch_size: int = 32, num_workers: Optional[int] = None,
                    limit: Optional[int] = None,
                    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders.

    Args:
        batch_size: Batch size for all splits.
        num_workers: Defaults to settings.num_workers (0 on Windows).
        limit: Per-split sample cap for smoke tests.
    """
    nw = settings.num_workers if num_workers is None else num_workers
    pin = settings.resolve_device() == "cuda"

    def make(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(FoodSubset(split, limit=limit), batch_size=batch_size,
                          shuffle=shuffle, num_workers=nw, pin_memory=pin)

    return make("train", True), make("val", False), make("test", False)


if __name__ == "__main__":
    tr, va, te = get_dataloaders(batch_size=8, limit=16)
    x, y = next(iter(tr))
    print(f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
    print(f"batch: x={tuple(x.shape)} y={tuple(y.shape)} labels={y.tolist()}")
