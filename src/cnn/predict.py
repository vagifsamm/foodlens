"""Single-image inference wrapper around a saved checkpoint."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CLASSES, settings
from src.cnn.dataset import eval_transform
from src.cnn.models import build_model

log = logging.getLogger(__name__)


class Predictor:
    """Loads a checkpoint once and classifies images.

    Args:
        model_name: "simple" | "effnet".
        device: Override; defaults to CPU. Inference is CPU-first by design
            (CLAUDE.md requirement) so the demo/API never compete with
            training for the 6 GB GPU.
    """

    def __init__(self, model_name: str = "effnet", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        ckpt_path = settings.models_dir / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {ckpt_path} - train first")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.classes: list[str] = ckpt.get("classes", CLASSES)
        self.model = build_model(model_name, num_classes=len(self.classes),
                                 pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform()
        self.meta = {k: v for k, v in ckpt.items() if k != "state_dict"}

    def to_tensor(self, img: Image.Image | np.ndarray) -> torch.Tensor:
        """PIL or BGR ndarray -> normalised batch tensor on device."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image | np.ndarray,
                topk: int = 5) -> list[tuple[str, float]]:
        """Return [(class_name, prob), ...] sorted desc, length topk."""
        probs = torch.softmax(self.model(self.to_tensor(img)), dim=1)[0]
        vals, idxs = probs.topk(min(topk, len(self.classes)))
        return [(self.classes[int(i)], float(v)) for v, i in zip(vals, idxs)]


@lru_cache(maxsize=2)
def get_predictor(model_name: str = "effnet") -> Predictor:
    """Cached predictor instance (model loads are expensive)."""
    return Predictor(model_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.cnn.predict <image> [model]")
    name = sys.argv[2] if len(sys.argv) > 2 else "effnet"
    p = Predictor(name)
    for cls, prob in p.predict(Image.open(Path(sys.argv[1])).convert("RGB")):
        print(f"{cls:<22}{prob:.3f}")
