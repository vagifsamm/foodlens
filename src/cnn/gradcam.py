"""Grad-CAM implemented with manual forward/backward hooks (no external lib).

Hooks the last convolutional block, weights activations by pooled gradients,
and overlays a jet heatmap at alpha=0.4 (spec section 5).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last conv block for a supported model."""
    if model_name == "simple":
        # Last conv layer of SimpleCNN.features (Conv,BN,ReLU,Pool x4).
        # nn.Module.__getattr__ is typed as Tensor|Module, so the stubs don't
        # know .features is iterable/indexable here; it is at runtime.
        convs = [m for m in model.features if isinstance(m, nn.Conv2d)]  # type: ignore[union-attr]
        return convs[-1]
    if model_name == "effnet":
        return model.features[-1]  # type: ignore[index,return-value]
    raise ValueError(f"Unknown model: {model_name}")


class GradCAM:
    """Minimal Grad-CAM over one target layer.

    Usage:
        cam = GradCAM(model, layer)
        heat = cam(input_tensor, class_idx)   # HxW float in [0,1]
        cam.close()
    """

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            layer.register_forward_hook(self._save_activation),
            layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _m: nn.Module, _i, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _m: nn.Module, _gi, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute the CAM for one image tensor (1,C,H,W)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()

        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,K,1,1)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Overlay a jet-colormapped CAM on a BGR image."""
    heat = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heat_u8 = (255 * heat).astype(np.uint8)
    heat_jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(heat_jet, alpha, img_bgr, 1 - alpha, 0)


if __name__ == "__main__":
    from src.cnn.models import build_model

    model = build_model("simple", pretrained=False)
    cam = GradCAM(model, target_layer(model, "simple"))
    heat = cam(torch.randn(1, 3, 224, 224))
    cam.close()
    print(f"cam shape={heat.shape} range=({heat.min():.2f},{heat.max():.2f})")
