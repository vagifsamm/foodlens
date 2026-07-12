"""Evaluation: metrics.json, confusion matrices, per-class F1, comparison chart.

Usage:
    python -m src.cnn.evaluate                 # evaluate all available checkpoints
    python -m src.cnn.evaluate --model effnet  # one model
    python -m src.cnn.evaluate --gradcam 10    # also dump Grad-CAM samples
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             f1_score)

from config import CLASSES, settings  # noqa: E402
from src.cnn.dataset import get_dataloaders  # noqa: E402
from src.cnn.models import count_params  # noqa: E402
from src.cnn.predict import Predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate")


@torch.no_grad()
def collect_predictions(pred: Predictor, bs: int = 64,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the test split; returns (targets, top1_preds, top5_hit_flags)."""
    _, _, test_loader = get_dataloaders(batch_size=bs)
    targets, top1, top5 = [], [], []
    for x, y in test_loader:
        logits = pred.model(x.to(pred.device))
        t5 = logits.topk(5, dim=1).indices.cpu()
        targets += y.tolist()
        top1 += t5[:, 0].tolist()
        top5 += (t5 == y.unsqueeze(1)).any(1).tolist()
    return np.array(targets), np.array(top1), np.array(top5)


@torch.no_grad()
def measure_cpu_latency(model_name: str, n: int = 20) -> float:
    """Mean CPU inference ms/image (batch size 1)."""
    pred = Predictor(model_name, device="cpu")
    x = torch.randn(1, 3, 224, 224)
    for _ in range(3):
        pred.model(x)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        pred.model(x)
    return (time.perf_counter() - t0) / n * 1000


def evaluate_model(model_name: str) -> dict:
    """Full metric computation + plots for one checkpoint."""
    pred = Predictor(model_name)
    targets, top1, top5_hits = collect_predictions(pred)

    report = classification_report(targets, top1, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "top1_acc": float((targets == top1).mean()),
        "top5_acc": float(top5_hits.mean()),
        "macro_f1": float(f1_score(targets, top1, average="macro")),
        "params": count_params(pred.model),
        "cpu_ms_per_image": round(measure_cpu_latency(model_name), 1),
        "per_class": {c: {k: round(report[c][k], 4)
                          for k in ("precision", "recall", "f1-score")}
                      for c in CLASSES},
    }

    cm = confusion_matrix(targets, top1, normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_title(f"Confusion matrix (normalised) - {model_name}")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(settings.reports_dir / f"confusion_matrix_{model_name}.png", dpi=130)
    plt.close(fig)

    f1s = [report[c]["f1-score"] for c in CLASSES]
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([CLASSES[i] for i in order], [f1s[i] for i in order], color="#e07b39")
    ax.set_xlabel("F1")
    ax.set_title(f"Per-class F1 - {model_name}")
    fig.tight_layout()
    fig.savefig(settings.reports_dir / f"per_class_f1_{model_name}.png", dpi=130)
    plt.close(fig)

    return metrics


def comparison_chart(all_metrics: dict[str, dict]) -> None:
    """SimpleCNN vs EfficientNet bar chart (goes in the slides)."""
    names = list(all_metrics)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [("top1_acc", "Top-1 accuracy", 1.0), ("macro_f1", "Macro F1", 1.0),
              ("params", "Parameters", None), ("cpu_ms_per_image", "CPU ms/img", None)]
    colors = ["#8093f1", "#e07b39"]
    for ax, (key, title, ymax) in zip(axes, panels):
        vals = [all_metrics[n][key] for n in names]
        ax.bar(names, vals, color=colors[:len(names)])
        ax.set_title(title)
        if ymax:
            ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            label = f"{v:,}" if key == "params" else f"{v:.3f}" if ymax else f"{v:.1f}"
            ax.text(i, v, label, ha="center", va="bottom", fontsize=9)
    fig.suptitle("SimpleCNN vs EfficientNet-B0")
    fig.tight_layout()
    fig.savefig(settings.reports_dir / "model_comparison.png", dpi=130)
    plt.close(fig)


def dump_gradcam_samples(model_name: str, n: int = 10, min_misclassified: int = 2) -> list[str]:
    """Save n Grad-CAM overlays incl. >= min_misclassified failure cases."""
    import cv2

    from src.cnn.dataset import FoodSubset
    from src.cnn.gradcam import GradCAM, overlay_heatmap, target_layer

    out_dir = settings.reports_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = Predictor(model_name)
    cam = GradCAM(pred.model, target_layer(pred.model, model_name))
    ds = FoodSubset("test")

    correct_saved, wrong_saved, paths = 0, 0, []
    rng = np.random.default_rng(42)
    for i in rng.permutation(len(ds.samples)).tolist():
        rel, label = ds.samples[i]
        img_path = settings.raw_dir / "food-101" / rel
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        x = pred.to_tensor(bgr)
        x.requires_grad_(False)
        with torch.enable_grad():
            heat = cam(x)
        top = pred.predict(bgr, topk=1)[0]
        is_wrong = top[0] != CLASSES[label]
        want_wrong = wrong_saved < min_misclassified
        want_correct = correct_saved < n - min_misclassified
        if (is_wrong and want_wrong) or (not is_wrong and want_correct):
            overlay = overlay_heatmap(bgr, heat)
            tag = "WRONG" if is_wrong else "ok"
            name = f"{model_name}_{tag}_true-{CLASSES[label]}_pred-{top[0]}_{i}.png"
            cv2.imwrite(str(out_dir / name), overlay)
            paths.append(name)
            wrong_saved += is_wrong
            correct_saved += not is_wrong
        if wrong_saved >= min_misclassified and correct_saved >= n - min_misclassified:
            break
    cam.close()
    log.info("Grad-CAM: %d saved (%d misclassified)", len(paths), wrong_saved)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simple", "effnet"], default=None)
    ap.add_argument("--gradcam", type=int, default=0)
    args = ap.parse_args()

    names = [args.model] if args.model else [
        m for m in ("simple", "effnet")
        if (settings.models_dir / f"{m}_best.pt").exists()]
    if not names:
        raise SystemExit("No checkpoints found - train first")

    metrics_path = settings.reports_dir / "metrics.json"
    all_metrics: dict[str, dict] = {}
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            all_metrics = json.load(f)

    for name in names:
        log.info("Evaluating %s ...", name)
        all_metrics[name] = evaluate_model(name)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    if len(all_metrics) >= 2:
        comparison_chart(all_metrics)

    print(f"\n{'model':<10}{'top1':>8}{'top5':>8}{'macroF1':>9}{'params':>12}{'cpu ms':>8}")
    for n_, m in all_metrics.items():
        print(f"{n_:<10}{m['top1_acc']:>8.3f}{m['top5_acc']:>8.3f}"
              f"{m['macro_f1']:>9.3f}{m['params']:>12,}{m['cpu_ms_per_image']:>8.1f}")

    if args.gradcam:
        dump_gradcam_samples(names[-1], n=args.gradcam)


if __name__ == "__main__":
    main()
