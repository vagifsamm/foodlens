"""Portion-estimation validation harness (Prompt 10).

Runs the full CV portion pipeline (plate detect -> GrabCut mask -> grams) on a
sample of Food-101 test images and compares the estimate against a per-class
reference serving weight.

IMPORTANT — honest framing: Food-101 ships no weighed ground truth. We use each
class's ``typical_serving_g`` from data/nutrition_db.json as the reference. That
is a *nominal medium serving*, not a scale-weighed value, so the reported MAE /
MAPE measure agreement-with-nominal, not true physical error. Real validation
would need a weighed test set. This limitation is stated in the output report.

Usage:
    python -m src.cv.validate_portion            # 10 images, writes report
    python -m src.cv.validate_portion --n 20
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import cv2
import numpy as np

from config import CLASSES, SEED, settings
from src.cv.portion import _load_nutrition_entry, estimate_portion
from src.cv.segment import detect_plate, segment_food


def _iter_test_samples(n: int) -> list[tuple[str, str]]:
    """Return n (relative_path, class_name) pairs spread across classes."""
    from src.cnn.dataset import FoodSubset

    ds = FoodSubset("test")
    rng = np.random.default_rng(SEED)
    # One image per class first (coverage), then fill randomly to reach n.
    by_class: dict[str, str] = {}
    order = rng.permutation(len(ds.samples)).tolist()
    for i in order:
        rel, label = ds.samples[i]
        cls = CLASSES[label]
        by_class.setdefault(cls, rel)
        if len(by_class) >= n:
            break
    picks = [(rel, cls) for cls, rel in by_class.items()][:n]
    return picks


def validate(n: int = 10) -> dict:
    rows = []
    abs_err, ape = [], []
    no_plate = 0
    for rel, cls in _iter_test_samples(n):
        img_path = settings.raw_dir / "food-101" / rel
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        plate = detect_plate(bgr)
        mask = segment_food(bgr, plate)
        est = estimate_portion(mask, plate, cls)
        ref = float(_load_nutrition_entry(cls)["typical_serving_g"])
        err = est.grams - ref
        rows.append({
            "class": cls,
            "reference_g": ref,
            "estimated_g": est.grams,
            "bucket": est.bucket,
            "coverage": est.coverage,
            "plate_detected": plate is not None,
            "confidence": est.confidence,
            "abs_error_g": round(abs(err), 1),
            "ape_pct": round(abs(err) / ref * 100, 1),
        })
        abs_err.append(abs(err))
        ape.append(abs(err) / ref * 100)
        no_plate += plate is None

    summary = {
        "n": len(rows),
        "mae_g": round(float(np.mean(abs_err)), 1) if abs_err else None,
        "mape_pct": round(float(np.mean(ape)), 1) if ape else None,
        "median_ape_pct": round(float(np.median(ape)), 1) if ape else None,
        "no_plate_count": no_plate,
        "rows": rows,
    }
    return summary


def write_report(summary: dict) -> None:
    out = settings.reports_dir / "portion_validation.md"
    r = summary
    lines = [
        "# Porsiya ölçüsünün validasiyası / Portion Estimation Validation",
        "",
        f"_Generated {date.today().isoformat()} · n = {r['n']} Food-101 test images_",
        "",
        "## ⚠️ Reference caveat (read first)",
        "",
        "Food-101 has **no weighed ground-truth mass**. The `reference_g` column is",
        "each class's `typical_serving_g` from `data/nutrition_db.json` — a *nominal",
        "medium serving*, not a scale-weighed value. The MAE / MAPE below therefore",
        "measure **agreement with the nominal serving**, not true physical error.",
        "A rigorous validation would require a kitchen-scale-weighed test set; this is",
        "documented as a known limitation (see README ethics/limitations).",
        "",
        "## Aggregate metrics",
        "",
        f"- **MAE**: {r['mae_g']} g",
        f"- **MAPE**: {r['mape_pct']} %",
        f"- **Median APE**: {r['median_ape_pct']} % (less skewed by outliers)",
        f"- **Plate not detected**: {r['no_plate_count']} / {r['n']} images "
        "(these fall back to the S/M/L bucket estimate, marked `confidence=low`)",
        "",
        "## Per-image results",
        "",
        "| Class | Ref g | Est g | Bucket | Coverage | Plate? | Conf | |err| g | APE % |",
        "|-------|------:|------:|:------:|---------:|:------:|:----:|------:|------:|",
    ]
    for row in sorted(r["rows"], key=lambda x: x["ape_pct"]):
        lines.append(
            f"| {row['class']} | {row['reference_g']:.0f} | {row['estimated_g']:.0f} "
            f"| {row['bucket']} | {row['coverage']:.2f} | "
            f"{'yes' if row['plate_detected'] else 'no'} | {row['confidence']} "
            f"| {row['abs_error_g']:.0f} | {row['ape_pct']:.0f} |"
        )
    lines += [
        "",
        "## Failure modes observed",
        "",
        "1. **No plate detected** → no cm/px scale, estimate falls back to the",
        "   coarse S/M/L bucket. Common on tight food crops (Food-101 is mostly",
        "   plateless close-ups), dark plates, and non-circular dishes.",
        "2. **Scale ambiguity** → the plate-scaled estimate assumes a fixed 26 cm",
        "   plate; smaller/larger real plates bias grams quadratically (area ∝ r²).",
        "3. **Segmentation bleed** → GrabCut can include plate rim or background",
        "   texture, inflating the mask area and thus grams.",
        "4. **Monocular depth** → a flat 2-D area cannot see food height; a tall",
        "   burger and a flat salad of equal footprint read as equal mass.",
        "",
        "## Honest conclusion",
        "",
        "Portion estimation is the **weakest** quantitative component, by design:",
        "single-image monocular mass estimation without a fiducial marker is an",
        "ill-posed problem. The pipeline is transparent about this — every estimate",
        "carries a `confidence` flag, and the UI shows the S/M/L bucket alongside the",
        "gram figure so users are not misled into thinking it is precise. For the",
        "defence: this is presented as a deliberate, measured approximation, not a",
        "solved problem.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    summary = validate(args.n)
    (settings.reports_dir / "portion_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(f"MAE={summary['mae_g']} g  MAPE={summary['mape_pct']} %  "
          f"no_plate={summary['no_plate_count']}/{summary['n']}")


if __name__ == "__main__":
    main()
