# Porsiya ölçüsünün validasiyası / Portion Estimation Validation

_Generated 2026-07-15 · n = 12 Food-101 test images_

## ⚠️ Reference caveat (read first)

Food-101 has **no weighed ground-truth mass**. The `reference_g` column is
each class's `typical_serving_g` from `data/nutrition_db.json` — a *nominal
medium serving*, not a scale-weighed value. The MAE / MAPE below therefore
measure **agreement with the nominal serving**, not true physical error.
A rigorous validation would require a kitchen-scale-weighed test set; this is
documented as a known limitation (see README ethics/limitations).

## Aggregate metrics

- **MAE**: 158.7 g
- **MAPE**: 131.2 %
- **Median APE**: 78.1 % (less skewed by outliers)
- **Plate not detected**: 1 / 12 images (these fall back to the S/M/L bucket estimate, marked `confidence=low`)

## Per-image results

| Class | Ref g | Est g | Bucket | Coverage | Plate? | Conf | |err| g | APE % |
|-------|------:|------:|:------:|---------:|:------:|:----:|------:|------:|
| club_sandwich | 230 | 246 | L | 0.52 | yes | high | 16 | 7 |
| spaghetti_bolognese | 300 | 220 | M | 0.46 | yes | high | 80 | 26 |
| fried_rice | 198 | 119 | S | 0.23 | no | low | 79 | 40 |
| omelette | 122 | 174 | M | 0.36 | yes | high | 52 | 43 |
| sushi | 200 | 92 | S | 0.22 | yes | high | 108 | 54 |
| lasagna | 250 | 434 | L | 0.74 | yes | high | 184 | 74 |
| falafel | 102 | 186 | M | 0.44 | yes | high | 84 | 82 |
| hamburger | 226 | 555 | L | 0.70 | yes | high | 329 | 145 |
| cheesecake | 125 | 413 | L | 0.78 | yes | high | 288 | 230 |
| pizza | 107 | 366 | L | 0.63 | yes | high | 259 | 242 |
| waffles | 75 | 309 | L | 0.73 | yes | high | 234 | 312 |
| donuts | 60 | 250 | L | 0.67 | yes | high | 190 | 318 |

## Failure modes observed

1. **No plate detected** → no cm/px scale, estimate falls back to the
   coarse S/M/L bucket. Common on tight food crops (Food-101 is mostly
   plateless close-ups), dark plates, and non-circular dishes.
2. **Scale ambiguity** → the plate-scaled estimate assumes a fixed 26 cm
   plate; smaller/larger real plates bias grams quadratically (area ∝ r²).
3. **Segmentation bleed** → GrabCut can include plate rim or background
   texture, inflating the mask area and thus grams.
4. **Monocular depth** → a flat 2-D area cannot see food height; a tall
   burger and a flat salad of equal footprint read as equal mass.

## Honest conclusion

Portion estimation is the **weakest** quantitative component, by design:
single-image monocular mass estimation without a fiducial marker is an
ill-posed problem. The pipeline is transparent about this — every estimate
carries a `confidence` flag, and the UI shows the S/M/L bucket alongside the
gram figure so users are not misled into thinking it is precise. For the
defence: this is presented as a deliberate, measured approximation, not a
solved problem.
