# PROJECT_SPEC.md — FoodLens

> Full technical specification. Claude Code: implement strictly against this.

---

## 1. Problem statement

People systematically underestimate what they eat. Manual calorie tracking apps fail
because logging is tedious. **FoodLens** removes the friction: one photo → identified
dish → estimated portion → nutrition breakdown → conversational dietary advice.

**Why this needs all three disciplines (this is the academic justification):**

| Requirement | Why it cannot be removed |
|---|---|
| **Computer Vision** | Classification alone gives "pizza" but not *how much* pizza. Portion size drives calories. Segmentation + area estimation is the only way to get grams from a single photo without a depth sensor. |
| **CNN** | Food is fine-grained visual recognition (101 visually-similar classes). Classical features (SIFT/HOG + SVM) collapse here — we prove this empirically with a baseline. |
| **NLP** | The output must be *actionable*. Users type "bugün səhər 2 kruassan yedim", ask "axşam bunu yeyə bilərəm?", and need a daily plan. That requires entity extraction, retrieval over nutrition/guideline knowledge, and generation. |

---

## 2. System architecture

```
                 ┌──────────────────────────────────────────────┐
   photo ───────▶│  CV LAYER (OpenCV)                           │
                 │  1. quality gate: blur (Laplacian var),      │
                 │     brightness, min-resolution               │
                 │  2. plate detection: Hough circles / contour │
                 │  3. food mask: GrabCut seeded by plate ROI   │
                 │  4. portion: mask_area / plate_area → S/M/L  │
                 │     → grams via per-class density table      │
                 └───────────────┬──────────────────────────────┘
                                 │ cropped, masked food image
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │  CNN LAYER (PyTorch)                         │
                 │  A. SimpleCNN  (from scratch, baseline)      │
                 │  B. EfficientNet-B0 (transfer, production)   │
                 │  → top-1 class + confidence + top-5          │
                 │  → Grad-CAM heatmap (explainability)         │
                 └───────────────┬──────────────────────────────┘
                                 │ (class, confidence, grams)
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │  NUTRITION ENGINE                            │
                 │  nutrition_db.json → kcal, protein, carb,    │
                 │  fat, fiber, sugar scaled by grams           │
                 └───────────────┬──────────────────────────────┘
                                 │
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │  NLP LAYER                                   │
                 │  1. meal_parser: free text → entities        │
                 │     "2 dilim pizza və 1 kola"                │
                 │     → [(pizza, 2, slice), (cola, 1, glass)]  │
                 │  2. retriever: embed query, retrieve top-k   │
                 │     from nutrition_db + guidelines/*.md      │
                 │  3. advisor: RAG → Azerbaijani advice        │
                 │  4. summarizer: daily log → summary + plan   │
                 └───────────────┬──────────────────────────────┘
                                 │
                    SQLite (MealLog)  +  Streamlit UI
```

---

## 3. Data

### 3.1 Image data
- Source: `torchvision.datasets.Food101` (auto-download, 101 classes × 1000 imgs).
- **Use a 25-class subset** to fit the timeline. Chosen for visual diversity and
  relevance:

```python
CLASSES = [
    "pizza", "hamburger", "french_fries", "caesar_salad", "sushi",
    "steak", "fried_rice", "spaghetti_bolognese", "pancakes", "omelette",
    "grilled_salmon", "chicken_curry", "donuts", "cheesecake", "ice_cream",
    "hot_dog", "dumplings", "falafel", "greek_salad", "lasagna",
    "ramen", "waffles", "tacos", "guacamole", "club_sandwich",
]
```
- Split: Food101's official train/test. Carve 10% of train as val (seed 42).
- Train transforms: RandomResizedCrop(224), HorizontalFlip, ColorJitter(0.2), RandAugment(n=2,m=7), Normalize(ImageNet stats).
- Eval transforms: Resize(256) → CenterCrop(224) → Normalize.

### 3.2 Nutrition DB (`data/nutrition_db.json`)
One entry per class. Generate with `scripts/build_nutrition_db.py` (hand-authored
values, cite source in README — e.g. USDA FoodData Central averages).

```json
{
  "pizza": {
    "az_name": "Pitsa",
    "per_100g": {"kcal": 266, "protein_g": 11.0, "carb_g": 33.0, "fat_g": 10.0, "fiber_g": 2.3, "sugar_g": 3.6, "sodium_mg": 598},
    "typical_serving_g": 107,
    "serving_name_az": "dilim",
    "density_g_per_cm2": 1.1,
    "tags": ["fast_food", "high_sodium", "high_carb"],
    "notes_az": "Yüksək natrium və doymuş yağ. Nazik xəmirli və tərəvəzli variant daha yaxşıdır."
  }
}
```

### 3.3 Guidelines corpus (`data/guidelines/*.md`)
6–10 short Azerbaijani markdown docs used by the RAG advisor:
`kalori_balansi.md`, `zulal_ehtiyaci.md`, `natrium_ve_teziq.md`, `seker_ve_qlikemik_indeks.md`,
`idman_oncesi_qidalanma.md`, `arıqlama_prinsipleri.md`, `diabet_ucun_qeydler.md`, `lifli_qidalar.md`.

Each: 200–400 words, plain factual guidance, **must end with a disclaimer line** that this
is not medical advice.

---

## 4. CV layer — detailed

### `src/cv/quality.py`
```python
def check_quality(img: np.ndarray) -> QualityReport
```
- `blur_score` = variance of Laplacian. Reject if `< 100`.
- `brightness` = mean of V channel (HSV). Reject if `< 40` or `> 220`.
- `resolution` — reject if min side `< 224`.
- Returns `QualityReport(ok: bool, reasons_az: list[str], scores: dict)`.
- UI must show the Azerbaijani reason ("Şəkil bulanıqdır, yenidən çəkin").

### `src/cv/segment.py`
```python
def detect_plate(img) -> Optional[Circle]   # HoughCircles, fallback: largest contour
def segment_food(img, plate: Optional[Circle]) -> np.ndarray  # binary mask, uint8
```
- GrabCut, initialised with a rect = plate bbox shrunk 10% (or centre 70% if no plate).
- Post-process: morphological open+close (5×5), keep largest connected component.
- Return mask + `masked_crop` for the CNN.

### `src/cv/portion.py`
```python
def estimate_portion(mask, plate: Optional[Circle], food_class: str) -> PortionEstimate
```
- `coverage = mask_area / plate_area` (or `/ image_area` if no plate found).
- Bucket: `<0.25 → S`, `0.25–0.5 → M`, `>0.5 → L`.
- grams = `typical_serving_g × {S: 0.6, M: 1.0, L: 1.6}` — **and** if a plate was detected,
  refine using real-world scale: assume standard plate Ø 26 cm → cm²/px → area_cm² →
  `grams = area_cm² × density_g_per_cm2`.
- Return both estimates + `confidence` (low if no plate detected) + allow **manual override
  in the UI** (slider). Document the limitation honestly in the report — monocular portion
  estimation is inherently approximate; this is a strength in the discussion section, not a flaw.

---

## 5. CNN layer — detailed

### `src/cnn/models.py`
1. **SimpleCNN** (baseline, from scratch — proves you can build a CNN):
   - 4 conv blocks: `Conv(3→32→64→128→256, k=3, p=1)` each + BatchNorm + ReLU + MaxPool(2)
   - GlobalAvgPool → Dropout(0.4) → Linear(256 → 25)
   - ~1.2M params. Expect **~55–65% top-1**.
2. **FoodNet** (production): `torchvision.models.efficientnet_b0(weights=IMAGENET1K_V1)`,
   replace classifier head with `Linear(1280, 25)`.
   - Phase 1: freeze backbone, train head, 3 epochs, lr=1e-3.
   - Phase 2: unfreeze all, 7 epochs, lr=1e-4, cosine schedule.
   - Expect **~88–93% top-1**, ~98% top-5.

### `src/cnn/train.py`
CLI:
```bash
python -m src.cnn.train --model simple  --epochs 15 --bs 64
python -m src.cnn.train --model effnet  --epochs 10 --bs 32 --mixed-precision
```
- AdamW, label smoothing 0.1, early stopping (patience 3 on val macro-F1).
- Save best to `models/{model}_best.pt` with a metadata dict (classes, transforms, metrics).
- Write per-epoch history to `reports/history_{model}.json`.

### `src/cnn/evaluate.py`
Produces into `reports/`:
- `metrics.json`: top-1, top-5, macro-F1, per-class precision/recall/F1, params, inference ms/img (CPU).
- `confusion_matrix_{model}.png`
- `per_class_f1_{model}.png`
- `model_comparison.png` — SimpleCNN vs EfficientNet bar chart. **This chart goes in the slides.**

### `src/cnn/gradcam.py`
- Hook the last conv block, compute Grad-CAM, return a `jet` heatmap overlaid at alpha=0.4.
- `scripts`: dump 10 sample overlays to `reports/gradcam/`.
- Discussion point for the report: does the model look at the food or at the plate/background?

---

## 6. NLP layer — detailed

### `src/nlp/meal_parser.py`
Free-text → structured entities. **This is the NER component.**
```python
parse_meal("bugün nahara 2 dilim pitsa və bir stəkan kola içdim")
# -> [MealEntity(food="pizza", qty=2.0, unit="dilim", raw="2 dilim pitsa"),
#     MealEntity(food="cola",  qty=1.0, unit="stəkan", raw="bir stəkan kola")]
```
Implementation:
1. Tokenise; map Azerbaijani number words (`bir, iki, üç, yarım, bir neçə`) + digits → float.
2. Unit lexicon: `dilim, ədəd, boşqab, stəkan, qaşıq, porsiya, qram, kq`.
3. Food matching: embed the noun phrase with `all-MiniLM-L6-v2`, cosine-match against
   class names + `az_name` + synonyms; accept if `sim > 0.55`, else mark `unknown`.
4. Return list of `MealEntity`. Unknown foods must be surfaced to the user, not silently dropped.

Unit test with ≥15 Azerbaijani sentences, including tricky ones ("yarım boşqab plov", "3 qaşıq düyü").

### `src/nlp/retriever.py`
- Chunk `data/guidelines/*.md` (≈200 tokens, 40 overlap) + one synthetic doc per nutrition class.
- Embed all chunks once → cache to `models/rag_index.npz` (numpy, cosine search — no vector DB needed).
- `retrieve(query: str, k: int = 4) -> list[Chunk]`.

### `src/nlp/llm.py`
Provider abstraction, selected via `LLM_PROVIDER` env:
- `anthropic` — `claude-sonnet-4-6` via the `anthropic` SDK (needs `ANTHROPIC_API_KEY`).
- `local` — `google/flan-t5-base` via `transformers` (CPU, works offline).
- `template` — deterministic Jinja templates, **always works, zero deps**. This is the fallback
  so the demo can never fail during the defence.

`generate(system: str, user: str) -> str` — same signature for all three.

### `src/nlp/advisor.py`
```python
def advise(meal: MealAnalysis, user_profile: UserProfile) -> str
```
- Build the RAG context from `retriever.retrieve(...)`.
- Prompt (Azerbaijani output, enforced):
  > Sən qidalanma məsləhətçisisən. Aşağıdakı yeməyi və istifadəçi profilini nəzərə alaraq
  > 3–4 cümləlik konkret, praktiki məsləhət ver. Yalnız verilən konteksti istifadə et.
  > Tibbi diaqnoz qoyma. Cavabı Azərbaycan dilində ver.
- Must include: kcal vs daily target %, one concrete swap suggestion, one warning if any
  `tags` include `high_sodium`/`high_sugar`.
- Append a fixed disclaimer.

### `src/nlp/summarizer.py`
```python
def daily_summary(logs: list[MealLog], profile: UserProfile) -> DailySummary
```
→ Azerbaijani paragraph: total kcal vs target, macro split, what was over/under,
plus **tomorrow's 3-meal plan** generated from the class list within the remaining budget.

---

## 7. Persistence & API

`src/db.py` — SQLAlchemy: `User(id, name, age, sex, height_cm, weight_kg, activity, goal, daily_kcal_target)`,
`MealLog(id, user_id, ts, image_path, food_class, confidence, grams, kcal, protein_g, carb_g, fat_g, source)`.
Compute `daily_kcal_target` with Mifflin-St Jeor × activity factor, adjusted by goal.

`src/api.py` — FastAPI:
| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | multipart image → full `MealAnalysis` JSON |
| POST | `/log` | persist a MealLog |
| POST | `/parse` | free text → MealEntity list (NLP-only path) |
| POST | `/advise` | MealAnalysis + profile → advice text |
| GET | `/summary/{user_id}?date=` | daily summary + plan |
| GET | `/health` | ok |

---

## 8. Streamlit demo (`app/streamlit_app.py`)

Tabs (all labels in Azerbaijani):
1. **Şəkil analizi** — upload/camera → 4-panel view: orijinal | seqmentasiya maskası |
   Grad-CAM | nəticə kartı (sinif, əminlik, qram, kcal, makrolar) + porsiya slider-i (manual override)
   + "Məsləhət al" düyməsi → NLP advice.
2. **Mətnlə əlavə et** — text box → meal_parser → confirmed entities → log.
3. **Gündəlik** — bugünkü loglar, kcal progress bar, makro pie chart, "Günü yekunlaşdır" → summary + sabahkı plan.
4. **Model** — metrics table, confusion matrix, SimpleCNN vs EfficientNet comparison chart.
   (This tab exists purely to impress the examiners — it makes the ML work visible.)

---

## 9. Evaluation & report artefacts

Must exist in `reports/` at the end:
- `metrics.json`
- `confusion_matrix_effnet.png`, `confusion_matrix_simple.png`
- `model_comparison.png`
- `per_class_f1_effnet.png`
- `gradcam/*.png` (10 samples, incl. ≥2 failure cases)
- `portion_validation.md` — measure the portion estimator against 10 manually-weighed
  reference photos; report MAE in grams. Be honest about the error.
- `nlp_eval.md` — meal_parser accuracy on the 15+ test sentences; RAG retrieval hit-rate on 10 queries.

---

## 10. Definition of Done

- [ ] `pip install -r requirements.txt && python -m src.cnn.train --model effnet` runs clean
- [ ] `streamlit run app/streamlit_app.py` gives a working end-to-end demo
- [ ] `pytest` green
- [ ] `reports/metrics.json` shows both models
- [ ] README with architecture diagram, results table, limitations, how-to-run
- [ ] Grad-CAM overlays saved
- [ ] Every user-visible string is in Azerbaijani

---

## 11. Explicit non-goals

- Multi-food-per-plate detection (mention as future work).
- Depth-accurate volume estimation.
- Medical-grade accuracy. **Every output carries a disclaimer.**
- Mobile app.
