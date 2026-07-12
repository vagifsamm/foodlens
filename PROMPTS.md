# PROMPTS.md — Claude Code addım-addım prompt zənciri (FoodLens)

> Qayda: **hər prompt-u ayrıca göndər**, nəticəni yoxla, sonra növbətiyə keç.
> Hamısını bir yerə yapışdırma — Claude Code kontekstini itirir və keyfiyyət düşür.
> Hər addımdan sonra `git commit` et.

---

## Hazırlıq (terminal)

```bash
mkdir foodlens && cd foodlens
git init
# CLAUDE.md və PROJECT_SPEC.md fayllarını bu qovluğa kopyala
claude
```

---

## PROMPT 0 — Kontekst yükləmə

```
Read CLAUDE.md and PROJECT_SPEC.md in this directory carefully.

Then, before writing any code:
1. Summarise back to me the architecture in 10 bullet points.
2. List every assumption you are making that could be wrong.
3. Flag anything in the spec that you think will NOT fit in a 2-day timeline,
   and propose the minimal cut.

Do not create any files yet. Wait for my go-ahead.
```

➡️ Cavabı oxu. Razılaşmadığın yer varsa düzəlt, sonra "go ahead" yaz.

---

## PROMPT 1 — Skelet + konfiqurasiya

```
Scaffold the repository exactly as specified in CLAUDE.md.

Create:
- The full directory tree with __init__.py files
- requirements.txt (pinned versions, CPU-friendly torch)
- .env.example and config.py using pydantic-settings
- .gitignore (data/raw, models/*.pt, __pycache__, .env)
- A Makefile with: install, data, train-simple, train-effnet, evaluate, api, demo, test
- Empty placeholder modules with correct function signatures and docstrings,
  each raising NotImplementedError

Do NOT implement logic yet. I want to review the skeleton first.
Then run `python -c "import src"` to prove it imports cleanly.
```

---

## PROMPT 2 — Data pipeline

```
Implement the data layer:

1. scripts/prepare_data.py — downloads torchvision Food101, filters to the 25 CLASSES
   from PROJECT_SPEC section 3.1, builds train/val/test splits (10% of train -> val, seed 42),
   and prints a class-distribution table.

2. src/cnn/dataset.py — Dataset + DataLoader factories with the exact train/eval transforms
   from the spec. Include a `get_dataloaders(batch_size, num_workers)` function.

3. scripts/build_nutrition_db.py — generates data/nutrition_db.json with a full entry for
   all 25 classes in the schema from spec section 3.2. Use realistic USDA-based values.
   Azerbaijani names and notes must be correct and natural — I am a native speaker and will check.

Run prepare_data.py and show me the class distribution + 1 sample batch shape.
```

---

## PROMPT 3 — CNN modelləri + training

```
Implement src/cnn/models.py and src/cnn/train.py per spec section 5.

- SimpleCNN: exactly the architecture in the spec. Print param count.
- FoodNet: EfficientNet-B0 transfer, two-phase (freeze head -> unfreeze all).
- train.py: CLI with --model, --epochs, --bs, --lr, --mixed-precision.
  AdamW, label smoothing 0.1, cosine schedule, early stopping on val macro-F1 (patience 3).
  Save best checkpoint + metadata to models/{model}_best.pt.
  Save per-epoch history to reports/history_{model}.json.
  Show a tqdm progress bar with live train/val loss and acc.

Before starting any real training run, do a 1-epoch smoke test on 200 images
and show me the output. Do NOT launch the full run without telling me the ETA.
```

➡️ Smoke test keçəndən sonra:
```
Now run the full training for --model simple.
```
sonra
```
Now run the full training for --model effnet.
```

---

## PROMPT 4 — Qiymətləndirmə + Grad-CAM

```
Implement src/cnn/evaluate.py and src/cnn/gradcam.py per spec section 5.

evaluate.py must write to reports/:
- metrics.json (top-1, top-5, macro-F1, per-class P/R/F1, param count, CPU ms/image)
- confusion_matrix_{model}.png (normalised, readable class labels)
- per_class_f1_{model}.png
- model_comparison.png (SimpleCNN vs EfficientNet: accuracy, F1, params, latency)

gradcam.py: manual hooks on the last conv block, jet overlay at alpha=0.4.
Dump 10 samples to reports/gradcam/, and make sure at least 2 of them are
MISCLASSIFIED examples — I want failure cases for the report.

Run both models through evaluate.py and print the comparison table to the terminal.
```

---

## PROMPT 5 — CV qatı

```
Implement the full CV layer per spec section 4:
- src/cv/quality.py (blur via Laplacian variance, brightness, resolution; Azerbaijani reasons)
- src/cv/segment.py (HoughCircles plate detection + GrabCut food mask + morphological cleanup)
- src/cv/portion.py (coverage -> S/M/L, plus plate-scale refinement -> grams)

Requirements:
- Every function must handle the "no plate detected" fallback gracefully.
- Return dataclasses (QualityReport, PortionEstimate), not raw tuples.
- Write tests/test_cv.py with synthetic images (a blurred one, a dark one, a plate with
  a coloured blob) asserting expected behaviour.

Then create a debug script that takes any image and saves a 4-panel figure:
original | plate detection | food mask | masked crop.
Run it on 3 Food101 test images and show me the results.
```

---

## PROMPT 6 — NLP: meal parser (NER)

```
Implement src/nlp/meal_parser.py per spec section 6.

- Azerbaijani number words + digits -> float ("bir", "iki", "üç", "yarım", "bir neçə")
- Unit lexicon: dilim, ədəd, boşqab, stəkan, qaşıq, porsiya, qram, kq
- Food matching via sentence-transformers (all-MiniLM-L6-v2) cosine similarity against
  class names + az_name + a synonym list. Threshold 0.55. Unknown -> flagged, never dropped.
- Return list[MealEntity] dataclasses.

Write tests/test_nlp.py with at least 15 Azerbaijani test sentences, including:
"yarım boşqab plov", "3 qaşıq düyü yedim", "2 dilim pitsa və bir stəkan kola",
"səhər omlet yedim", "heç nə yeməmişəm".

Run the tests and show me the pass rate + any failures with explanations.
```

---

## PROMPT 7 — NLP: RAG + advisor + summarizer

```
Implement:

1. data/guidelines/ — write 8 Azerbaijani markdown docs (200-400 words each) as listed in
   spec section 3.3. Factual, plain language, each ending with a not-medical-advice disclaimer.

2. src/nlp/retriever.py — chunk + embed + cache to models/rag_index.npz, cosine top-k retrieval.

3. src/nlp/llm.py — provider abstraction: anthropic | local (flan-t5-base) | template.
   The `template` provider must produce genuinely useful Azerbaijani advice with zero
   external deps — this is the demo-day safety net. Default provider = template unless
   ANTHROPIC_API_KEY is set.

4. src/nlp/advisor.py and src/nlp/summarizer.py per spec section 6.
   Output language: Azerbaijani, always. Always append the disclaimer.

Test: run advise() for a 320g pizza with a 2000 kcal target user, using all three
providers, and show me the three outputs side by side.
```

---

## PROMPT 8 — Pipeline + DB + API

```
Implement src/pipeline.py, src/db.py and src/api.py per spec sections 6-7.

pipeline.analyze(image) must chain: quality gate -> segment -> portion -> CNN predict
-> Grad-CAM -> nutrition lookup -> return a MealAnalysis dataclass with everything the UI needs.
If the quality gate fails, return early with the Azerbaijani reason.

db.py: SQLAlchemy models + Mifflin-St Jeor daily target calculation.
api.py: all endpoints from the spec table, with pydantic response models.

Write tests/test_pipeline.py — end-to-end on one real Food101 image, asserting the
MealAnalysis has non-null class, grams, kcal and advice.

Start the API and show me the /docs endpoint output for POST /analyze.
```

---

## PROMPT 9 — Streamlit demo

```
Build app/streamlit_app.py per spec section 8. Four tabs, all Azerbaijani labels.

Design requirements:
- Clean, modern, NOT default-Streamlit-looking. Custom CSS, a coherent colour palette
  (warm food-appropriate tones), card-based result layout, generous spacing.
- Tab 1: upload OR camera input -> 4-panel visual (orijinal / maska / Grad-CAM / nəticə kartı),
  portion override slider, "Məsləhət al" button.
- Tab 2: free-text meal entry -> parsed entities shown as editable chips -> confirm -> log.
- Tab 3: today's log, kcal progress bar, macro donut chart, "Günü yekunlaşdır" button.
- Tab 4: model metrics table + confusion matrix + model comparison chart.
- Sidebar: user profile (age, sex, height, weight, activity, goal) -> computes kcal target.
- Graceful loading states and error messages, in Azerbaijani.

Run it and confirm every tab works end-to-end.
```

---

## PROMPT 10 — Validasiya artefaktları (bunlar hesabat üçün kritikdir)

```
Produce the evaluation artefacts from spec section 9:

1. reports/portion_validation.md — take 10 Food101 test images, have me provide the
   "true" grams (ask me for them), compute the estimator's MAE and MAPE, and write an
   honest analysis of when it fails (no plate, top-down vs angled, dark plates).

2. reports/nlp_eval.md — meal_parser accuracy on the test sentences, RAG retrieval
   hit-rate on 10 hand-written queries with expected source docs.

3. notebooks/results.ipynb — all charts assembled, ready to screenshot for slides.

Be brutally honest in the analysis sections. Examiners reward acknowledged limitations
far more than they punish them.
```

---

## PROMPT 11 — README + təqdimat materialı

```
Write README.md:
- Azerbaijani section first, English section after
- Architecture diagram in Mermaid
- Results table: SimpleCNN vs EfficientNet (accuracy, F1, params, latency)
- Sample screenshots (reference the reports/ images)
- Full setup + run instructions, verified by actually running them from scratch
- Limitations & future work
- Ethics note: not medical advice, portion estimation is approximate, dataset bias
  (Food-101 is Western-food heavy — say so explicitly)

Then write PRESENTATION_OUTLINE.md — 12 slides in Azerbaijani:
problem, why CV+CNN+NLP are all necessary, architecture, dataset, CNN comparison,
Grad-CAM findings, CV portion method + its error, NLP layer, live demo, results,
limitations, future work. One line of speaker notes per slide.
```

---

## PROMPT 12 — Yekun keyfiyyət yoxlaması

```
Final review pass:

1. Run pytest, ruff, and mypy. Fix everything.
2. Verify the Definition of Done checklist in PROJECT_SPEC section 10, item by item.
   Report each as PASS/FAIL with evidence.
3. Do a clean-clone test: what breaks if someone clones this repo and follows the README?
4. Look for anything an examiner could attack: hardcoded values, leaked test data,
   unrealistic claims in the README, missing disclaimers. List them and fix them.
5. Tell me the 3 weakest parts of this project and how I should answer if asked about them
   in the defence.
```

---

## Vaxt bölgüsü (2 gün)

| Vaxt | İş | Prompt |
|---|---|---|
| Gün 1, 09:00–10:00 | Setup, skelet, data | 0, 1, 2 |
| Gün 1, 10:00–13:00 | CNN train (SimpleCNN + EfficientNet fonda işləyir) | 3 |
| Gün 1, 13:00–15:00 | CV qatı (training fonda gedərkən) | 5 |
| Gün 1, 15:00–17:00 | Evaluate + Grad-CAM | 4 |
| Gün 1, 17:00–19:00 | NLP parser | 6 |
| Gün 2, 09:00–12:00 | RAG + advisor + summarizer | 7 |
| Gün 2, 12:00–14:00 | Pipeline + API | 8 |
| Gün 2, 14:00–17:00 | Streamlit demo | 9 |
| Gün 2, 17:00–19:00 | Validasiya, README, slaydlar, final review | 10, 11, 12 |

---

## Vacib xırdalıqlar

- **Training-i fonda işlət**, sən isə paralel olaraq CV/NLP prompt-larını sür. Claude Code
  ayrı terminal sessiyasında işləyə bilər.
- **`LLM_PROVIDER=template` defolt qalsın.** Müdafiə günü internet/API problemi olarsa demo dayanmasın.
- **Grad-CAM-da səhv nümunələr mütləq olsun** — komissiya "modelin zəif tərəfi nədir?" soruşacaq,
  sən artıq cavabı hazır göstərəcəksən.
- **Porsiya xətasını gizlətmə.** MAE-ni açıq yaz və niyə belə olduğunu izah et — bu, layihəni
  zəiflətmir, əksinə mühəndis yetkinliyi göstərir.
- Hər prompt-dan sonra `git commit -m "..."` — geri qayıtmaq lazım olsa xilas edər.
