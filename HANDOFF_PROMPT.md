# HANDOFF PROMPT — copy everything below this line into Claude Code

---

You are continuing **FoodLens**, an AI Engineering course final project that is
already ~80% built. Work autonomously; do not ask me to write code.

## Get the code

```
git clone https://github.com/vagifsamm/foodlens.git
cd foodlens
```

## Read these first, fully, in this order

1. `CLAUDE.md` — binding project rules (tech stack, layout, coding standards)
2. `PROJECT_SPEC.md` — the complete technical specification
3. `PROGRESS.md` — live checklist: exactly what is done and what remains
4. `DECISIONS.md` — 11 logged engineering decisions; do NOT re-litigate them
5. `PROMPTS.md` — original build order (Prompts 0-12); remaining work maps to Prompts 10-12
6. `BASLA.md` — how the owner runs the demo (one-click launchers)

## Current state (verified working)

- All code layers are implemented and tested: CV (OpenCV quality gate, HoughCircles
  plate detection, GrabCut segmentation, portion-to-grams), CNN (SimpleCNN from
  scratch + EfficientNet-B0 transfer, Grad-CAM via manual hooks), NLP (Azerbaijani
  meal parser/NER, hybrid RAG retriever, advisor, summarizer, 3-provider LLM
  abstraction with `template` as the zero-dependency default), FastAPI, SQLite,
  Streamlit demo (4 tabs, all Azerbaijani, custom styling).
- Tests: 38 passed, 2 skipped (the 2 need trained checkpoints). Run: `pytest tests/ -q`
- Data: Food101 25-class subset (16,875 train / 1,875 val / 6,250 test),
  `data/nutrition_db.json` complete for all 25 classes.
- `reports/nlp_eval.md` exists: parser 20/20, RAG hit-rate 8/10 (honest misses documented).

## Check before doing anything: are trained checkpoints present?

Look for `models/simple_best.pt` and `models/effnet_best.pt` in the repo.
- **If present**: training is done, skip to "Remaining work".
- **If absent**: train first. On a CUDA GPU (~1h total):
  `python -m src.cnn.train --model simple --epochs 15 --bs 64 --num-workers 4`
  then `python -m src.cnn.train --model effnet --epochs 10 --bs 32 --num-workers 4`.
  On CPU, reduce the dataset (edit `limit` in dataloaders) to stay under ~60 min.
  Do NOT use `--mixed-precision` (see DECISIONS.md #11 — it NaNs BatchNorm on GTX 16xx).

## Remaining work, in order (details in PROMPTS.md, Prompts 10-12)

1. **Evaluate both models**: `python -m src.cnn.evaluate --gradcam 10`
   -> `reports/metrics.json`, confusion matrices, per-class F1, `model_comparison.png`,
   10 Grad-CAM overlays **including at least 2 misclassified examples** (the evaluate
   script handles this automatically).
2. **`reports/portion_validation.md`** (Prompt 10): run the portion estimator on 10
   Food101 test images, compare against reference grams (ask the owner for
   real weighed values, or clearly mark provisional estimates), report MAE and MAPE
   honestly, analyse failure modes (no plate detected, angled shots, dark plates).
3. **`notebooks/results.ipynb`**: assemble all charts from `reports/` for the slides.
4. **`README.md`** (Prompt 11): Azerbaijani section first, then English. Mermaid
   architecture diagram, results table (SimpleCNN vs EfficientNet), setup/run
   instructions verified from scratch, limitations, ethics note (not medical advice,
   portion estimation approximate, Food-101 Western-food bias).
5. **`PRESENTATION_OUTLINE.md`** (Prompt 11): 12 slides in Azerbaijani with one line
   of speaker notes each.
6. **Final review** (Prompt 12): run `pytest`, `ruff check .`, `mypy` and fix issues;
   verify the Definition of Done checklist in PROJECT_SPEC.md section 10 item by item
   with PASS/FAIL evidence; do a clean-clone test; list the 3 weakest parts of the
   project and how to defend them to examiners.

## Non-negotiable rules

- Every user-facing string and generated advice: **Azerbaijani**. Code/comments/commits: English.
- `LLM_PROVIDER=template` stays the default — the demo must run offline with zero API keys.
- Never replace the CNN/CV/NLP layers with API shortcuts; the grade depends on them
  being visibly real.
- Report errors honestly (portion MAE, RAG misses); examiners reward acknowledged limitations.
- Seeds stay at 42. `git commit` after each step.
- Windows host uses `run.ps1` (not Makefile); on macOS/Linux just call the same
  `python -m ...` commands directly.
- If something is ambiguous: make the reasonable choice, log it in `DECISIONS.md`, keep moving.
