# CLAUDE.md — FoodLens

Project rules for Claude Code. Read this before every task.

## Project

**FoodLens** — an AI system that takes a photo of a meal and returns:
1. What food it is (CNN classification)
2. How much of it there is (CV-based portion estimation)
3. Calories + macros (nutrition lookup)
4. Personalised nutrition advice + daily meal plan (NLP/RAG + LLM)

This is a **university final project for an AI Engineering course**. The grading
requires visible, non-trivial use of **Computer Vision, CNN, and NLP**. Therefore:

> **Never replace a required component with an API shortcut.**
> The CNN must be trained in this repo. The CV pipeline must be real OpenCV code.
> The NLP layer must include a parser + retrieval, not just "send text to an LLM".

## Hard constraints

- Timeline: **2 days**. Prefer working code over perfect code.
- Must run on a laptop CPU for **inference**. Training may use Colab/GPU but must
  also complete on CPU within ~60 min on the reduced dataset.
- No paid dataset. No manual annotation.
- Deterministic seeds everywhere (`seed=42`) so results are reproducible in the report.

## Tech stack (do not swap without asking)

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| CV | OpenCV (`opencv-python`) |
| CNN | PyTorch + torchvision |
| Dataset | `torchvision.datasets.Food101` (subset of 25 classes) |
| Explainability | Grad-CAM (implemented manually with hooks, no heavy lib) |
| NLP embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| NLP generation | Pluggable: `anthropic` API → `flan-t5-base` local → template fallback |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Storage | SQLite via SQLAlchemy |
| Config | `pydantic-settings` + `.env` |

## Repo layout (keep it exactly like this)

```
foodlens/
├── CLAUDE.md
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── data/
│   ├── nutrition_db.json        # per-class nutrition facts (per 100g + typical serving)
│   ├── guidelines/              # .md docs used by the RAG advisor
│   └── raw/                     # Food101 downloads (gitignored)
├── src/
│   ├── __init__.py
│   ├── cv/
│   │   ├── quality.py           # blur / brightness / "is there food?" checks
│   │   ├── segment.py           # plate detection + GrabCut food mask
│   │   └── portion.py           # mask area -> portion size (S/M/L) -> grams
│   ├── cnn/
│   │   ├── dataset.py           # Food101 subset, transforms, dataloaders
│   │   ├── models.py            # SimpleCNN (from scratch) + EfficientNet-B0 transfer
│   │   ├── train.py             # CLI: --model simple|effnet
│   │   ├── evaluate.py          # top-1/top-5, macro-F1, confusion matrix, per-class
│   │   ├── gradcam.py           # Grad-CAM heatmaps
│   │   └── predict.py           # single-image inference wrapper
│   ├── nlp/
│   │   ├── meal_parser.py       # NER: free text -> [(food, quantity, unit)]
│   │   ├── retriever.py         # embed + retrieve from nutrition_db + guidelines
│   │   ├── advisor.py           # RAG prompt -> advice text
│   │   ├── summarizer.py        # daily log -> natural-language summary + next-day plan
│   │   └── llm.py               # provider abstraction (anthropic | local | template)
│   ├── pipeline.py              # end-to-end: image -> CV -> CNN -> nutrition -> NLP
│   ├── db.py                    # SQLAlchemy models: User, MealLog
│   └── api.py                   # FastAPI endpoints
├── app/
│   └── streamlit_app.py         # demo UI (Azerbaijani strings)
├── scripts/
│   ├── prepare_data.py
│   └── build_nutrition_db.py
├── notebooks/
│   └── results.ipynb            # charts for the report
├── models/                      # saved .pt checkpoints (gitignored except .gitkeep)
├── reports/                     # confusion matrix, gradcam samples, metrics.json
└── tests/
    ├── test_cv.py
    ├── test_nlp.py
    └── test_pipeline.py
```

## Coding rules

- Type hints on every public function. Google-style docstrings.
- No hardcoded paths — everything from `config.py`.
- Every module must be importable and runnable standalone (`if __name__ == "__main__"` smoke test).
- Log with `logging`, not `print` (except CLI output).
- Write the test **in the same commit** as the feature.
- If a step would take >30 min of compute, stop and tell me first.

## UI / language rules

- **All user-facing strings in the Streamlit app and all generated advice: Azerbaijani.**
- Code, comments, docstrings, commit messages: English.
- README: Azerbaijani first, English section after.

## What "done" means

A `make demo` (or documented command) that:
1. loads the trained EfficientNet checkpoint,
2. accepts a food photo,
3. shows: segmentation mask + Grad-CAM overlay + predicted class + estimated grams
   + calories/macros + Azerbaijani advice text,
4. logs the meal to SQLite and can produce a daily summary.

Plus `reports/metrics.json` comparing SimpleCNN vs EfficientNet.
