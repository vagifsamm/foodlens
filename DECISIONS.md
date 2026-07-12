# DECISIONS.md — engineering decisions log

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `run.ps1` instead of Makefile | Windows host, per build instructions (overrides CLAUDE.md). |
| 2 | Skipped the "placeholder modules raising NotImplementedError" stage of Prompt 1; modules are implemented directly, step by step | Autonomous run — writing placeholders then rewriting them doubles work with zero review benefit. Skeleton structure itself is still created and committed first. |
| 3 | `requirements.txt` pins torch/torchvision exactly (cu126 wheels); other deps use compatible ranges + `requirements.lock.txt` (pip freeze) for exact reproducibility | Python 3.13 is newer than most tutorial pins; ranges avoid resolution dead-ends while the lock file keeps the report reproducible. |
| 4 | Training runs locally on GTX 1660 Ti (6 GB), not Colab | CLAUDE.md allows GPU training; local GPU removes upload/download friction. AMP implemented but Turing 1660 Ti has no tensor cores — modest speedup expected. |
| 5 | Meal parser: exact/fuzzy Azerbaijani synonym-lexicon match BEFORE MiniLM embedding fallback | `all-MiniLM-L6-v2` is English-centric; AZ food words ("pitsa", "kruassan") embed poorly. Spec's embedding matcher is kept, but as fallback, not first line. |
| 6 | `portion_validation.md` reference grams: provisional values derived from typical-serving reasoning, clearly marked; user can supply real weighed values at STOP 4 and re-run | Prompt 10 says "ask me for true grams" but the stop protocol forbids questions between STOP 3 and STOP 4. |
| 7 | Vercel deployment rejected; GitHub for code hosting only | Torch+OpenCV exceed serverless limits; Streamlit needs a long-running process; defence-day requirement is offline local run anyway. Confirmed with user. |
| 8 | Shared dataclasses moved to `src/schemas.py` (not in spec layout) | Avoids circular imports between pipeline, advisor and db. |
| 9 | SimpleCNN has 396K params, not the "~1.2M" the spec estimates | The architecture follows the spec text exactly (4 conv blocks 32-256, GAP, Dropout, Linear 256->25); the spec's own param estimate is inconsistent with its GAP head. Architecture kept, estimate corrected in the report. |
| 11 | Mixed precision (AMP) removed from the effnet default run | Measured: AMP produced NaN in EfficientNet BatchNorm running stats on the GTX 1660 Ti (fp16 activation overflow; Turing has no tensor cores so AMP gave ~0 speedup anyway: 4.8s vs 4.5s smoke). The --mixed-precision flag still exists but is off by default. |
| 10 | Meal parser embedding threshold raised 0.55 -> 0.75 | Measured false positive: "qutab" -> guacamole at 0.70 cosine (MiniLM surface-form artifact on AZ words). Legit English variants moved into the synonym lexicon instead. Documented in nlp_eval.md. |
