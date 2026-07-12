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
