---
name: apex-review
description: >
  Customised code review for APEX that checks the current diff against the project's
  hard contracts: no silent failure, no normalisation leakage, fixed seeds, mandatory
  confidence gates, error-code discipline, and config-driven thresholds. Invoke as
  /apex-review [feature-id] before committing. Complements the built-in /code-review
  (general bug hunt) and /security-review — run those too for anything user-facing.
argument-hint: [feature-id]
---

Review the current changes for feature **$ARGUMENTS** (default: the working diff).

First gather context: run `git diff` (and `git diff --staged`) to see what changed,
and read the matching spec in `specs/` for the contracts this code must honour.

Then check the APEX contract, reporting each as PASS / FAIL / NEEDS REVIEW with
file:line evidence. Order findings by severity (leakage and silent failure first):

1. **No silent failure.** No bare `except`, no `except Exception: pass`, no lap/row
   dropped without a logged error code, no `return None` swallowing an error. Every
   failure path returns a typed result with an error code and writes to
   `data_quality_log` where the spec requires it.
2. **No normalisation/scaler leakage.** Any `.fit()`/`.fit_transform()` touches
   training data only; held-out circuits never enter a fit path; scalers loaded from
   `artefacts/`. If the diff touches ML, delegate a deeper pass to the **ml-auditor**
   subagent and fold its findings in.
3. **Reproducibility.** Every randomness source seeds from `config.yaml`. New
   training touches MLflow logging.
4. **Confidence gates present** on every model output path (not just happy path).
5. **No magic numbers.** Thresholds/ranges imported from `config.yaml`; error-code
   strings come from the errors Enum.
6. **Failsafe coverage.** For modules with documented failure modes, delegate to the
   **failsafe-auditor** subagent and include its failure-mode table.
7. **Tests.** Each new acceptance criterion and each failure mode has a test that
   exercises it. Vectorised pandas/numpy (no per-row Python loops). Type hints +
   Pydantic on boundaries.

Finish with a ranked, numbered fix list. For each item give the file, the line, the
contract violated, and the minimal change. Do not apply fixes unless asked — output
the review only.
