# Feature F20 — Evaluation Framework (the full §11 run — four pillars + RQ verdicts)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.

**Status:** SPEC WRITTEN
**Master spec sections:** §11 **Evaluation Framework** in full — §11.1 (Classifier), §11.2 (Fingerprint
Stability), §11.3 (Degradation Detector), §11.4 (Failsafe / RQ5); §11 intro ("Every evaluation reports
failure cases alongside success metrics… equal space to where APEX fails"); §2.4 (the five research
questions RQ1–RQ5); §3.2 (held-out set — Belgium, Italy — "used only for final generalisation
evaluation"); §5.1 (classifier targets + the three baselines + normalisation ablation); §13 Project Plan
**Week 20** (deliverable: "Full evaluation run. Classifier metrics. Fingerprint stability. Degradation
validation. Failsafe tests. **All results tables completed. All research questions answered.** … Complete
evaluation chapter results. **Every RQ answered with data.**"); §15 (contribution + limitations statement).
**Layer / Week:** Evaluation & writing / Wk 20
**Depends on:** **F06–F15** (INDEX build order). Consumes read-only (calls or harvests, **refits
nothing**): **F06a** `ClassifierTrainingResult` (§11.1 held-out metrics + baselines + normalisation
ablation, computed at train time); **F06b** `load_model`/`predict` (optional artefact-gated re-score);
**F07** `season_stability` (§11.2 year-on-year Jensen-Shannon); **F08** `stability_report` /
`teammate_proximity` / `project_unknown` (§11.2); **F09** `run_ablation` (corner-type ablation); **F10**
`lead_time_report` (§11.3); **F15** `run_failsafe_suite` + `FailsafeReport` (§11.4). **All error codes
pre-seeded; F20 adds none.**

## 1. Purpose
F20 is APEX's **evaluation chapter made executable**: a single orchestrator that runs the master's four
§11 evaluation pillars, gathers every metric against its documented target, and maps the results onto the
five research questions — producing the Week-20 deliverable ("all results tables completed, every RQ
answered with data"). It **reimplements no metric maths**: it *calls* the read-only evaluation surfaces the
earlier features already shipped (`run_failsafe_suite`, `lead_time_report`, `stability_report`,
`teammate_proximity`, `project_unknown`, `season_stability`, `run_ablation`) and *harvests* the classifier's
held-out §11.1 numbers from the champion `ClassifierTrainingResult`. Faithful to the constitution's
non-negotiable #1 and §11's "report failure cases alongside success", **every pillar that cannot run
(missing artefact, missing held-out data) returns an explicit `skipped` state with a reason — never a
fabricated number** — and every metric carries its pass/fail verdict against the master target.

## 2. Upstream / downstream contracts
- **Consumes (all read-only, refits nothing):**
  - **F15** — `run_failsafe_suite(config, *, artefact_dir=None) → FailsafeReport` and the
    `FailsafeReport`/`FailsafeTestResult` models (`app/eval/failsafe.py`); the harness primitives in
    `app/eval/harness.py` (`assemble`, `fake_prediction`, `fake_projection`, `measure_post`, `p95`) for the
    §11.4 real-artefact rate extension (Tests 1/6 → §4.0-e).
  - **F10** — `lead_time_report(matrix, *, handle: MonitorHandle, config, log_path=None) → LeadTimeReport`
    and `load_detector(*, config, artefact_dir=None, log_path=None) → MonitorHandle` (`app/monitoring/detector.py`).
  - **F08** — `stability_report(laps, *, config, artefact_dir=None, log_path=None) → StabilityReport`;
    `teammate_proximity(handle_or_coords, teams, *, config) → TeammateProximityResult`;
    `project_unknown(laps, *, circuit, handle: StyleSpaceHandle, config, log_path=None) → ProjectionResult`;
    `load_style_space(*, config, artefact_dir=None, log_path=None) → StyleSpaceHandle` (`app/ml/stylespace.py`).
  - **F07** — `season_stability(per_season_shap, *, config, log_path=None, entity_id=…) → SeasonStabilityResult`
    (`app/ml/explain.py`; `stability_score` = mean pairwise JS, `pairwise_js`, `partial`).
  - **F09** — `run_ablation(frame, *, config, log_path=None) → AblationReport` (`app/ml/corners.py`).
  - **F06a** — the champion's `ClassifierTrainingResult` (from `train_classifier(...)`, persisted /
    recoverable from the `apex-classifier` MLflow experiment): the §11.1 fields
    `held_out_macro_f1`, `held_out_per_driver_f1`, `baseline_macro_f1`, `beats_all_baselines`,
    `normalisation_ablation_macro_f1`, `confusion_matrix_path`, `calibration_curve_path`,
    `target_macro_f1`, `target_met`, `per_driver_floor`, `per_driver_floor_met`. **F06b** `load_model` +
    the F06a helpers for the optional re-score (§4.0-c).
  - **F04/F12/F13** — the held-out feature matrix (Belgium/Italy laps with a `split` column) is upstream
    data; the `drivers` table supplies the driver→team map for teammate proximity (may be passed in).
  - `config.yaml` → new `eval:` block; **references** (never re-declares) `classifier.target_macro_f1`
    (0.78), `classifier.per_driver_floor` (0.70), `classifier.baselines`, `stylespace.stability_variance_threshold`
    (0.05), `stylespace.stability_n_runs` (10), `stylespace.nearest_neighbours_k` (3),
    `monitoring.sensitivity_grid_sd`, `shap.stability_min_seasons`, `failsafe.*` (the whole block),
    `api.latency_slo_ms` (via the harness), `held_out_circuits` ([Belgium, Italy]), and `seeds`.
- **Produces:**
  - `app/eval/evaluate.py` — the four-pillar orchestrator: `run_full_evaluation(...) → EvaluationReport`,
    the pillar runners (`_pillar_classifier`, `_pillar_stability`, `_pillar_degradation`, `_pillar_failsafe`,
    `_pillar_ablation`), the RQ mapper (`_rq_verdicts`), and the thin classifier-harvest seam
    (`evaluate_classifier`). The typed models `MetricResult`, `PillarResult`, `RQVerdict`, `EvaluationReport`
    (with `render_pillar_tables()` / `render_rq_table()` / `render_report()`).
  - **`artefacts/evaluation_report.json`** — the results-table artefact (content-hashed on the reproducible
    OUTCOME, timestamp-free — the F04/F10/F15 idiom) + the rendered markdown the dissertation embeds.
  - New typed `EvalConfig` (`app/contracts/config.py`) + the `eval:` YAML block; `app/eval/__init__.py`
    re-exports `EvaluationReport` / `run_full_evaluation`.
  - **No new error codes; no new model artefact** (F20 measures/harvests existing behaviour; the report is
    generated evidence, not committed — like `failsafe_report.json`).
- **Consumed by:** **F21/F22** (the dissertation's Evaluation, Discussion, Conclusion chapters embed the
  rendered tables + the RQ verdicts) and the RQ5 evidence trail (via the folded-in `FailsafeReport`). F20 is
  the last executable feature before writing; nothing in `app/` consumes it.

## 3. Inputs & outputs (typed)
- **Input:** `run_full_evaluation(matrix, *, config, artefact_dir=None, teams=None, log_path=None)`:
  - `matrix: pd.DataFrame | None` — the assembled evaluation feature matrix: the 28-feature columns plus
    `driver`, `circuit`, `circuit_type`, `season`, `compound`, `split` (`train`/`val`/`test`), and the stint
    columns `lead_time_report` needs (`pit_lap`, `lap_number`, safety-car flag). **Produced upstream**
    (F00–F03 data-prep) — F20 consumes it; the held-out rows are the `circuit ∈ held_out_circuits` slice,
    the ONLY sanctioned evaluation touch-point (divergence #61). `None` → the data-dependent pillars skip.
  - `config: AppConfig`; `artefact_dir` — where the committed champion / umap / detector / normalisation
    artefacts live; `teams: Mapping[str, str] | None` — driver→team map for §11.2 teammate proximity
    (from the `drivers` table; `None` → the proximity metric skips); `log_path` — DQ-log override.
- **`MetricResult`** (Pydantic): `{ name: str, value: float | int | None, target: float | int | None,
  comparator: Literal[">", ">=", "<", "<=", "in_range", "=="], passes: bool | None, finding: str | None }`.
  `value=None` ⇒ not computed (data/artefact absent) ⇒ `passes=None` (never a fabricated pass); `finding`
  carries the §11 failure-analysis note (e.g. "VER-PER confused as Red Bull teammates — expected").
- **`PillarResult`** (Pydantic): `{ pillar: Literal["classifier","stability","degradation","failsafe",
  "ablation"], metrics: list[MetricResult], skipped: bool = False, skip_reason: str | None = None,
  findings: list[str] = [], native: dict[str, Any] = {} }`. `native` optionally serialises the underlying
  typed result (the `FailsafeReport`, `LeadTimeReport`, `AblationReport`, `StabilityReport`) so the
  dissertation can drill down without a second run.
- **`RQVerdict`** (Pydantic): `{ rq: Literal["RQ1","RQ2","RQ3","RQ4","RQ5"], question: str,
  evidence: list[str] (metric names / pillars), verdict: Literal["SUPPORTED","PARTIAL","NOT_SUPPORTED",
  "INSUFFICIENT_DATA"], findings: list[str] = [] }`. `INSUFFICIENT_DATA` when the evidence pillar(s) skipped
  (offline / no artefacts) — the honest "we built the machine, the numbers need the trained run" state.
- **`EvaluationReport`** (Pydantic): `{ pillars: list[PillarResult], rq_verdicts: list[RQVerdict],
  n_pillars: int, n_pillars_run: int, n_pillars_skipped: int, all_targets_met: bool, content_hash: str }`.
  Timestamp-free; `content_hash` covers the reproducible **outcome** (per metric: name + target + comparator
  + pass/fail; per RQ: verdict) — **not** the run-specific stochastic values (UMAP coords vary within
  Procrustes tolerance; failsafe timing is per-run), so "same artefacts + seeds → same verdicts → same hash"
  (reproducibility #3, mirrors `FailsafeReport`, divergence #58). `render_report() → str` emits the full
  markdown evaluation chapter (the four pillar tables + the RQ verdict table + the failure findings).

## 4. Method / algorithm

### 4.0 Design notes (the decisions a builder must not re-litigate)
- **(a) F20 orchestrates; it does not recompute (divergence #53).** Each §11 pillar's metric maths already
  ships in the feature that owns it. F20 *calls* the read-only surfaces (`run_failsafe_suite`,
  `lead_time_report`, `stability_report`, `teammate_proximity`, `project_unknown`, `season_stability`,
  `run_ablation`) and *harvests* the classifier's held-out numbers from the champion
  `ClassifierTrainingResult`. A second copy of the evaluation maths would be a divergence risk (two sites can
  disagree) and a leakage risk (a second held-out scoring outside F06a's `_assert_no_leakage` guard). F20's
  own code is roll-up + gate-against-target + RQ-mapping + render — no model fitting, no metric arithmetic
  beyond comparing a harvested value to its configured target.
- **(b) No committed artefacts/matrix → framework offline, real numbers artefact-gated (divergence #54).**
  APEX commits **no** model artefacts (`artefacts/` holds only `mlflow.db`) and no feature matrix — the
  established pattern (F13's no-artefact 503, F15's `@pytest.mark.artefacts` rate tests). So F20 ships the
  **evaluation framework** (orchestrator + typed report + RQ mapping + render) and runs its **mechanism**
  offline: the §11.4 pillar runs its harness *mechanism* via `run_failsafe_suite` (it needs no committed
  artefacts), and the three data/artefact-dependent pillars return `skipped=True` + reason when their inputs
  are absent, so the offline report is honest (§11.4 mechanism + three skips + RQ5 partially answered,
  RQ1–RQ4 `INSUFFICIENT_DATA`). The **real dissertation numbers** are produced by a single
  `@pytest.mark.artefacts` (+ evaluation-matrix) run once the champion/umap/detector/normalisation artefacts
  and the held-out matrix are present. The Week-20 deliverable = *the machine* (built + mechanism-tested
  here) *plus* that gated run. This is stated, not stubbed.
- **(c) §11.1 classifier is harvested, not re-scored (divergence #55).** There is **no standalone
  re-scorable classifier evaluator** — all §11.1 numbers (held-out macro-F1, per-driver F1, confusion matrix,
  calibration curve, the three baselines, the normalisation ablation) are computed **inside**
  `train_classifier` and returned on `ClassifierTrainingResult` (+ logged to the `apex-classifier` MLflow
  experiment; the confusion-matrix / calibration-curve PNGs are already-written artefacts). F20's
  `evaluate_classifier(result, …)` reads those fields and gates them against
  `classifier.target_macro_f1` / `per_driver_floor` / `beats_all_baselines`. Re-computing would touch
  held-out data a second time; harvesting the champion's own logged held-out evaluation IS the §11.1
  evidence. The optional `@pytest.mark.artefacts` cross-check (re-score via `load_model` + the F06a helpers
  `_macro_f1_on` / `_per_driver_f1` / `_calibration_curve` / `_normalisation_ablation`) is a *verification*
  path, not the primary one; if it needs those helpers promoted from private, that is a tiny additive F06a
  seam-completion (the `labels.json` precedent), not F20 reaching under the hood.
- **(d) §11.2 circuit generalisation REUSES the §11.1 held-out F1 (divergence #56).** The style-space module
  ships no "circuit generalisation" function. The master's §11.2 "Circuit generalisation — F1 > 0.70 on
  Belgium and Italy" is the *same held-out per-driver F1* the classifier pillar already reports; F20 surfaces
  it under both pillars (labelled as shared), and adds the **feature-degradation-on-unseen-circuits** finding
  by comparing per-driver F1 on held-out vs training circuits from the harvested `ClassifierTrainingResult`.
  No second computation.
- **(e) §11.4 is F15's `run_failsafe_suite`, extended for the two rate tests (references F15 divergence
  #39/#41).** F20 does **not** reimplement the 7 tests — it calls `run_failsafe_suite(config, …) →
  FailsafeReport` and folds the report into the degradation-and-failsafe section. When committed artefacts
  are present, F20 additionally runs the two artefact-gated *rate* assertions F15 deferred: Test 1's ≥90%
  `LOW_CONFIDENCE` rate on real ambiguous input (avg of two drivers' laps through the real champion) and
  Test 6's real 5-SD detection through the real UMAP — using the F15 harness primitives (`assemble`,
  `measure_post`). Offline these two extensions skip (the mechanism is already PASS in the folded report).
- **(f) RQ2 is under-instrumented by the master → answered PARTIAL, honestly (divergence #57).** §11 has a
  metric table for the classifier (RQ1), stability (RQ1/RQ4), degradation (RQ3) and failsafe (RQ5) — but
  **no metric table for RQ2** ("how strongly does compatibility correlate with stint-performance
  consistency"). F20 does not fabricate one: it answers RQ2 from what exists — F11's per-parameter
  reliability finding ("only ~5 setup parameters are reliably inferrable", already an honest research result)
  + §11.2 teammate proximity as a compatibility proxy — and marks RQ2 `PARTIAL` with a finding that the
  correlation study is under-instrumented in the current system. This is the §11-intro contract ("documented
  as findings") applied to a research question, and a stated dissertation limitation (§15.2 style).
- **(g) F20 never raises; a pillar fault is a skip, not a crash (divergence #59, lifts F15 §4.0-h to the
  pillar level).** `run_full_evaluation` wraps each pillar in its own try; a pillar-internal fault (or the
  system-under-evaluation raising) → that `PillarResult.skipped=True` + a finding describing it; one
  pillar's fault never aborts the run. A report with any skip returns `n_pillars_run < n_pillars` — surfaced
  in the render, never swallowed. `run_full_evaluation` is **async** (it awaits `run_failsafe_suite`).
- **(h) The `split` column is consumed, never produced (divergence #61).** The train/val/held-out
  assignment (`matrix["split"]`) is produced upstream (F00–F03 data-prep); it appears in `app/` only as a
  consumer. F20 reads `matrix["split"]` + `config.held_out_circuits`; a missing column or a slice with **no
  held-out rows** → the affected pillar skips with a reason (never a silent pass on training data). Belgium
  and Italy are quarantined from F00 onward and are F20's ONLY evaluation touch-point (INDEX dependency note).
- **(i) F20 refits nothing, fabricates nothing.** No `.fit()` anywhere in `app/eval/evaluate.py`; every
  number is harvested from a typed result or returned by a read-only surface; the report is generated
  evidence written to `artefacts/`, never committed. Stated so `/apex-review` ticks it off.

### 4.1 The four pillars + the F09 ablation (metric · target · source · offline?)
Targets are quoted verbatim from §11. "Source" is the exact surface F20 calls/harvests. "Offline" = can the
metric be produced with no committed artefacts (mechanism) vs needs the artefacts + matrix (→ the gated run).

| Pillar | Metric (§11) | Target (verbatim) | Source (call/harvest) | Offline? |
|---|---|---|---|---|
| **classifier §11.1** | Macro-averaged F1 | > 0.78 on held-out (Belgium, Italy) | harvest `ClassifierTrainingResult.held_out_macro_f1` vs `classifier.target_macro_f1` | 🔶 gated |
| classifier | Per-driver F1 | > 0.70 for all 8 drivers | harvest `held_out_per_driver_f1` vs `classifier.per_driver_floor` | 🔶 gated |
| classifier | Confusion matrix | visual heatmap (findings) | harvest `confusion_matrix_path` (artefact already written) | 🔶 gated |
| classifier | Calibration curve | probs ≈ empirical freq | harvest `calibration_curve_path` | 🔶 gated |
| classifier | Baseline comparison | XGBoost beats all 3 baselines | harvest `baseline_macro_f1` + `beats_all_baselines` | 🔶 gated |
| classifier | Normalisation ablation | F1 drops significantly w/o norm | harvest `normalisation_ablation_macro_f1` | 🔶 gated |
| **stability §11.2** | Year-on-year stability (JS) | < 0.15 for stable drivers | call `season_stability` per driver vs `eval.stability_js_target` | 🔶 gated |
| stability | Teammate proximity | intra-team dist < inter-team | call `teammate_proximity(handle, teams)` → `passes` | 🔶 gated |
| stability | Circuit generalisation | F1 > 0.70 on Belgium & Italy | REUSE §11.1 held-out per-driver F1 (§4.0-d) | 🔶 gated |
| stability | Unknown-driver projection | correct NN for 6/8 drivers-as-unknown | call `project_unknown` re-inserting each known driver vs `eval.unknown_nn_correct_min` | 🔶 gated |
| stability | UMAP stability | Procrustes var < 0.05 over 10 seeds | call `stability_report` vs `stylespace.stability_variance_threshold` | 🔶 gated |
| **degradation §11.3** | Mean alert lead time | 2–5 laps before pit | call `lead_time_report` → `mean_lead_time_laps` vs `eval.lead_time_min/max_laps` | 🔶 gated |
| degradation | True positive rate | > 70% alerted within 5 laps | `LeadTimeReport.tp_rate` vs `eval.tp_rate_target` | 🔶 gated |
| degradation | False positive rate | < 20% alert > 5 laps early | `LeadTimeReport.fp_rate` vs `eval.fp_rate_max` | 🔶 gated |
| degradation | Compound stratification | reported per Soft/Med/Hard/Inter | `LeadTimeReport.per_compound` (finding) | 🔶 gated |
| degradation | Threshold sensitivity | TP at 1/1.5/2/2.5 SD | `LeadTimeReport.sensitivity` + `optimal_threshold_per_circuit_type` | 🔶 gated |
| **failsafe §11.4** | 7 RQ5 tests | pass or documented as findings | call `run_failsafe_suite` → `FailsafeReport` (fold whole) | ✅ mechanism · 🔶 rates (Tests 1/6) |
| **ablation (F09)** | corner-type improves classifier? | IMPROVES/NEUTRAL/DEGRADES | call `run_ablation` → `AblationReport.finding` | 🔶 gated |

### 4.1.1 The RQ verdict mapping (§2.4 — "every RQ answered with data")
| RQ | Question (§2.4, abbreviated) | Evidence pillars/metrics | Verdict rule |
|---|---|---|---|
| **RQ1** | stable, discriminative fingerprint that **generalises to unseen circuits** | classifier held-out macro-F1 (>0.78) + per-driver F1 (>0.70) + normalisation ablation; stability circuit-generalisation + year-on-year JS | SUPPORTED iff macro-F1 & per-driver targets met AND ablation shows the expected drop; PARTIAL if targets relaxed (risk-register 0.72 path); else NOT_SUPPORTED |
| **RQ2** | compatibility ↔ stint-performance consistency correlation | F11 per-parameter reliability + teammate proximity **proxy** (§4.0-f) | **PARTIAL** by construction — no §11 metric table; finding records under-instrumentation |
| **RQ3** | does deviation **precede** lap-time degradation, by how many laps | degradation mean lead time (2–5), TP (>70%), FP (<20%), compound strat, threshold sensitivity | SUPPORTED iff mean lead time in [2,5] AND TP target met AND FP target met |
| **RQ4** | style space continuous enough to **project unknown drivers** meaningfully | unknown-driver NN (≥6/8), UMAP Procrustes stability (<0.05), teammate proximity | SUPPORTED iff NN target met AND Procrustes target met |
| **RQ5** | principal failure modes + how failsafes affect reliability | the folded `FailsafeReport` (7 tests) | SUPPORTED iff all 7 PASS (or documented as findings per §11.4) |
| any | evidence pillar skipped (offline / no artefacts) | — | **INSUFFICIENT_DATA** (never SUPPORTED without data) |

### 4.2 Orchestration architecture (`app/eval/evaluate.py`)
- **`async def run_full_evaluation(matrix, *, config, artefact_dir=None, teams=None, log_path=None) →
  EvaluationReport`** — loads the committed handles best-effort (`load_model` / `load_style_space` /
  `load_detector`, each of which raises its own `*LoadError` when the artefact is absent → caught → that
  pillar skips), runs the five pillar runners (each wrapped in its own try, §4.0-g), calls
  `run_failsafe_suite`, assembles the `EvaluationReport`, computes the RQ verdicts, writes
  `artefacts/evaluation_report.json`, and returns the report. Never raises.
- **Pillar runners** (each `→ PillarResult`, each self-contained, each skips-with-reason on absent input):
  `_pillar_classifier(result, matrix, config)` (harvest, §4.0-c); `_pillar_stability(handle, matrix, teams,
  config)` (calls the four F08/F07 surfaces); `_pillar_degradation(handle, matrix, config)` (calls
  `lead_time_report`); `_pillar_failsafe(config, artefact_dir)` (`await run_failsafe_suite` + the two
  artefact-gated rate extensions, §4.0-e); `_pillar_ablation(matrix, config)` (`run_ablation` +
  the harvested normalisation ablation).
- **`evaluate_classifier(result: ClassifierTrainingResult | None, *, config, log_path=None) → PillarResult`**
  — the thin harvest seam (public, so F21/F22 can call it directly on a persisted training result). `None` →
  skipped("no champion training result available — run F06a `train_classifier`").
- **`_rq_verdicts(pillars) → list[RQVerdict]`** — pure function applying the §4.1.1 rules to the pillar
  metrics; an RQ whose evidence pillar skipped → `INSUFFICIENT_DATA`.
- **Report writer** `_write_report(report, config, artefact_dir)` — the F04/F10/F15 idiom
  (`json.dumps(model_dump(), sort_keys=True, indent=2, allow_nan=False)`, embedded `content_hash` = sha256 of
  the canonical **outcome** body, no timestamp; §4.3).

### 4.3 The evaluation results-table artefact (§13 Week-20 deliverable)
`EvaluationReport` → `artefacts/evaluation_report.json` written with the reproducibility idiom. The
`content_hash` (mirrors `FailsafeReport._canonical`) covers the canonical **outcome** — for each metric its
`name`/`target`/`comparator`/`passes`, and each `RQVerdict.rq`/`verdict` — **not** the run-specific stochastic
`value`s (UMAP coordinates, failsafe timings). So a re-run on the same committed artefacts + seeds yields the
same verdicts and the same hash (divergence #58). `render_report()` emits the markdown the dissertation
embeds: one table per pillar (Metric · Value · Target · Pass · Finding), the RQ verdict table (RQ · Verdict ·
Evidence · Finding), and a failures roll-up. The artefact is generated, **not committed** (like every
`artefacts/*`); the gated run regenerates it.

## 5. config.yaml keys this feature adds
```yaml
# --- F20 evaluation framework (§11 — all four pillars + RQ verdicts; §13 Week-20 deliverable) ------
# F20 orchestrates the existing pillar surfaces and GATES each harvested/returned metric against its
# master target. Targets already owned by a feature block are REFERENCED, never re-declared here:
#   classifier.target_macro_f1 (0.78), classifier.per_driver_floor (0.70), classifier.baselines,
#   stylespace.stability_variance_threshold (0.05), stylespace.stability_n_runs (10),
#   stylespace.nearest_neighbours_k (3), monitoring.sensitivity_grid_sd, shap.stability_min_seasons,
#   the whole failsafe: block, api.latency_slo_ms, held_out_circuits, seeds.
# This block holds ONLY the §11 targets that NO existing block owns. F20 adds no error code.
eval:
  stability_js_target: 0.15         # §11.2 — year-on-year JS divergence below this = stable (F07 season_stability)
  unknown_nn_correct_min: 6         # §11.2 — min drivers-as-unknown projected to the correct nearest neighbour (of the known-driver count)
  lead_time_min_laps: 2             # §11.3 — mean alert lead time lower bound (laps before pit)
  lead_time_max_laps: 5             # §11.3 — mean alert lead time upper bound
  tp_rate_target: 0.70              # §11.3 — min stint true-positive rate (alerted within 5 laps before pit)
  fp_rate_max: 0.20                 # §11.3 — max stint false-positive rate (alert > 5 laps early)
  report_filename: evaluation_report.json  # results-table artefact under paths.artefacts_dir
```
New typed `EvalConfig` in `app/contracts/config.py` (standing rule — feature config lives in `config.py`,
`extra="forbid"`), wired as `AppConfig.eval_: EvalConfig | None = Field(default=None, alias="eval")` (the
attribute is named `eval_` to avoid shadowing the builtin, set by the YAML alias `eval:` — the F02
`lap_validation`/`validate` precedent). Validators (in-model): `0 < stability_js_target`;
`unknown_nn_correct_min ≥ 1`; `0 < lead_time_min_laps < lead_time_max_laps`; `0 < tp_rate_target ≤ 1`;
`0 ≤ fp_rate_max < 1`; non-empty `report_filename`. Cross-block (on `AppConfig`, F10/F11 precedent): an
`eval:` block **requires** the `classifier`, `stylespace`, `monitoring`, and `failsafe` blocks present (F20
references their targets/surfaces) — a missing sibling is a load-time `CONFIG_001`, not a first-use crash.

## 6. Error codes & failure modes (the failsafe contract F20 asserts + surfaces)
F20 adds **no** error codes — it *harvests/asserts* the pre-seeded ones each pillar emits, and represents a
"cannot evaluate" condition as a typed `skipped` state (never an error code, mirroring `FailsafeTestResult.
skipped`). The table below is the contract F20 preserves.

| Failure mode | Detection | Code / behaviour | System response | Logged? | User-facing (report) |
|--------------|-----------|------------------|-----------------|---------|----------------------|
| No champion artefact / training result | `load_model` → `ModelLoadError` / `result is None` | `MODEL_NOT_FOUND` (from F06b) | classifier + stability(circuit-gen) + ablation → `skipped` + reason; RQ1 `INSUFFICIENT_DATA` | per F06b | "classifier pillar skipped — no committed champion; run F06a" |
| No UMAP/PCA artefact | `load_style_space` → `StyleSpaceLoadError` | `MODEL_NOT_FOUND` / `UMAP_FALLBACK` | stability pillar → `skipped` + reason; RQ4 `INSUFFICIENT_DATA` | per F08 | "stability pillar skipped — no committed style space" |
| No detector artefact | `load_detector` → `DetectorLoadError` | `MODEL_NOT_FOUND` | degradation pillar → `skipped`; RQ3 `INSUFFICIENT_DATA` | per F10 | "degradation pillar skipped — no committed detector" |
| `matrix is None` / no held-out rows / no `split` col | column/slice check | (no code) typed `skipped` | affected pillars → `skipped` + reason (never score on training data, §4.0-h) | no | "skipped — no held-out (Belgium/Italy) rows in matrix" |
| Malformed eval matrix passed to a surface | the surface's own guard | `INVALID_INPUT` (from F09/F10) | that surface returns its DEGRADED/INVALID result; F20 records it as a metric `value=None` + finding | per surface | finding column |
| A target not met (real number below target) | F20 gate vs config target | (no code) `MetricResult.passes=False` | reported honestly with the §11 failure-analysis finding (NOT a failure of F20) | no | e.g. "macro-F1 0.74 < 0.78 — relaxed-target path (risk register)" |
| Pillar-internal fault (F20 bug or SUT raises) | per-pillar try/except (§4.0-g) | (no code) `PillarResult.skipped=True` + finding | run continues; `n_pillars_run < n_pillars` surfaced | no | finding in the render |
| §11.4 sub-test fault | inside `run_failsafe_suite` | (F15's) `FailsafeTestResult.passed=False` | folded verbatim; RQ5 reflects it | per F15 | folded RQ5 table |

## 7. Acceptance criteria (each item must be mechanically testable by /spec-check)
- [ ] `run_full_evaluation` is **async**, **never raises**, and returns an `EvaluationReport` with exactly
      the five pillars (`classifier`, `stability`, `degradation`, `failsafe`, `ablation`) and exactly five
      `RQVerdict`s (RQ1–RQ5).
- [ ] **§11.4 pillar** calls `run_failsafe_suite(config, …)` and folds the returned `FailsafeReport` into the
      report's `failsafe` pillar (it does NOT reimplement the 7 tests); RQ5 SUPPORTED iff all 7 PASS/skip per
      §11.4. Runs offline (mechanism) with no committed artefacts.
- [ ] **§11.1 pillar** harvests `held_out_macro_f1` / `held_out_per_driver_f1` / `baseline_macro_f1` /
      `beats_all_baselines` / `normalisation_ablation_macro_f1` from a `ClassifierTrainingResult` and gates
      them against `classifier.target_macro_f1` (0.78) / `per_driver_floor` (0.70) / `beats_all_baselines`
      — **reading the config values, no magic numbers**. `result=None` → `skipped` + reason (no fabricated 0.0).
- [ ] **§11.2 pillar** calls `stability_report` (Procrustes var vs `stylespace.stability_variance_threshold`),
      `teammate_proximity`, `project_unknown` (re-inserting each known driver as unknown; correct-NN count vs
      `eval.unknown_nn_correct_min`), and `season_stability` (year-on-year JS vs `eval.stability_js_target`);
      the circuit-generalisation metric **reuses** the §11.1 held-out per-driver F1 (§4.0-d), not recomputed.
- [ ] **§11.3 pillar** calls `lead_time_report` and gates `mean_lead_time_laps` in
      `[eval.lead_time_min_laps, eval.lead_time_max_laps]`, `tp_rate` ≥ `eval.tp_rate_target`, `fp_rate` ≤
      `eval.fp_rate_max`; `per_compound` + `sensitivity` + `optimal_threshold_per_circuit_type` surfaced as
      findings.
- [ ] **F09 ablation** pillar calls `run_ablation` and reports its `IMPROVES/NEUTRAL/DEGRADES` finding + the
      harvested normalisation ablation.
- [ ] **RQ mapping** — each of RQ1–RQ5 gets a verdict from the §4.1.1 rules; an RQ whose evidence pillar
      skipped → `INSUFFICIENT_DATA`; **RQ2 is `PARTIAL` by construction** with a finding recording that no
      §11 metric table instruments it (§4.0-f).
- [ ] Any pillar with an absent artefact/matrix returns `skipped=True` + `skip_reason` and a matching
      `INSUFFICIENT_DATA` RQ verdict — **never a fabricated number** (a metric's `value=None ⇒ passes=None`).
- [ ] `run_full_evaluation` writes **`artefacts/evaluation_report.json`** (content-hashed, timestamp-free;
      the outcome `content_hash` is stable across two runs on the same inputs — stochastic `value`s are not
      hashed) and `render_report()` emits the four pillar tables + the RQ verdict table + a failures roll-up.
- [ ] A pillar-internal fault yields `skipped=True` + a finding and **does not abort the run** (§4.0-g); the
      run **never writes to the real `cache/` or `data_quality_log`** during the §11.4 sub-run (F15 already
      redirects both — F20 inherits it).
- [ ] **No `.fit()` / no refit / no metric-maths reimplementation** in `app/eval/evaluate.py`; every number
      is harvested or returned by a read-only surface; the report is generated, never committed.
- [ ] All targets read from `config.yaml` (`eval:` or the referenced sibling blocks); **no magic numbers**;
      the `eval:` cross-block validator requires `classifier`/`stylespace`/`monitoring`/`failsafe`.
- [ ] `mypy app` clean, `ruff` clean; the real full-run assertions are `@pytest.mark.artefacts`-gated
      (skipped offline, mirroring F15's rate tests + F12's `@pytest.mark.postgres`).

## 8. Test plan
- **`tests/eval/test_evaluation.py`** — the mechanism suite, offline:
  - `run_full_evaluation(matrix=None, config=…)` returns a report whose `failsafe` pillar ran (mechanism,
    7 results) and whose `classifier`/`stability`/`degradation`/`ablation` pillars are `skipped` with
    reasons; RQ5 answered, RQ1–RQ4 `INSUFFICIENT_DATA`.
  - `evaluate_classifier` fed a **synthetic** `ClassifierTrainingResult` with numbers above target →
    all `passes=True`, RQ1 SUPPORTED; fed one below target → `passes=False` + the §11.1 failure-analysis
    finding, RQ1 NOT_SUPPORTED (or PARTIAL on the relaxed-target path); fed `None` → `skipped`.
  - stability/degradation/ablation pillars driven with **injected fakes / tiny synthetic matrices** (the F15
    `assemble`/`fake_projection` pattern) so the *call + gate + finding* wiring is asserted without artefacts.
  - RQ2 is always `PARTIAL` with the under-instrumentation finding.
- **Report tests** — `run_full_evaluation` writes `evaluation_report.json`; the **outcome content-hash is
  stable on a second run** with the same inputs; `render_report()` contains all four pillar tables + the RQ
  table; a run leaves the real `cache/` + `data_quality_log` untouched; a pillar that raises → `skipped`, run
  still returns five pillars.
- **`@pytest.mark.artefacts` real full run** — when `artefacts/classifier.pkl` + `umap_model.pkl` +
  `stint_variation.json` + `normalisation.json` and an evaluation matrix are present: assert real macro-F1 vs
  0.78, real Procrustes var vs 0.05, real lead time in [2,5], the two §11.4 rate extensions (Test 1 ≥90%,
  Test 6 5-SD) — the honest deferred numbers. Skipped offline.
- **Config/property/edge** — `eval:` without a required sibling block fails validation; `lead_time_min_laps
  ≥ lead_time_max_laps` fails; `tp_rate_target > 1` fails; a matrix with no held-out rows → the data pillars
  skip; an all-NaN metric → `value=None`, `passes=None` (never a fabricated pass).
- **Consolidation, not duplication** — F20 *reuses* the pillar surfaces' own unit tests (F08 `stability_report`,
  F10 `lead_time_report`, F09 `run_ablation`, F15 `run_failsafe_suite`); those stay, F20 asserts only the
  orchestration + gating + RQ-mapping layer on top.

## 9. Definition of done
All acceptance criteria MET with passing tests (offline — no committed artefacts; the real-number
assertions `@pytest.mark.artefacts`-gated), `/apex-review F20` clean on the contract checklist (no
`.fit()`, no re-derived scaler, no magic numbers, no silent skip), `mypy app` + `ruff` clean, INDEX row →
DONE. `failsafe-auditor` run on `app/eval/evaluate.py` (it should find the orchestrator itself fails loud —
every skip is typed + reasoned, no bare excepts). Record in the dissertation limitations: (1) F20 ships the
evaluation **framework + mechanism**; the real numbers require the trained artefacts + held-out matrix, run
via the gated path (divergence #54); (2) §11.1 classifier metrics are **harvested** from the champion's
own held-out evaluation, not recomputed (#53/#55); (3) **RQ2 is answered PARTIAL** — the master provides no
§11 metric table for the compatibility↔consistency correlation, and F20 surfaces this as a finding, not a
fabricated result (#57); (4) UMAP-dependent metrics are reproducible only within Procrustes tolerance, so
the report's `content_hash` certifies the **verdicts**, not the stochastic coordinates (#58); (5) the
evaluation results table is emitted as `evaluation_report.json` + rendered markdown — the master names the
deliverable ("results tables", "evaluation chapter results"), not a file/format (#60).

## 10. Risks & mitigations (from master risk register where relevant)
- **A silent fabricated number would corrupt the dissertation's evidence** (the top risk for an evaluation
  feature) → §4.0-b/§4.0-h: every un-runnable metric is `value=None`/`passes=None` and every un-runnable
  pillar is `skipped` + reason; the render shows PASS/FAIL/SKIP distinctly; the RQ verdict is
  `INSUFFICIENT_DATA` rather than SUPPORTED. This is the §11-intro "honest system" contract enforced in code.
- **XGBoost macro-F1 below 0.78 on held-out** (risk register, Wk 6) → F20 does **not** treat a below-target
  number as an F20 failure: it reports it with the §11.1 failure-analysis finding and, on the risk-register
  relaxed-target path (0.72 + honest analysis), marks RQ1 `PARTIAL`. The evaluation's job is to report the
  truth, not to pass.
- **Degradation mean lead time below 2 laps** (risk register, Wk 12) → surfaced as `MetricResult.passes=False`
  + the `sensitivity`/`optimal_threshold_per_circuit_type` findings that motivate the register's mitigation;
  RQ3 reflects it honestly.
- **UMAP projection unstable** (risk register, Wk 8) → the Procrustes-variance metric is exactly the
  register's trigger; F20 reports it against 0.05 and records the "PCA-primary fallback" implication.
- **Scope creep — reimplementing a pillar's maths** → §4.0-a draws the line: F20 calls/harvests, never
  recomputes; `/apex-review` checks there is no metric arithmetic beyond value-vs-target comparison and no
  `.fit()`.
- **Normalisation leakage** → N/A: F20 refits nothing and touches held-out data only through surfaces that
  already guard their fit sites; the §11.1 numbers are harvested from F06a's leakage-guarded evaluation.
  Stated so `/apex-review` ticks it off.
- **RQ2 under-instrumentation read as a system failure** → §4.0-f: F20 answers it PARTIAL with an explicit
  finding; the dissertation states it as a limitation (§15.2 style), turning a gap into an honest
  contribution rather than a hidden hole.
