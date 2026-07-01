# Feature F07 — SHAP Explainability (beeswarm · force · radar · NL templates)

> One spec, one session. Self-contained: build this from this file + CLAUDE.md only.
> F07 turns every F06 classifier output into a **SHAP decomposition** — a global beeswarm
> artefact, a per-prediction force explanation, a 6-axis fingerprint radar, and a natural-language
> style sentence. Explainability is **best-effort and never blocks a prediction**: a SHAP fault
> degrades to `SHAP_UNAVAILABLE`, it never suppresses the F06b `PredictionResult`.

**Status:** SPEC WRITTEN
**Master spec sections:** §6 (Explainability Architecture), §6.1 (SHAP Implementation — library,
scope, global/local/radar/NL/compute-time), §6.2 (SHAP Failure Handling — the three failure
modes), §4.2 (Full Feature Matrix — the ID→Domain mapping the radar aggregates over), §8.2
(Classifier module: "SHAP failure does not block prediction"), §8.4 (`predictions.shap_values`
JSONB / `shap_available` BOOL; `fingerprints.shap_global` JSONB / `radar_values` JSONB(6D))
**Layer / Week:** ML / Wk 7
**Depends on:** **F06a** (the calibrated champion `classifier.pkl` + `labels.json` + the real
`normalisation.json`; the normalised training-fit matrix used as the beeswarm background),
**F06b** (the loaded `ModelHandle`; F07 fills the `shap_values` / `shap_available` fields F06b
leaves `None` / `False`), F04 (`apply_normalisation` — F07 explains the **same 28 normalised
features** the model consumed).

## 1. Purpose
Every APEX prediction must arrive with a full SHAP decomposition — explainability is a first-class
output, not an add-on (master §6). F07 computes, from the F06 champion: (a) a **global** beeswarm
over the training laps, stored once as an artefact; (b) a **local** force explanation per
prediction — which features pushed toward or away from the predicted driver — persisted as JSONB;
(c) a **6-axis fingerprint radar** aggregating SHAP across the six feature domains (throttle,
braking, speed, gear, engine, cornering); and (d) a **natural-language style sentence** generated
by a template engine from the top-2 domains. Per the failsafe contract (§6.2), a SHAP failure
returns the prediction **without** SHAP tagged `SHAP_UNAVAILABLE` — it never blocks the prediction,
never fabricates a dominant trait, and never silently ships an inconsistent attribution.

## 2. Upstream / downstream contracts
- **Consumes (the seam):**
  - The **F06b `ModelHandle`** — the loaded champion `CalibratedClassifierCV(FrozenEstimator(
    XGBClassifier))`, its `drivers` (from `labels.json`), and `normalisation_params`. F07 loads
    **read-only**; it never trains, tunes, or re-fits (a static scan enforces this, as in F06b).
  - The **base XGBoost tree model** extracted from the calibrated champion (§4 `_base_tree_model`)
    — TreeExplainer explains the tree margins, not the calibrated wrapper (the calibrator is a
    monotone post-hoc transform).
  - The **normalised training-fit feature matrix** (F06a's `_apply_to_frame(splits.fit)` output —
    train circuits only, **never** Belgium/Italy) as the beeswarm/global-SHAP background.
  - `artefacts/labels.json` (index→driver-name) via the handle; the §4.2 **feature-domain map**
    (code constant `DOMAIN_FEATURES`).
- **Produces:**
  - `ExplanationResult` — the per-prediction local SHAP + force list + 6-domain radar + NL sentence
    (fills F06b's `PredictionResult.shap_values` / `shap_available`; the `force` list is the §8.4
    `predictions.shap_values` JSONB payload).
  - `GlobalExplainabilityResult` — the beeswarm artefact `artefacts/<shap.beeswarm_filename>`
    (`shap_beeswarm.json`) + `per_driver_shap` (→ §8.4 `fingerprints.shap_global`) +
    `per_driver_radar` (→ §8.4 `fingerprints.radar_values`, 6D).
  - `SeasonStabilityResult` — the **multi-season stability** score per driver (§6.1 row): pairwise
    Jensen-Shannon divergence between a driver's per-season SHAP distributions → a scalar drift
    measure (→ §8.4 `fingerprints.stability_score`; served by `GET /drivers/{id}/stability` and the
    `GET /drivers/{id}/fingerprint` "multi-season stability" field).
  - `ShapExplainer` — the loaded TreeExplainer handle (built once at startup, shared read-only).
- **Consumed by:** **F13** — serialises `ExplanationResult` into `POST /predict`
  (`shap_values`/`shap_available`) and persists `predictions`; serves `GET /drivers/{id}/fingerprint`
  (SHAP radar + NL + `stability`), `GET /drivers/{id}/stability` (JS scores; **206 when <3 seasons**),
  and `GET /compare/{id1}/{id2}` (**`delta_shap`** = signed per-domain radar difference,
  **`most_distinctive_features`** = top-5 features by `|SHAP delta|` — both derived from F07's
  per-driver SHAP + radar). **F11** (personalisation reads the 6-domain radar / NL style as a
  fingerprint input); **F16/F17** (the animated UI is built around these SHAP values — beeswarm,
  force, radar, delta radar); **F10** reuses the same explainer to decompose a degradation deviation
  (§5.3 alert payload) — F07 owns the explainer, F10 calls it.

## 3. Inputs & outputs (typed)
All ops vectorised (pandas 3.x / numpy 2.x). Module is **read-only** (no `.fit(` / `fit_normaliser`
/ `optuna` / training symbol — static-scan asserted, as F06b).

- **Build explainer (startup, once):**
  `build_explainer(handle: ModelHandle, *, config: AppConfig, log_path: Path | None = None)
  -> ShapExplainer`. Extracts the base XGBoost, constructs `shap.TreeExplainer(base,
  feature_perturbation="tree_path_dependent")` (exact, not approximate — §6.1). On any failure
  (unexpected champion structure, SHAP import/build error) it logs `SHAP_UNAVAILABLE` and returns a
  **not-ready** `ShapExplainer(ready=False)` — the serving layer then degrades every explanation to
  `SHAP_UNAVAILABLE` (prediction still served). It **does not raise** (explainability never blocks).
- **`ShapExplainer`** (frozen): `{ explainer, base_values: dict[str, float] (per-driver expected
  value / tree base margin), drivers: list[str], feature_ids: tuple[str, ...] (== FEATURE_IDS, 28),
  ready: bool }`.
- **Explain one prediction (per call, read-only, never raises):**
  `explain_prediction(feature_vector: dict[str, float | None], *, circuit: str, predicted_class:
  str, explainer: ShapExplainer, handle: ModelHandle, config: AppConfig, log_path: Path | None =
  None) -> ExplanationResult`. Takes the **raw** 28-feature vector + `circuit` and **re-normalises
  internally** via F04's read-only `apply_normalisation` (F06b's `predict` does not expose the
  normalised vector, so F07 re-derives exactly what the model saw — deterministic, no `.fit`).
  `predicted_class` is F06b's `top_prediction`; F07 is only invoked on a **PREDICTED and accepted**
  path (HIGH/MEDIUM) — a gated result (`None` prediction) carries no SHAP.
- **`ExplanationResult(BaseResult)`:**
  ```
  { shap_values: dict[str, float] | None,   # 28 F-id -> signed SHAP for predicted_class; None on SHAP_UNAVAILABLE
    base_value: float,                        # explainer expected value for predicted_class; 0.0 when unavailable
    predicted_class: str,
    force: list[Contribution],                # ordered by |shap| desc — the §8.4 shap_values JSONB payload; [] when unavailable
    radar: dict[str, float],                  # exactly the 6 domain scores (throttle..cornering); {} when unavailable
    dominant_domains: list[str],              # top-2 by |magnitude|; [] when generic/balanced OR unavailable
    nl_description: str,                       # style sentence; generic-balanced text when no dominant domain; "" when unavailable
    shap_available: bool,                      # True only on a returned attribution (incl. CONSISTENCY_WARN); False on SHAP_UNAVAILABLE
    consistency_ok: bool,                      # additivity check (§6.2) within tolerance
    consistency_delta: float,                  # |base_value + Σ shap - tree_margin(predicted_class)|
    latency_ms: float,
    status, error_code, flags, logged }
  ```
  `Contribution` = `{ feature_id: str, name: str, shap: float, feature_value: float | None,
  direction: "toward" | "away" }` (`toward` when `shap > 0` for the predicted class).
- **Build global explainability (offline, once after training):**
  `build_global_explainability(explainer: ShapExplainer, background: pd.DataFrame, driver_labels:
  Sequence[str], *, config: AppConfig, artefact_dir: Path | None = None, log_path: Path | None =
  None) -> GlobalExplainabilityResult`. `background` = the **normalised training-fit** matrix (28
  feature cols + a driver column); F07 asserts no held-out circuit is present is the caller's job —
  F07 documents that `background` must already exclude Belgium/Italy. Never raises; a SHAP fault →
  `status=DEGRADED`, `SHAP_UNAVAILABLE`, empty payloads, logged.
- **`GlobalExplainabilityResult(BaseResult)`:** `{ beeswarm_artefact_path: str, per_driver_shap:
  dict[str, dict[str, float]] (driver -> mean|SHAP| per feature), per_driver_radar: dict[str,
  dict[str, float]] (driver -> 6 domain scores), n_background: int, shap_available: bool,
  latency_ms: float, status, error_code, flags, logged }`.
- **Multi-season stability (§6.1 row):**
  `season_stability(per_season_shap: dict[str, dict[str, list[float]]], *, config: AppConfig,
  log_path: Path | None = None) -> SeasonStabilityResult`. Input maps `season -> {feature_id ->
  the SHAP values of that feature across that season's laps}` for **one** driver (the caller runs
  the explainer per season; each season's laps are its own normalised training laps — held-out
  circuits excluded, as always). Never raises.
- **`SeasonStabilityResult(BaseResult)`:** `{ stability_score: float | None (mean pairwise JS
  divergence across seasons — lower = more stable / less drift; None when <2 seasons), pairwise_js:
  dict[str, float] (e.g. "2022|2023" -> JS), seasons_available: int, partial: bool (True when
  seasons_available < shap.stability_min_seasons), status, error_code, flags, logged }`.
- **Pure helpers (no I/O — unit-testable in isolation):**
  `aggregate_domains(shap_by_feature: dict[str, float]) -> dict[str, float]` (the 6 radar scores);
  `describe_style(radar: dict[str, float], *, config: AppConfig) -> StyleDescription` (`{ sentence:
  str, dominant_domains: list[str], generic: bool }`).
- **Exceptions:** none on any serving path. `explain_prediction`, `build_explainer`, and
  `build_global_explainability` **never raise** — every fault is a coded result / not-ready handle.

## 4. Method / algorithm
**`_base_tree_model(champion)` — extract the XGBoost the tree explainer needs.** The champion is
`CalibratedClassifierCV(FrozenEstimator(XGBClassifier))`; TreeExplainer cannot explain the
calibrated wrapper, so F07 navigates `champion.calibrated_classifiers_[0].estimator` (the
`FrozenEstimator`) → `.estimator` (the fitted `XGBClassifier`) with a **fail-loud guard**: if the
structure is not as expected, or the resolved object is not an `xgb.XGBClassifier`, log
`SHAP_UNAVAILABLE` and mark the explainer not-ready (never `AttributeError`, never a wrong model).
Prefit calibration produces exactly one `calibrated_classifiers_` entry, so index 0 is total.
*Rationale:* the isotonic/sigmoid calibrator is a monotone post-hoc transform of the class score;
SHAP over the base tree margins is the standard, documented approach and preserves attribution
ranking (noted in §9 limitations).

**`build_explainer` (startup):** `shap.TreeExplainer(base, feature_perturbation=
"tree_path_dependent")` → exact tree SHAP (§6.1 "exact, not approximate"). Record `base_values`
(the explainer `expected_value`, one per class) keyed by driver name via `handle.drivers`. Any
exception → logged `SHAP_UNAVAILABLE`, `ready=False`.

**`explain_prediction` (per call) — never raises:**
1. If `explainer.ready is False` → return `SHAP_UNAVAILABLE` immediately (prediction unaffected).
2. Build `x` = the 28 normalised features in `FEATURE_IDS` order (a NULL → `np.nan`, as F06b feeds
   the model); resolve the predicted class index `k = drivers.index(predicted_class)` (a
   `predicted_class` not in `drivers` → `SHAP_UNAVAILABLE`, logged — never a silent mis-key).
3. **Guarded** `explainer(x)` (SHAP 0.45+ Explanation API) → take the predicted-class slice
   `values[..., k]` and `base_values[..., k]`. **On ANY exception** (model/version mismatch, shape
   error) → catch, log full detail (`SHAP_UNAVAILABLE`), return the coded result **without SHAP**;
   `status=DEGRADED`, `shap_values=None`, `shap_available=False`, `force=[]`, `radar={}`,
   `dominant_domains=[]`, `nl_description=""`. **The prediction is never suppressed** (§6.2 FM1).
4. `shap_values = { fid: signed_shap_j }` (28 keys). Build `force` = the features sorted by `|shap|`
   desc, each tagged `toward` (`shap > 0`) / `away`.
5. **Consistency (additivity) check (§6.2 FM2):** `delta = |base_value + Σ shap_values -
   tree_margin_k(x)|`, where `tree_margin_k` is the base XGBoost raw margin for class `k`. If
   `delta > shap.consistency_tolerance` → `consistency_ok=False`, add `SHAP_CONSISTENCY_WARN` to
   `flags`, `status=DEGRADED`, log one row — but **return the SHAP values** with the warning flag
   (§6.2: "Return SHAP values with warning flag"), `shap_available=True`.
6. **Radar:** `radar = aggregate_domains(shap_values)` — for each of the 6 domains, score =
   `Σ |shap_j|` over that domain's §4.2 features (magnitude of the domain's contribution). Exactly
   6 keys (throttle, braking, speed, gear, engine, cornering).
7. **NL (§6.1 template engine):** `describe_style(radar, config)` — rank the 6 domains by score;
   if the spread is degenerate — **all six within `shap.nl_dominance_margin` (10%) of each other**
   (i.e. `(max-min) <= margin * max`) → `generic=True`, `dominant_domains=[]`, and the **verbatim**
   generic sentence: *"This driver shows balanced characteristics across all domains with no single
   dominant style trait."* (§6.2 FM3 — do not fabricate a dominant trait). Otherwise pick the top
   `shap.nl_top_domains` (2) domains by magnitude, **dropping any with a zero (or negative)
   contribution** (naming a domain that contributed nothing is the same FM3 fabrication) — so a
   radar with a single non-zero domain yields a one-domain sentence, not a padded two. Compose the
   surviving domains' template phrases into the sentence. A generic description is **not** an error
   (`status=OK`).
8. **Latency:** stamp `latency_ms`; if `> shap.single_lap_warn_ms` (50) add `PERF_WARN` to `flags`
   + one DQ row (does not by itself degrade `status`).

**`aggregate_domains` (pure):** sum of `|shap|` per §4.2 domain over `DOMAIN_FEATURES`. F22–F28
(Interaction / Phase / Degradation, 9 features) are **not** part of the 6 radar domains — the
master names exactly six (§6.1 "the 6 feature domains (throttle, braking, speed, gear, engine,
cornering)"); those 9 features appear in the beeswarm/force (all 28) but not the radar. Documented
divergence (§9), pinned by a validator on `DOMAIN_FEATURES`.

**`build_global_explainability` (offline, once) — never raises:** run the explainer over the
`background` matrix; the **beeswarm** payload = per-feature `{ shap, feature_value }` per lap
(each dot = a lap, x-axis = SHAP value, colour = feature value — §6.1), written to
`artefacts/<shap.beeswarm_filename>` deterministically (sorted keys, no timestamp). `per_driver_shap`
= mean `|SHAP|` per feature within each driver's laps (→ `fingerprints.shap_global`);
`per_driver_radar` = `aggregate_domains` of each driver's mean `|SHAP|` (→ `fingerprints.radar_values`,
6D). Timed against `shap.batch_warn_ms` (2000 for 100 laps → `PERF_WARN`). A SHAP fault → DEGRADED +
`SHAP_UNAVAILABLE`, empty payloads, logged — the artefact build never crashes the training pipeline.

**`season_stability` (multi-season drift, §6.1) — never raises:** for each pair of available
seasons, bin each feature's SHAP values into `shap.stability_js_bins` (20) normalised histograms
and compute the **Jensen-Shannon divergence** (base-2, so ∈ [0, 1]) per feature, then average over
features → the pair's JS. A pair with **no comparable feature** (e.g. a season whose SHAP is
entirely NaN/empty) is **uncomparable — dropped, not scored 0.0** (all-NaN is *missing data*, not
zero drift; scoring it 0 would silently report perfect stability). `stability_score` = mean over the
**comparable** season-pairs (lower = more stable; "quantifies how much each driver's fingerprint
drifted year-on-year" — §6.1). With `seasons_available < shap.stability_min_seasons` (3) set
`partial=True`, `status=DEGRADED`; with `< 2` seasons — or when **no pair is comparable** — no score
exists → `stability_score=None` (never a fabricated 0.0), `status=DEGRADED`, logged. F07 reports
`seasons_available`; the **206-vs-200** decision for `GET /drivers/{id}/stability` is F13's HTTP
mapping, not F07's.

**status contract:** clean local explanation (incl. generic-balanced NL) → `OK`;
`SHAP_CONSISTENCY_WARN` → `DEGRADED` (SHAP returned + flagged); `SHAP_UNAVAILABLE` → `DEGRADED`
(prediction preserved, no SHAP). F07 never emits `FAILED` — the prediction is owned by F06b and is
never blocked by explainability.

## 5. config.yaml keys this feature adds
F07 adds a new typed `shap:` block (a `ShapConfig` sub-model on `AppConfig`, `extra="forbid"`, per
the standing typed-config rule). It **reuses** the existing `SHAP_UNAVAILABLE` / `SHAP_CONSISTENCY_WARN`
error codes (already seeded in `errors.py` §6.2) and the existing `PERF_WARN` flag — **no new error
code**. The six radar domains and their feature membership are **structural** (from §4.2) and live in
code as `DOMAIN_FEATURES` (like `FEATURE_IDS`), not config.
```yaml
shap:
  consistency_tolerance: 0.01     # |base + Σshap - tree_margin| above this -> SHAP_CONSISTENCY_WARN (§6.2)
                                  #   master says "far from expected" without a number -> documented design decision
  nl_dominance_margin: 0.10       # all 6 domain scores within this rel. band -> generic description (§6.1/§6.2)
  nl_top_domains: 2               # top-N domains by |magnitude| generate the style sentence (§6.1 "Top 2 domains")
  single_lap_warn_ms: 50          # single-lap SHAP over this -> PERF_WARN (§6.1 "< 50ms" budget)
  batch_warn_ms: 2000             # batch-of-100 SHAP over this -> PERF_WARN (§6.1 "< 2 seconds" budget)
  beeswarm_filename: shap_beeswarm.json  # global beeswarm artefact under paths.artefacts_dir (§6.1)
  stability_js_bins: 20           # histogram bins for the per-feature Jensen-Shannon divergence (§6.1)
  stability_min_seasons: 3        # fewer available seasons -> partial stability (F13 returns 206) (§8.5)
```
**Validator placement:** intra-block checks (`0 < consistency_tolerance`, `0 < nl_dominance_margin
< 1`, `nl_top_domains` in `1..6`, positive warn budgets, non-empty `beeswarm_filename`,
`stability_js_bins >= 2`, `stability_min_seasons >= 2`) live on `ShapConfig`; a mis-edit there is a
load-time `CONFIG_001`. `DOMAIN_FEATURES` is a **code constant** (not config), so it gets a
module-import validator (`_validate_domain_features`): its 6 keys are exactly the six radar domains
and its values partition `F01..F21` with no overlap/gap. A mis-edit is a fail-loud import-time
`ValueError` (the module refuses to import), not `CONFIG_001` — same "fail loud, never silent"
intent, different layer (code vs YAML).

**Build-env note.** `shap` is **not yet installed** (verified this session). F07 adds `shap` 0.45+
(§6.1). Per the F06a precedent, the pinned wheel will be the newest release with a Python-3.14 wheel
(the venv is Python 3.14 / numpy 2 / pandas 3); record the resolved version in `requirements.lock`
and back-note any transitive stack movement, exactly as F06a did for the ML deps.

## 6. Error codes & failure modes (the failsafe contract)
Explainability **never raises** and **never blocks a prediction**. All three master §6.2 modes are
handled; F07 adds **no new error code** (both SHAP codes pre-exist).

| Failure mode | Detection | Error code / flag | System response | Logged? | User-facing message |
|--------------|-----------|-------------------|-----------------|---------|---------------------|
| SHAP computation raises (e.g. model-version mismatch); or explainer not-ready; or `predicted_class ∉ drivers` | try/except around `explainer(x)` + startup build | `SHAP_UNAVAILABLE`, `status=DEGRADED` | **return the prediction without SHAP**; `shap_values=None`, `shap_available=False`, `force=[]`, `radar={}`, `nl_description=""`; log full traceback detail | yes | "Explanation unavailable for this prediction." (prediction still shown) |
| SHAP values sum far from expected (additivity/consistency check fails) | `|base + Σshap − tree_margin| > consistency_tolerance` | `SHAP_CONSISTENCY_WARN` (flag), `status=DEGRADED` | **return SHAP values with the warning flag**; `shap_available=True`; include in monitoring | yes | none (SHAP shown with a consistency warning badge) |
| NL template finds no dominant domain (all 6 within 10%) | `(max−min) <= nl_dominance_margin * max` | *(not an error)* `status=OK`, `dominant_domains=[]` | return the **verbatim generic** sentence; do **not** fabricate a dominant trait | no | "This driver shows balanced characteristics across all domains with no single dominant style trait." |
| Single-lap SHAP > 50 ms / batch > 2 s | per-call timing | `PERF_WARN` (flag) | return result + `PERF_WARN` in `flags` | yes | none (monitoring only) |
| Multi-season stability with <3 seasons | `seasons_available < stability_min_seasons` | *(not an error)* `status=DEGRADED`, `partial=True` | compute what pairs exist; `<2` → `stability_score=None`; F13 maps to **206** | yes | "Multi-season stability partial: only N season(s) of data." |

The generic-balanced description is a **success** path (`status=OK`), not a failure — it is listed
here because §6.2 documents it as a handled mode. The **User-facing message** column is illustrative
copy F13/F16 render; tests assert `error_code` / `flags` / `status` / field values, not prose.

## 7. Acceptance criteria (each item mechanically testable by /spec-check)
- [ ] **Exact tree SHAP, not the calibrated wrapper:** `build_explainer` constructs
      `shap.TreeExplainer` over the **base `XGBClassifier`** extracted from
      `champion.calibrated_classifiers_[0].estimator.estimator`; a test asserts the explained object
      is an `xgb.XGBClassifier` (not the `CalibratedClassifierCV`/`FrozenEstimator`).
- [ ] **28-feature local SHAP:** `explain_prediction` returns `shap_values` with exactly the 28
      `FEATURE_IDS` keys for the predicted class on a clean path; `force` lists all 28 ordered by
      `|shap|` desc with a `toward`/`away` direction matching `sign(shap)`.
- [ ] **SHAP never blocks the prediction (§6.2 FM1):** a stub explainer whose call raises → result
      is `SHAP_UNAVAILABLE`, `status=DEGRADED`, `shap_values is None`, `shap_available is False`,
      and **no exception propagates**; the F06b `PredictionResult` it augments is unchanged.
- [ ] **Not-ready explainer degrades cleanly:** `build_explainer` on a champion whose base tree
      cannot be extracted returns `ready=False` (no raise); `explain_prediction` on it →
      `SHAP_UNAVAILABLE`.
- [ ] **Consistency check (§6.2 FM2):** when `|base + Σshap − tree_margin| > consistency_tolerance`
      → `SHAP_CONSISTENCY_WARN` in `flags`, `status=DEGRADED`, **`shap_available is True`** (values
      still returned); within tolerance → `consistency_ok is True`, no warn flag. Both sides of the
      tolerance asserted with a stub.
- [ ] **6-axis radar over the named domains:** `aggregate_domains` returns exactly the 6 keys
      `{throttle,braking,speed,gear,engine,cornering}`; each score equals `Σ|shap|` over that
      domain's §4.2 features; F22–F28 contribute to **none** of the 6 axes.
- [ ] **`DOMAIN_FEATURES` integrity:** a validator asserts the 6 domains partition `F01..F21`
      exactly (no overlap, no gap) with the §4.2 membership.
- [ ] **NL top-2 domains:** a radar with two clearly dominant domains → `dominant_domains` is those
      two (by magnitude) and the sentence references them; `generic is False`, `status=OK`.
- [ ] **NL generic fallback (§6.2 FM3):** a radar with all 6 domains within `nl_dominance_margin`
      (10%) → `generic is True`, `dominant_domains == []`, and the **verbatim** generic sentence;
      the code never fabricates a dominant trait; `status=OK` (not an error).
- [ ] **Global beeswarm artefact:** `build_global_explainability` writes
      `artefacts/<shap.beeswarm_filename>` (each entry a lap with per-feature `{shap,feature_value}`)
      deterministically (byte-identical on re-run for the same background); returns
      `per_driver_shap` (mean|SHAP| per feature) and `per_driver_radar` (6D per driver).
- [ ] **Global build never raises:** a SHAP fault in the global path → `status=DEGRADED`,
      `SHAP_UNAVAILABLE`, empty payloads, `logged=True`, no exception.
- [ ] **Multi-season stability (§6.1):** `season_stability` over 3 seasons returns
      `stability_score` = mean pairwise **Jensen-Shannon divergence** (∈ [0,1]) with a `pairwise_js`
      entry per season-pair; identical per-season distributions → `stability_score == 0.0`;
      **`seasons_available < stability_min_seasons` → `partial=True`, `status=DEGRADED`**; **<2
      seasons → `stability_score is None`** (never fabricated 0.0). Never raises.
- [ ] **Held-out never explained:** `build_global_explainability` is exercised only on the
      normalised **training-fit** background; a test asserts Belgium/Italy rows are absent from the
      supplied background (documents the caller contract) — no `.fit` / no held-out laps touch F07.
- [ ] **No retrain / no fit:** static source scan — no `.fit(`, `fit_normaliser`, `optuna`,
      `CalibratedClassifierCV(` in the module; F07 only builds explainers + explains.
- [ ] **Latency flags (both sides):** single-lap timed 51 ms → `PERF_WARN`; 49/50 ms → absent
      (monkeypatched timer); batch > `batch_warn_ms` → `PERF_WARN`. Latency alone does not set
      `status=DEGRADED`.
- [ ] **Logged + DQ row:** for each logged mode (`SHAP_UNAVAILABLE`, `SHAP_CONSISTENCY_WARN`,
      `PERF_WARN`) `result.logged is True` AND a `data_quality_log` row with the matching code is
      written (asserted against a tmp log).
- [ ] **Config-driven:** the code reads `shap.{consistency_tolerance,nl_dominance_margin,
      nl_top_domains,single_lap_warn_ms,batch_warn_ms,beeswarm_filename}`; a literal scan finds no
      `0.01`/`0.10`/`50`/`2000` threshold literal in the module body.
- [ ] **Config validator:** `AppConfig` load raises when `nl_dominance_margin` ∉ (0,1),
      `nl_top_domains` ∉ 1..6, `consistency_tolerance <= 0`, a warn budget `<= 0`, or
      `beeswarm_filename` empty.
- [ ] Every error code / flag above has a unit test that triggers it and asserts the result.

## 8. Test plan
- **Fixtures (`tests/ml/conftest.py`, extend):** reuse F06a's module-scoped `trained` run (real
  `classifier.pkl` + `normalisation.json` + `labels.json` under tmp) → `load_model` → a genuine
  `ModelHandle`, so `build_explainer` builds a real TreeExplainer over the real champion and one
  end-to-end `explain_prediction` runs on an actual lap. Add a `StubExplainer` (a `ShapExplainer`
  with a hand-set `values`/`base_values`/`tree_margin`) to pin the consistency and gate boundaries
  deterministically, and a raising variant for the `SHAP_UNAVAILABLE` path.
- **Happy path (real champion):** clean accepted lap → 28-key `shap_values`, `force` ordered,
  `radar` 6 keys, `consistency_ok is True`, `shap_available is True`, `status=OK`,
  `nl_description` non-empty.
- **Boundary fixtures (via StubExplainer):** consistency delta just over / under
  `consistency_tolerance`; radar with 2 dominant domains vs all-6-within-10 %; single-lap latency
  49/50/51 ms.
- **Never-raises / SHAP_UNAVAILABLE:** stub `explainer(x)` raises → coded `SHAP_UNAVAILABLE`, no
  exception, prediction preserved; `build_explainer` on a champion with an unexpected structure →
  `ready=False`; `predicted_class` not in `drivers` → `SHAP_UNAVAILABLE`.
- **Radar / NL purity:** `aggregate_domains` and `describe_style` tested as pure functions —
  hand-built SHAP dict → exact 6 scores; hand-built radar → exact top-2 / generic sentence.
- **Global:** small background (a few laps across ≥2 drivers) → artefact written, byte-identical on
  re-run, `per_driver_shap` / `per_driver_radar` shaped correctly; a raising explainer → DEGRADED +
  empty payloads.
- **Stability (pure):** hand-built per-season SHAP dicts — identical seasons → `stability_score ==
  0.0`; divergent seasons → higher score; 2 seasons → `partial=True`; 1 season → `stability_score
  is None`. JS divergence symmetric and bounded to [0,1].
- **Static scans:** no `.fit(` / `fit_normaliser` / `optuna` in the module; threshold-literal scan
  for `0.01`/`0.10`/`50`/`2000`.
- **Config validator:** assert `AppConfig` rejects each bad `shap:` value.

## 9. Definition of done
All acceptance criteria MET with passing tests; **ml-auditor PASS** (no refit / no `.fit` at
explain time; loaded normalisation reused, never recomputed; held-out laps never enter the beeswarm
background; SHAP failures degrade, never inflate or block); **failsafe-auditor PASS** (all three
§6.2 modes handled + logged + surfaced; prediction never suppressed); `/apex-review F07` clean (no
silent failure — SHAP faults are coded + logged, never a bare `except: pass`; config-driven
thresholds; error-code discipline; no magic numbers); `/spec-check F07` green; `mypy app` clean.
INDEX F07 row → **DONE**. **Limitations for the dissertation:** (1) **SHAP explains the base
XGBoost margins, not the calibrated probabilities** — the isotonic/sigmoid calibrator is a monotone
post-hoc transform; attribution *ranking* is preserved but the values are pre-calibration logits,
not probability contributions (standard practice; state it). (2) **Radar uses exactly the 6 named
domains (F01–F21); F22–F28 (Interaction / Phase / Degradation, 9 features) are shown in beeswarm/
force but not the radar** — the master names six domains; documented divergence, flag alongside the
F06a/F06b 28-vs-34 seam. (3) **`consistency_tolerance` is a design decision** — the master says
"far from expected" without a number; the value is config-driven and calibrated against the
additivity residual observed on training laps, reported in the dissertation. (4) Batch SHAP runs
synchronously here; the **Celery async batch path is F14** — F07 provides the batch function it will
call. (5) The `_base_tree_model` extraction depends on the sklearn ≥1.6 `FrozenEstimator` prefit
structure (one `calibrated_classifiers_` entry); a future multi-fold calibration would need a
mean-over-folds explainer — pinned by the acceptance test so a structure change fails loudly.
(6) **Multi-season stability** requires ≥2 seasons of a driver's laps to compute any JS divergence
and ≥3 (`stability_min_seasons`) for a full score; with the training data (2022–2024) this holds
for known drivers but is `partial`/`None` for unknown or single-season uploads (F13 → 206). The
JS orientation (lower = more stable) and the 20-bin histogram are documented design choices — the
master gives the method (Jensen-Shannon) but not the binning.

## 10. Risks & mitigations (from master risk register where relevant)
- **An unexplained prediction ships silently (§6 headline):** every explain path is coded — a SHAP
  fault is `SHAP_UNAVAILABLE` + logged with full detail, never a swallowed exception; the F13 layer
  can render "explanation unavailable" but the prediction is preserved (§6.2 FM1). failsafe-auditor
  gates this.
- **Explaining the wrong model (calibrated wrapper vs base tree):** `_base_tree_model` is guarded
  and asserted (`isinstance XGBClassifier`); a structure change → `ready=False` → `SHAP_UNAVAILABLE`,
  never a silently wrong attribution.
- **Attribution inconsistency passed off as truth (§6.2 FM2):** the additivity check compares
  `base + Σshap` to the tree margin every call; a mismatch flags `SHAP_CONSISTENCY_WARN` and
  surfaces to monitoring — values are shown *with* the warning, never silently trusted.
- **Fabricated dominant trait (§6.2 FM3):** the 10 %-band check forces the verbatim generic sentence
  when no domain dominates; a boundary test pins it.
- **Normalisation leakage via the background (non-negotiable #2):** the beeswarm background is the
  F06a **training-fit normalised** matrix only; F07 never calls `.fit` / `fit_normaliser` (static
  scan) and never sees Belgium/Italy — the held-out generalisation story is preserved.
- **SHAP dep migration (build env):** `shap` 0.45+ not yet installed; pin the newest Python-3.14
  wheel per the F06a precedent, record in `requirements.lock`, and back-note any transitive movement.
