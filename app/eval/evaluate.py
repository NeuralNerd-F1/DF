"""F20 — the evaluation framework: one async orchestrator over the master's four §11 pillars.

F20 is APEX's evaluation chapter made executable (§11 + §2.4 + §13 Week-20). It **orchestrates; it
does not recompute** (divergence #53): each §11 pillar's metric maths already ships in the feature
that owns it, so ``run_full_evaluation`` *calls* the read-only surfaces (``run_failsafe_suite``,
``lead_time_report``, ``stability_report``, ``teammate_proximity``, ``project_unknown``,
``season_stability``, ``run_ablation``) and *harvests* the classifier's held-out numbers from the
champion :class:`ClassifierTrainingResult`. It **fits nothing** and does no metric arithmetic beyond
comparing a harvested/returned value to its configured target.

Faithful to non-negotiable #1 and §11's "report failure cases alongside success": every pillar that
cannot run (missing artefact / held-out data) returns an explicit ``skipped`` state with a reason —
**never a fabricated number** (a metric's ``value=None`` ⇒ ``passes=None``) — and every metric
carries its pass/fail verdict against the master target. ``run_full_evaluation`` **never raises**: a
pillar-internal fault (or the system-under-eval raising) becomes that pillar's ``skipped=True``
+ a finding (§4.0-g), never an aborted run.

Because APEX commits no model artefacts (only ``artefacts/mlflow.db``) and no feature matrix, F20
runs its **mechanism** offline (the §11.4 failsafe pillar runs via ``run_failsafe_suite``; the three
data/artefact pillars ``skip``); the **real dissertation numbers** come from a single
``@pytest.mark.artefacts`` (+ matrix) run once the artefacts land (divergence #54). F20 adds **no**
error code and **no** model artefact — the ``evaluation_report.json`` is generated evidence, never
committed (like ``failsafe_report.json``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.eval.failsafe import run_failsafe_suite

if TYPE_CHECKING:
    import pandas as pd

    from app.contracts.config import AppConfig
    from app.ml.classifier import ClassifierTrainingResult
    from app.ml.stylespace import StyleSpaceHandle
    from app.monitoring.detector import MonitorHandle

_LOG = logging.getLogger(__name__)

Comparator = Literal[">", ">=", "<", "<=", "in_range", "=="]
PillarName = Literal["classifier", "stability", "degradation", "failsafe", "ablation"]
RQName = Literal["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"]
Verdict = Literal["SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "INSUFFICIENT_DATA"]

_PILLARS: tuple[PillarName, ...] = (
    "classifier",
    "stability",
    "degradation",
    "failsafe",
    "ablation",
)


# --- typed models --------------------------------------------------------------------------------
class MetricResult(BaseModel):
    """One §11 metric gated against its master target. ``value=None`` ⇒ not computed (data/artefact
    absent) ⇒ ``passes=None`` — never a fabricated pass. ``finding`` carries a §11 failure-analysis
    note. For an ``in_range`` gate the ``target`` is a two-element ``[lo, hi]`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | bool | None = None
    target: float | int | bool | list[float] | None = None
    comparator: Comparator | None = None
    passes: bool | None = None
    finding: str | None = None


class PillarResult(BaseModel):
    """One §11 evaluation pillar. ``skipped=True`` (+ ``skip_reason``) when the pillar's input is
    absent (§4.0-g/§4.0-h) — never a silent pass. ``native`` optionally serialises the underlying
    typed result (``FailsafeReport``/``LeadTimeReport``/``AblationReport``/``StabilityReport``)
    so the dissertation can drill down without a second run."""

    model_config = ConfigDict(extra="forbid")

    pillar: PillarName
    metrics: list[MetricResult] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    findings: list[str] = Field(default_factory=list)
    native: dict[str, Any] = Field(default_factory=dict)


class RQVerdict(BaseModel):
    """A research-question verdict (§2.4 — "every RQ answered w/ data"). ``INSUFFICIENT_DATA`` when
    the evidence pillar(s) skipped (offline / no artefacts) — the honest "we built the machine, the
    numbers need the trained run" state. RQ2 is ``PARTIAL`` by construction (§4.0-f)."""

    model_config = ConfigDict(extra="forbid")

    rq: RQName
    question: str
    evidence: list[str] = Field(default_factory=list)
    verdict: Verdict
    findings: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """The full §11 evaluation results table (§13 Week-20 deliverable). Timestamp-free;
    ``content_hash`` covers the reproducible **outcome** (per metric: name + target + comparator +
    pass/fail; per RQ: verdict) — **not** the run-specific stochastic ``value``s (UMAP coords vary
    within Procrustes tolerance; failsafe timing is per-run) — so "same artefacts + seeds → same
    verdicts → same hash" (reproducibility #3, mirrors ``FailsafeReport``, divergence #58)."""

    model_config = ConfigDict(extra="forbid")

    pillars: list[PillarResult] = Field(default_factory=list)
    rq_verdicts: list[RQVerdict] = Field(default_factory=list)
    n_pillars: int = 0
    n_pillars_run: int = 0
    n_pillars_skipped: int = 0
    all_targets_met: bool = False
    content_hash: str = ""

    def render_pillar_tables(self) -> str:
        """One markdown table per pillar (Metric · Value · Target · Pass · Finding)."""
        blocks: list[str] = []
        for p in self.pillars:
            head = f"### {p.pillar.capitalize()} pillar"
            if p.skipped:
                blocks.append(f"{head}\n\n_SKIPPED — {p.skip_reason or 'no reason given'}_")
                continue
            lines = [
                head,
                "",
                "| Metric | Value | Target | Pass | Finding |",
                "|--------|-------|--------|------|---------|",
            ]
            for m in p.metrics:
                value = "—" if m.value is None else str(m.value)
                target = "—" if m.target is None else str(m.target)
                verdict = "SKIP" if m.passes is None else ("PASS" if m.passes else "FAIL")
                comp = f" {m.comparator}" if m.comparator else ""
                lines.append(
                    f"| {m.name} | {value} | {comp.strip()} {target} | {verdict} "
                    f"| {m.finding or '—'} |"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_rq_table(self) -> str:
        """The RQ verdict table (RQ · Verdict · Evidence · Finding)."""
        lines = [
            "### Research questions (§2.4)",
            "",
            "| RQ | Verdict | Evidence | Finding |",
            "|----|---------|----------|---------|",
        ]
        for v in self.rq_verdicts:
            evidence = ", ".join(v.evidence) or "—"
            finding = "; ".join(v.findings) or "—"
            lines.append(f"| {v.rq} | {v.verdict} | {evidence} | {finding} |")
        return "\n".join(lines)

    def render_report(self) -> str:
        """The full markdown evaluation chapter: the four pillar tables + the RQ verdict table + a
        failures roll-up (the §11 "equal space to where APEX fails" contract)."""
        failures: list[str] = []
        for p in self.pillars:
            if p.skipped:
                failures.append(f"- **{p.pillar}** — SKIPPED: {p.skip_reason}")
            for m in p.metrics:
                if m.passes is False:
                    failures.append(
                        f"- **{p.pillar}.{m.name}** — FAIL: {m.finding or 'target not met'}"
                    )
        roll = "\n".join(failures) if failures else "- none"
        return (
            "## Evaluation results (§11)\n\n"
            f"_{self.n_pillars_run}/{self.n_pillars} pillars ran; "
            f"{self.n_pillars_skipped} skipped. "
            f"all_targets_met={self.all_targets_met}. content_hash={self.content_hash[:12]}…_\n\n"
            f"{self.render_pillar_tables()}\n\n"
            f"{self.render_rq_table()}\n\n"
            f"### Failures & skips roll-up\n{roll}"
        )


# --- gate + serialisation helpers ----------------------------------------------------------------
def _compare(value: float | int | bool, comparator: Comparator, target: Any) -> bool:
    """Pure value-vs-target comparison (the ONLY 'maths' F20 does — no metric arithmetic)."""
    if comparator == ">":
        return bool(value > target)
    if comparator == ">=":
        return bool(value >= target)
    if comparator == "<":
        return bool(value < target)
    if comparator == "<=":
        return bool(value <= target)
    if comparator == "==":
        return bool(value == target)
    # in_range: target is [lo, hi], inclusive
    lo, hi = target
    return bool(lo <= value <= hi)


def _gate(
    name: str,
    value: float | int | bool | None,
    target: float | int | bool | list[float] | None,
    comparator: Comparator,
    *,
    finding: str | None = None,
) -> MetricResult:
    """Build a :class:`MetricResult`, gating ``value`` against ``target``. ``value=None`` (or NaN)
    ⇒ ``passes=None``; a **NaN target** (e.g. a harvested ``held_out_macro_f1`` from an empty test
    split, used as the ablation-drop target) also ⇒ ``passes=None`` — a metric that could not be
    computed is never a fabricated pass or a spurious FAIL (§4.0-b: ``x < nan`` is False)."""
    if value is not None and isinstance(value, float) and math.isnan(value):
        value = None
    if value is None:
        return MetricResult(
            name=name, value=None, target=target, comparator=comparator, passes=None,
            finding=finding,
        )
    if target is None:  # no configured target (absent eval: block) → cannot gate, not a pass
        return MetricResult(
            name=name, value=value, target=None, comparator=comparator, passes=None,
            finding=finding or "no target configured (config block absent)",
        )
    if isinstance(target, float) and not math.isfinite(target):  # NaN/inf target = not computable
        return MetricResult(
            name=name, value=value, target=None, comparator=comparator, passes=None,
            finding=finding or "target not computable (NaN) — reported as skipped, not FAIL",
        )
    passes = _compare(value, comparator, target)
    return MetricResult(
        name=name, value=value, target=target, comparator=comparator, passes=passes,
        finding=finding,
    )


def _skipped(pillar: PillarName, reason: str) -> PillarResult:
    """A pillar that could not run — typed + reasoned, never a silent pass (§4.0-h)."""
    return PillarResult(pillar=pillar, skipped=True, skip_reason=reason)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce a folded native payload into JSON-native values so the ``allow_nan=False``
    artefact write never trips: non-finite floats (NaN/±inf) → ``None`` (e.g. an empty-compound
    ``LeadTimeReport``); numpy scalars → their Python scalar; any other non-native type → ``str`` (a
    visible fallback, never a crash). Pydantic ``model_dump()`` coerces typed fields; this is
    the belt for a stray value nested in a ``native: dict[str, Any]``."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (str, int)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    item = getattr(obj, "item", None)  # numpy / pandas scalar → Python scalar
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # noqa: BLE001 - fall through to the str fallback
            pass
    return str(obj)  # last-resort visible fallback (never a serialisation crash)


def _held_out_matrix(
    matrix: pd.DataFrame | None, config: AppConfig
) -> tuple[pd.DataFrame | None, str | None]:
    """The sanctioned evaluation slice: the held-out circuits (Belgium/Italy) — F20's **ONLY**
    evaluation touch-point (§3.2, §4.0-h, div #61). Returns ``(slice, None)`` or ``(None, reason)``.
    A matrix with no ``circuit``/``split`` column, or **no held-out rows**, → ``(None, reason)`` so
    the generalisation pillars (degradation lead-time, unknown-driver projection, the §11.4 rate
    extensions) skip with a reason rather than silently score on training data — the top risk for
    an evaluation feature (§10). The fit/self-quarantine surfaces (UMAP Procrustes, F09 ablation)
    keep the full matrix; they exclude held-out internally via F04's leakage guards."""
    if matrix is None:
        return None, "no feature matrix supplied"
    import pandas as pd

    cols = set(matrix.columns)
    if "circuit" not in cols and "split" not in cols:
        return None, (
            "matrix has no 'circuit'/'split' column — cannot isolate the held-out slice (§4.0-h)"
        )
    held = list(config.held_out_circuits or [])
    if "circuit" in cols and held:
        sliced = matrix[matrix["circuit"].isin(held)]
    else:
        sliced = matrix[matrix["split"].astype(str).str.lower() == "test"]
    if len(sliced) == 0:
        return None, (
            f"no held-out {held or 'test-split'} rows in matrix — refusing to score on training "
            "data (§4.0-h)"
        )
    return pd.DataFrame(sliced).copy(), None


def _canonical(report: EvaluationReport) -> str:
    """Canonical body for the content hash: the reproducible **outcome** only (per metric
    name/target/comparator/passes + each pillar's skipped flag; per RQ its verdict) — NOT the
    run-specific stochastic ``value``s or ``native`` (mirrors ``FailsafeReport._canonical``,
    §4.3). So the hash certifies "same inputs → same verdicts", the reproducibility that matters."""
    pillars = [
        {
            "pillar": p.pillar,
            "skipped": p.skipped,
            "metrics": [
                {
                    "name": m.name,
                    "target": m.target,
                    "comparator": m.comparator,
                    "passes": m.passes,
                }
                for m in p.metrics
            ],
        }
        for p in report.pillars
    ]
    rqs = [{"rq": v.rq, "verdict": v.verdict} for v in report.rq_verdicts]
    body = {"pillars": pillars, "rqs": rqs, "all_targets_met": report.all_targets_met}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_report(
    report: EvaluationReport, config: AppConfig, artefact_dir: str | Path | None
) -> Path:
    """Write the results table to ``artefacts/evaluation_report.json`` (F04/F10/F15 idiom: full
    ``model_dump()`` with sort_keys, indent=2, allow_nan=False, embedded hash, no timestamp).
    The artefact is generated, **never committed** (like every ``artefacts/*``)."""
    adir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    fname = config.eval_.report_filename if config.eval_ else "evaluation_report.json"
    path = adir / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(report.model_dump()), sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --- pillar: §11.1 classifier (harvest, §4.0-c) --------------------------------------------------
def evaluate_classifier(
    result: ClassifierTrainingResult | None,
    *,
    config: AppConfig,
    log_path: str | Path | None = None,
) -> PillarResult:
    """Harvest the champion's held-out §11.1 numbers and gate them against the configured targets
    (public seam — F21/F22 call it directly on a persisted training result). Harvests, never
    re-scores: re-computing would touch held-out data a second time outside F06a's leakage guard
    (§4.0-c). ``result=None`` → ``skipped`` (no fabricated 0.0)."""
    if result is None:
        return _skipped(
            "classifier",
            "no champion training result available — run F06a train_classifier "
            "(or inject classifier_result); harvested, never re-scored (§4.0-c)",
        )
    if config.classifier is None:  # guarded by the eval: cross-block validator, belt-and-braces
        return _skipped("classifier", "classifier: config block absent")

    target_f1 = config.classifier.target_macro_f1
    floor = config.classifier.per_driver_floor
    per_driver = result.held_out_per_driver_f1 or {}
    worst_driver_f1 = min(per_driver.values()) if per_driver else None
    worst_name = min(per_driver, key=lambda d: per_driver[d]) if per_driver else "?"

    metrics = [
        _gate("held_out_macro_f1", result.held_out_macro_f1, target_f1, ">"),
        _gate(
            "held_out_per_driver_f1_min",
            worst_driver_f1,
            floor,
            ">",
            finding=(None if worst_driver_f1 is None else f"worst driver: {worst_name}"),
        ),
        # circuit generalisation (§11.2) REUSES the §11.1 held-out per-driver F1 (§4.0-d) — surfaced
        # under both pillars (shared), not a second computation.
        _gate(
            "circuit_generalisation_f1_min",
            worst_driver_f1,
            floor,
            ">",
            finding="§11.2 circuit generalisation = the shared held-out per-driver F1",
        ),
        _gate("beats_all_baselines", result.beats_all_baselines, True, "=="),
        # normalisation ablation: F1 should DROP without norm (§11.1 "drops significantly").
        _gate(
            "normalisation_ablation_drop",
            result.normalisation_ablation_macro_f1,
            result.held_out_macro_f1,
            "<",
            finding=(
                f"held-out {result.held_out_macro_f1:.3f} vs no-norm "
                f"{result.normalisation_ablation_macro_f1:.3f}"
            ),
        ),
    ]
    findings: list[str] = []
    if result.baseline_macro_f1:
        findings.append(
            "baselines: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(result.baseline_macro_f1.items()))
        )
    if per_driver:
        below = sorted(d for d, f1 in per_driver.items() if f1 <= floor)
        if below:
            findings.append(f"per-driver F1 at/below {floor}: {below} (§11.1 failure analysis)")
    return PillarResult(
        pillar="classifier",
        metrics=metrics,
        findings=findings,
        native={"classifier_training_result": _json_safe(result.model_dump())},
    )


# --- pillar: §11.2 stability (call the F08/F07 surfaces) -----------------------------------------
def _pillar_stability(
    handle: StyleSpaceHandle | None,
    matrix: pd.DataFrame | None,
    held_out: pd.DataFrame | None,
    teams: Any,
    config: AppConfig,
    *,
    classifier_result: ClassifierTrainingResult | None,
    held_out_reason: str | None = None,
    log_path: str | Path | None = None,
) -> PillarResult:
    """§11.2 fingerprint stability: UMAP Procrustes stability, teammate proximity, unknown-driver
    projection, and year-on-year Jensen-Shannon — each CALLED from the F08/F07 surfaces (§4.0-a).
    Procrustes fits over the full ``matrix`` (self-quarantines held-out via F04); the unknown-driver
    projection scores on the ``held_out`` slice ONLY (§4.0-h). Absent style-space handle → skip."""
    if handle is None:
        return _skipped("stability", "no committed style space (umap_model.pkl absent)")
    if config.stylespace is None:
        return _skipped("stability", "stylespace: config block absent")

    from app.ml.stylespace import stability_report, teammate_proximity

    metrics: list[MetricResult] = []
    findings: list[str] = []
    native: dict[str, Any] = {}

    # UMAP Procrustes stability over N seeds (needs the training laps → full matrix)
    if matrix is None:
        metrics.append(_gate("umap_procrustes_variance", None,
                             config.stylespace.stability_variance_threshold, "<="))
        findings.append("Procrustes stability skipped — no feature matrix supplied")
    else:
        rep = stability_report(matrix, config=config, log_path=log_path)
        native["stability_report"] = _json_safe(rep.model_dump())
        metrics.append(
            _gate("umap_procrustes_variance", rep.procrustes_variance,
                  config.stylespace.stability_variance_threshold, "<=")
        )

    # teammate proximity (intra-team < inter-team) — needs the driver→team map
    if teams is None:
        metrics.append(_gate("teammate_proximity", None, True, "=="))
        findings.append("teammate proximity skipped — no driver→team map supplied")
    else:
        prox = teammate_proximity(handle, teams, config=config)
        native["teammate_proximity"] = _json_safe(prox.model_dump())
        metrics.append(_gate("teammate_proximity", prox.passes, True, "=="))

    # unknown-driver projection: re-insert each known driver (held-out laps ONLY) as "unknown"
    nn_target = config.eval_.unknown_nn_correct_min if config.eval_ else None
    nn_metric, nn_finding = _unknown_projection_metric(
        handle, held_out, nn_target, config, log_path, held_out_reason,
    )
    metrics.append(nn_metric)
    if nn_finding:
        findings.append(nn_finding)

    # year-on-year Jensen-Shannon (§11.2, via F07 season_stability) — needs per-season SHAP
    js_target = config.eval_.stability_js_target if config.eval_ else None
    metrics.append(
        _gate("year_on_year_js", None, js_target, "<",
              finding="season_stability skipped — needs per-season SHAP (artefact-gated)")
    )
    return PillarResult(pillar="stability", metrics=metrics, findings=findings, native=native)


def _unknown_projection_metric(
    handle: StyleSpaceHandle,
    held_out: pd.DataFrame | None,
    target: int | None,
    config: AppConfig,
    log_path: str | Path | None,
    held_out_reason: str | None,
) -> tuple[MetricResult, str | None]:
    """Re-insert each known driver's **held-out** laps as an 'unknown' projection and count how many
    land on the correct nearest neighbour (== themselves). The assigned driver is ``nn[0].driver``
    (F08 has no scalar assigned-driver field; the list sorts by descending cosine); empty list
    (raw-NN fallback) = incorrect. Gated vs ``eval.unknown_nn_correct_min``. A held-out slice
    that is absent / empty (§4.0-h) → skip with the reason, never score on training data."""
    if held_out is None:
        return _gate("unknown_driver_nn_correct", None, target, ">="), (
            held_out_reason or "unknown-driver projection skipped — no held-out feature matrix"
        )
    from app.ml.stylespace import project_unknown

    drivers = list(getattr(handle, "drivers", []) or [])
    if not drivers or "driver" not in held_out.columns or "circuit" not in held_out.columns:
        return _gate("unknown_driver_nn_correct", None, target, ">="), (
            "unknown-driver projection skipped — held-out slice lacks driver/circuit columns"
        )
    correct = 0
    for drv in drivers:
        laps = held_out[held_out["driver"] == drv]
        if laps.empty:
            continue
        circuit = str(laps["circuit"].iloc[0])
        proj = project_unknown(
            laps, circuit=circuit, handle=handle, config=config, log_path=log_path
        )
        nn = proj.nearest_neighbours
        if nn and nn[0].driver == drv:
            correct += 1
    return _gate("unknown_driver_nn_correct", correct, target, ">="), (
        f"{correct}/{len(drivers)} known drivers projected to the correct nearest neighbour"
    )


# --- pillar: §11.3 degradation (call lead_time_report) -------------------------------------------
def _pillar_degradation(
    handle: MonitorHandle | None,
    held_out: pd.DataFrame | None,
    config: AppConfig,
    *,
    held_out_reason: str | None = None,
    log_path: str | Path | None = None,
) -> PillarResult:
    """§11.3 degradation lead time: CALL ``lead_time_report`` on the held-out stints ONLY (§4.0-h)
    and gate its mean lead time / TP / FP against the ``eval:`` targets; surface per-compound +
    sensitivity as findings (§4.0-a). Absent detector / held-out slice / eval: block → skip."""
    if handle is None:
        return _skipped("degradation", "no committed detector (stint_variation.json absent)")
    if held_out is None:
        return _skipped(
            "degradation",
            held_out_reason or "no held-out feature matrix — nothing to score",
        )
    if config.eval_ is None:
        return _skipped("degradation", "eval: config block absent")

    from app.monitoring.detector import lead_time_report

    rep = lead_time_report(held_out, handle=handle, config=config, log_path=log_path)
    ev = config.eval_
    metrics = [
        _gate("mean_lead_time_laps", rep.mean_lead_time_laps,
              [float(ev.lead_time_min_laps), float(ev.lead_time_max_laps)], "in_range"),
        _gate("true_positive_rate", rep.tp_rate, ev.tp_rate_target, ">="),
        _gate("false_positive_rate", rep.fp_rate, ev.fp_rate_max, "<="),
    ]
    findings: list[str] = [f"n_stints={rep.n_stints}"]
    if rep.per_compound:
        findings.append(f"per-compound reported: {sorted(rep.per_compound)}")
    if rep.optimal_threshold_per_circuit_type:
        findings.append(
            f"optimal τ per circuit-type: {rep.optimal_threshold_per_circuit_type}"
        )
    if rep.sensitivity:
        findings.append(f"threshold sensitivity grid: {sorted(rep.sensitivity)}")
    return PillarResult(
        pillar="degradation", metrics=metrics, findings=findings,
        native={"lead_time_report": _json_safe(rep.model_dump())},
    )


# --- pillar: §11.4 failsafe (fold run_failsafe_suite, §4.0-e) ------------------------------------
async def _pillar_failsafe(
    config: AppConfig, artefact_dir: str | Path | None, held_out: pd.DataFrame | None
) -> PillarResult:
    """§11.4 RQ5: CALL ``run_failsafe_suite`` (F15) and fold the whole ``FailsafeReport`` in — F20
    does NOT reimplement the 7 tests (§4.0-e). Runs offline (mechanism, no artefacts). When the
    committed artefacts + a held-out slice are present, additionally runs the two artefact-gated
    *rate* extensions F15 deferred (Test 1 ≥90% LOW_CONFIDENCE rate; Test 6 real 5-SD detection);
    offline both skip."""
    report = await run_failsafe_suite(config, artefact_dir=artefact_dir)
    failed = report.total - report.passed - report.skipped
    metrics = [
        _gate(
            "failsafe_tests_pass_or_documented",
            report.passed + report.skipped,
            report.total,
            "==",
            finding=(None if failed == 0 else f"{failed} of {report.total} §11.4 tests FAILED"),
        )
    ]
    metrics.extend(_failsafe_rate_extensions(config, artefact_dir, held_out))
    findings = [
        "Mechanism validated in-process; production p95 + real model rates via the artefact-gated "
        "run (F15 div #39/#41).",
    ]
    return PillarResult(
        pillar="failsafe", metrics=metrics, findings=findings,
        native={"failsafe_report": _json_safe(report.model_dump())},
    )


def _failsafe_rate_extensions(
    config: AppConfig, artefact_dir: str | Path | None, held_out: pd.DataFrame | None
) -> list[MetricResult]:
    """The two §11.4 rate assertions F15 deferred (§4.0-e). Offline (champion/umap or held-out slice
    absent) → two ``value=None`` metrics (artefact-gated), honest. When artefacts + held-out slice
    are present, measure the real rates via the surfaces (``predict`` for the LOW_CONFIDENCE
    rate on ambiguous input; ``project_unknown`` for 5-SD OOD detection); any fault degrades to
    ``value=None`` + a finding (never a crash, never a fabricated pass). ``config.failsafe`` is
    guaranteed present here — the pillar reached this only because ``run_failsafe_suite`` (which
    requires it) already succeeded."""
    assert config.failsafe is not None  # noqa: S101 - guaranteed post-run_failsafe_suite
    rate_target = config.failsafe.ambiguous_low_conf_rate
    adir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    champ = (
        adir / config.classifier.model_artefact_filename
        if config.classifier is not None
        else None
    )
    umap = adir / config.stylespace.umap_model_filename if config.stylespace is not None else None
    champ_present = champ is not None and champ.is_file()
    umap_present = umap is not None and umap.is_file()

    if not (champ_present and held_out is not None):
        test1 = _gate("test1_low_confidence_rate", None, rate_target, ">=",
                      finding="artefact-gated — real ≥90% rate needs champion + held-out matrix")
    else:  # pragma: no cover - artefact-gated, exercised only in the @pytest.mark.artefacts run
        test1 = _test1_low_confidence_rate(config, artefact_dir, held_out, rate_target)

    if not (umap_present and held_out is not None):
        test6 = _gate("test6_ood_5sd_detection", None, True, "==",
                      finding="artefact-gated — real 5-SD detection needs umap + held-out matrix")
    else:  # pragma: no cover - artefact-gated
        test6 = _test6_ood_5sd_detection(config, artefact_dir, held_out)
    return [test1, test6]


def _test1_low_confidence_rate(  # pragma: no cover - artefact-gated (@pytest.mark.artefacts)
    config: AppConfig, artefact_dir: str | Path | None, held_out: pd.DataFrame, rate_target: float
) -> MetricResult:
    """§11.4 Test 1 real rate: the fraction of *ambiguous* inputs (each the elementwise mean of two
    drivers' held-out laps) the real champion returns as LOW_CONFIDENCE — must be ≥ the configured
    floor. A fault → ``value=None`` + a finding (visible, never a fabricated pass)."""
    from app.contracts.enums import ConfidenceState
    from app.ml.inference import load_model, predict
    from app.pipeline.features import FEATURE_IDS

    assert config.failsafe is not None  # noqa: S101 - guaranteed post-run_failsafe_suite
    name = "test1_low_confidence_rate"
    try:
        handle = load_model(config=config, artefact_dir=artefact_dir)
        n = config.failsafe.n_ambiguous_samples
        drivers = sorted(str(d) for d in held_out["driver"].dropna().unique())
        if len(drivers) < 2:
            return _gate(name, None, rate_target, ">=",
                         finding="fewer than 2 held-out drivers — cannot form ambiguous pairs")
        feat_cols = [fid for fid in FEATURE_IDS if fid in held_out.columns]
        low = total = 0
        for i in range(n):
            da = held_out[held_out["driver"] == drivers[i % len(drivers)]]
            db = held_out[held_out["driver"] == drivers[(i + 1) % len(drivers)]]
            if da.empty or db.empty:
                continue
            a, b = da.iloc[i % len(da)], db.iloc[i % len(db)]  # vary the lap across iterations
            fv: dict[str, float | None] = {
                fid: (float(a[fid]) + float(b[fid])) / 2.0 for fid in feat_cols
            }
            pred = predict(fv, circuit=str(a["circuit"]), handle=handle, config=config)
            total += 1
            if pred.confidence_state == ConfidenceState.LOW_CONFIDENCE:
                low += 1
        if total == 0:
            return _gate(name, None, rate_target, ">=",
                         finding="no ambiguous pairs could be formed from the held-out slice")
        return _gate(name, low / total, rate_target, ">=",
                     finding=f"{low}/{total} ambiguous inputs returned LOW_CONFIDENCE")
    except Exception as exc:  # noqa: BLE001 - a rate fault is a visible skip, never a false pass
        return _gate(name, None, rate_target, ">=", finding=f"rate not computed: {exc}")


def _test6_ood_5sd_detection(  # pragma: no cover - artefact-gated (@pytest.mark.artefacts)
    config: AppConfig, artefact_dir: str | Path | None, held_out: pd.DataFrame
) -> MetricResult:
    """§11.4 Test 6 real detection: a held-out lap set shifted ``ood_injection_sd`` SDs outside
    training must project to ``ood=True`` through the UMAP. A fault → ``value=None`` + finding."""
    import pandas as pd

    from app.ml.stylespace import load_style_space, project_unknown
    from app.pipeline.features import FEATURE_IDS

    assert config.failsafe is not None  # noqa: S101 - guaranteed post-run_failsafe_suite
    name = "test6_ood_5sd_detection"
    try:
        handle = load_style_space(config=config, artefact_dir=artefact_dir)
        inj = config.failsafe.ood_injection_sd
        drivers = sorted(str(d) for d in held_out["driver"].dropna().unique())
        laps = pd.DataFrame(held_out[held_out["driver"] == drivers[0]]).copy()
        feat_cols = [fid for fid in FEATURE_IDS if fid in laps.columns]
        for fid in feat_cols:
            col = laps[fid].astype(float)
            laps[fid] = col + inj * (col.std(ddof=0) or 1.0)
        proj = project_unknown(
            laps, circuit=str(laps["circuit"].iloc[0]), handle=handle, config=config
        )
        return _gate(name, bool(proj.ood), True, "==",
                     finding=f"5-SD-injected ood={proj.ood} (pc_sd={proj.ood_max_pc_sd})")
    except Exception as exc:  # noqa: BLE001 - a detection fault is a visible skip, never a pass
        return _gate(name, None, True, "==", finding=f"detection not computed: {exc}")


# --- pillar: F09 ablation (call run_ablation) ----------------------------------------------------
def _pillar_ablation(
    matrix: pd.DataFrame | None,
    config: AppConfig,
    *,
    classifier_result: ClassifierTrainingResult | None,
    log_path: str | Path | None = None,
) -> PillarResult:
    """F09 corner-type ablation: CALL ``run_ablation`` and report its IMPROVES/NEUTRAL/DEGRADES
    finding + the harvested normalisation ablation (§4.0-a). Absent matrix → skip (ablation needs
    the feature frame); the harvested normalisation number is surfaced regardless when available."""
    metrics: list[MetricResult] = []
    findings: list[str] = []
    native: dict[str, Any] = {}
    if classifier_result is not None:
        metrics.append(
            _gate(
                "normalisation_ablation_macro_f1",
                classifier_result.normalisation_ablation_macro_f1,
                classifier_result.held_out_macro_f1,
                "<",
                finding="harvested from the champion (§11.1 normalisation ablation)",
            )
        )
    if matrix is None:
        if not metrics:
            return _skipped("ablation", "no feature matrix supplied — run_ablation needs the frame")
        findings.append("corner-type ablation skipped — no feature matrix supplied")
        return PillarResult(pillar="ablation", metrics=metrics, findings=findings, native=native)

    from app.ml.corners import run_ablation

    rep = run_ablation(matrix, config=config, log_path=log_path)
    native["ablation_report"] = _json_safe(rep.model_dump())
    findings.append(f"corner-type ablation: {rep.finding or 'no finding'} (delta={rep.delta})")
    return PillarResult(pillar="ablation", metrics=metrics, findings=findings, native=native)


# --- RQ verdict mapping (§4.1.1) -----------------------------------------------------------------
_RQ_QUESTIONS: dict[RQName, str] = {
    "RQ1": "Can a stable, discriminative driver fingerprint generalise to unseen circuits?",
    "RQ2": "Does car compatibility correlate with stint-performance consistency?",
    "RQ3": "Does behavioural deviation precede lap-time degradation, and by how many laps?",
    "RQ4": "Is the style space continuous enough to project unknown drivers meaningfully?",
    "RQ5": "What are the principal failure modes, and how do the failsafes affect reliability?",
}


def _tri_verdict(passes: list[bool | None]) -> Verdict:
    """The uniform three-way rule for an 'iff all targets met' RQ: all computed targets met →
    SUPPORTED; none met → NOT_SUPPORTED; mixed (or a computed-but-unmet subset) → PARTIAL. A metric
    whose value could not be computed (``passes is None``) does not count as met."""
    computed = [p for p in passes if p is not None]
    if not computed:
        return "INSUFFICIENT_DATA"
    if all(computed):
        return "SUPPORTED"
    if not any(computed):
        return "NOT_SUPPORTED"
    return "PARTIAL"


def _rq_verdicts(pillars: list[PillarResult]) -> list[RQVerdict]:
    """Apply the §4.1.1 rules to the pillar metrics. An RQ whose evidence pillar skipped →
    ``INSUFFICIENT_DATA`` (never SUPPORTED w/o data). RQ2 is ``PARTIAL`` by design (§4.0-f)."""
    by_name = {p.pillar: p for p in pillars}

    def metric(pillar: PillarName, name: str) -> MetricResult | None:
        p = by_name.get(pillar)
        if p is None or p.skipped:
            return None
        return next((m for m in p.metrics if m.name == name), None)

    def passes(pillar: PillarName, name: str) -> bool | None:
        m = metric(pillar, name)
        return None if m is None else m.passes

    verdicts: list[RQVerdict] = []

    # RQ1 — fingerprint generalises to unseen circuits
    if by_name.get("classifier", PillarResult(pillar="classifier")).skipped:
        rq1: Verdict = "INSUFFICIENT_DATA"
        rq1_find = ["classifier pillar skipped — no committed champion (§4.0-c)"]
    else:
        rq1 = _tri_verdict([
            passes("classifier", "held_out_macro_f1"),
            passes("classifier", "held_out_per_driver_f1_min"),
            passes("classifier", "normalisation_ablation_drop"),
        ])
        rq1_find = (
            ["PARTIAL = the risk-register relaxed-target path (honest below-target analysis)"]
            if rq1 == "PARTIAL"
            else []
        )
    verdicts.append(RQVerdict(
        rq="RQ1", question=_RQ_QUESTIONS["RQ1"],
        evidence=["classifier.held_out_macro_f1", "classifier.held_out_per_driver_f1_min",
                  "classifier.normalisation_ablation_drop",
                  "classifier.circuit_generalisation_f1_min"],
        verdict=rq1, findings=rq1_find,
    ))

    # RQ2 — PARTIAL by construction (§4.0-f): no §11 metric table instruments the correlation
    verdicts.append(RQVerdict(
        rq="RQ2", question=_RQ_QUESTIONS["RQ2"],
        evidence=["personalisation reliability (F11 finding)", "stability.teammate_proximity"],
        verdict="PARTIAL",
        findings=[
            "No §11 metric table instruments the compatibility↔consistency correlation; answered "
            "from F11's honest '~5 setup params reliably inferrable' finding + teammate proximity "
            "as a compatibility proxy. Under-instrumented — a dissertation limitation (§15.2).",
        ],
    ))

    # RQ3 — deviation precedes degradation
    if by_name.get("degradation", PillarResult(pillar="degradation")).skipped:
        rq3: Verdict = "INSUFFICIENT_DATA"
    else:
        rq3 = _tri_verdict([
            passes("degradation", "mean_lead_time_laps"),
            passes("degradation", "true_positive_rate"),
            passes("degradation", "false_positive_rate"),
        ])
    verdicts.append(RQVerdict(
        rq="RQ3", question=_RQ_QUESTIONS["RQ3"],
        evidence=["degradation.mean_lead_time_laps", "degradation.true_positive_rate",
                  "degradation.false_positive_rate"],
        verdict=rq3,
    ))

    # RQ4 — style space continuous enough to project unknown drivers
    if by_name.get("stability", PillarResult(pillar="stability")).skipped:
        rq4: Verdict = "INSUFFICIENT_DATA"
    else:
        rq4 = _tri_verdict([
            passes("stability", "unknown_driver_nn_correct"),
            passes("stability", "umap_procrustes_variance"),
        ])
    verdicts.append(RQVerdict(
        rq="RQ4", question=_RQ_QUESTIONS["RQ4"],
        evidence=["stability.unknown_driver_nn_correct", "stability.umap_procrustes_variance",
                  "stability.teammate_proximity"],
        verdict=rq4,
    ))

    # RQ5 — failure modes + failsafe reliability (the folded FailsafeReport)
    fs = by_name.get("failsafe", PillarResult(pillar="failsafe"))
    all_pass = passes("failsafe", "failsafe_tests_pass_or_documented")
    if fs.skipped or all_pass is None:  # suite not run / metric absent → honest INSUFFICIENT_DATA
        rq5: Verdict = "INSUFFICIENT_DATA"
        rq5_find: list[str] = []
    else:
        rq5 = "SUPPORTED" if all_pass else "PARTIAL"
        rq5_find = [] if all_pass else ["one or more §11.4 tests documented as findings (§11.4)"]
    verdicts.append(RQVerdict(
        rq="RQ5", question=_RQ_QUESTIONS["RQ5"],
        evidence=["failsafe.failsafe_tests_pass_or_documented"], verdict=rq5, findings=rq5_find,
    ))
    return verdicts


# --- orchestrator ---------------------------------------------------------------------------------
def _recover_champion_result(config: AppConfig) -> ClassifierTrainingResult | None:
    """Best-effort recovery of the champion :class:`ClassifierTrainingResult` (§4.0-c). APEX commits
    no champion, and reconstructing the full typed result from MLflow metrics/params is a follow-up
    seam — so offline this returns ``None`` (→ the classifier pillar skips honestly, never a
    fabricated number). The gated run supplies the result via the ``classifier_result`` kwarg."""
    _LOG.debug("champion result not auto-recovered — inject via classifier_result")
    return None


def _run_pillar(pillar: PillarName, fn: Callable[[], PillarResult]) -> PillarResult:
    """Run one pillar in its own try (§4.0-g): a pillar-internal fault (or the SUT raising) → that
    pillar ``skipped=True`` + a finding, never an aborted run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a pillar fault must not abort the run (§4.0-g)
        _LOG.warning("evaluation pillar %s faulted: %s", pillar, exc)
        return PillarResult(
            pillar=pillar, skipped=True,
            skip_reason=f"pillar faulted: {type(exc).__name__}: {exc}",
            findings=[f"{type(exc).__name__}: {exc}"],
        )


def _try_load(loader: Callable[[], Any]) -> Any:
    """Best-effort handle load: any ``*LoadError`` (absent artefact) → ``None`` → pillar skips."""
    try:
        return loader()
    except Exception as exc:  # noqa: BLE001 - absent/corrupt artefact → pillar skips, never a crash
        _LOG.debug("handle load failed (pillar will skip): %s", exc)
        return None


async def run_full_evaluation(
    matrix: pd.DataFrame | None,
    *,
    config: AppConfig,
    artefact_dir: str | Path | None = None,
    teams: Any = None,
    log_path: str | Path | None = None,
    classifier_result: ClassifierTrainingResult | None = None,
) -> EvaluationReport:
    """Run the master's four §11 pillars (+ the F09 ablation), gate every metric against its target,
    map the results onto RQ1–RQ5, write ``artefacts/evaluation_report.json``, and return the report.

    **Never raises** (§4.0-g): each pillar runs in its own try; a fault becomes that pillar's
    ``skipped`` state. Offline (no artefacts / ``matrix=None``): the failsafe pillar runs its
    mechanism; classifier/stability/degradation/ablation skip; RQ5 is answered, RQ1/RQ3/RQ4 are
    ``INSUFFICIENT_DATA``, RQ2 is ``PARTIAL`` by construction (§4.0-b/§4.0-f).

    ``classifier_result`` is the harvest source for §11.1 (§4.0-c); when ``None`` it is recovered
    best-effort (offline → ``None`` → classifier pillar skips). ``teams`` is the driver→team map for
    §11.2 teammate proximity (``None`` → that metric skips).

    Handle-load probes + pillar surfaces log to an **eval-scoped** ``eval_dq.jsonl`` under
    ``artefact_dir`` (not the operational ``data_quality_log``) when the caller passes no
    ``log_path``: an eval run finding no committed artefact is an expected condition, so its
    fail-loud ``MODEL_NOT_FOUND`` rows must not pollute the operational log (§7 no-pollution AC).
    The §11.4 sub-run redirects its own cache/dq internally (F15 ``_harness_config``)."""
    if classifier_result is None:
        classifier_result = _recover_champion_result(config)

    adir = Path(artefact_dir) if artefact_dir is not None else Path(config.paths.artefacts_dir)
    eval_log: str | Path = log_path if log_path is not None else str(adir / "eval_dq.jsonl")

    style_handle = _try_load(lambda: _load_style_space(config, artefact_dir, eval_log))
    detector_handle = _try_load(lambda: _load_detector(config, artefact_dir, eval_log))

    # §4.0-h: isolate the sanctioned held-out (Belgium/Italy) slice ONCE. The generalisation
    # evaluators (degradation lead-time, unknown-driver projection, the §11.4 rate extensions) score
    # on this slice ONLY; the fit-and-self-quarantine surfaces (UMAP Procrustes, F09 ablation) keep
    # the full matrix. A missing split/circuit column or no held-out rows → those pillars skip.
    held_out, held_out_reason = _held_out_matrix(matrix, config)

    pillars: list[PillarResult] = [
        _run_pillar(
            "classifier",
            lambda: evaluate_classifier(classifier_result, config=config, log_path=eval_log),
        ),
        _run_pillar(
            "stability",
            lambda: _pillar_stability(
                style_handle, matrix, held_out, teams, config,
                classifier_result=classifier_result, held_out_reason=held_out_reason,
                log_path=eval_log,
            ),
        ),
        _run_pillar(
            "degradation",
            lambda: _pillar_degradation(
                detector_handle, held_out, config,
                held_out_reason=held_out_reason, log_path=eval_log,
            ),
        ),
    ]
    # the §11.4 pillar is async (it awaits run_failsafe_suite) — its own try, §4.0-g
    try:
        pillars.append(await _pillar_failsafe(config, artefact_dir, held_out))
    except Exception as exc:  # noqa: BLE001 - a failsafe-suite fault must not abort the run
        _LOG.warning("failsafe pillar faulted: %s", exc)
        pillars.append(PillarResult(
            pillar="failsafe", skipped=True,
            skip_reason=f"pillar faulted: {type(exc).__name__}: {exc}",
            findings=[f"{type(exc).__name__}: {exc}"],
        ))
    pillars.append(
        _run_pillar(
            "ablation",
            lambda: _pillar_ablation(
                matrix, config, classifier_result=classifier_result, log_path=eval_log,
            ),
        )
    )
    # keep the canonical §11 pillar order regardless of append timing
    pillars.sort(key=lambda p: _PILLARS.index(p.pillar))

    rq_verdicts = _rq_verdicts(pillars)
    n_skipped = sum(1 for p in pillars if p.skipped)
    all_metrics = [m for p in pillars for m in p.metrics]
    all_targets_met = n_skipped == 0 and bool(all_metrics) and all(
        m.passes is True for m in all_metrics
    )
    report = EvaluationReport(
        pillars=pillars,
        rq_verdicts=rq_verdicts,
        n_pillars=len(_PILLARS),
        n_pillars_run=len(pillars) - n_skipped,
        n_pillars_skipped=n_skipped,
        all_targets_met=all_targets_met,
    )
    # Hashing + persistence must never abort the run (§4.0-g): the report OBJECT is the
    # deliverable, the file is a side effect. A fault → log + return the report anyway.
    try:
        report.content_hash = hashlib.sha256(_canonical(report).encode("utf-8")).hexdigest()
    except Exception as exc:  # noqa: BLE001 - a hash fault must not lose the whole report
        _LOG.warning("evaluation content-hash failed: %s", exc)
    try:
        _write_report(report, config, artefact_dir)
    except Exception as exc:  # noqa: BLE001 - a write fault must not lose the whole report
        _LOG.warning("evaluation report write failed (report still returned): %s", exc)
    return report


# --- thin load wrappers (lazy imports keep `app.eval` light; each raises its own *LoadError) ------
def _load_style_space(
    config: AppConfig, artefact_dir: str | Path | None, log_path: str | Path | None
) -> StyleSpaceHandle:
    from app.ml.stylespace import load_style_space

    return load_style_space(config=config, artefact_dir=artefact_dir, log_path=log_path)


def _load_detector(
    config: AppConfig, artefact_dir: str | Path | None, log_path: str | Path | None
) -> MonitorHandle:
    from app.monitoring.detector import load_detector

    return load_detector(config=config, artefact_dir=artefact_dir, log_path=log_path)
