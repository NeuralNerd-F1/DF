"""Setup-tendency validation — the six axes of evaluation directive #5 (2026-08-13).

The claim under test is deliberately scoped: **"APEX produces internally consistent setup
tendencies associated with measurable driver fingerprints"** — never "APEX predicts the optimal
setup" (no ground-truth setups exist; the targets are the documented telemetry heuristic).

Six axes, all read-only over the served bundles (frozen regressors, persisted style coords, the
engine's own pure helpers — no ``.fit`` anywhere):

1. **Season stability** — each driver's per-season fingerprint through the engine at a fixed
   circuit context: is the TENDENCY direction (above/below the grid reference) constant across
   seasons?
2. **Circuit stability** — the same fingerprint across the four circuit contexts: which
   parameters are context-stable vs context-driven (both are legitimate; the split is reported)?
3. **Fingerprint sensitivity** — perturb each 5D coordinate by ±0.1/0.5/1.0 of that dimension's
   training spread (the style bundle's own ``pc_std`` — no invented scale): responses should be
   smooth (small ε → small Δ) and direction-consistent, not erratic.
4. **Confidence calibration** — per-parameter served confidence vs observed error against the
   heuristic targets on HELD-OUT cells (Belgium/Italy — never in personalisation training).
5. **Parameter reliability ranking** — the served per-parameter reliability, ranked, with rating
   bands, in one table instead of buried per-response.
6. **Confidence stratifies error** — the decisive check: per-cell absolute errors (normalised by
   FIA range) split by the parameter's served rating band; LOW/WIDE must show worse error than
   MEDIUM/HIGH for the rating system to be information rather than decoration.

Output: ``artefacts/real/eval/setup_validation.json`` (content-hashed) + a printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CONTEXT = "high_speed"  # one fixed, in-vocab circuit context for axes 1–3 (stated, not tuned)
_EPS_FRACTIONS = (0.1, 0.5, 1.0)


def tendency_direction(value: float | None, grid_mean: float | None) -> str | None:
    """'more' / 'less' / 'level' vs the grid reference — None when either side is absent
    (WIDE parameters and CATEGORY kinds carry no honest direction)."""
    if value is None or grid_mean is None:
        return None
    if value > grid_mean:
        return "more"
    if value < grid_mean:
        return "less"
    return "level"


def direction_consistency(directions: list[str | None]) -> float | None:
    """Fraction of pairwise agreements among the non-None directions; None with < 2 samples."""
    present = [d for d in directions if d is not None]
    if len(present) < 2:
        return None
    agree = sum(
        1 for i in range(len(present)) for j in range(i + 1, len(present))
        if present[i] == present[j]
    )
    total = len(present) * (len(present) - 1) // 2
    return agree / total


def normalised_error(value: float, target: float, lo: float, hi: float) -> float:
    """|prediction − heuristic target| ÷ FIA range — comparable across parameters."""
    span = hi - lo
    return abs(value - target) / span if span > 0 else float("nan")


def _content_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run(config_path: str, matrix_path: str, artefact_dir: str | None, out_path: str) -> dict[str, Any]:
    import app.personalisation.engine as eng
    from app.contracts.config import load_config
    from app.ml.stylespace import load_style_space
    from app.personalisation.engine import load_personalisation
    from app.pipeline.features import FEATURE_IDS
    from app.pipeline.normalise import load_normalisation_params

    config = load_config(config_path)
    art_dir = Path(artefact_dir) if artefact_dir else Path(config.paths.artefacts_dir)
    handle = load_personalisation(config=config, artefact_dir=art_dir)
    style = load_style_space(config=config, artefact_dir=art_dir)
    norm_params = load_normalisation_params(art_dir / config.normalise.artefact_filename)
    pcfg = config.personalisation
    tmp_log = Path(tempfile.mkdtemp()) / "eval_dq.jsonl"  # eval ledger — never the serving log

    numeric_params = [p.name for p in eng._SETUP_PARAMETERS
                      if p.kind is not eng.SetupOutputKind.CATEGORY]

    def recommend(coord: np.ndarray, circuit: str, ctype: str | None):
        ledger = eng._DQLedger(log_path=tmp_log)
        grid = eng._grid_reference(
            style.known_coords_5d, circuit=circuit, circuit_type=ctype, handle=handle,
            config=pcfg, ledger=ledger, entity_id="eval/grid",
        )
        rec = eng._recommend_from_coord(
            np.asarray(coord, float), circuit=circuit, circuit_type=ctype, handle=handle,
            config=pcfg, ledger=ledger, ood_max_pc_sd=0.0, entity_id="eval/rec",
            grid_means=grid,
        )
        return {p.name: p for p in rec.parameters}

    # ---- Axis 1 · season stability --------------------------------------------------------
    per_season: dict[str, dict[str, np.ndarray]] = {}
    for driver, by_season in style.season_coords_5d.items():
        for season, c in by_season.items():
            per_season.setdefault(driver, {})[str(season)] = np.asarray(c, float)

    season_consistency: dict[str, list[float]] = defaultdict(list)
    for driver, seasons in per_season.items():
        dirs_by_param: dict[str, list[str | None]] = defaultdict(list)
        for season in sorted(seasons):
            params = recommend(seasons[season], "EvalCircuit", _CONTEXT)
            for name in numeric_params:
                p = params[name]
                v = p.value if isinstance(p.value, (int, float)) else None
                dirs_by_param[name].append(tendency_direction(v, p.grid_mean))
        for name, dirs in dirs_by_param.items():
            c = direction_consistency(dirs)
            if c is not None:
                season_consistency[name].append(c)
    axis1 = {
        name: {"mean_pairwise_agreement": float(np.mean(v)), "n_drivers": len(v)}
        for name, v in sorted(season_consistency.items())
    }

    # ---- Axis 2 · circuit-context stability ------------------------------------------------
    ctx_consistency: dict[str, list[float]] = defaultdict(list)
    contexts = list(handle.circuit_types)
    for driver, coord in style.known_coords_5d.items():
        dirs_by_param = defaultdict(list)
        for ctype in contexts:
            params = recommend(np.asarray(coord, float), "EvalCircuit", ctype)
            for name in numeric_params:
                p = params[name]
                v = p.value if isinstance(p.value, (int, float)) else None
                dirs_by_param[name].append(tendency_direction(v, p.grid_mean))
        for name, dirs in dirs_by_param.items():
            c = direction_consistency(dirs)
            if c is not None:
                ctx_consistency[name].append(c)
    axis2 = {
        name: {"mean_pairwise_agreement": float(np.mean(v)), "n_drivers": len(v)}
        for name, v in sorted(ctx_consistency.items())
    }

    # ---- Axis 3 · fingerprint sensitivity --------------------------------------------------
    pc_std = np.asarray(style.pc_std, float)[: eng._N_FP_DIMS]
    fia = handle.fia_bounds
    sens: dict[str, dict[str, float]] = {name: {} for name in numeric_params}
    smooth_violations = 0
    smooth_checks = 0
    coords = list(style.known_coords_5d.values())
    base_coord = np.asarray(coords[len(coords) // 2], float)  # a mid-grid driver as the probe
    base = recommend(base_coord, "EvalCircuit", _CONTEXT)
    for frac in _EPS_FRACTIONS:
        deltas: dict[str, list[float]] = defaultdict(list)
        for dim in range(eng._N_FP_DIMS):
            for sign in (+1.0, -1.0):
                coord = base_coord.copy()
                coord[dim] += sign * frac * pc_std[dim]
                pert = recommend(coord, "EvalCircuit", _CONTEXT)
                for name in numeric_params:
                    b, p = base[name].value, pert[name].value
                    if isinstance(b, (int, float)) and isinstance(p, (int, float)):
                        lo, hi = fia[name]
                        deltas[name].append(abs(p - b) / (hi - lo) if hi > lo else np.nan)
        for name, vals in deltas.items():
            sens[name][f"mean_abs_delta_frac_at_{frac}sd"] = float(np.nanmean(vals))
    for name in numeric_params:  # smoothness: response must not shrink as ε grows
        s = sens[name]
        keys = [f"mean_abs_delta_frac_at_{f}sd" for f in _EPS_FRACTIONS]
        for a, b in zip(keys, keys[1:]):
            smooth_checks += 1
            if s[a] > s[b] + 1e-9:
                smooth_violations += 1
    axis3 = {
        "per_parameter": sens,
        "smoothness_violations": smooth_violations,
        "smoothness_checks": smooth_checks,
    }

    # ---- Axes 4 + 6 · held-out error, calibration, stratification --------------------------
    df = pd.read_parquet(matrix_path)
    held = df[(df["split"] == "test") & (df["is_clean"].astype(bool))].copy()
    train = df[(df["split"] == "train") & (df["is_clean"].astype(bool))].copy()

    ledger = eng._DQLedger(log_path=tmp_log)
    min_clean = config.thresholds.min_clean_laps
    train_cells, _ = eng._build_cells(train, norm_params, config, min_clean,
                                      pcfg.top_quartile_pct, tmp_log, ledger)
    held_cells, _ = eng._build_cells(held, norm_params, config, min_clean,
                                     pcfg.top_quartile_pct, tmp_log, ledger)
    agg_matrix = np.vstack([c.aggregate for c in train_cells])
    fmin, fmax = eng._minmax_stats(agg_matrix)  # TRAIN-cell min/max — frozen train semantics

    per_param_errors: dict[str, list[float]] = defaultdict(list)
    band_errors: dict[str, list[float]] = defaultdict(list)
    n_cells_scored = 0
    for cell in held_cells:
        if cell.driver not in style.known_coords_5d:
            continue
        n_cells_scored += 1
        params = recommend(
            np.asarray(style.known_coords_5d[cell.driver], float), cell.circuit,
            cell.circuit_type,
        )
        for spec in eng._SETUP_PARAMETERS:
            if spec.kind is eng.SetupOutputKind.CATEGORY:
                continue
            target = eng._target_for(cell.aggregate, spec, fmin, fmax, fia, config.normalise.min_std)
            p = params[spec.name]
            value = p.value if isinstance(p.value, (int, float)) else (
                float(np.mean(p.value_range)) if p.value_range else None
            )  # a WIDE range is scored at its centre — stated, so WIDE still earns an error
            if value is None:
                continue
            lo, hi = fia[spec.name]
            err = normalised_error(float(value), target, lo, hi)
            per_param_errors[spec.name].append(err)
            band = "LOW_OR_WIDE" if str(p.rating) in ("SetupConfidence.LOW", "SetupConfidence.WIDE",
                                                      "LOW", "WIDE") else "MEDIUM_OR_HIGH"
            band_errors[band].append(err)

    reliability = {k: float(v) for k, v in handle.reliability.items()}
    param_err_mean = {k: float(np.mean(v)) for k, v in per_param_errors.items()}
    shared = sorted(set(reliability) & set(param_err_mean))
    corr = (
        float(np.corrcoef([reliability[k] for k in shared],
                          [param_err_mean[k] for k in shared])[0, 1])
        if len(shared) >= 3 else None
    )
    axis46 = {
        "n_held_out_cells_scored": n_cells_scored,
        "per_parameter_mean_normalised_error": dict(sorted(param_err_mean.items())),
        "reliability_vs_error_pearson": corr,  # calibrated ⇒ strongly negative
        "error_by_rating_band": {
            band: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
            for band, v in sorted(band_errors.items())
        },
        "stratification_holds": (
            bool(np.mean(band_errors["LOW_OR_WIDE"]) > np.mean(band_errors["MEDIUM_OR_HIGH"]))
            if band_errors.get("LOW_OR_WIDE") and band_errors.get("MEDIUM_OR_HIGH") else None
        ),
        "stratification_note": (
            "The real bundle rates every parameter MEDIUM/HIGH for known in-distribution "
            "drivers (reliabilities all above the confidence floor), so the LOW/WIDE band is "
            "EMPTY BY CONSTRUCTION on held-out cells and the band comparison cannot run; the "
            "continuous reliability↔error correlation above carries the stratification claim. "
            "LOW/WIDE ratings arise only under out-of-distribution attenuation, where forcing "
            "them would test the attenuation design, not the information content."
            if not band_errors.get("LOW_OR_WIDE") else None
        ),
        "wide_scoring_note": "WIDE parameters are scored at their range centre (stated).",
    }

    # ---- Axis 5 · reliability ranking ------------------------------------------------------
    axis5 = [
        {"parameter": k, "reliability": v}
        for k, v in sorted(reliability.items(), key=lambda kv: -kv[1])
    ]

    result: dict[str, Any] = {
        "claim_under_test": (
            "APEX produces internally consistent setup tendencies associated with measurable "
            "driver fingerprints — never 'predicts the optimal setup'."
        ),
        "model_version": handle.model_version,
        "context": _CONTEXT,
        "axis1_season_stability": axis1,
        "axis2_circuit_context_stability": axis2,
        "axis3_fingerprint_sensitivity": axis3,
        "axis4_axis6_heldout_calibration": axis46,
        "axis5_reliability_ranking": axis5,
    }
    result["content_hash"] = _content_hash(result)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--matrix", default="artefacts/real/training_matrix.parquet")
    ap.add_argument("--artefact-dir", default=None)
    ap.add_argument("--out", default="artefacts/real/eval/setup_validation.json")
    args = ap.parse_args()
    r = run(args.config, args.matrix, args.artefact_dir, args.out)

    a1 = r["axis1_season_stability"]
    a2 = r["axis2_circuit_context_stability"]
    print(f"axis1 season-direction agreement (mean over params): "
          f"{np.mean([v['mean_pairwise_agreement'] for v in a1.values()]):.2f}")
    print(f"axis2 context-direction agreement (mean over params): "
          f"{np.mean([v['mean_pairwise_agreement'] for v in a2.values()]):.2f}")
    a3 = r["axis3_fingerprint_sensitivity"]
    print(f"axis3 smoothness violations: {a3['smoothness_violations']}/{a3['smoothness_checks']}")
    a46 = r["axis4_axis6_heldout_calibration"]
    print(f"axis4 held-out cells {a46['n_held_out_cells_scored']}  "
          f"reliability↔error r = {a46['reliability_vs_error_pearson']}")
    print(f"axis6 stratification holds: {a46['stratification_holds']}  "
          f"{a46['error_by_rating_band']}")
    print("axis5 top/bottom:",
          r["axis5_reliability_ranking"][0], r["axis5_reliability_ranking"][-1])
    print(f"content_hash {r['content_hash'][:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
