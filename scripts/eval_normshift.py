"""Unseen-circuit normalisation — Evaluation A (zero-shot) vs Evaluation B (calibrated).

Evaluation directive #3 (2026-08-13): "Don't hide 0.4068; understand it. … Do not simply patch
the 0.4068." The held-out circuits have no per-circuit normalisation statistics — principle 2
working as specified — so every held-out lap normalises through the GLOBAL fallback. This script
separates the two deployment assumptions that number conflates:

- **Evaluation A — zero-shot unseen circuit.** Global normalisation, exactly as served. The
  headline 0.4068 stays; it is reproduced here (full test split) as a cross-check, never altered.
- **Evaluation B — new circuit after calibration running.** The first N clean laps of the unseen
  circuit (chronological — what a team would actually have first) are used ONLY to compute the
  six per-circuit normalisation statistics; the classifier stays FROZEN; those calibration laps
  are EXCLUDED from scoring. N is swept, so the result is a curve of recoverable performance vs
  calibration data — the domain-shift finding.

**Paired protocol:** at each N, Evaluation A is re-scored on the SAME remaining-lap set as
Evaluation B, so the A/B delta is never confounded with the scored-set change.

**Leakage statement (protocol B, deliberate and eval-only):** the per-circuit statistics computed
here live in memory only — the persisted ``normalisation.json`` is never touched (asserted by
hash), no ``.fit()`` runs on the classifier, and the calibration/scoring split is disjoint by
construction (tested). This is an EVALUATION of assumption B, not a change to the serving system.

Output: ``artefacts/real/eval/normshift.json`` (content-hashed) + a printed A/B curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def calibration_split(
    df: pd.DataFrame, circuit: str, n_calibration: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calibration, remainder) for ONE held-out circuit: the first ``n_calibration`` CLEAN laps
    in chronological order (season, then lap order within it) calibrate; everything else scores.
    Disjoint by construction — the same index never appears in both."""
    circ = df[df["circuit"] == circuit].sort_values(["season", "session", "lap_number"])
    clean = circ[circ["is_clean"].astype(bool)]
    calib_idx = clean.index[:n_calibration]
    calib = circ.loc[circ.index.isin(calib_idx)]
    remainder = circ.loc[~circ.index.isin(calib_idx)]
    return calib, remainder


def circuit_stats(
    calib: pd.DataFrame, features: list[str], *, ddof: int, min_std: float,
    global_stats: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Per-feature (mean, std) from the calibration laps only. A degenerate feature (all-NULL or
    zero variance below ``min_std``) falls back to the GLOBAL stats for that feature — the same
    fallback semantics the serving normaliser applies, so B is never quietly worse-defined."""
    out: dict[str, tuple[float, float]] = {}
    for f in features:
        if f not in calib.columns:  # absent column = degenerate → global fallback, never a crash
            out[f] = global_stats[f]
            continue
        vals = pd.to_numeric(calib[f], errors="coerce").dropna()
        if len(vals) < 2:
            out[f] = global_stats[f]
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=ddof))
        out[f] = (mean, std) if std >= min_std else global_stats[f]
    return out


def normalise_frame(
    frame: pd.DataFrame, features: list[str], all_features: list[str],
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Vectorised transform: the six normalised features z-scored by ``stats``, the rest verbatim;
    NULLs stay NaN (XGBoost handles them natively, mirroring the serving path)."""
    x = np.array(
        frame[all_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        copy=True,  # pandas-3 copy-on-write can hand back a read-only view
    )
    for j, f in enumerate(all_features):
        if f in stats and f in features:
            mean, std = stats[f]
            x[:, j] = (x[:, j] - mean) / std
    return x


def macro_f1(model: Any, drivers: list[str], x: np.ndarray, y_names: pd.Series) -> float:
    from sklearn.metrics import f1_score

    index_of = {d: i for i, d in enumerate(drivers)}
    y_true = y_names.map(index_of).to_numpy()
    y_pred = np.asarray(model.predict(x), dtype=int)
    return float(f1_score(y_true, y_pred, average="macro"))


def _content_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run(
    config_path: str, matrix_path: str, artefact_dir: str | None, out_path: str,
    sweep: list[int],
) -> dict[str, Any]:
    from app.contracts.config import load_config
    from app.ml.inference import load_model
    from app.pipeline.features import FEATURE_IDS
    from app.pipeline.normalise import load_normalisation_params

    config = load_config(config_path)
    art_dir = Path(artefact_dir) if artefact_dir else Path(config.paths.artefacts_dir)
    handle = load_model(config=config, artefact_dir=art_dir)
    params_path = art_dir / config.normalise.artefact_filename
    artefact_before = sha256(params_path.read_bytes()).hexdigest()
    params = load_normalisation_params(params_path)

    features = list(params.normalised_features)
    all_features = list(FEATURE_IDS)
    min_std = config.normalise.min_std
    ddof = params.ddof
    global_stats = {f: (params.global_[f].mean, params.global_[f].std) for f in features}

    df = pd.read_parquet(matrix_path)
    test = df[df["split"] == "test"].copy()
    held_out = sorted(test["circuit"].unique())

    # Evaluation A on the FULL test split — the headline reproduction, untouched.
    x_full = normalise_frame(test, features, all_features, global_stats)
    f1_a_full = macro_f1(handle.model, handle.drivers, x_full, test["driver"])

    curve: list[dict[str, Any]] = []
    for n in sweep:
        scored_parts: list[pd.DataFrame] = []
        scored_b_stats: dict[str, dict[str, tuple[float, float]]] = {}
        n_calib_used: dict[str, int] = {}
        for circuit in held_out:
            calib, remainder = calibration_split(test, circuit, n)
            n_calib_used[circuit] = len(calib)
            scored_parts.append(remainder)
            scored_b_stats[circuit] = circuit_stats(
                calib, features, ddof=ddof, min_std=min_std, global_stats=global_stats
            )
        scored = pd.concat(scored_parts)

        # Paired scoring: A and B on the SAME remainder set.
        xa = normalise_frame(scored, features, all_features, global_stats)
        f1_a = macro_f1(handle.model, handle.drivers, xa, scored["driver"])

        xb_parts: list[np.ndarray] = []
        yb_parts: list[pd.Series] = []
        for circuit in held_out:
            part = scored[scored["circuit"] == circuit]
            xb_parts.append(
                normalise_frame(part, features, all_features, scored_b_stats[circuit])
            )
            yb_parts.append(part["driver"])
        f1_b = macro_f1(
            handle.model, handle.drivers, np.vstack(xb_parts), pd.concat(yb_parts)
        )

        curve.append(
            {
                "n_calibration_per_circuit": n,
                "n_calibration_used": n_calib_used,
                "n_scored": int(len(scored)),
                "macro_f1_A_global": f1_a,
                "macro_f1_B_calibrated": f1_b,
                "delta_B_minus_A": f1_b - f1_a,
            }
        )

    artefact_after = sha256(params_path.read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "matrix": matrix_path,
        "model_version": handle.model_version,
        "held_out_circuits": held_out,
        "normalised_features": features,
        "protocol": (
            "A = zero-shot (global fallback, as served). B = per-circuit stats from the first N "
            "chronological clean laps, classifier FROZEN, calibration laps excluded from scoring; "
            "A re-scored on the identical remainder at each N (paired). Eval-only: the persisted "
            "normalisation artefact is untouched (hash-asserted)."
        ),
        "evaluation_A_full_test_macro_f1": f1_a_full,
        "curve": curve,
        "normalisation_artefact_untouched": artefact_before == artefact_after,
    }
    result["content_hash"] = _content_hash(result)
    if not result["normalisation_artefact_untouched"]:  # pragma: no cover — the tripwire
        raise RuntimeError("normalisation.json changed during an eval-only run")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--matrix", default="artefacts/real/training_matrix.parquet")
    ap.add_argument("--artefact-dir", default=None)
    ap.add_argument("--out", default="artefacts/real/eval/normshift.json")
    ap.add_argument("--sweep", type=int, nargs="+", default=[30, 60, 100, 200])
    args = ap.parse_args()
    r = run(args.config, args.matrix, args.artefact_dir, args.out, args.sweep)

    print(f"Evaluation A (full test split, zero-shot): macro-F1 {r['evaluation_A_full_test_macro_f1']:.4f}")
    print(f"{'N/circ':>7} {'A (paired)':>11} {'B (calib)':>10} {'Δ':>8}")
    for row in r["curve"]:
        print(
            f"{row['n_calibration_per_circuit']:>7} {row['macro_f1_A_global']:>11.4f}"
            f" {row['macro_f1_B_calibrated']:>10.4f} {row['delta_B_minus_A']:>+8.4f}"
        )
    print(f"artefact untouched: {r['normalisation_artefact_untouched']}")
    print(f"content_hash {r['content_hash'][:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
