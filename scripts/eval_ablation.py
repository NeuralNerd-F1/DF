"""The ablation matrix — feature-domain, normalisation, and corner ablations (RQ2 by removal).

Evaluation directive #4 (2026-08-13): domain-only representations vs the full 28, the
circuit-normalisation ablation, and the corner-decomposition row, under ONE protocol.

**Protocol (identical for every row, stated in the artefact):**
- Train split → fit a fresh XGBoost with the CHAMPION's hyperparameters (extracted from the
  served bundle; seed and ``n_jobs=1`` pinned). No per-row re-tuning: hyperparameters were tuned
  once on the full representation, which slightly favours the 28-feature row — reported, not
  hidden. No per-row calibration either: every row scores raw argmax macro-F1, so rows are
  comparable to each other (the served champion's calibrated pipeline is a different object; the
  full-28 row here is the PROTOCOL's own baseline, not a re-statement of the headline).
- Evaluation: held-out circuits under zero-shot (Evaluation A) conditions — global-fallback
  normalisation — matching how the headline number is measured.
- Normalised rows use the PERSISTED train-only statistics (per-circuit for train circuits, global
  for held-out), read-only. The "− circuit normalisation" row feeds raw features on both sides.

**Rows deliberately absent, with reasons (never silently dropped):**
- "28 + fingerprint": the fingerprint is derived FROM the driver — appending it to a
  driver-identification task is label leakage; the row would measure the leak, not behaviour.
- "28 + corner decomposition": the corner sub-features are computed from raw X/Y telemetry and
  were never extracted into the real matrix; filling the row requires a Stage-B re-extraction
  pass over the cached telemetry (recorded as the follow-up), not a quick join.

Output: ``artefacts/real/eval/ablation.json`` (content-hashed) + a printed table including the
harvested Stage-B baselines for context.
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

# The domain partition of the 28 features (asserted complete + disjoint against FEATURE_IDS).
DOMAINS: dict[str, tuple[str, ...]] = {
    "throttle": ("F01", "F02", "F03", "F04", "F05"),
    "braking": ("F06", "F07", "F08", "F09", "F10"),
    "speed": ("F11", "F12", "F13", "F14"),
    "gear": ("F15", "F16"),
    "engine": ("F17", "F18"),
    "cornering": ("F19", "F20", "F21"),
    "interaction_consistency": ("F22", "F23", "F24", "F25", "F26", "F27", "F28"),
}

_KEPT_PARAM_KEYS = (
    "objective", "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
    "tree_method", "random_state", "n_jobs",
)


def champion_params(model: Any) -> dict[str, Any]:
    """Dig the base XGBClassifier's hyperparameters out of the served calibrated bundle.
    Fails loud if the structure is not what the bundle contract promises."""
    cc = getattr(model, "calibrated_classifiers_", None)
    if not cc:
        raise ValueError("champion bundle has no calibrated_classifiers_ — not the F06 contract")
    est = cc[0].estimator
    inner = getattr(est, "estimator", est)  # FrozenEstimator wraps the booster
    params = inner.get_params()
    return {k: params[k] for k in _KEPT_PARAM_KEYS if k in params}


def normalise_split(
    df: pd.DataFrame, features: list[str], all_features: list[str], params: Any,
) -> np.ndarray:
    """Read-only mirror of the serving transform: per-circuit stats where the circuit was fitted,
    global fallback otherwise; non-normalised features verbatim; NULLs stay NaN."""
    x = np.array(
        df[all_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True
    )
    circuits = df["circuit"].astype(str).to_numpy()
    for j, f in enumerate(all_features):
        if f not in features:
            continue
        g = params.global_[f]
        means = np.full(len(df), g.mean)
        stds = np.full(len(df), g.std)
        for circuit, stats in params.per_circuit.items():
            mask = circuits == circuit
            if mask.any() and f in stats:
                means[mask] = stats[f].mean
                stds[mask] = stats[f].std
        x[:, j] = (x[:, j] - means) / stds
    return x


def fit_and_score(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    params: dict[str, Any],
) -> float:
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    return float(f1_score(y_test, model.predict(x_test), average="macro"))


def _content_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run(
    config_path: str, matrix_path: str, artefact_dir: str | None, out_path: str,
) -> dict[str, Any]:
    from app.contracts.config import load_config
    from app.ml.inference import load_model
    from app.pipeline.features import FEATURE_IDS
    from app.pipeline.normalise import load_normalisation_params

    config = load_config(config_path)
    art_dir = Path(artefact_dir) if artefact_dir else Path(config.paths.artefacts_dir)
    handle = load_model(config=config, artefact_dir=art_dir)
    params = champion_params(handle.model)
    norm = load_normalisation_params(art_dir / config.normalise.artefact_filename)

    # The partition must cover the 28 exactly — a drifted vocabulary fails loud, never silently.
    flat = [f for fs in DOMAINS.values() for f in fs]
    if sorted(flat) != sorted(FEATURE_IDS) or len(flat) != len(set(flat)):
        raise ValueError("DOMAINS is not a complete disjoint partition of FEATURE_IDS")

    df = pd.read_parquet(matrix_path)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    all_features = list(FEATURE_IDS)
    norm_features = list(norm.normalised_features)

    index_of = {d: i for i, d in enumerate(handle.drivers)}
    y_train = train["driver"].map(index_of).to_numpy()
    y_test = test["driver"].map(index_of).to_numpy()

    # Normalised design matrices are built ONCE over all 28; each row slices its columns.
    xn_train = normalise_split(train, norm_features, all_features, norm)
    xn_test = normalise_split(test, norm_features, all_features, norm)
    xr_train = np.array(
        train[all_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True
    )
    xr_test = np.array(
        test[all_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True
    )
    col = {f: j for j, f in enumerate(all_features)}

    rows: list[dict[str, Any]] = []

    def add_row(name: str, features: tuple[str, ...], *, raw: bool = False) -> None:
        idx = [col[f] for f in features]
        xt, xe = (xr_train, xr_test) if raw else (xn_train, xn_test)
        f1 = fit_and_score(xt[:, idx], y_train, xe[:, idx], y_test, params)
        rows.append({"representation": name, "n_features": len(features), "held_out_macro_f1": f1})
        print(f"  {name:<28} d={len(features):>2}  macro-F1 {f1:.4f}")

    print("ablation rows (held-out, zero-shot conditions):")
    for name, features in DOMAINS.items():
        add_row(f"{name}_only", features)
    add_row("all_28", tuple(all_features))
    add_row("all_28_no_normalisation", tuple(all_features), raw=True)

    result: dict[str, Any] = {
        "matrix": matrix_path,
        "champion_model_version": handle.model_version,
        "champion_hyperparameters": params,
        "protocol": (
            "Fresh XGBoost per row with the champion's hyperparameters (tuned once on the full "
            "representation — slightly favours all_28, stated); raw argmax macro-F1 on the "
            "held-out circuits under zero-shot (global-fallback) normalisation; no per-row "
            "tuning or calibration."
        ),
        "rows": rows,
        "rows_deliberately_absent": {
            "all_28_plus_fingerprint": (
                "label leakage — the fingerprint is derived from the driver; the row would "
                "measure the leak, not behaviour"
            ),
            "all_28_plus_corner_decomposition": (
                "corner sub-features are not in the real matrix; requires a Stage-B telemetry "
                "re-extraction pass (recorded follow-up), not a join"
            ),
        },
        "context_baselines_stage_b": {
            "majority": 0.0122, "logistic_regression": 0.2445, "random_forest": 0.3749,
            "served_champion_calibrated": 0.4068,
            "note": "harvested from the Stage-B training summary for context, not recomputed",
        },
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
    ap.add_argument("--out", default="artefacts/real/eval/ablation.json")
    args = ap.parse_args()
    r = run(args.config, args.matrix, args.artefact_dir, args.out)
    print(f"content_hash {r['content_hash'][:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
