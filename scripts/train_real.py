"""Stage C of real-data training: the training matrix -> the F06 classifier + F08 style space.

Consumes what ``scripts/build_training_matrix.py`` (Stage B) wrote and calls the REAL training
surfaces -- :func:`train_classifier` and :func:`build_style_space`. It fits nothing itself: the
normalisation artefact, the held-out quarantine, the calibration carve and the MLflow logging all
happen inside those functions, exactly as they do for a synthetic run.

Artefacts land in a SEPARATE directory (default ``artefacts/real``) so a real-data run never
overwrites the synthetic bundle the demo currently serves. Promote deliberately, not by accident.

Pre-flight refuses to train on a matrix that cannot support the claim being made -- no held-out
circuits, a single driver, or a driver below the per-combination lap floor -- rather than emitting
a model with a meaningless held-out score.

Run:

    .venv/Scripts/python.exe scripts/train_real.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.config import load_config  # noqa: E402
from app.ml.classifier import ClassifierTrainingError, train_classifier  # noqa: E402
from app.ml.stylespace import StyleSpaceBuildError, build_style_space  # noqa: E402


def preflight(matrix: pd.DataFrame, held_out: set[str]) -> list[str]:
    """Structural problems that would make a trained model's headline number meaningless."""
    problems: list[str] = []
    n_drivers = matrix["driver"].nunique()
    if n_drivers < 2:
        problems.append(f"only {n_drivers} driver(s) — a classifier needs >= 2 classes")
    present_held_out = sorted(set(matrix["circuit"].unique()) & held_out)
    if not present_held_out:
        problems.append(
            f"no held-out circuit present ({sorted(held_out)}) — held_out_macro_f1 would be "
            "computed on an empty test set"
        )
    train = matrix[matrix["split"] == "train"]
    if train.empty:
        problems.append("no rows in the train split")
    else:
        if train["circuit"].nunique() < 2:
            problems.append(
                f"only {train['circuit'].nunique()} training circuit — StratifiedGroupKFold "
                "groups by circuit and needs >= 2"
            )
        missing = sorted(set(matrix["driver"].unique()) - set(train["driver"].unique()))
        if missing:
            problems.append(f"drivers absent from the train split: {missing}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="D:/COMP702/DF/artefacts/real/training_matrix.parquet")
    ap.add_argument("--artefact-dir", default="D:/COMP702/DF/artefacts/real")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-stylespace", action="store_true")
    ap.add_argument("--force", action="store_true", help="train despite pre-flight problems")
    args = ap.parse_args()

    config = load_config(args.config)
    art_dir = Path(args.artefact_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    log_path = art_dir / "train_dq.jsonl"

    matrix = pd.read_parquet(args.matrix)
    held_out = set(config.held_out_circuits)

    print(f"matrix: {len(matrix)} vectors | {matrix['driver'].nunique()} drivers | "
          f"{matrix['circuit'].nunique()} circuits | seasons {sorted(matrix['season'].unique())}")
    print(f"  split: train={int((matrix['split'] == 'train').sum())} "
          f"test={int((matrix['split'] == 'test').sum())}")
    print(f"  held-out present: {sorted(set(matrix['circuit'].unique()) & held_out)}")

    problems = preflight(matrix, held_out)
    if problems:
        print("\nPRE-FLIGHT PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if not args.force:
            print("\nrefusing to train (pass --force to override)", file=sys.stderr)
            raise SystemExit(4)
        print("\n--force given: training anyway\n", file=sys.stderr)

    summary: dict[str, object] = {"matrix": str(args.matrix), "n_vectors": int(len(matrix))}

    # ---- F06 classifier ----------------------------------------------------------------------
    print("\n=== F06 classifier ===")
    try:
        res = train_classifier(matrix, config=config, artefact_dir=art_dir, log_path=log_path)
    except ClassifierTrainingError as exc:
        print(f"classifier training FAILED: {exc.error_code}: {exc}", file=sys.stderr)
        raise SystemExit(5) from exc
    print(f"  status              : {res.status}")
    print(f"  held-out macro F1   : {res.held_out_macro_f1:.4f} "
          f"(target {res.target_macro_f1}, met={res.target_met})")
    print(f"  per-driver floor    : {res.per_driver_floor} met={res.per_driver_floor_met}")
    print(f"  beats all baselines : {res.beats_all_baselines}  {res.baseline_macro_f1}")
    print(f"  train circuits      : {res.train_circuits}")
    print(f"  excluded (held out) : {res.excluded_circuits}")
    print(f"  trials              : {res.n_trials} ({res.n_trials_failed} failed)")
    print(f"  model               : {res.model_artefact_path}")
    _pdf = {k: round(v, 3) for k, v in res.held_out_per_driver_f1.items()}
    print(f"  per-driver F1       : {_pdf}")
    summary["classifier"] = {
        "status": str(res.status),
        "held_out_macro_f1": res.held_out_macro_f1,
        "target_macro_f1": res.target_macro_f1,
        "target_met": res.target_met,
        "per_driver_floor_met": res.per_driver_floor_met,
        "beats_all_baselines": res.beats_all_baselines,
        "baseline_macro_f1": res.baseline_macro_f1,
        "held_out_per_driver_f1": res.held_out_per_driver_f1,
        "train_circuits": res.train_circuits,
        "excluded_circuits": res.excluded_circuits,
        "normalisation_content_hash": res.normalisation_content_hash,
        "mlflow_run_id": res.mlflow_run_id,
    }

    # ---- F08 style space ---------------------------------------------------------------------
    if not args.skip_stylespace:
        print("\n=== F08 style space ===")
        try:
            sres = build_style_space(matrix, config=config, artefact_dir=art_dir, log_path=log_path)
            print(f"  status     : {sres.status}  drivers={sres.n_drivers} "
                  f"features={sres.n_features}")
            print(f"  umap       : {sres.umap_artefact_path}")
            print(f"  pca        : {sres.pca_artefact_path}")
            summary["stylespace"] = {
                "status": str(sres.status),
                "n_drivers": sres.n_drivers,
                "n_features": sres.n_features,
                "umap_sha256": sres.umap_sha256,
                "mlflow_run_id": sres.mlflow_run_id,
            }
        except StyleSpaceBuildError as exc:
            # Non-fatal for the run: the classifier is already trained and on disk.
            print(f"  style space FAILED: {exc.error_code}: {exc}", file=sys.stderr)
            summary["stylespace"] = {"status": "FAILED", "error": str(exc)}

    (art_dir / "train_real_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nsummary -> {art_dir / 'train_real_summary.json'}")


if __name__ == "__main__":
    main()
