"""RQ4 on the real corpus — lead-time distribution, false-positive characterisation, compound
breakdown.

Evaluation directive #2 (2026-08-13): "Complete real-corpus RQ4. Actual lead-time distribution,
false-positive rate, compound breakdown, not just replay demonstration."

**Read-only over the served detector.** Every stint in the 2022–2025 matrix is run ONCE through
the real ``monitor_stint`` (full stint, causal per-lap states — the same function the replay
engine wraps), and the per-lap state sequences are summarised. Nothing is fitted, calibrated or
persisted to any model artefact.

Honesty framing, stated up front and carried into the artefact:

- **No ground-truth degradation labels exist.** The stint's end (the driver actually pitted or the
  race ended) is the only observable event, and it is a PROXY: stints end for strategy, safety
  cars and chequered flags too, not only degradation. Every "lead time to stint end" below is a
  proxy-lead-time and is named as such.
- **"False positive" therefore has no single number.** Instead the artefact reports the
  distributions an examiner needs: the alert rate, laps-remaining-after-alert quantiles, and the
  **recovered-alert fraction** (stints where the deviation fell back under τ and the state
  returned to NOMINAL after an ALERT — the strongest self-labelled false-positive proxy).
- **Two splits, reported separately**: stints on training circuits (τ calibrated on data from the
  same circuits — in-calibration behaviour) and stints on the held-out circuits (Belgium/Italy —
  the generalisation number; per-driver × circuit-type τ cells apply unchanged because
  calibration is keyed by circuit TYPE, not circuit).

Output: ``artefacts/real/eval/rq4_detector.json`` (content-hashed) + a printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ALERT = "ALERT"
_NOMINAL = "NOMINAL"
_WATCH = "WATCH"


def stint_summary(states: list[str], alert: Any) -> dict[str, Any]:
    """Summarise ONE stint's causal per-lap state sequence + its (possibly None) alert payload.

    Pure over the detector's own outputs: lead/lag numbers come from the served
    ``first_crossing_lap`` / ``alert_lap`` and the position of the ALERT states in the sequence —
    never recomputed from raw deviations."""
    n = len(states)
    counts = Counter(states)
    alerted = alert is not None
    out: dict[str, Any] = {
        "n_laps": n,
        "alerted": alerted,
        "watched": counts[_WATCH] > 0 or counts[_ALERT] > 0,
        "n_alert_laps": counts[_ALERT],
        "n_watch_laps": counts[_WATCH],
    }
    if alerted:
        # Confirmation lag: crossing -> confirmed alert (both served lap NUMBERS).
        out["confirmation_lag_laps"] = int(alert.alert_lap - alert.first_crossing_lap)
        # Proxy lead time: laps between the confirmed alert and the stint's end.
        alert_idx = next((i for i, s in enumerate(states) if s == _ALERT), None)
        out["laps_remaining_after_alert"] = (n - 1 - alert_idx) if alert_idx is not None else None
        # Recovered alert = the state came BACK to NOMINAL after an ALERT lap — the deviation
        # dropped under τ again; the strongest self-labelled false-positive proxy.
        recovered = alert_idx is not None and any(
            s == _NOMINAL for s in states[alert_idx + 1 :]
        )
        out["recovered"] = bool(recovered)
        out["alert_confidence"] = str(alert.alert_confidence)
    return out


def _quantiles(values: list[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "mean": float(arr.mean()),
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll one group of stint summaries into the reported metrics."""
    n = len(summaries)
    alerted = [s for s in summaries if s["alerted"]]
    watched_only = [s for s in summaries if s["watched"] and not s["alerted"]]
    recovered = [s for s in alerted if s.get("recovered")]
    return {
        "n_stints": n,
        "alert_rate": (len(alerted) / n) if n else None,
        "watch_only_rate": (len(watched_only) / n) if n else None,
        "recovered_alert_fraction": (len(recovered) / len(alerted)) if alerted else None,
        "confirmation_lag_laps": _quantiles(
            [s["confirmation_lag_laps"] for s in alerted if "confirmation_lag_laps" in s]
        ),
        "laps_remaining_after_alert": _quantiles(
            [
                s["laps_remaining_after_alert"]
                for s in alerted
                if s.get("laps_remaining_after_alert") is not None
            ]
        ),
        "alert_confidence_counts": dict(
            Counter(s.get("alert_confidence") for s in alerted if s.get("alert_confidence"))
        ),
    }


def modal_compound(values: pd.Series) -> str:
    """The stint's compound = the modal non-null per-lap value; 'UNKNOWN' when none present."""
    v = values.dropna()
    v = v[~v.astype(str).str.upper().isin(("NAN", "NONE", ""))]
    if v.empty:
        return "UNKNOWN"
    return str(v.mode().iloc[0])


def _content_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run(
    config_path: str, matrix_path: str, artefact_dir: str | None, out_path: str,
    min_stint_laps: int,
) -> dict[str, Any]:
    from app.contracts.config import load_config
    from app.monitoring.detector import load_detector, monitor_stint
    from app.pipeline.features import FEATURE_IDS

    config = load_config(config_path)
    art_dir = Path(artefact_dir) if artefact_dir else Path(config.paths.artefacts_dir)
    handle = load_detector(config=config, artefact_dir=art_dir)
    held_out = set(config.held_out_circuits)

    df = pd.read_parquet(matrix_path)
    frame_cols = ["lap_number", "tyre_age", "track_status", "is_clean", "has_interpolation",
                  *FEATURE_IDS]

    groups = df.groupby(["season", "circuit", "circuit_type", "driver", "stint"], dropna=True)
    rows: list[dict[str, Any]] = []
    skipped_short = 0
    for (season, circuit, ctype, driver, stint), g in groups:
        g = g.sort_values("lap_number")
        if len(g) < min_stint_laps:
            skipped_short += 1
            continue
        frame = g[frame_cols].copy()
        res = monitor_stint(
            frame,
            driver=str(driver),
            circuit=str(circuit),
            circuit_type=str(ctype),
            handle=handle,
            config=config,
        )
        states = [lap.state.value for lap in res.laps]
        s = stint_summary(states, res.alert)
        s["split"] = "held_out" if str(circuit) in held_out else "train"
        s["compound"] = modal_compound(g["compound"])
        s["season"] = str(season)
        rows.append(s)

    by_split = {
        split: aggregate([r for r in rows if r["split"] == split])
        for split in ("train", "held_out")
    }
    compounds = sorted({r["compound"] for r in rows})
    by_compound = {
        c: aggregate([r for r in rows if r["compound"] == c]) for c in compounds
    }

    result: dict[str, Any] = {
        "matrix": matrix_path,
        "model_version": handle.model_version,
        "min_stint_laps": min_stint_laps,
        "n_stints_evaluated": len(rows),
        "n_stints_skipped_short": skipped_short,
        "proxy_note": (
            "Stint end (pit stop / race end) is the only observable event and is a PROXY for "
            "degradation; no ground-truth labels exist. 'recovered_alert_fraction' (state "
            "returned to NOMINAL after an ALERT) is the strongest self-labelled false-positive "
            "proxy."
        ),
        "by_split": by_split,
        "by_compound": by_compound,
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
    ap.add_argument("--out", default="artefacts/real/eval/rq4_detector.json")
    ap.add_argument(
        "--min-stint-laps", type=int, default=12,
        help="skip stints too short for the monitor to activate AND confirm (8-lap floor + "
             "rolling window + 2-lap confirmation); the skip count is reported, never silent",
    )
    args = ap.parse_args()
    r = run(args.config, args.matrix, args.artefact_dir, args.out, args.min_stint_laps)

    print(f"stints evaluated {r['n_stints_evaluated']}  (skipped short: {r['n_stints_skipped_short']})")
    for split, m in r["by_split"].items():
        lead = m["laps_remaining_after_alert"]
        lag = m["confirmation_lag_laps"]
        print(
            f"[{split}] n={m['n_stints']}  alert rate {m['alert_rate']:.2%}"
            f"  watch-only {m['watch_only_rate']:.2%}"
            f"  recovered(FP proxy) {m['recovered_alert_fraction']:.2%}"
            if m["n_stints"] and m["alert_rate"] is not None
            else f"[{split}] n={m['n_stints']}"
        )
        if lead:
            print(
                f"        proxy lead (laps left after alert): p50 {lead['p50']:.0f}"
                f"  p25 {lead['p25']:.0f}  p75 {lead['p75']:.0f}"
            )
        if lag:
            print(f"        confirmation lag: p50 {lag['p50']:.0f}  mean {lag['mean']:.1f}")
    for c, m in r["by_compound"].items():
        if m["n_stints"]:
            print(f"[compound {c}] n={m['n_stints']}  alert rate {m['alert_rate']:.2%}")
    print(f"content_hash {r['content_hash'][:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
