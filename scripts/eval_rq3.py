"""RQ3 on the real corpus — teammate proximity, stability, and quantitative cluster evidence.

Evaluation directive #1 (2026-08-13): "Complete real-corpus RQ3. Teammate proximity + stability +
quantitative cluster evidence." Master §11.2 names the metric: intra-team distance < inter-team
distance in the 5D style space.

**Read-only over the served style bundle.** Every coordinate analysed here is the persisted
artefact's own output — the per-``(driver, season)`` coords the handle reconstructs via a pure
``transform`` (never a new fit), so this script cannot leak, refit, or move anything. It computes:

- **Teammate proximity** (§11.2): per season, the 5D distance of every corpus teammate pair vs all
  non-teammate pairs; pooled across seasons with a seeded permutation test (teams reassigned at
  random within each season) for the one-sided hypothesis *intra < inter*.
- **Nearest-neighbour evidence**: of the drivers who HAVE a corpus teammate that season, the
  fraction whose nearest neighbour in 5D is that teammate (chance level reported alongside).
- **Season stability**: per driver, the distance between consecutive-season positions divided by
  the median inter-driver distance of the later season — a ratio < 1 means a driver's own seasons
  sit closer together than different drivers do (identity is stable relative to the space).

The teammate map below is a DATA table (public sporting fact, per season, restricted to pairs
where both drivers are in the 13-driver corpus), not a tunable. VER–TSU 2025 is deliberately
EXCLUDED: the pairing began mid-season, so the season aggregate spans two cars and the pair would
test car-change noise, not teammate proximity — recorded here rather than silently included.

Output: ``artefacts/real/eval/rq3_style_space.json`` (content-hashed, seed from config) + a
printed summary. Never writes to any model artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Per-season teammate pairs among the 13 corpus drivers (both members present all four seasons).
TEAMMATE_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "2022": (("LEC", "SAI"), ("HAM", "RUS"), ("ALO", "OCO"), ("GAS", "TSU")),
    "2023": (("LEC", "SAI"), ("HAM", "RUS"), ("GAS", "OCO"), ("ALO", "STR")),
    "2024": (("LEC", "SAI"), ("HAM", "RUS"), ("GAS", "OCO"), ("ALO", "STR")),
    "2025": (("LEC", "HAM"), ("ALB", "SAI"), ("ALO", "STR")),
}
EXCLUDED_PAIRS: dict[str, str] = {
    "VER-TSU 2025": "pairing began mid-season; the season aggregate spans two cars",
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def season_pair_distances(
    coords: dict[str, np.ndarray], pairs: tuple[tuple[str, str], ...]
) -> tuple[list[float], list[float]]:
    """(intra, inter) pair distances for ONE season. ``coords`` maps driver → 5D position; pairs
    not fully present in ``coords`` are skipped (a missing per-season coord is honest absence)."""
    drivers = sorted(coords)
    pair_set = {frozenset(p) for p in pairs if p[0] in coords and p[1] in coords}
    intra: list[float] = []
    inter: list[float] = []
    for a, b in combinations(drivers, 2):
        d = _dist(coords[a], coords[b])
        (intra if frozenset((a, b)) in pair_set else inter).append(d)
    return intra, inter


def pooled_permutation_pvalue(
    per_season: dict[str, dict[str, np.ndarray]],
    pairs_by_season: dict[str, tuple[tuple[str, str], ...]],
    *,
    n_iter: int,
    seed: int,
) -> float:
    """One-sided permutation p for *intra < inter*, pooled across seasons. Within each season the
    coordinate↔driver assignment is shuffled (team structure kept fixed), the pooled mean intra
    distance recomputed; p = fraction of permutations with mean intra ≤ the observed mean."""
    rng = np.random.default_rng(seed)
    observed: list[float] = []
    for season, coords in per_season.items():
        intra, _ = season_pair_distances(coords, pairs_by_season.get(season, ()))
        observed.extend(intra)
    if not observed:
        return float("nan")
    obs_mean = float(np.mean(observed))

    hits = 0
    for _ in range(n_iter):
        perm: list[float] = []
        for season, coords in per_season.items():
            drivers = sorted(coords)
            shuffled = dict(zip(rng.permutation(drivers).tolist(), [coords[d] for d in drivers]))
            intra, _ = season_pair_distances(shuffled, pairs_by_season.get(season, ()))
            perm.extend(intra)
        if perm and float(np.mean(perm)) <= obs_mean:
            hits += 1
    return hits / n_iter


def nearest_teammate_fraction(
    coords: dict[str, np.ndarray], pairs: tuple[tuple[str, str], ...]
) -> tuple[int, int]:
    """(hits, eligible) for ONE season: among drivers with a corpus teammate present, how many
    have that teammate as their 5D nearest neighbour."""
    mate: dict[str, str] = {}
    for a, b in pairs:
        if a in coords and b in coords:
            mate[a], mate[b] = b, a
    hits = 0
    for d, m in mate.items():
        others = {o: _dist(coords[d], coords[o]) for o in coords if o != d}
        if others and min(others, key=lambda o: others[o]) == m:
            hits += 1
    return hits, len(mate)


def season_drift_ratios(
    per_season: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Per driver, per consecutive season pair: own-drift ÷ median inter-driver distance of the
    later season. < 1 ⇒ the driver's identity moved less than the typical spacing between
    different drivers — the stability claim, made relative rather than absolute."""
    seasons = sorted(per_season)
    out: dict[str, dict[str, float]] = {}
    for prev, curr in zip(seasons, seasons[1:]):
        a, b = per_season[prev], per_season[curr]
        inter = [
            _dist(b[x], b[y]) for x, y in combinations(sorted(b), 2)
        ]
        med = float(np.median(inter)) if inter else float("nan")
        for d in sorted(set(a) & set(b)):
            out.setdefault(d, {})[f"{prev}->{curr}"] = (
                _dist(a[d], b[d]) / med if med and np.isfinite(med) and med > 0 else float("nan")
            )
    return out


def team_silhouette(
    coords: dict[str, np.ndarray], pairs: tuple[tuple[str, str], ...]
) -> float | None:
    """Silhouette of team labels for ONE season, restricted to drivers in corpus teammate pairs
    (singleton 'teams' carry no cluster information). None when fewer than two teams qualify.
    n is small by construction (≤ 8 points) — reported with that caveat, never over-claimed."""
    from sklearn.metrics import silhouette_score

    members: dict[str, int] = {}
    for i, (a, b) in enumerate(pairs):
        if a in coords and b in coords:
            members[a] = i
            members[b] = i
    if len(set(members.values())) < 2:
        return None
    x = np.stack([coords[d] for d in sorted(members)])
    labels = [members[d] for d in sorted(members)]
    return float(silhouette_score(x, labels))


def _content_hash(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run(config_path: str, artefact_dir: str | None, out_path: str, n_iter: int) -> dict[str, Any]:
    from app.contracts.config import load_config
    from app.ml.stylespace import load_style_space

    config = load_config(config_path)
    handle = load_style_space(config=config, artefact_dir=artefact_dir)
    seed = config.seeds.numpy

    per_season: dict[str, dict[str, np.ndarray]] = {}
    for driver, by_season in handle.season_coords_5d.items():
        for season, c in by_season.items():
            per_season.setdefault(str(season), {})[driver] = np.asarray(c, dtype=float)

    seasons_out: dict[str, Any] = {}
    all_intra: list[float] = []
    all_inter: list[float] = []
    nn_hits = 0
    nn_eligible = 0
    for season in sorted(per_season):
        pairs = TEAMMATE_PAIRS.get(season, ())
        intra, inter = season_pair_distances(per_season[season], pairs)
        h, e = nearest_teammate_fraction(per_season[season], pairs)
        nn_hits += h
        nn_eligible += e
        all_intra.extend(intra)
        all_inter.extend(inter)
        seasons_out[season] = {
            "n_drivers": len(per_season[season]),
            "n_teammate_pairs": len(intra),
            "mean_intra": float(np.mean(intra)) if intra else None,
            "mean_inter": float(np.mean(inter)) if inter else None,
            "median_intra": float(np.median(intra)) if intra else None,
            "median_inter": float(np.median(inter)) if inter else None,
            "nearest_neighbour_is_teammate": {"hits": h, "eligible": e},
            "team_silhouette": team_silhouette(per_season[season], pairs),
        }

    p_value = pooled_permutation_pvalue(
        per_season, TEAMMATE_PAIRS, n_iter=n_iter, seed=seed
    )
    drift = season_drift_ratios(per_season)
    drift_values = [v for by in drift.values() for v in by.values() if np.isfinite(v)]

    result: dict[str, Any] = {
        "engine": handle.engine,
        "model_version": handle.model_version,
        "seed": seed,
        "n_permutations": n_iter,
        "teammate_pairs": {s: [list(p) for p in ps] for s, ps in TEAMMATE_PAIRS.items()},
        "excluded_pairs": EXCLUDED_PAIRS,
        "per_season": seasons_out,
        "pooled": {
            "n_intra": len(all_intra),
            "n_inter": len(all_inter),
            "mean_intra": float(np.mean(all_intra)) if all_intra else None,
            "mean_inter": float(np.mean(all_inter)) if all_inter else None,
            "intra_over_inter": (
                float(np.mean(all_intra) / np.mean(all_inter))
                if all_intra and all_inter and np.mean(all_inter) > 0
                else None
            ),
            "permutation_p_intra_lt_inter": p_value,
            "nearest_neighbour_is_teammate": {
                "hits": nn_hits,
                "eligible": nn_eligible,
                # chance = teammate picked among the (n−1) other drivers, averaged per season
                "chance_level": (
                    float(
                        np.mean(
                            [
                                1.0 / (s["n_drivers"] - 1)
                                for s in seasons_out.values()
                                if s["n_drivers"] > 1
                            ]
                        )
                    )
                    if seasons_out
                    else None
                ),
            },
        },
        "season_stability": {
            "per_driver_drift_over_inter_median": drift,
            "median_ratio": float(np.median(drift_values)) if drift_values else None,
            "fraction_below_1": (
                float(np.mean([v < 1.0 for v in drift_values])) if drift_values else None
            ),
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
    ap.add_argument("--artefact-dir", default=None, help="default: config paths.artefacts_dir")
    ap.add_argument("--out", default="artefacts/real/eval/rq3_style_space.json")
    ap.add_argument("--permutations", type=int, default=10_000)
    args = ap.parse_args()
    r = run(args.config, args.artefact_dir, args.out, args.permutations)

    p = r["pooled"]
    nn = p["nearest_neighbour_is_teammate"]
    print(f"engine={r['engine']}  seasons={sorted(r['per_season'])}")
    print(
        f"pooled intra {p['mean_intra']:.3f} vs inter {p['mean_inter']:.3f}"
        f"  ratio {p['intra_over_inter']:.3f}  permutation p={p['permutation_p_intra_lt_inter']:.4f}"
    )
    print(
        f"nearest-neighbour-is-teammate {nn['hits']}/{nn['eligible']}"
        f"  (chance ≈ {nn['chance_level']:.2f})"
    )
    s = r["season_stability"]
    print(
        f"season drift ratio median {s['median_ratio']:.3f}"
        f"  fraction<1 {s['fraction_below_1']:.2f}"
    )
    print(f"content_hash {r['content_hash'][:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
