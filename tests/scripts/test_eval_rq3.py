"""RQ3 evaluation mechanisms — the pure metric functions, on synthetic geometry.

The real numbers live in the content-hashed artefact; these tests pin the MACHINERY: pair
bookkeeping, the permutation test's direction and determinism, the nearest-teammate metric, and
the drift ratio — each on coordinates constructed so the right answer is known by eye.
"""

from __future__ import annotations

import numpy as np

from scripts import eval_rq3 as rq3


def _clustered_season() -> dict[str, np.ndarray]:
    """Two tight teammate clusters far apart + one singleton — intra ≪ inter by construction."""
    return {
        "A1": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "A2": np.array([0.1, 0.0, 0.0, 0.0, 0.0]),
        "B1": np.array([10.0, 0.0, 0.0, 0.0, 0.0]),
        "B2": np.array([10.1, 0.0, 0.0, 0.0, 0.0]),
        "LON": np.array([5.0, 5.0, 0.0, 0.0, 0.0]),
    }


PAIRS = (("A1", "A2"), ("B1", "B2"))


def test_season_pair_distances_partitions_all_pairs() -> None:
    intra, inter = rq3.season_pair_distances(_clustered_season(), PAIRS)
    assert len(intra) == 2
    assert len(intra) + len(inter) == 5 * 4 // 2  # every unordered pair lands exactly once
    assert max(intra) < min(inter)  # the construction: teammates tight, everyone else far


def test_season_pair_distances_skips_missing_drivers() -> None:
    coords = _clustered_season()
    coords.pop("B2")
    intra, _ = rq3.season_pair_distances(coords, PAIRS)
    assert len(intra) == 1  # the broken pair is honest absence, not a crash or a zero


def test_permutation_p_is_small_for_real_structure_and_large_for_none() -> None:
    per_season = {"2022": _clustered_season(), "2023": _clustered_season()}
    pairs = {"2022": PAIRS, "2023": PAIRS}
    p_structured = rq3.pooled_permutation_pvalue(per_season, pairs, n_iter=500, seed=42)
    assert p_structured < 0.05  # teammates genuinely closer → small one-sided p

    rng = np.random.default_rng(0)
    noise = {
        "2022": {d: rng.normal(size=5) for d in _clustered_season()},
    }
    p_noise = rq3.pooled_permutation_pvalue(noise, {"2022": PAIRS}, n_iter=500, seed=42)
    assert p_noise > 0.05  # no structure → the test must NOT manufacture significance


def test_permutation_p_is_deterministic_under_the_seed() -> None:
    per_season = {"2022": _clustered_season()}
    pairs = {"2022": PAIRS}
    a = rq3.pooled_permutation_pvalue(per_season, pairs, n_iter=200, seed=7)
    b = rq3.pooled_permutation_pvalue(per_season, pairs, n_iter=200, seed=7)
    assert a == b


def test_nearest_teammate_fraction() -> None:
    hits, eligible = rq3.nearest_teammate_fraction(_clustered_season(), PAIRS)
    assert (hits, eligible) == (4, 4)  # all four paired drivers sit next to their teammate
    # Break one cluster: move A2 next to B-team — A1's NN stays A2 (still nearest to A1),
    # but A2's NN becomes a B driver.
    coords = _clustered_season()
    coords["A2"] = np.array([10.05, 0.0, 0.0, 0.0, 0.0])
    hits2, eligible2 = rq3.nearest_teammate_fraction(coords, PAIRS)
    assert eligible2 == 4
    assert hits2 < 4


def test_season_drift_ratio_relative_to_inter_median() -> None:
    s22 = _clustered_season()
    s23 = {d: c + np.array([0.05, 0, 0, 0, 0]) for d, c in s22.items()}  # tiny uniform drift
    ratios = rq3.season_drift_ratios({"2022": s22, "2023": s23})
    vals = [v for by in ratios.values() for v in by.values()]
    assert vals and all(np.isfinite(v) for v in vals)
    assert all(v < 0.5 for v in vals)  # drift 0.05 ≪ median inter-driver spacing


def test_team_silhouette_positive_for_clusters_none_for_one_team() -> None:
    s = rq3.team_silhouette(_clustered_season(), PAIRS)
    assert s is not None and s > 0.5
    assert rq3.team_silhouette(_clustered_season(), (("A1", "A2"),)) is None


def test_content_hash_is_stable_and_key_order_free() -> None:
    a = rq3._content_hash({"x": 1, "y": [1, 2]})
    b = rq3._content_hash({"y": [1, 2], "x": 1})
    assert a == b
