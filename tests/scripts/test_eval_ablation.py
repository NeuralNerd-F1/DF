"""Ablation-matrix mechanisms — partition integrity, hyperparameter extraction, the
normalisation mirror. The trained rows live in the content-hashed artefact."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.pipeline.features import FEATURE_IDS

from scripts import eval_ablation as ab


def test_domains_partition_the_28_exactly() -> None:
    flat = [f for fs in ab.DOMAINS.values() for f in fs]
    assert sorted(flat) == sorted(FEATURE_IDS)
    assert len(flat) == len(set(flat))  # disjoint — no feature counted in two domains


def test_champion_params_requires_the_bundle_contract() -> None:
    with pytest.raises(ValueError, match="calibrated_classifiers_"):
        ab.champion_params(SimpleNamespace(calibrated_classifiers_=None))


def test_champion_params_digs_through_frozen_wrapper() -> None:
    class Inner:
        def get_params(self):
            return {"n_estimators": 7, "max_depth": 3, "irrelevant": "x", "n_jobs": 1}

    frozen = SimpleNamespace(estimator=Inner())
    model = SimpleNamespace(calibrated_classifiers_=[SimpleNamespace(estimator=frozen)])
    p = ab.champion_params(model)
    assert p == {"n_estimators": 7, "max_depth": 3, "n_jobs": 1}  # only the whitelisted keys


def test_normalise_split_uses_per_circuit_then_global() -> None:
    params = SimpleNamespace(
        global_={"F09": SimpleNamespace(mean=0.0, std=2.0)},
        per_circuit={"Bahrain": {"F09": SimpleNamespace(mean=10.0, std=5.0)}},
    )
    df = pd.DataFrame(
        {"circuit": ["Bahrain", "Belgium"], "F01": [1.0, 2.0], "F09": [15.0, 4.0]}
    )
    x = ab.normalise_split(df, ["F09"], ["F01", "F09"], params)
    assert x[0, 0] == 1.0 and x[1, 0] == 2.0  # non-normalised feature verbatim
    assert x[0, 1] == pytest.approx(1.0)  # Bahrain row → per-circuit stats
    assert x[1, 1] == pytest.approx(2.0)  # Belgium (unknown) row → global fallback


def test_fit_and_score_runs_a_real_tiny_fit() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] > 0).astype(int)  # perfectly separable on feature 0
    f1 = ab.fit_and_score(
        x[:60], y[:60], x[60:], y[60:],
        {"n_estimators": 20, "max_depth": 2, "random_state": 42, "n_jobs": 1},
    )
    assert f1 > 0.9
