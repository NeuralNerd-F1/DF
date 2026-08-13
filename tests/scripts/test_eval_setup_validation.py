"""Setup-validation mechanisms — the pure scoring helpers, on constructed values."""

from __future__ import annotations

import math

from scripts import eval_setup_validation as sv


def test_tendency_direction_honest_absence() -> None:
    assert sv.tendency_direction(5.0, 4.0) == "more"
    assert sv.tendency_direction(3.0, 4.0) == "less"
    assert sv.tendency_direction(4.0, 4.0) == "level"
    assert sv.tendency_direction(None, 4.0) is None  # WIDE → no direction, never invented
    assert sv.tendency_direction(5.0, None) is None  # no grid reference → no direction


def test_direction_consistency_pairwise() -> None:
    assert sv.direction_consistency(["more", "more", "more"]) == 1.0
    assert sv.direction_consistency(["more", "less"]) == 0.0
    # 2 agreements of 3 pairs when one season disagrees.
    assert sv.direction_consistency(["more", "more", "less"]) == 1 / 3
    # None entries (WIDE seasons) are excluded, not counted as disagreement.
    assert sv.direction_consistency(["more", None, "more"]) == 1.0
    assert sv.direction_consistency(["more", None]) is None  # <2 usable samples → honest None


def test_normalised_error_by_fia_range() -> None:
    assert sv.normalised_error(6.0, 4.0, 0.0, 12.0) == 2.0 / 12.0
    assert sv.normalised_error(6.0, 6.0, 0.0, 12.0) == 0.0
    assert math.isnan(sv.normalised_error(1.0, 2.0, 5.0, 5.0))  # zero span → NaN, never a fake 0


def test_content_hash_key_order_free() -> None:
    assert sv._content_hash({"a": 1, "b": 2}) == sv._content_hash({"b": 2, "a": 1})
