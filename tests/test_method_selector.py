"""
tests/test_method_selector.py

Unit tests for engine/method_selector.py.

Tests each of the four recommendation branches plus edge cases.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from method_selector import recommend_method

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_triangle.csv"


def _make_cdfs(avg: float) -> pd.Series:
    """Build a uniform CDF series so that the mean equals avg."""
    return pd.Series(
        {f"dev_{p}": avg for p in [12, 24, 36, 48, 60, 72]}, name="CDF to Ultimate"
    )


def _make_results(current_periods: list[int]) -> list[dict]:
    return [
        {
            "accident_year": 2018 + i,
            "current_period": p,
            "cl_ibnr": 100_000,
            "bf_ibnr": 110_000,
        }
        for i, p in enumerate(current_periods)
    ]


# ---------------------------------------------------------------------------
# Branch 1: avg CDF ≤ 1.10 → CL
# ---------------------------------------------------------------------------

class TestCLRecommendation:

    def test_mature_portfolio_recommends_cl(self):
        cdfs = _make_cdfs(1.05)
        results = _make_results([72, 60, 48, 36, 24, 12])
        method, rationale = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert method == "CL"

    def test_cl_rationale_mentions_cdf(self):
        cdfs = _make_cdfs(1.05)
        results = _make_results([72, 60, 48, 36, 24, 12])
        _, rationale = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert "1.05" in rationale or "CDF" in rationale

    def test_boundary_at_exactly_110(self):
        cdfs = _make_cdfs(1.10)
        results = _make_results([72, 60, 48, 36, 24, 12])
        method, _ = recommend_method(cdfs, results, has_premiums=False, elr=0.65)
        assert method == "CL"


# ---------------------------------------------------------------------------
# Branch 2: avg CDF > 1.25, premiums present → CC
# ---------------------------------------------------------------------------

class TestCCRecommendation:

    def test_immature_with_premiums_recommends_cc(self):
        cdfs = _make_cdfs(1.80)
        results = _make_results([12, 24, 36, 12, 24, 12])
        method, rationale = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert method == "CC"

    def test_cc_rationale_mentions_premium(self):
        cdfs = _make_cdfs(1.80)
        results = _make_results([12, 24, 36, 12, 24, 12])
        _, rationale = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert "premium" in rationale.lower() or "Cape Cod" in rationale


# ---------------------------------------------------------------------------
# Branch 3: avg CDF > 1.25, no premiums → BF
# ---------------------------------------------------------------------------

class TestBFRecommendationImmature:

    def test_immature_no_premiums_recommends_bf(self):
        cdfs = _make_cdfs(1.80)
        results = _make_results([12, 24, 36, 12, 24, 12])
        method, rationale = recommend_method(cdfs, results, has_premiums=False, elr=0.65)
        assert method == "BF"

    def test_bf_rationale_mentions_elr(self):
        cdfs = _make_cdfs(1.80)
        results = _make_results([12, 24, 36, 12, 24, 12])
        _, rationale = recommend_method(cdfs, results, has_premiums=False, elr=0.72)
        assert "0.72" in rationale


# ---------------------------------------------------------------------------
# Branch 4: avg CDF 1.10–1.25 (moderate immaturity) → BF
# ---------------------------------------------------------------------------

class TestBFRecommendationModerate:

    def test_moderate_immaturity_with_premiums_recommends_bf(self):
        cdfs = _make_cdfs(1.18)
        results = _make_results([60, 48, 36, 24, 12, 12])
        method, _ = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert method == "BF"

    def test_moderate_immaturity_no_premiums_recommends_bf(self):
        cdfs = _make_cdfs(1.18)
        results = _make_results([60, 48, 36, 24, 12, 12])
        method, rationale = recommend_method(cdfs, results, has_premiums=False, elr=0.65)
        assert method == "BF"
        assert "0.65" in rationale

    def test_boundary_just_above_110(self):
        cdfs = _make_cdfs(1.11)
        results = _make_results([72, 60, 48, 36, 24, 12])
        method, _ = recommend_method(cdfs, results, has_premiums=False, elr=0.65)
        assert method == "BF"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_cdfs_returns_bf(self):
        method, rationale = recommend_method(
            pd.Series(dtype=float), [], has_premiums=False, elr=0.65
        )
        assert method == "BF"
        assert "unavailable" in rationale.lower()

    def test_rationale_is_nonempty_string(self):
        cdfs = _make_cdfs(1.40)
        results = _make_results([24, 12])
        _, rationale = recommend_method(cdfs, results, has_premiums=True, elr=0.65)
        assert isinstance(rationale, str)
        assert len(rationale) > 20
