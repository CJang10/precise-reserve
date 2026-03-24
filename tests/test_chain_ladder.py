"""
tests/test_chain_ladder.py

Unit tests for engine/chain_ladder.py.

Verifies LDFs, CDFs, ultimates, and IBNR against:
  - Triangle A: the bundled sample (6 AYs, real actuarial data)
  - Triangle B: a synthetic triangle with analytically exact LDFs
  - Triangle C: a minimal 3-AY triangle (boundary condition)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make engine importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from chain_ladder import ChainLadder, DEV_COLUMNS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_triangle.csv"

PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}


def _load_sample() -> pd.DataFrame:
    """Load the bundled sample triangle (drop the premium column)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
    from data_loader import load_triangle
    return load_triangle(_SAMPLE_CSV)


def _make_triangle_b() -> pd.DataFrame:
    """
    Synthetic triangle with exact LDF = 1.5 for every period.

    All rows have dev_12 = 1 000, dev_24 = 1 500, dev_36 = 2 250, …
    Volume-weighted LDF = 1.5 exactly for every transition.
    """
    base = 1_000
    ratio = 1.5
    data = {}
    for i, col in enumerate(DEV_COLUMNS):
        val = base * (ratio ** i)
        data[col] = [val, val * 1.1, val * 0.9, val * 1.05, val * 0.95, val]

    df = pd.DataFrame(data, index=[2018, 2019, 2020, 2021, 2022, 2023])
    df.index.name = "accident_year"

    # Apply lower-left triangle pattern: each subsequent year has one fewer period
    for i, year in enumerate(df.index):
        # year 2018 (i=0) keeps all columns; year 2023 (i=5) keeps only dev_12
        for j, col in enumerate(DEV_COLUMNS):
            if j >= len(DEV_COLUMNS) - i:
                df.at[year, col] = np.nan
    return df


def _make_triangle_c() -> pd.DataFrame:
    """
    Minimal 3-AY triangle designed for exact manual verification.

    AY 2021 is fully developed; AY 2022 and 2023 are immature.

        AY   12      24      36      48      60      72
        2021 100,000 150,000 180,000 200,000 210,000 215,000
        2022 110,000 165,000 198,000
        2023 120,000

    LDF(12→24) = (150k + 165k) / (100k + 110k) = 315k / 210k = 1.5000
    LDF(24→36) = (180k + 198k) / (150k + 165k) = 378k / 315k = 1.2000
    LDF(36→48) = 200k / 180k ≈ 1.1111
    LDF(48→60) = 210k / 200k = 1.05
    LDF(60→72) = 215k / 210k ≈ 1.02381
    """
    data = {
        "dev_12": [100_000, 110_000, 120_000],
        "dev_24": [150_000, 165_000, np.nan],
        "dev_36": [180_000, 198_000, np.nan],
        "dev_48": [200_000, np.nan,   np.nan],
        "dev_60": [210_000, np.nan,   np.nan],
        "dev_72": [215_000, np.nan,   np.nan],
    }
    df = pd.DataFrame(data, index=[2021, 2022, 2023])
    df.index.name = "accident_year"
    return df


# ---------------------------------------------------------------------------
# Triangle A — sample data
# ---------------------------------------------------------------------------

class TestChainLadderSample:
    """Tests using the bundled sample triangle."""

    @pytest.fixture(autouse=True)
    def setup(self):
        triangle = _load_sample()
        self.cl = ChainLadder(triangle)
        self.cl.run()

    def test_ldf_count(self):
        """One LDF per adjacent pair of development periods."""
        assert len(self.cl.ldfs) == len(DEV_COLUMNS) - 1

    def test_ldfs_positive(self):
        """All LDFs must be strictly positive for a standard triangle."""
        assert (self.cl.ldfs > 0).all()

    def test_ldfs_above_unity(self):
        """All LDFs in the sample are > 1.0 (paid losses increase over time)."""
        assert (self.cl.ldfs > 1.0).all()

    def test_cdf_count(self):
        """One CDF per development period column."""
        assert len(self.cl.cdfs) == len(DEV_COLUMNS)

    def test_cdfs_geq_tail_factor(self):
        """All CDFs must be ≥ tail_factor (default 1.0)."""
        assert (self.cl.cdfs >= 1.0).all()

    def test_cdf_rightmost_equals_tail_factor(self):
        """CDF at the last period equals the tail factor."""
        assert self.cl.cdfs["dev_72"] == pytest.approx(self.cl.tail_factor)

    def test_cdf_monotone_decreasing(self):
        """CDFs must be non-increasing from left (most immature) to right (most mature)."""
        cdf_values = [self.cl.cdfs[col] for col in DEV_COLUMNS]
        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] >= cdf_values[i + 1]

    def test_ibnr_nonnegative(self):
        """IBNR must be ≥ 0 for every accident year in a non-declining triangle."""
        assert (self.cl.summary["ibnr"] >= 0).all()

    def test_ultimate_geq_paid(self):
        """Ultimate must be ≥ paid-to-date for every accident year."""
        assert (self.cl.summary["ultimate"] >= self.cl.summary["paid_to_date"]).all()

    def test_ibnr_equals_ultimate_minus_paid(self):
        """IBNR = Ultimate - Paid (up to rounding)."""
        diff = (
            self.cl.summary["ultimate"]
            - self.cl.summary["paid_to_date"]
            - self.cl.summary["ibnr"]
        )
        assert (diff.abs() <= 1).all()  # rounding tolerance of $1

    def test_most_mature_year_low_ibnr(self):
        """The most mature accident year (2018, dev_72) should have zero IBNR."""
        assert self.cl.summary.at[2018, "ibnr"] == pytest.approx(0, abs=1)

    def test_pct_developed_between_0_and_100(self):
        """Percent developed must be in (0, 100]."""
        pct = self.cl.summary["pct_developed"]
        assert (pct > 0).all()
        assert (pct <= 100.0).all()

    def test_summary_index_matches_triangle(self):
        triangle = _load_sample()
        assert list(self.cl.summary.index) == list(triangle.index)


# ---------------------------------------------------------------------------
# Triangle B — synthetic, analytically exact LDFs
# ---------------------------------------------------------------------------

class TestChainLadderSynthetic:
    """Tests against a triangle where every LDF is exactly 1.5."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.triangle = _make_triangle_b()
        self.cl = ChainLadder(self.triangle)
        self.cl.run()

    def test_all_ldfs_equal_ratio(self):
        """Volume-weighted LDFs should equal exactly 1.5 for every transition."""
        for label, ldf in self.cl.ldfs.items():
            assert ldf == pytest.approx(1.5, rel=1e-6), f"LDF {label} = {ldf}"

    def test_cdf_product_formula(self):
        """CDF(dev_12) should equal the product of all 5 LDFs times the tail factor."""
        product = 1.0
        for ldf in self.cl.ldfs.values:
            product *= ldf
        product *= self.cl.tail_factor
        assert self.cl.cdfs["dev_12"] == pytest.approx(product, rel=1e-6)


# ---------------------------------------------------------------------------
# Triangle C — minimal 3-AY triangle
# ---------------------------------------------------------------------------

class TestChainLadderMinimal:
    """Tests using the minimum-size triangle (3 accident years)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.triangle = _make_triangle_c()
        self.cl = ChainLadder(self.triangle)
        self.cl.run()

    def test_ldf_12_24_correct(self):
        """Verify LDF(12→24) by hand: (150k+165k) / (100k+110k) = 315/210 = 1.5."""
        assert self.cl.ldfs["12→24"] == pytest.approx(315_000 / 210_000, rel=1e-6)

    def test_ldf_24_36_correct(self):
        """Verify LDF(24→36) by hand: (180k+198k) / (150k+165k) = 378/315 = 1.2."""
        assert self.cl.ldfs["24→36"] == pytest.approx(378_000 / 315_000, rel=1e-6)

    def test_ibnr_computed_for_all_years(self):
        assert len(self.cl.summary) == 3


# ---------------------------------------------------------------------------
# Tail factor tests
# ---------------------------------------------------------------------------

class TestTailFactor:

    def test_tail_factor_propagates_to_cdf(self):
        """A tail factor of 1.05 must be reflected in the CDF at dev_72."""
        triangle = _load_sample()
        cl = ChainLadder(triangle, tail_factor=1.05)
        cl.run()
        assert cl.cdfs["dev_72"] == pytest.approx(1.05)

    def test_tail_factor_below_unity_raises(self):
        with pytest.raises(ValueError, match="tail_factor must be"):
            ChainLadder(_load_sample(), tail_factor=0.99)

    def test_tail_factor_above_two_raises(self):
        with pytest.raises(ValueError, match="exceeds 2.0"):
            ChainLadder(_load_sample(), tail_factor=2.01)

    def test_fit_tail_returns_float_geq_one(self):
        triangle = _load_sample()
        cl = ChainLadder(triangle)
        cl.run()
        tail = cl.fit_tail()
        assert isinstance(tail, float)
        assert tail >= 1.0

    def test_fit_tail_stored_on_instance(self):
        triangle = _load_sample()
        cl = ChainLadder(triangle)
        cl.run()
        tail = cl.fit_tail()
        assert cl.fitted_tail_factor == tail
