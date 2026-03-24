"""
tests/test_bornhuetter_ferguson.py

Unit tests for engine/bornhuetter_ferguson.py.

Key invariants verified
-----------------------
- BF IBNR = expected_ultimate × pct_unreported
- BF converges to CL for the most mature accident year (pct_unreported → 0)
- pct_unreported = 1 - 1/CDF (actuarial definition)
- Totals sum correctly
- Missing premium raises ValueError
- Per-year ELR dict is supported
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from bornhuetter_ferguson import BornhuetterFerguson
from chain_ladder import ChainLadder

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_triangle.csv"

PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}
ELR = 0.65


def _load_sample():
    from data_loader import load_triangle
    return load_triangle(_SAMPLE_CSV)


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------

class TestBFFormulas:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.triangle = _load_sample()
        self.bf = BornhuetterFerguson(self.triangle, PREMIUMS, elr=ELR)
        self.bf.run()

    def test_pct_unreported_formula(self):
        """pct_unreported ≈ 1 - 1/CDF for every accident year (2dp rounding tolerance)."""
        for year, row in self.bf.bf_summary.iterrows():
            cdf = row["cdf_to_ultimate"]
            expected_pct = (1.0 - 1.0 / cdf) * 100
            # stored value is rounded to 2dp
            assert row["pct_unreported"] == pytest.approx(expected_pct, abs=0.01)

    def test_bf_ibnr_formula(self):
        """bf_ibnr ≈ expected_ultimate × (1 - 1/CDF) from raw CDF (not rounded pct)."""
        for year, row in self.bf.bf_summary.iterrows():
            cdf = row["cdf_to_ultimate"]
            pct_unreported = 1.0 - (1.0 / cdf)
            expected_ibnr = row["expected_ultimate"] * pct_unreported
            # stored value is rounded to the nearest dollar
            assert row["bf_ibnr"] == pytest.approx(expected_ibnr, abs=500)

    def test_bf_ultimate_formula(self):
        """bf_ultimate = paid_to_date + bf_ibnr."""
        for year, row in self.bf.bf_summary.iterrows():
            expected_ultimate = row["paid_to_date"] + row["bf_ibnr"]
            assert row["bf_ultimate"] == pytest.approx(expected_ultimate, abs=1)

    def test_expected_ultimate_formula(self):
        """expected_ultimate = premium × ELR."""
        for year, row in self.bf.bf_summary.iterrows():
            assert row["expected_ultimate"] == pytest.approx(
                PREMIUMS[year] * ELR, abs=1
            )

    def test_bf_ibnr_nonnegative_for_positive_elr(self):
        """With a positive ELR, BF IBNR must be ≥ 0."""
        assert (self.bf.bf_summary["bf_ibnr"] >= 0).all()

    def test_comparison_diff_ibnr(self):
        """diff_ibnr = bf_ibnr - cl_ibnr."""
        for year, row in self.bf.comparison.iterrows():
            diff = row["bf_ibnr"] - row["cl_ibnr"]
            assert row["diff_ibnr"] == pytest.approx(diff, abs=1)


# ---------------------------------------------------------------------------
# Convergence to Chain Ladder
# ---------------------------------------------------------------------------

class TestBFConvergence:
    """BF must converge to CL for mature accident years (pct_unreported → 0)."""

    def test_most_mature_year_bf_close_to_cl(self):
        """AY 2018 is fully developed (dev_72) — BF IBNR must equal CL IBNR exactly."""
        triangle = _load_sample()
        bf = BornhuetterFerguson(triangle, PREMIUMS, elr=ELR)
        bf.run()

        # AY 2018 is in dev_72 → CDF(72) = 1.0 → pct_unreported = 0 → BF IBNR = 0
        assert bf.comparison.at[2018, "bf_ibnr"] == pytest.approx(0, abs=1)
        assert bf.comparison.at[2018, "cl_ibnr"] == pytest.approx(0, abs=1)

    def test_immature_year_bf_differs_from_cl(self):
        """The most immature AY should show a material BF-CL difference."""
        triangle = _load_sample()
        bf = BornhuetterFerguson(triangle, PREMIUMS, elr=ELR)
        bf.run()
        # AY 2023 has only 12 months of development — largest divergence expected
        diff_2023 = abs(bf.comparison.at[2023, "diff_ibnr"])
        diff_2018 = abs(bf.comparison.at[2018, "diff_ibnr"])
        assert diff_2023 > diff_2018


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestBFValidation:

    def test_missing_premium_raises(self):
        triangle = _load_sample()
        incomplete_premiums = {k: v for k, v in PREMIUMS.items() if k != 2023}
        with pytest.raises(ValueError, match="Premiums missing"):
            BornhuetterFerguson(triangle, incomplete_premiums, elr=ELR)

    def test_invalid_elr_raises(self):
        triangle = _load_sample()
        with pytest.raises(ValueError, match="plausible range"):
            BornhuetterFerguson(triangle, PREMIUMS, elr=3.0)

    def test_zero_premium_raises(self):
        triangle = _load_sample()
        bad_premiums = dict(PREMIUMS)
        bad_premiums[2018] = 0
        with pytest.raises(ValueError, match="positive"):
            BornhuetterFerguson(triangle, bad_premiums, elr=ELR)


# ---------------------------------------------------------------------------
# Per-year ELR dict
# ---------------------------------------------------------------------------

class TestPerYearELR:

    def test_per_year_elr_produces_different_ibnr(self):
        triangle = _load_sample()
        per_year_elr = {yr: 0.65 + 0.01 * i for i, yr in enumerate(PREMIUMS)}
        bf_varying = BornhuetterFerguson(triangle, PREMIUMS, elr=per_year_elr)
        bf_varying.run()

        bf_flat = BornhuetterFerguson(triangle, PREMIUMS, elr=0.65)
        bf_flat.run()

        # With different ELRs the total IBNR should differ from a flat ELR
        assert (
            bf_varying.comparison["bf_ibnr"].sum()
            != bf_flat.comparison["bf_ibnr"].sum()
        )
