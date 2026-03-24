"""
tests/test_cape_cod.py

Unit tests for engine/cape_cod.py.

Key invariants verified
-----------------------
- CC ELR derivation: cc_elr = Σ paid / Σ used_up_premium
- used_up_premium = premium × (1 / CDF)
- CC IBNR formula: cc_ibnr = (premium × cc_elr) × pct_unreported
- CC ultimate = paid + cc_ibnr
- CC ELR is in (0, 2) for a valid triangle
- Compared to BF: CC uses data-derived ELR; BF uses a priori ELR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from cape_cod import CapeCod
from bornhuetter_ferguson import BornhuetterFerguson

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_triangle.csv"

PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}


def _load_sample():
    from data_loader import load_triangle
    return load_triangle(_SAMPLE_CSV)


# ---------------------------------------------------------------------------
# ELR derivation
# ---------------------------------------------------------------------------

class TestCCELRDerivation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.triangle = _load_sample()
        self.cc = CapeCod(self.triangle, PREMIUMS)
        self.cc.run()

    def test_elr_in_plausible_range(self):
        """CC ELR must be in (0, 2) for a valid commercial lines triangle."""
        assert 0.0 < self.cc.cc_elr < 2.0

    def test_elr_formula_correctness(self):
        """Verify cc_elr = Σ paid / Σ used_up_premium from first principles."""
        total_paid = 0.0
        total_used_up = 0.0
        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            paid = float(row[last_col])
            cdf = float(self.cc.cl.cdfs[last_col])
            premium = float(PREMIUMS[year])
            total_paid += paid
            total_used_up += premium * (1.0 / cdf)
        expected_elr = total_paid / total_used_up
        assert self.cc.cc_elr == pytest.approx(expected_elr, rel=1e-6)

    def test_elr_stored_on_summary(self):
        """Every row in cc_summary references the same derived ELR."""
        for year, row in self.cc.cc_summary.iterrows():
            assert row["cc_elr"] == pytest.approx(self.cc.cc_elr, rel=1e-6)


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------

class TestCCFormulas:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.triangle = _load_sample()
        self.cc = CapeCod(self.triangle, PREMIUMS)
        self.cc.run()

    def test_cc_ibnr_formula(self):
        """cc_ibnr ≈ (premium × cc_elr) × (1 - 1/CDF) for every accident year."""
        for year, row in self.cc.cc_summary.iterrows():
            cdf = row["cdf_to_ultimate"]
            pct_unreported = 1.0 - (1.0 / cdf)
            expected_ibnr = float(PREMIUMS[year]) * self.cc.cc_elr * pct_unreported
            # stored value is rounded to nearest dollar; allow generous tolerance
            assert row["cc_ibnr"] == pytest.approx(expected_ibnr, abs=500)

    def test_cc_ultimate_formula(self):
        """cc_ultimate = paid_to_date + cc_ibnr."""
        for year, row in self.cc.cc_summary.iterrows():
            expected = row["paid_to_date"] + row["cc_ibnr"]
            assert row["cc_ultimate"] == pytest.approx(expected, abs=1)

    def test_cc_ibnr_nonnegative(self):
        """CC IBNR must be ≥ 0 for a triangle with positive CDF and ELR."""
        assert (self.cc.cc_summary["cc_ibnr"] >= 0).all()

    def test_most_mature_year_near_zero_ibnr(self):
        """AY 2018 is at dev_72 (tail_factor=1.0) → CDF = 1.0 → CC IBNR ≈ 0."""
        assert self.cc.cc_summary.at[2018, "cc_ibnr"] == pytest.approx(0, abs=1)


# ---------------------------------------------------------------------------
# CC vs BF relationship
# ---------------------------------------------------------------------------

class TestCCvsBF:
    """Cape Cod and BF share the same structural formula but differ in ELR source."""

    def test_same_elr_produces_same_ibnr(self):
        """When BF ELR equals the derived CC ELR, BF and CC IBNR should match."""
        triangle = _load_sample()
        cc = CapeCod(triangle, PREMIUMS)
        cc.run()

        # Use the derived CC ELR as the a priori ELR for BF
        bf = BornhuetterFerguson(triangle, PREMIUMS, elr=cc.cc_elr)
        bf.run()

        for year in triangle.index:
            assert cc.cc_summary.at[year, "cc_ibnr"] == pytest.approx(
                bf.comparison.at[year, "bf_ibnr"], abs=1
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestCCValidation:

    def test_missing_premium_raises(self):
        triangle = _load_sample()
        incomplete = {k: v for k, v in PREMIUMS.items() if k != 2023}
        with pytest.raises(ValueError, match="Premiums missing"):
            CapeCod(triangle, incomplete)

    def test_zero_premium_raises(self):
        triangle = _load_sample()
        bad = dict(PREMIUMS)
        bad[2018] = 0
        with pytest.raises(ValueError, match="positive"):
            CapeCod(triangle, bad)
