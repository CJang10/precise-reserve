"""
bornhuetter_ferguson.py

Bornhuetter-Ferguson (BF) reserving method.

The BF method is a credibility blend of two approaches:
  - Chain Ladder   : projects ultimate purely from observed development patterns
  - Expected Losses: estimates ultimate as Premium × A Priori Expected Loss Ratio (ELR)

The blend is determined by how mature each accident year is. For each year:
  - The "reported" portion of losses is taken from actual paid data (credible)
  - The "unreported" portion is estimated from the a priori expectation (stable)

This makes BF more robust than Chain Ladder for immature accident years, where
thin data causes CL ultimates to be highly sensitive to early development volatility.

Formula
-------
    expected_ultimate  = Premium × ELR
    pct_unreported     = 1 − (1 / CDF)          [expected fraction still to emerge]
    bf_ibnr            = expected_ultimate × pct_unreported
    bf_ultimate        = paid_to_date + bf_ibnr

As a year matures (CDF → 1.0), pct_unreported → 0 and BF converges to CL,
because almost all losses have been observed and the prior has little influence.

Workflow
--------
1. _run_chain_ladder()     — derive CDFs from the triangle via ChainLadder
2. calculate_bf_ultimate() — apply the BF formula for each accident year
3. compare()               — join BF and CL results into a single comparison table
4. run()                   — orchestrate and return the comparison DataFrame
"""

import pandas as pd
import numpy as np
from pathlib import Path

from chain_ladder import ChainLadder, DEV_COLUMNS


class BornhuetterFerguson:
    """
    Bornhuetter-Ferguson reserving model.

    Parameters
    ----------
    triangle : pd.DataFrame
        Loss development triangle produced by data_loader.load_triangle().
        Rows are accident years, columns are dev_12 … dev_72.
    premiums : dict or pd.Series
        Earned premium for each accident year, keyed by accident year integer.
        Example: {2018: 13_500_000, 2019: 14_200_000, ...}
    elr : float, optional
        A priori expected loss ratio (default 0.65).
        This represents the actuary's initial expectation of the loss ratio
        before considering the emerging paid loss experience.

    Attributes
    ----------
    cl : ChainLadder
        Fitted Chain Ladder model (provides CDFs and CL results for comparison).
    bf_summary : pd.DataFrame or None
        BF results table after calculate_bf_ultimate().
    comparison : pd.DataFrame or None
        Side-by-side BF vs CL table after compare().
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        premiums: dict | pd.Series,
        elr: float | dict[int, float] = 0.65,
        tail_factor: float = 1.0,
    ) -> None:
        self.triangle = triangle.copy()
        self.premiums = pd.Series(premiums, name="premium", dtype=float)
        self.tail_factor = tail_factor

        if isinstance(elr, dict):
            missing_elr = [yr for yr in self.triangle.index if yr not in elr]
            if missing_elr:
                raise ValueError(
                    f"ELR missing for accident year(s): {missing_elr}. "
                    "Provide an ELR entry for every row in the triangle, or pass a single scalar."
                )
            for yr, e in elr.items():
                if not (0.0 < e < 2.0):
                    raise ValueError(
                        f"ELR for accident year {yr} ({e:.2f}) is outside a plausible range (0, 2)."
                    )
            self.elr: float | dict[int, float] = {int(k): float(v) for k, v in elr.items()}
        else:
            if not (0.0 < elr < 2.0):
                raise ValueError(f"ELR of {elr:.2f} is outside a plausible range (0, 2).")
            self.elr = float(elr)

        self.cl: ChainLadder | None = None
        self.bf_summary: pd.DataFrame | None = None
        self.comparison: pd.DataFrame | None = None

        self._validate_premiums()

    def _validate_premiums(self) -> None:
        """Ensure premiums are provided for every accident year in the triangle."""
        missing = [yr for yr in self.triangle.index if yr not in self.premiums.index]
        if missing:
            raise ValueError(
                f"Premiums missing for accident year(s): {missing}. "
                "Provide a premium entry for every row in the triangle."
            )
        if (self.premiums <= 0).any():
            raise ValueError("All premium values must be positive.")

    def _elr_for(self, year: int) -> float:
        """Return the ELR for a given accident year (supports scalar or per-year dict)."""
        if isinstance(self.elr, dict):
            return self.elr[year]
        return self.elr

    # ------------------------------------------------------------------
    # Step 1: Derive CDFs from Chain Ladder
    # ------------------------------------------------------------------

    def _run_chain_ladder(self) -> ChainLadder:
        """
        Fit the Chain Ladder model and cache the result.

        The BF method does not re-derive its own development factors — it reuses
        the volume-weighted CDFs from the Chain Ladder. This ensures both methods
        start from the same pattern assumptions and isolates the only structural
        difference: how the unreported portion is estimated.

        Returns
        -------
        ChainLadder
            Fitted model with ldfs, cdfs, projected_triangle, and summary populated.
        """
        self.cl = ChainLadder(self.triangle, tail_factor=self.tail_factor)
        self.cl.run()
        return self.cl

    # ------------------------------------------------------------------
    # Step 2: BF ultimate and IBNR
    # ------------------------------------------------------------------

    def calculate_bf_ultimate(self) -> pd.DataFrame:
        """
        Apply the Bornhuetter-Ferguson formula to each accident year.

        For each accident year the calculation proceeds as follows:

        (a) expected_ultimate  = Premium × ELR
               The a priori loss estimate — what we expected before seeing any data.

        (b) pct_unreported  = 1 − (1 / CDF)
               The expected fraction of ultimate losses not yet paid as of the
               evaluation date. Derived from the Chain Ladder CDF.
               - Mature year  (CDF near 1.0): pct_unreported ≈ 0  → minimal IBNR
               - Immature year (CDF >> 1.0):  pct_unreported >> 0 → most losses unseen

        (c) bf_ibnr  = expected_ultimate × pct_unreported
               The amount expected to emerge in future periods, sourced from
               the a priori expectation rather than development of actual losses.
               This is the BF method's key stabiliser for immature years.

        (d) bf_ultimate  = paid_to_date + bf_ibnr
               Actual paid losses plus the expected future emergence. Unlike CL,
               the paid losses are not multiplied by any factor — they are taken
               as fully credible for what has already happened.

        Returns
        -------
        pd.DataFrame
            One row per accident year, indexed by accident_year, with columns:

            premium           : earned premium for the accident year
            elr               : a priori expected loss ratio (same for all rows)
            expected_ultimate : premium × ELR
            paid_to_date      : latest observed cumulative paid loss
            current_period    : months of development at the evaluation date
            cdf_to_ultimate   : CDF from current period to ultimate (from CL)
            pct_unreported    : 1 − 1/CDF, the expected unreported fraction
            bf_ultimate       : BF projected ultimate loss
            bf_ibnr           : BF IBNR reserve (bf_ultimate − paid_to_date)
        """
        if self.cl is None:
            self._run_chain_ladder()

        records = []
        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            paid = row[last_col]
            period = int(last_col.replace("dev_", ""))
            cdf = self.cl.cdfs[last_col]

            premium = self.premiums[year]
            year_elr = self._elr_for(year)
            expected_ultimate = premium * year_elr

            # The unreported fraction: complement of the reciprocal of the CDF.
            # 1/CDF is the "percent paid to date" implied by the development pattern.
            pct_unreported = 1.0 - (1.0 / cdf)

            bf_ibnr = expected_ultimate * pct_unreported
            bf_ultimate = paid + bf_ibnr

            records.append(
                {
                    "accident_year": year,
                    "premium": premium,
                    "elr": year_elr,
                    "expected_ultimate": round(expected_ultimate),
                    "paid_to_date": paid,
                    "current_period": period,
                    "cdf_to_ultimate": round(cdf, 4),
                    "pct_unreported": round(pct_unreported * 100, 2),
                    "bf_ultimate": round(bf_ultimate),
                    "bf_ibnr": round(bf_ibnr),
                }
            )

        self.bf_summary = pd.DataFrame(records).set_index("accident_year")
        return self.bf_summary

    # ------------------------------------------------------------------
    # Step 3: Comparison table
    # ------------------------------------------------------------------

    def compare(self) -> pd.DataFrame:
        """
        Build a side-by-side comparison of BF and Chain Ladder reserve estimates.

        Joining BF and CL on the same rows makes it easy to see where the methods
        agree (mature years with credible data) and where they diverge (immature
        years where the a priori expectation pulls BF away from CL).

        The diff_ibnr column (BF − CL) is a useful diagnostic:
          - Positive: BF is more conservative than CL (ELR implies more loss)
          - Negative: BF is more optimistic than CL (ELR implies less loss)
          - Near zero: methods agree — year is mature or ELR ≈ implied CL LR

        Returns
        -------
        pd.DataFrame
            One row per accident year with columns:

            paid_to_date      : latest observed cumulative paid
            current_period    : months of development
            premium           : earned premium
            expected_ultimate : premium × ELR (a priori)
            cl_ultimate       : Chain Ladder projected ultimate
            cl_ibnr           : Chain Ladder IBNR
            bf_ultimate       : Bornhuetter-Ferguson projected ultimate
            bf_ibnr           : Bornhuetter-Ferguson IBNR
            diff_ibnr         : bf_ibnr − cl_ibnr  (method divergence)
        """
        if self.bf_summary is None:
            self.calculate_bf_ultimate()

        cl = self.cl.summary[["ultimate", "ibnr"]].rename(
            columns={"ultimate": "cl_ultimate", "ibnr": "cl_ibnr"}
        )
        bf = self.bf_summary[
            ["paid_to_date", "current_period", "premium",
             "expected_ultimate", "bf_ultimate", "bf_ibnr"]
        ]

        combined = bf.join(cl)
        combined["diff_ibnr"] = combined["bf_ibnr"] - combined["cl_ibnr"]
        combined = combined[
            ["paid_to_date", "current_period", "premium", "expected_ultimate",
             "cl_ultimate", "cl_ibnr", "bf_ultimate", "bf_ibnr", "diff_ibnr"]
        ]

        self.comparison = combined
        return self.comparison

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full BF workflow and return the comparison table.

        Steps
        -----
        1. _run_chain_ladder()     — fit CL to obtain CDFs
        2. calculate_bf_ultimate() — apply BF formula
        3. compare()               — join BF and CL results

        Returns
        -------
        pd.DataFrame
            Side-by-side BF vs CL comparison (same as compare() output).
        """
        self._run_chain_ladder()
        self.calculate_bf_ultimate()
        return self.compare()


# ------------------------------------------------------------------
# Pretty-print helper
# ------------------------------------------------------------------

def print_results(model: BornhuetterFerguson) -> None:  # pragma: no cover
    """Print the BF vs Chain Ladder comparison table and aggregate totals."""
    df = model.comparison

    currency_cols = [
        "paid_to_date", "premium", "expected_ultimate",
        "cl_ultimate", "cl_ibnr", "bf_ultimate", "bf_ibnr", "diff_ibnr",
    ]

    elr_display = "per-year" if isinstance(model.elr, dict) else f"{model.elr:.0%}"
    print("=" * 100)
    print(f"  BORNHUETTER-FERGUSON vs CHAIN LADDER   |   A Priori ELR: {elr_display}")
    print("=" * 100)

    display = df.copy().astype(object)
    for col in currency_cols:
        display[col] = df[col].map(lambda x: f"${x:>12,.0f}")
    display["current_period"] = df["current_period"].map(lambda x: f"{x} mo")

    print(display.to_string())

    print("-" * 100)
    totals = {col: df[col].sum() for col in currency_cols}
    print(
        f"  {'TOTAL':30s}"
        f"  CL Ultimate: ${totals['cl_ultimate']:>12,.0f}"
        f"  CL IBNR: ${totals['cl_ibnr']:>12,.0f}"
        f"  BF Ultimate: ${totals['bf_ultimate']:>12,.0f}"
        f"  BF IBNR: ${totals['bf_ibnr']:>12,.0f}"
        f"  Diff: ${totals['diff_ibnr']:>10,.0f}"
    )
    print("=" * 100)

    print()
    print("  KEY METRICS")
    print("-" * 50)
    total_paid = totals["paid_to_date"]
    total_premium = totals["premium"]
    print(f"  Total Paid-to-Date   : ${total_paid:>12,.0f}")
    print(f"  Total Premium        : ${total_premium:>12,.0f}")
    print(f"  CL Total IBNR        : ${totals['cl_ibnr']:>12,.0f}   ({totals['cl_ibnr']/total_premium:.1%} of premium)")
    print(f"  BF Total IBNR        : ${totals['bf_ibnr']:>12,.0f}   ({totals['bf_ibnr']/total_premium:.1%} of premium)")
    print(f"  BF vs CL Difference  : ${totals['diff_ibnr']:>12,.0f}")
    print("=" * 50)


if __name__ == "__main__":  # pragma: no cover
    from data_loader import load_triangle

    data_path = Path(__file__).parent.parent / "data" / "claims_triangle.csv"
    triangle = load_triangle(data_path)

    # Earned premiums for each accident year (small personal auto insurer).
    # 2020 reflects a mid-year COVID premium refund reducing earned exposure.
    premiums = {
        2018: 13_500_000,
        2019: 14_200_000,
        2020: 12_800_000,
        2021: 15_000_000,
        2022: 15_800_000,
        2023: 16_500_000,
    }

    model = BornhuetterFerguson(triangle, premiums, elr=0.65)
    model.run()
    print_results(model)
