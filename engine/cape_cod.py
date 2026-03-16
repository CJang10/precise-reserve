"""
cape_cod.py

Cape Cod (Stanard-Bühlmann) reserving method.

The Cape Cod method is a data-driven variant of Bornhuetter-Ferguson. Rather than
relying on an externally specified a priori ELR, it derives the ELR from the actual
emerging loss experience weighted by each accident year's maturity — the "used-up
premium" approach.

Key difference from BF
----------------------
BF  : cc_expected_ult = premium × ELR_external   (actuary's prior belief)
CC  : cc_expected_ult = premium × ELR_derived     (inferred from the data itself)

Because the ELR is derived from the same triangle, Cape Cod lies between Chain
Ladder (fully data-driven, sensitive to thin data) and BF (fully prior-driven,
insensitive to emerging experience). For triangles with sufficient history the CC
ELR converges to the actual aggregate loss ratio.

Key formulas
------------
    used_up_premium[y]  = premium[y] × (1 / CDF[y])
        The portion of earned premium "consumed" by observed development:
        immature years (large CDF → small 1/CDF) contribute little, so their
        noisy early data is naturally down-weighted in the ELR estimate.

    cc_elr              = Σ paid_to_date / Σ used_up_premium
        Volume-weighted actual loss ratio across all accident years.

    cc_expected_ult[y]  = premium[y] × cc_elr
    pct_unreported[y]   = 1 − (1 / CDF[y])
    cc_ibnr[y]          = cc_expected_ult[y] × pct_unreported[y]
    cc_ultimate[y]      = paid_to_date[y] + cc_ibnr[y]

Workflow
--------
1. _run_chain_ladder()     — derive CDFs from the triangle via ChainLadder
2. calculate_cc_elr()      — derive the CC ELR from used-up premium
3. calculate_cc_ultimate() — apply the CC formula for each accident year
4. compare()               — join CC and CL results into a comparison table
5. run()                   — orchestrate all steps and return the comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path

from chain_ladder import ChainLadder, DEV_COLUMNS


class CapeCod:
    """
    Cape Cod (Stanard-Bühlmann) reserving model.

    Parameters
    ----------
    triangle : pd.DataFrame
        Loss development triangle produced by data_loader.load_triangle().
        Rows are accident years, columns are dev_12 … dev_72.
    premiums : dict or pd.Series
        Earned premium for each accident year, keyed by accident year integer.
        Example: {2018: 13_500_000, 2019: 14_200_000, ...}
    tail_factor : float, optional
        Tail factor passed through to the internal ChainLadder (default 1.0).
        Affects CDFs and therefore both the used-up premium weights and the
        unreported fraction applied in the CC formula.

    Attributes
    ----------
    cl : ChainLadder
        Fitted Chain Ladder model (provides CDFs and CL results for comparison).
    cc_elr : float or None
        Derived Cape Cod ELR after calculate_cc_elr().
    cc_summary : pd.DataFrame or None
        CC results table after calculate_cc_ultimate().
    comparison : pd.DataFrame or None
        Side-by-side CC vs CL table after compare().
    """

    def __init__(
        self,
        triangle: pd.DataFrame,
        premiums: dict | pd.Series,
        tail_factor: float = 1.0,
    ) -> None:
        self.triangle = triangle.copy()
        self.premiums = pd.Series(premiums, name="premium", dtype=float)
        self.tail_factor = tail_factor

        self.cl: ChainLadder | None = None
        self.cc_elr: float | None = None
        self.cc_summary: pd.DataFrame | None = None
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

    # ------------------------------------------------------------------
    # Step 1: Derive CDFs from Chain Ladder
    # ------------------------------------------------------------------

    def _run_chain_ladder(self) -> ChainLadder:
        """
        Fit the Chain Ladder model and cache the result.

        CC reuses CL development factors so both methods share the same
        pattern assumptions. The only structural difference is in how the
        unreported portion is priced: CL multiplies paid data forward,
        CC anchors to an experience-derived expected ultimate.

        Returns
        -------
        ChainLadder
            Fitted model with ldfs, cdfs, projected_triangle, and summary populated.
        """
        self.cl = ChainLadder(self.triangle, tail_factor=self.tail_factor)
        self.cl.run()
        return self.cl

    # ------------------------------------------------------------------
    # Step 2: Derive the Cape Cod ELR
    # ------------------------------------------------------------------

    def calculate_cc_elr(self) -> float:
        """
        Derive the Cape Cod ELR from the actual loss experience.

        For each accident year, the "used-up premium" is the share of earned
        premium corresponding to the portion of losses already observed:

            used_up_premium[y] = premium[y] × (1 / CDF[y])

        where 1/CDF[y] is the percent-paid-to-date implied by the development
        pattern. Immature years (CDF >> 1 → small 1/CDF) contribute little,
        which naturally down-weights their noisy early development in the ELR.

        The CC ELR is the aggregate experience loss ratio:

            cc_elr = Σ paid_to_date / Σ used_up_premium

        Returns
        -------
        float
            The derived Cape Cod ELR.

        Notes
        -----
        Unlike the BF ELR this requires no external benchmark. The cost is
        parameter uncertainty: if the triangle is small or dominated by a single
        large accident year the derived ELR may be unstable.
        """
        if self.cl is None:
            self._run_chain_ladder()

        total_paid = 0.0
        total_used_up_premium = 0.0

        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            paid = float(row[last_col])
            cdf = float(self.cl.cdfs[last_col])
            pct_developed = 1.0 / cdf          # reciprocal of CDF = % paid to date
            premium = float(self.premiums[year])

            total_paid += paid
            total_used_up_premium += premium * pct_developed

        if total_used_up_premium == 0.0:
            raise ValueError(
                "Total used-up premium is zero — cannot derive the Cape Cod ELR. "
                "Check that all CDFs are finite and premiums are positive."
            )

        self.cc_elr = total_paid / total_used_up_premium
        return self.cc_elr

    # ------------------------------------------------------------------
    # Step 3: CC ultimate and IBNR
    # ------------------------------------------------------------------

    def calculate_cc_ultimate(self) -> pd.DataFrame:
        """
        Apply the Cape Cod formula to each accident year.

        For each accident year:

        (a) cc_expected_ultimate = premium × cc_elr
               The expected total loss based on the experience-derived ELR.

        (b) pct_unreported = 1 − (1 / CDF)
               Expected fraction of ultimate losses not yet paid. Identical
               to the BF formula — the only difference is the ELR source.

        (c) cc_ibnr = cc_expected_ultimate × pct_unreported
               Future emergence, anchored to the data-derived expectation.

        (d) cc_ultimate = paid_to_date + cc_ibnr
               Actual paid losses plus the CC-estimated future emergence.

        Returns
        -------
        pd.DataFrame
            One row per accident year, indexed by accident_year, with columns:

            premium              : earned premium
            cc_elr               : derived Cape Cod ELR (same for all rows)
            cc_expected_ultimate : premium × cc_elr
            paid_to_date         : latest observed cumulative paid loss
            current_period       : months of development at evaluation date
            cdf_to_ultimate      : CDF from current period to ultimate (from CL)
            pct_unreported       : 1 − 1/CDF (expected unreported fraction, %)
            cc_ultimate          : Cape Cod projected ultimate loss
            cc_ibnr              : Cape Cod IBNR reserve
        """
        if self.cc_elr is None:
            self.calculate_cc_elr()

        records = []
        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            paid = float(row[last_col])
            period = int(last_col.replace("dev_", ""))
            cdf = float(self.cl.cdfs[last_col])
            premium = float(self.premiums[year])

            cc_expected_ultimate = premium * self.cc_elr
            pct_unreported = 1.0 - (1.0 / cdf)
            cc_ibnr = cc_expected_ultimate * pct_unreported
            cc_ultimate = paid + cc_ibnr

            records.append(
                {
                    "accident_year": year,
                    "premium": premium,
                    "cc_elr": round(self.cc_elr, 6),
                    "cc_expected_ultimate": round(cc_expected_ultimate),
                    "paid_to_date": paid,
                    "current_period": period,
                    "cdf_to_ultimate": round(cdf, 4),
                    "pct_unreported": round(pct_unreported * 100, 2),
                    "cc_ultimate": round(cc_ultimate),
                    "cc_ibnr": round(cc_ibnr),
                }
            )

        self.cc_summary = pd.DataFrame(records).set_index("accident_year")
        return self.cc_summary

    # ------------------------------------------------------------------
    # Step 4: Comparison table (CC vs CL)
    # ------------------------------------------------------------------

    def compare(self) -> pd.DataFrame:
        """
        Build a side-by-side comparison of Cape Cod and Chain Ladder results.

        Returns
        -------
        pd.DataFrame
            One row per accident year with columns:

            paid_to_date         : latest observed cumulative paid
            current_period       : months of development
            premium              : earned premium
            cc_expected_ultimate : premium × cc_elr (derived a priori)
            cl_ultimate          : Chain Ladder projected ultimate
            cl_ibnr              : Chain Ladder IBNR
            cc_ultimate          : Cape Cod projected ultimate
            cc_ibnr              : Cape Cod IBNR
            diff_ibnr            : cc_ibnr − cl_ibnr (method divergence)
        """
        if self.cc_summary is None:
            self.calculate_cc_ultimate()

        cl_cols = self.cl.summary[["ultimate", "ibnr"]].rename(
            columns={"ultimate": "cl_ultimate", "ibnr": "cl_ibnr"}
        )
        cc_cols = self.cc_summary[
            ["paid_to_date", "current_period", "premium",
             "cc_expected_ultimate", "cc_ultimate", "cc_ibnr"]
        ]

        combined = cc_cols.join(cl_cols)
        combined["diff_ibnr"] = combined["cc_ibnr"] - combined["cl_ibnr"]
        combined = combined[
            ["paid_to_date", "current_period", "premium", "cc_expected_ultimate",
             "cl_ultimate", "cl_ibnr", "cc_ultimate", "cc_ibnr", "diff_ibnr"]
        ]

        self.comparison = combined
        return self.comparison

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full Cape Cod workflow and return the comparison table.

        Steps
        -----
        1. _run_chain_ladder()     — fit CL to obtain CDFs
        2. calculate_cc_elr()      — derive ELR from used-up premium
        3. calculate_cc_ultimate() — apply CC formula
        4. compare()               — join CC and CL results

        Returns
        -------
        pd.DataFrame
            Side-by-side CC vs CL comparison (same as compare() output).
        """
        self._run_chain_ladder()
        self.calculate_cc_elr()
        self.calculate_cc_ultimate()
        return self.compare()


# ------------------------------------------------------------------
# Pretty-print helper
# ------------------------------------------------------------------

def print_results(model: CapeCod) -> None:
    """Print the CC vs Chain Ladder comparison table and aggregate totals."""
    df = model.comparison

    currency_cols = [
        "paid_to_date", "premium", "cc_expected_ultimate",
        "cl_ultimate", "cl_ibnr", "cc_ultimate", "cc_ibnr", "diff_ibnr",
    ]

    print("=" * 100)
    print(f"  CAPE COD vs CHAIN LADDER   |   Derived CC ELR: {model.cc_elr:.1%}")
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
        f"  CC Ultimate: ${totals['cc_ultimate']:>12,.0f}"
        f"  CC IBNR: ${totals['cc_ibnr']:>12,.0f}"
        f"  Diff: ${totals['diff_ibnr']:>10,.0f}"
    )
    print("=" * 100)

    print()
    print("  KEY METRICS")
    print("-" * 50)
    total_premium = totals["premium"]
    print(f"  Derived CC ELR       : {model.cc_elr:.1%}")
    print(f"  Total Paid-to-Date   : ${totals['paid_to_date']:>12,.0f}")
    print(f"  CL Total IBNR        : ${totals['cl_ibnr']:>12,.0f}"
          f"   ({totals['cl_ibnr']/total_premium:.1%} of premium)")
    print(f"  CC Total IBNR        : ${totals['cc_ibnr']:>12,.0f}"
          f"   ({totals['cc_ibnr']/total_premium:.1%} of premium)")
    print(f"  CC vs CL Difference  : ${totals['diff_ibnr']:>12,.0f}")
    print("=" * 50)


if __name__ == "__main__":
    from data_loader import load_triangle

    data_path = Path(__file__).parent.parent / "data" / "claims_triangle.csv"
    triangle = load_triangle(data_path)

    premiums = {
        2018: 13_500_000,
        2019: 14_200_000,
        2020: 12_800_000,
        2021: 15_000_000,
        2022: 15_800_000,
        2023: 16_500_000,
    }

    model = CapeCod(triangle, premiums)
    model.run()
    print_results(model)
