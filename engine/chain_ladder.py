"""
chain_ladder.py

Deterministic Chain Ladder (Development Method) for loss reserving.

The Chain Ladder method projects a partially-observed loss development triangle
to ultimate by assuming that future development will follow the same proportional
patterns observed in historical data.

Workflow
--------
1. calculate_ldfs()       — volume-weighted age-to-age factors from the triangle
2. _calculate_cdfs()      — cumulative factors from each period to ultimate
3. project_to_ultimate()  — fill in the upper-right of the triangle
4. calculate_ibnr()       — derive IBNR as (ultimate - paid-to-date)
5. run()                  — orchestrate all four steps and return a summary
"""

import pandas as pd
import numpy as np
from pathlib import Path

DEV_PERIODS = [12, 24, 36, 48, 60, 72]
DEV_COLUMNS = [f"dev_{p}" for p in DEV_PERIODS]


class ChainLadder:
    """
    Deterministic Chain Ladder reserving model.

    Parameters
    ----------
    triangle : pd.DataFrame
        Loss development triangle produced by data_loader.load_triangle().
        Rows are accident years, columns are dev_12 … dev_72.
        Upper-right cells (future periods) must be NaN.
    tail_factor : float, optional
        Multiplier applied beyond the last observed development period (default 1.0,
        meaning the triangle is assumed fully run-off at 72 months). A value of
        1.05 implies 5% of ultimate losses are still to emerge after dev_72.
        Must be >= 1.0. Use fit_tail() to get a data-driven recommendation.

    Attributes
    ----------
    ldfs : pd.Series or None
        Age-to-age (link ratio) development factors after calculate_ldfs().
    cdfs : pd.Series or None
        Cumulative development factors to ultimate after _calculate_cdfs().
    projected_triangle : pd.DataFrame or None
        Completed triangle after project_to_ultimate().
    summary : pd.DataFrame or None
        IBNR summary table after calculate_ibnr().
    fitted_tail_factor : float or None
        Recommended tail factor from fit_tail(), populated only after that method runs.
    """

    def __init__(self, triangle: pd.DataFrame, tail_factor: float = 1.0) -> None:
        if tail_factor < 1.0:
            raise ValueError(
                f"tail_factor must be >= 1.0 (got {tail_factor}). "
                "A tail factor below 1.0 would imply negative future development."
            )
        if tail_factor > 2.0:
            raise ValueError(
                f"tail_factor of {tail_factor} exceeds 2.0 — verify this is intentional."
            )
        self.triangle = triangle.copy()
        self.tail_factor = tail_factor
        self.ldfs: pd.Series | None = None
        self.cdfs: pd.Series | None = None
        self.projected_triangle: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None
        self.fitted_tail_factor: float | None = None

    # ------------------------------------------------------------------
    # Step 1: Age-to-age development factors
    # ------------------------------------------------------------------

    def calculate_ldfs(self) -> pd.Series:
        """
        Compute volume-weighted average loss development factors (LDFs).

        For each pair of consecutive development periods (j, j+1), the LDF is
        the ratio of the column sums restricted to rows where *both* periods are
        observed:

            LDF(j → j+1) = Σ triangle[:, j+1] / Σ triangle[:, j]
                           (summed over accident years with data in both columns)

        Volume-weighting (summing values rather than averaging ratios) gives more
        credibility to larger, more fully-developed accident years and suppresses
        the distorting influence of volatile small-year ratios.

        Returns
        -------
        pd.Series
            LDFs indexed by transition label, e.g. "12→24", "24→36", …, "60→72".
        """
        ldfs = {}

        for i in range(len(DEV_COLUMNS) - 1):
            col_from = DEV_COLUMNS[i]
            col_to = DEV_COLUMNS[i + 1]

            # Restrict to rows where both columns are observed (non-NaN)
            mask = self.triangle[col_from].notna() & self.triangle[col_to].notna()
            if mask.sum() == 0:
                raise ValueError(
                    f"No overlapping data found for transition "
                    f"{DEV_PERIODS[i]}→{DEV_PERIODS[i+1]}. "
                    "Cannot compute a development factor."
                )

            numerator = self.triangle.loc[mask, col_to].sum()
            denominator = self.triangle.loc[mask, col_from].sum()
            label = f"{DEV_PERIODS[i]}→{DEV_PERIODS[i + 1]}"
            ldfs[label] = numerator / denominator

        self.ldfs = pd.Series(ldfs, name="LDF")
        return self.ldfs

    # ------------------------------------------------------------------
    # Step 2: Cumulative development factors
    # ------------------------------------------------------------------

    def _calculate_cdfs(self) -> pd.Series:
        """
        Compute cumulative development factors (CDFs) from each period to ultimate.

        A CDF answers: "By what multiple must we inflate current paid losses to
        reach the ultimate (fully-developed) value?"

        CDFs are built by multiplying LDFs from right to left:

            CDF(last period → ult) = tail factor  (assumed 1.0 here)
            CDF(j → ult)           = LDF(j → j+1) × CDF(j+1 → ult)

        The tail factor of 1.0 assumes the triangle is fully run-off at 72 months.
        For long-tailed lines, this would be replaced with a fitted tail factor.

        Returns
        -------
        pd.Series
            CDFs indexed by development period column name, e.g. "dev_12".
        """
        if self.ldfs is None:
            self.calculate_ldfs()

        ldf_values = self.ldfs.values  # ordered 12→24, 24→36, …, 60→72
        cdfs = {}

        # Rightmost column: start from the tail factor (1.0 means fully run-off at dev_72).
        cdf = self.tail_factor
        cdfs[DEV_COLUMNS[-1]] = cdf

        # Multiply backward through the LDF chain
        for i in range(len(ldf_values) - 1, -1, -1):
            cdf = cdf * ldf_values[i]
            cdfs[DEV_COLUMNS[i]] = cdf

        self.cdfs = pd.Series(cdfs, name="CDF to Ultimate")
        return self.cdfs

    # ------------------------------------------------------------------
    # Step 3: Project the triangle to ultimate
    # ------------------------------------------------------------------

    def project_to_ultimate(self) -> pd.DataFrame:
        """
        Fill in the upper-right of the triangle to produce ultimate loss estimates.

        For each accident year, the most recent diagonal value (paid-to-date) is
        stepped forward one period at a time using the age-to-age LDFs until the
        final development period is reached:

            projected[year, j+1] = projected[year, j] × LDF(j → j+1)

        Accident years that are already at the final development period (i.e. fully
        developed) pass through unchanged with IBNR = 0.

        Returns
        -------
        pd.DataFrame
            Completed triangle (same shape as input) with all NaN cells replaced
            by projected values. Original observed cells are untouched.
        """
        if self.cdfs is None:
            self._calculate_cdfs()

        projected = self.triangle.copy()

        # Iterate the *original* triangle so mutations to `projected` don't
        # affect the starting values used for each row's projection.
        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            if last_col is None:
                continue  # fully missing row — nothing to project

            last_col_pos = DEV_COLUMNS.index(last_col)

            # Step forward from the current position, one period at a time
            current_value = row[last_col]
            for j in range(last_col_pos + 1, len(DEV_COLUMNS)):
                ldf_label = f"{DEV_PERIODS[j - 1]}→{DEV_PERIODS[j]}"
                current_value = current_value * self.ldfs[ldf_label]
                projected.at[year, DEV_COLUMNS[j]] = current_value

        self.projected_triangle = projected
        return self.projected_triangle

    # ------------------------------------------------------------------
    # Step 4: IBNR reserves
    # ------------------------------------------------------------------

    def calculate_ibnr(self) -> pd.DataFrame:
        """
        Compute IBNR (Incurred But Not Reported) reserves for each accident year.

        IBNR is the amount of loss expected to emerge in the future beyond what
        has already been paid. It is the primary output of a loss reserving exercise:

            IBNR = Ultimate - Paid-to-Date

        where Ultimate = Paid-to-Date × CDF(current period → ultimate).

        A high IBNR relative to paid losses indicates an immature accident year
        (few months of development) or a long-tailed line of business.

        Returns
        -------
        pd.DataFrame
            One row per accident year, indexed by accident_year, with columns:

            paid_to_date     : latest observed cumulative paid loss
            current_period   : months of development as of the evaluation date
            cdf_to_ultimate  : cumulative factor applied to reach ultimate
            ultimate         : projected total loss at full development
            ibnr             : estimated reserve (ultimate minus paid-to-date)
            pct_developed    : paid_to_date / ultimate, a maturity measure
        """
        if self.projected_triangle is None:
            self.project_to_ultimate()

        records = []
        for year, row in self.triangle.iterrows():
            last_col = row.last_valid_index()
            paid = row[last_col]
            period = int(last_col.replace("dev_", ""))
            cdf = self.cdfs[last_col]
            # Apply tail factor: dev_72 value represents development through 72 months;
            # multiplying by tail_factor projects to true ultimate beyond the triangle.
            ultimate = self.projected_triangle.at[year, DEV_COLUMNS[-1]] * self.tail_factor
            ibnr = ultimate - paid
            pct_developed = paid / ultimate if ultimate > 0 else np.nan

            records.append(
                {
                    "accident_year": year,
                    "paid_to_date": paid,
                    "current_period": period,
                    "cdf_to_ultimate": round(cdf, 4),
                    "ultimate": round(ultimate),
                    "ibnr": round(ibnr),
                    "pct_developed": round(pct_developed * 100, 1),
                }
            )

        self.summary = pd.DataFrame(records).set_index("accident_year")
        return self.summary

    # ------------------------------------------------------------------
    # Tail factor fitting
    # ------------------------------------------------------------------

    def fit_tail(self, n_ldfs: int = 3) -> float:
        """
        Fit an exponential decay curve to the last ``n_ldfs`` LDFs and extrapolate
        a recommended tail factor beyond the last development period.

        Method
        ------
        The LDF excess above 1.0 — i.e. ``y(t) = LDF(t) - 1.0`` — is modelled as:

            y(t) = a · exp(−b · t)

        Taking logs gives a linear model that is solved with ``numpy.polyfit``:

            log y(t) = log a − b · t

        The fitted curve is then extrapolated by discretely multiplying future-period
        LDFs (at 12-month increments beyond the last observed period) until the
        incremental LDF falls below 1.0001, at which point the tail is considered
        converged.

        Parameters
        ----------
        n_ldfs : int
            Number of trailing LDFs to include in the fit (default 3).

        Returns
        -------
        float
            Recommended tail factor (≥ 1.0). Stored in ``self.fitted_tail_factor``.

        Notes
        -----
        Returns 1.0 (no tail) when:
          - Any of the n_ldfs used for fitting is ≤ 1.0 (flat/declining pattern)
          - The fitted decay rate b ≤ 0 (LDFs not decreasing)
          - Fewer than 2 LDFs are available
        """
        if self.ldfs is None:
            self.calculate_ldfs()

        ldf_values = self.ldfs.values  # e.g. [LDF12→24, LDF24→36, ..., LDF60→72]
        n = min(n_ldfs, len(ldf_values))

        if n < 2:
            self.fitted_tail_factor = 1.0
            return 1.0

        # x = "from" period of each selected transition, e.g. [36, 48, 60]
        x = np.array(DEV_PERIODS[-(n + 1):-1], dtype=float)
        y = ldf_values[-n:] - 1.0  # excess above 1.0; must be strictly positive to log

        if np.any(y <= 0):
            # Flat or sub-unity LDFs — no credible tail to fit
            self.fitted_tail_factor = 1.0
            return 1.0

        # Log-linear fit: log(y) = log(a) - b*x
        log_y = np.log(y)
        slope, intercept = np.polyfit(x, log_y, 1)
        b = -slope       # decay rate (must be positive for exponential decay)
        a = np.exp(intercept)

        if b <= 0:
            # LDFs are not decaying — extrapolation not meaningful
            self.fitted_tail_factor = 1.0
            return 1.0

        # Multiply extrapolated LDFs from (last_period + 12) until convergence
        tail = 1.0
        t = float(DEV_PERIODS[-1] + 12)
        for _ in range(200):
            ldf_t = 1.0 + a * np.exp(-b * t)
            if ldf_t < 1.0001:
                break
            tail *= ldf_t
            t += 12.0

        self.fitted_tail_factor = round(tail, 4)
        return self.fitted_tail_factor

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full Chain Ladder method end-to-end and return an IBNR summary.

        Steps
        -----
        1. calculate_ldfs()      — age-to-age volume-weighted factors
        2. _calculate_cdfs()     — cumulative factors to ultimate
        3. project_to_ultimate() — complete the development triangle
        4. calculate_ibnr()      — derive IBNR reserves

        Returns
        -------
        pd.DataFrame
            IBNR summary (same as calculate_ibnr() output).
        """
        self.calculate_ldfs()
        self._calculate_cdfs()
        self.project_to_ultimate()
        return self.calculate_ibnr()


# ------------------------------------------------------------------
# Pretty-print helpers (CLI only, excluded from coverage)
# ------------------------------------------------------------------

def _fmt_currency(series: pd.Series) -> pd.Series:  # pragma: no cover
    return series.map(lambda x: f"${x:>12,.0f}")


def print_results(model: ChainLadder) -> None:  # pragma: no cover
    """Print LDFs, CDFs, and the IBNR summary table to stdout."""
    pd.set_option("display.float_format", "{:,.0f}".format)

    print("=" * 60)
    print("  AGE-TO-AGE LOSS DEVELOPMENT FACTORS (LDFs)")
    print("=" * 60)
    for label, ldf in model.ldfs.items():
        print(f"  {label:>8}   {ldf:.4f}")

    print()
    print("=" * 60)
    print("  CUMULATIVE DEVELOPMENT FACTORS (CDFs) TO ULTIMATE")
    print("=" * 60)
    for col, cdf in model.cdfs.items():
        period = col.replace("dev_", "").rjust(3)
        print(f"  {period} months   {cdf:.4f}")

    print()
    print("=" * 60)
    print("  IBNR RESERVE SUMMARY")
    print("=" * 60)
    summary = model.summary.copy()
    fmt_cols = ["paid_to_date", "ultimate", "ibnr"]
    display = summary.copy()
    for col in fmt_cols:
        display[col] = _fmt_currency(summary[col])
    display["cdf_to_ultimate"] = summary["cdf_to_ultimate"].map("{:.4f}".format)
    display["pct_developed"] = summary["pct_developed"].map("{:.1f}%".format)
    print(display.to_string())

    total_ibnr = model.summary["ibnr"].sum()
    total_ultimate = model.summary["ultimate"].sum()
    total_paid = model.summary["paid_to_date"].sum()
    print("-" * 60)
    print(f"  Total Paid-to-Date : ${total_paid:>12,.0f}")
    print(f"  Total Ultimate     : ${total_ultimate:>12,.0f}")
    print(f"  Total IBNR         : ${total_ibnr:>12,.0f}")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    from data_loader import load_triangle

    data_path = Path(__file__).parent.parent / "data" / "claims_triangle.csv"
    triangle = load_triangle(data_path)

    model = ChainLadder(triangle)
    model.run()
    print_results(model)
