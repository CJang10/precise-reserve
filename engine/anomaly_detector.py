"""
engine/anomaly_detector.py

Anomaly detection for development triangles and reserve outputs.

Flags unusual patterns that may indicate data quality issues or methodological
concerns before an actuary relies on the output.

Flags produced
--------------
- ldf_below_unity      : LDF < 1.0 on any age-to-age period
- concentration_risk   : Single AY contributes > 40% of total paid losses
- insufficient_periods : Fewer than 3 complete development periods
- tail_double_count    : Tail factor > 1.15 AND final-period LDF > 1.10
- bf_cl_divergence     : BF-CL IBNR diverge > 15% on any accident year
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_DEV_COLUMNS = ["dev_12", "dev_24", "dev_36", "dev_48", "dev_60", "dev_72"]


def detect_anomalies(
    triangle: pd.DataFrame,
    ldfs: "pd.Series",
    results: list[dict[str, Any]],
    tail_factor: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Run all anomaly checks and return a list of warning objects.

    Parameters
    ----------
    triangle : pd.DataFrame
        Raw development triangle (rows = accident years, cols = dev_12..dev_72).
    ldfs : pd.Series
        Volume-weighted LDFs from the fitted ChainLadder (indexed by "12→24" etc.).
    results : list[dict]
        Per-accident-year result dicts from the /upload response. Each dict must
        contain: accident_year, cl_ibnr, bf_ibnr, paid_to_date, current_period.
    tail_factor : float
        User-specified tail factor (default 1.0).

    Returns
    -------
    list[dict]
        Each warning is a dict with keys:
        - type          : str  — machine-readable flag name
        - severity      : "warn" | "error"
        - affected_years: list[int] — accident years that triggered this flag ([] = portfolio-level)
        - message       : str  — plain-English explanation
    """
    warnings: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Flag 1: LDF < 1.0 (paid amounts decreasing between periods)
    # ------------------------------------------------------------------
    below_unity = {label: ldf for label, ldf in ldfs.items() if ldf < 1.0}
    if below_unity:
        periods_str = ", ".join(
            f"{label} ({ldf:.4f})" for label, ldf in below_unity.items()
        )
        warnings.append(
            {
                "type": "ldf_below_unity",
                "severity": "error",
                "affected_years": [],
                "message": (
                    f"One or more age-to-age LDFs are below 1.0: {periods_str}. "
                    "This implies paid losses are decreasing between development periods, "
                    "which may indicate data errors, salvage recoveries, or case reserve "
                    "reductions. Investigate before relying on Chain Ladder projections."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Flag 2: Single AY > 40% of total paid losses (concentration risk)
    # ------------------------------------------------------------------
    paid_by_year = {r["accident_year"]: r["paid_to_date"] for r in results}
    total_paid = sum(paid_by_year.values())
    if total_paid > 0:
        concentrated = [
            yr for yr, paid in paid_by_year.items() if paid / total_paid > 0.40
        ]
        if concentrated:
            pct_str = ", ".join(
                f"AY {yr} ({paid_by_year[yr] / total_paid * 100:.1f}%)"
                for yr in concentrated
            )
            warnings.append(
                {
                    "type": "concentration_risk",
                    "severity": "warn",
                    "affected_years": concentrated,
                    "message": (
                        f"High loss concentration detected: {pct_str} of total paid losses. "
                        "A single accident year dominating the triangle may distort "
                        "volume-weighted LDFs. Consider excluding or separately developing "
                        "this year."
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Flag 3: Fewer than 3 complete development periods
    # A period is "complete" if at least 2 accident years have data in it.
    # ------------------------------------------------------------------
    complete_periods = sum(
        1 for col in _DEV_COLUMNS if triangle[col].notna().sum() >= 2
    )
    if complete_periods < 3:
        warnings.append(
            {
                "type": "insufficient_periods",
                "severity": "warn",
                "affected_years": [],
                "message": (
                    f"Only {complete_periods} development period(s) have at least 2 data "
                    "points. Chain Ladder requires at least 3 credible development periods. "
                    "Results should be treated as highly preliminary."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Flag 4: Tail factor > 1.15 AND final-period LDF > 1.10 (double-counting)
    # ------------------------------------------------------------------
    final_ldf = ldfs.get("60→72", 1.0)
    if tail_factor > 1.15 and final_ldf > 1.10:
        warnings.append(
            {
                "type": "tail_double_count",
                "severity": "warn",
                "affected_years": [],
                "message": (
                    f"Potential tail double-counting: tail factor is {tail_factor:.3f} "
                    f"and the 60→72 LDF is {final_ldf:.4f}. "
                    "When the final observed development factor is already elevated, "
                    "a large additional tail factor may overcorrect. Consider whether "
                    "the tail factor was derived independently of the triangle."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Flag 5: BF-CL divergence > 15% on any accident year
    # ------------------------------------------------------------------
    divergent_years = []
    for r in results:
        cl_ibnr = r.get("cl_ibnr", 0)
        bf_ibnr = r.get("bf_ibnr", 0)
        base = max(abs(cl_ibnr), abs(bf_ibnr))
        if base > 0 and abs(bf_ibnr - cl_ibnr) / base > 0.15:
            divergent_years.append(r["accident_year"])

    if divergent_years:
        warnings.append(
            {
                "type": "bf_cl_divergence",
                "severity": "warn",
                "affected_years": divergent_years,
                "message": (
                    f"BF and Chain Ladder IBNR diverge by more than 15% for accident "
                    f"year(s): {', '.join(str(y) for y in divergent_years)}. "
                    "Large divergence typically signals immature years where the a priori "
                    "ELR materially disagrees with observed development patterns. "
                    "Review the ELR assumption and consider weighting the two methods "
                    "explicitly."
                ),
            }
        )

    return warnings
