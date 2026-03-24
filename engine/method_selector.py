"""
engine/method_selector.py

Rule-based method recommendation engine.

Recommends the most appropriate reserving method (CL, BF, or CC) based on
portfolio maturity characteristics derived from the development triangle.

Decision rules (from PRD Milestone 2.4 / Roadmap 2.4)
------------------------------------------------------
- CL  : avg CDF ≤ 1.10  — mature portfolio; data speaks for itself
- CC  : avg CDF > 1.25 AND premium data present
- BF  : avg CDF > 1.25 AND no premium data (default prior-based approach)
- BF  : avg CDF 1.10–1.25 (moderate immaturity; credibility blend is appropriate)

The recommendation cites specific data characteristics (avg CDF, immature year
count, ELR delta) so actuaries can evaluate and override it.
"""

from __future__ import annotations

import pandas as pd


def recommend_method(
    cdfs: "pd.Series",
    results: list[dict],
    has_premiums: bool,
    elr: float,
) -> tuple[str, str]:
    """
    Recommend a reserving method based on portfolio maturity.

    Parameters
    ----------
    cdfs : pd.Series
        Cumulative development factors from the fitted ChainLadder, indexed by
        column name (e.g. "dev_12", "dev_24", …).
    results : list[dict]
        Per-accident-year result dicts. Each must contain:
        current_period (int), cl_ibnr (float), bf_ibnr (float).
    has_premiums : bool
        True if explicit premium data was supplied by the caller.
    elr : float
        The a priori ELR in use (for the BF rationale string).

    Returns
    -------
    tuple[str, str]
        (recommended_method, recommendation_rationale)
        recommended_method       : "CL" | "BF" | "CC"
        recommendation_rationale : plain-English explanation
    """
    if cdfs is None or len(cdfs) == 0:
        return "BF", (
            "Could not assess maturity — CDF data unavailable. "
            "Defaulting to BF as a conservative choice."
        )

    # ── Maturity metrics ──────────────────────────────────────────────────────
    avg_cdf = float(cdfs.mean())

    n_total = len(results)
    n_immature = sum(1 for r in results if r.get("current_period", 72) < 48)
    n_mature = sum(1 for r in results if r.get("current_period", 72) >= 60)
    pct_mature = (n_mature / n_total * 100) if n_total > 0 else 0.0

    # ── Decision rules ────────────────────────────────────────────────────────
    if avg_cdf <= 1.10:
        rationale = (
            f"The average CDF to ultimate is {avg_cdf:.3f}, indicating a mature "
            f"portfolio ({n_mature} of {n_total} accident years are ≥ 60 months "
            "developed). Chain Ladder is appropriate when the triangle is credible "
            "and development patterns are stable — the data speaks for itself."
        )
        return "CL", rationale

    if avg_cdf > 1.25:
        if has_premiums:
            rationale = (
                f"The average CDF to ultimate is {avg_cdf:.3f}, indicating an "
                f"immature portfolio ({n_immature} of {n_total} accident years have "
                "fewer than 48 months of development). Premium data is available, "
                "allowing Cape Cod to derive a data-driven ELR from used-up premium. "
                "CC is preferred over BF when premiums are credible because the ELR "
                "is endogenously derived rather than externally assumed."
            )
            return "CC", rationale
        else:
            rationale = (
                f"The average CDF to ultimate is {avg_cdf:.3f}, indicating an "
                f"immature portfolio ({n_immature} of {n_total} accident years have "
                "fewer than 48 months of development). No premium data is available "
                f"for Cape Cod; using BF with the supplied ELR of {elr:.2f}. "
                "BF is the conservative default when premium data is absent."
            )
            return "BF", rationale

    # avg_cdf between 1.10 and 1.25 — moderately immature
    if has_premiums:
        rationale = (
            f"The average CDF to ultimate is {avg_cdf:.3f} — the portfolio is "
            f"moderately immature ({pct_mature:.0f}% of accident years are ≥ 60 months "
            "developed). BF provides a credibility blend between observed development "
            "and a priori expectations. Cape Cod is also viable given premium data "
            "is available — consider running both for comparison."
        )
    else:
        rationale = (
            f"The average CDF to ultimate is {avg_cdf:.3f} — the portfolio is "
            f"moderately immature. BF with ELR {elr:.2f} blends observed development "
            "with a priori expectations, appropriate when neither the triangle nor an "
            "external prior fully dominates."
        )
    return "BF", rationale
