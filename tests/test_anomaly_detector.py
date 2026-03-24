"""
tests/test_anomaly_detector.py

Unit tests for engine/anomaly_detector.py.

Tests each of the five anomaly flags in isolation using synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from anomaly_detector import detect_anomalies
from chain_ladder import ChainLadder, DEV_COLUMNS

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "claims_triangle.csv"

PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}


def _load_sample_and_run():
    from data_loader import load_triangle
    triangle = load_triangle(_SAMPLE_CSV)
    cl = ChainLadder(triangle)
    cl.run()
    return triangle, cl


def _make_results(triangle, cl, bf_ibnr_override=None):
    """Build a minimal results list from the CL summary."""
    results = []
    for year, row in cl.summary.iterrows():
        cl_ibnr = float(row["ibnr"])
        bf_ibnr = bf_ibnr_override.get(year, cl_ibnr) if bf_ibnr_override else cl_ibnr
        results.append({
            "accident_year": int(year),
            "paid_to_date": float(row["paid_to_date"]),
            "current_period": int(row["current_period"]),
            "cl_ibnr": cl_ibnr,
            "bf_ibnr": bf_ibnr,
        })
    return results


# ---------------------------------------------------------------------------
# No anomalies on clean data
# ---------------------------------------------------------------------------

def test_clean_triangle_no_warnings():
    triangle, cl = _load_sample_and_run()
    results = _make_results(triangle, cl)
    warnings = detect_anomalies(triangle, cl.ldfs, results, tail_factor=1.0)
    # The sample data may have BF-CL divergence for immature years — filter that
    non_divergence = [w for w in warnings if w["type"] != "bf_cl_divergence"]
    assert non_divergence == [], f"Unexpected warnings: {non_divergence}"


# ---------------------------------------------------------------------------
# Flag 1: LDF < 1.0
# ---------------------------------------------------------------------------

def test_ldf_below_unity_flagged():
    triangle, cl = _load_sample_and_run()
    results = _make_results(triangle, cl)

    # Manually inject a sub-unity LDF
    bad_ldfs = cl.ldfs.copy()
    bad_ldfs["24→36"] = 0.95

    warnings = detect_anomalies(triangle, bad_ldfs, results, tail_factor=1.0)
    types = [w["type"] for w in warnings]
    assert "ldf_below_unity" in types

    ldf_warn = next(w for w in warnings if w["type"] == "ldf_below_unity")
    assert ldf_warn["severity"] == "error"
    assert "24→36" in ldf_warn["message"]


# ---------------------------------------------------------------------------
# Flag 2: Concentration risk
# ---------------------------------------------------------------------------

def test_concentration_risk_flagged():
    triangle, cl = _load_sample_and_run()

    # Inflate one year's paid to dominate > 40%
    results = _make_results(triangle, cl)
    total = sum(r["paid_to_date"] for r in results)
    results[0]["paid_to_date"] = total * 10  # 2018 now > 40%

    warnings = detect_anomalies(triangle, cl.ldfs, results, tail_factor=1.0)
    types = [w["type"] for w in warnings]
    assert "concentration_risk" in types

    warn = next(w for w in warnings if w["type"] == "concentration_risk")
    assert 2018 in warn["affected_years"]


# ---------------------------------------------------------------------------
# Flag 3: Insufficient development periods
# ---------------------------------------------------------------------------

def test_insufficient_periods_flagged():
    # Only 1 data point per period (just 2 AYs → most periods have 0 or 1 rows)
    data = {
        "dev_12": [100_000, 110_000, np.nan, np.nan, np.nan, np.nan],
        "dev_24": [150_000, np.nan, np.nan, np.nan, np.nan, np.nan],
        "dev_36": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "dev_48": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "dev_60": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "dev_72": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    triangle = pd.DataFrame(data, index=[2018, 2019, 2020, 2021, 2022, 2023])
    triangle.index.name = "accident_year"

    # Only dev_12 has ≥ 2 data points; dev_24 has exactly 1 → insufficient
    cl_stub = type("CL", (), {"ldfs": pd.Series({"12→24": 1.5, "24→36": 1.2, "36→48": 1.05, "48→60": 1.02, "60→72": 1.01})})()

    results = [
        {"accident_year": 2018, "paid_to_date": 150_000, "current_period": 24, "cl_ibnr": 10_000, "bf_ibnr": 10_000},
        {"accident_year": 2019, "paid_to_date": 110_000, "current_period": 12, "cl_ibnr": 60_000, "bf_ibnr": 60_000},
    ]

    warnings = detect_anomalies(triangle, cl_stub.ldfs, results, tail_factor=1.0)
    types = [w["type"] for w in warnings]
    assert "insufficient_periods" in types


# ---------------------------------------------------------------------------
# Flag 4: Tail double-count
# ---------------------------------------------------------------------------

def test_tail_double_count_flagged():
    triangle, cl = _load_sample_and_run()

    # Force the 60→72 LDF > 1.10 for this test
    high_ldfs = cl.ldfs.copy()
    high_ldfs["60→72"] = 1.15

    results = _make_results(triangle, cl)
    warnings = detect_anomalies(triangle, high_ldfs, results, tail_factor=1.20)
    types = [w["type"] for w in warnings]
    assert "tail_double_count" in types


def test_tail_double_count_not_flagged_low_tail():
    triangle, cl = _load_sample_and_run()
    results = _make_results(triangle, cl)
    warnings = detect_anomalies(triangle, cl.ldfs, results, tail_factor=1.05)
    types = [w["type"] for w in warnings]
    assert "tail_double_count" not in types


# ---------------------------------------------------------------------------
# Flag 5: BF-CL divergence
# ---------------------------------------------------------------------------

def test_bf_cl_divergence_flagged():
    triangle, cl = _load_sample_and_run()

    # Override 2023 BF IBNR to be 30% higher than CL IBNR (triggers flag)
    cl_2023 = cl.summary.at[2023, "ibnr"]
    overrides = {2023: cl_2023 * 1.35}

    results = _make_results(triangle, cl, bf_ibnr_override=overrides)
    warnings = detect_anomalies(triangle, cl.ldfs, results, tail_factor=1.0)
    types = [w["type"] for w in warnings]
    assert "bf_cl_divergence" in types

    warn = next(w for w in warnings if w["type"] == "bf_cl_divergence")
    assert 2023 in warn["affected_years"]


def test_bf_cl_no_divergence_when_equal():
    triangle, cl = _load_sample_and_run()
    # When BF == CL, no divergence flag
    results = _make_results(triangle, cl)  # bf_ibnr_override = None → bf == cl
    warnings = detect_anomalies(triangle, cl.ldfs, results, tail_factor=1.0)
    types = [w["type"] for w in warnings]
    assert "bf_cl_divergence" not in types


# ---------------------------------------------------------------------------
# Warning structure
# ---------------------------------------------------------------------------

def test_warning_has_required_keys():
    triangle, cl = _load_sample_and_run()
    high_ldfs = cl.ldfs.copy()
    high_ldfs["60→72"] = 1.15

    results = _make_results(triangle, cl)
    warnings = detect_anomalies(triangle, high_ldfs, results, tail_factor=1.20)

    for w in warnings:
        assert "type" in w
        assert "severity" in w
        assert "affected_years" in w
        assert "message" in w
        assert w["severity"] in ("warn", "error")
        assert isinstance(w["affected_years"], list)
