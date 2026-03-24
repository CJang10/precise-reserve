"""
tests/test_api.py

Integration tests for api/main.py.

Covers
------
- All 400 error paths (invalid file type, empty file, missing columns, etc.)
- Successful /upload end-to-end with the sample triangle
- /health, /sample, /export endpoints
- New v1.1 response fields: warnings, recommended_method, commentary, key_risk_flag
"""

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make both engine/ and api/ importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))
sys.path.insert(0, str(_ROOT / "api"))

from api.main import app  # noqa: E402

client = TestClient(app)

_SAMPLE_CSV = _ROOT / "data" / "claims_triangle.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload_csv(content: str | bytes, filename: str = "triangle.csv") -> dict:
    """POST /upload with an in-memory CSV."""
    if isinstance(content, str):
        content = content.encode()
    response = client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(content), "text/csv")},
    )
    return response


def _upload_sample() -> dict:
    """Upload the bundled sample CSV."""
    with open(_SAMPLE_CSV, "rb") as f:
        return client.post(
            "/upload",
            files={"file": ("claims_triangle.csv", f, "text/csv")},
        )


# ---------------------------------------------------------------------------
# Health / Sample / Export
# ---------------------------------------------------------------------------

class TestUtilityEndpoints:

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_sample_returns_csv(self):
        resp = client.get("/sample")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_export_returns_xlsx(self):
        resp = client.get("/export")
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /upload — 400 error paths
# ---------------------------------------------------------------------------

class TestUploadErrors:

    def test_wrong_file_type(self):
        # Use application/pdf MIME type to trigger the type check
        response = client.post(
            "/upload",
            files={"file": ("triangle.pdf", io.BytesIO(b"fake content"), "application/pdf")},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["error"] == "InvalidFileType"

    def test_empty_file(self):
        resp = _upload_csv(b"")
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "EmptyFile"

    def test_missing_accident_year_column(self):
        csv = "year,dev_12,dev_24,dev_36,dev_48,dev_60,dev_72\n2020,100,200,300,400,450,480\n"
        resp = _upload_csv(csv)
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] in (
            "MissingIndexColumn", "InvalidColumns", "InvalidTriangle", "InsufficientData"
        )

    def test_missing_dev_columns(self):
        csv = "accident_year,dev_12,dev_24\n2020,100000,150000\n2021,110000,160000\n2022,120000,\n"
        resp = _upload_csv(csv)
        assert resp.status_code == 400

    def test_non_numeric_values(self):
        csv = (
            "accident_year,dev_12,dev_24,dev_36,dev_48,dev_60,dev_72\n"
            "2020,abc,200000,300000,400000,450000,480000\n"
            "2021,110000,160000,240000,320000,,\n"
            "2022,120000,180000,,,, \n"
        )
        resp = _upload_csv(csv)
        assert resp.status_code == 400

    def test_fewer_than_three_accident_years(self):
        csv = (
            "accident_year,dev_12,dev_24,dev_36,dev_48,dev_60,dev_72\n"
            "2021,100000,150000,180000,,, \n"
            "2022,110000,165000,,,,\n"
        )
        resp = _upload_csv(csv)
        assert resp.status_code == 400

    def test_invalid_elr_query_param(self):
        with open(_SAMPLE_CSV, "rb") as f:
            resp = client.post(
                "/upload?elr=5.0",
                files={"file": ("claims_triangle.csv", f, "text/csv")},
            )
        # ELR > 2 triggers validation error — either 400 or 422
        assert resp.status_code in (400, 422)

    def test_invalid_tail_factor(self):
        with open(_SAMPLE_CSV, "rb") as f:
            resp = client.post(
                "/upload?tail_factor=0.5",
                files={"file": ("claims_triangle.csv", f, "text/csv")},
            )
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /upload — successful response
# ---------------------------------------------------------------------------

class TestUploadSuccess:

    @pytest.fixture(autouse=True)
    def setup(self):
        resp = _upload_sample()
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}\n{resp.text}"
        self.body = resp.json()

    # ── Response structure ────────────────────────────────────────────────────
    def test_has_results(self):
        assert "results" in self.body
        assert len(self.body["results"]) == 6  # 6 accident years

    def test_has_totals(self):
        assert "totals" in self.body

    def test_has_assumptions(self):
        assert "assumptions" in self.body

    def test_has_warnings(self):
        assert "warnings" in self.body
        assert isinstance(self.body["warnings"], list)

    def test_has_recommended_method(self):
        assert "recommended_method" in self.body
        assert self.body["recommended_method"] in ("CL", "BF", "CC")

    def test_has_recommendation_rationale(self):
        assert "recommendation_rationale" in self.body
        assert isinstance(self.body["recommendation_rationale"], str)
        assert len(self.body["recommendation_rationale"]) > 0

    def test_has_key_risk_flag(self):
        assert "key_risk_flag" in self.body
        assert isinstance(self.body["key_risk_flag"], bool)

    def test_commentary_field_present(self):
        assert "commentary" in self.body
        # commentary may be None if ANTHROPIC_API_KEY is not set — that's acceptable

    # ── Numerical invariants ──────────────────────────────────────────────────
    def test_ibnr_nonnegative(self):
        for r in self.body["results"]:
            assert r["cl_ibnr"] >= 0
            assert r["bf_ibnr"] >= 0
            assert r["cc_ibnr"] >= 0

    def test_ultimate_geq_paid(self):
        for r in self.body["results"]:
            assert r["cl_ultimate"] >= r["paid_to_date"]
            assert r["bf_ultimate"] >= r["paid_to_date"]
            assert r["cc_ultimate"] >= r["paid_to_date"]

    def test_total_cl_ibnr_equals_sum(self):
        total = sum(r["cl_ibnr"] for r in self.body["results"])
        assert self.body["totals"]["cl_ibnr"] == pytest.approx(total, abs=1)

    def test_total_bf_ibnr_equals_sum(self):
        total = sum(r["bf_ibnr"] for r in self.body["results"])
        assert self.body["totals"]["bf_ibnr"] == pytest.approx(total, abs=1)

    def test_total_cc_ibnr_equals_sum(self):
        total = sum(r["cc_ibnr"] for r in self.body["results"])
        assert self.body["totals"]["cc_ibnr"] == pytest.approx(total, abs=1)

    def test_most_mature_year_zero_ibnr(self):
        ay2018 = next(r for r in self.body["results"] if r["accident_year"] == 2018)
        assert ay2018["cl_ibnr"] == pytest.approx(0, abs=1)

    def test_loss_ratios_present(self):
        for r in self.body["results"]:
            assert r["cl_loss_ratio"] is not None
            assert r["bf_loss_ratio"] is not None
            assert r["cc_loss_ratio"] is not None

    def test_accident_years_ascending(self):
        years = [r["accident_year"] for r in self.body["results"]]
        assert years == sorted(years)

    # ── Assumptions ───────────────────────────────────────────────────────────
    def test_default_elr_echoed(self):
        assert self.body["assumptions"]["elr"] == pytest.approx(0.65)

    def test_cc_elr_in_range(self):
        cc_elr = self.body["assumptions"]["cc_elr"]
        assert 0 < cc_elr < 2

    def test_fitted_tail_factor_geq_one(self):
        assert self.body["assumptions"]["fitted_tail_factor"] >= 1.0


# ---------------------------------------------------------------------------
# /upload — custom parameters
# ---------------------------------------------------------------------------

class TestUploadWithParams:

    def test_custom_elr(self):
        with open(_SAMPLE_CSV, "rb") as f:
            resp = client.post(
                "/upload?elr=0.75",
                files={"file": ("claims_triangle.csv", f, "text/csv")},
            )
        assert resp.status_code == 200
        assert resp.json()["assumptions"]["elr"] == pytest.approx(0.75)

    def test_custom_tail_factor(self):
        with open(_SAMPLE_CSV, "rb") as f:
            resp = client.post(
                "/upload?tail_factor=1.05",
                files={"file": ("claims_triangle.csv", f, "text/csv")},
            )
        assert resp.status_code == 200
        assert resp.json()["assumptions"]["tail_factor"] == pytest.approx(1.05)

    def test_premium_json_form_field(self):
        premiums = json.dumps({
            2018: 13_500_000, 2019: 14_200_000, 2020: 12_800_000,
            2021: 15_000_000, 2022: 15_800_000, 2023: 16_500_000,
        })
        with open(_SAMPLE_CSV, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("claims_triangle.csv", f, "text/csv")},
                data={"premiums": premiums},
            )
        assert resp.status_code == 200
