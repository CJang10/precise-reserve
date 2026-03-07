"""
api/main.py

FastAPI backend for the precise-reserve actuarial engine.

Endpoints
---------
GET  /health   — Liveness check
POST /upload   — Accept a claims triangle CSV, run Chain Ladder + BF, return IBNR JSON

Running
-------
    # from project root:
    uvicorn api.main:app --reload

    # or directly:
    python api/main.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ------------------------------------------------------------------
# Path setup — make /engine importable from /api
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

from chain_ladder import ChainLadder                      # noqa: E402
from bornhuetter_ferguson import BornhuetterFerguson      # noqa: E402
from data_loader import load_triangle                     # noqa: E402

# ------------------------------------------------------------------
# Default premiums (small personal auto insurer, AY 2018–2023)
# Override by passing `premiums` JSON in the request form.
# ------------------------------------------------------------------
_DEFAULT_PREMIUMS: dict[int, float] = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="precise-reserve",
    description=(
        "Actuarial loss reserving API. "
        "Upload a claims development triangle and receive Chain Ladder "
        "and Bornhuetter-Ferguson IBNR estimates."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class AccidentYearResult(BaseModel):
    """Reserve estimates for a single accident year."""
    accident_year: int
    paid_to_date: float
    current_period: int
    cl_ultimate: float
    cl_ibnr: float
    bf_ultimate: float
    bf_ibnr: float
    diff_ibnr: float


class Totals(BaseModel):
    """Aggregate totals across all accident years."""
    paid_to_date: float
    cl_ultimate: float
    cl_ibnr: float
    bf_ultimate: float
    bf_ibnr: float
    diff_ibnr: float


class Assumptions(BaseModel):
    """Model assumptions echoed back in the response."""
    elr: float
    premiums: dict[int, float]

    @field_validator("elr")
    @classmethod
    def elr_must_be_positive(cls, v: float) -> float:
        if not (0.0 < v < 2.0):
            raise ValueError(f"elr must be between 0 and 2, got {v}")
        return v


class ReserveResponse(BaseModel):
    """Full IBNR response returned by POST /upload."""
    results: list[AccidentYearResult]
    totals: Totals
    assumptions: Assumptions


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_premiums(
    triangle: pd.DataFrame,
    cl: ChainLadder,
    elr: float,
    premiums_raw: str | None,
) -> dict[int, float]:
    """
    Determine which premiums to use for the BF model.

    Priority
    --------
    1. Caller-supplied JSON string {"2018": 13500000, ...}
    2. Built-in defaults when the triangle covers AY 2018–2023 exactly
    3. Implied premiums derived from CL ultimates — premium[y] = cl_ultimate[y] / ELR
       (this makes BF converge toward CL, but keeps the endpoint functional for any triangle)
    """
    if premiums_raw is not None:
        try:
            parsed = json.loads(premiums_raw)
            return {int(k): float(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Could not parse `premiums` field: {exc}. "
                    'Expected a JSON object, e.g. {"2018": 13500000, "2019": 14200000}'
                ),
            )

    if set(triangle.index.tolist()) == set(_DEFAULT_PREMIUMS):
        return _DEFAULT_PREMIUMS

    # Fallback: back-calculate implied premium from CL ultimate and ELR.
    # Documents the assumption in the response so callers know it was estimated.
    return {
        int(year): float(row["ultimate"]) / elr
        for year, row in cl.summary.iterrows()
    }


def _build_response(bf: BornhuetterFerguson, elr: float) -> ReserveResponse:
    """Serialize the BF comparison DataFrame into a ReserveResponse."""
    df = bf.comparison

    results = [
        AccidentYearResult(
            accident_year=int(year),
            paid_to_date=float(row["paid_to_date"]),
            current_period=int(row["current_period"]),
            cl_ultimate=float(row["cl_ultimate"]),
            cl_ibnr=float(row["cl_ibnr"]),
            bf_ultimate=float(row["bf_ultimate"]),
            bf_ibnr=float(row["bf_ibnr"]),
            diff_ibnr=float(row["diff_ibnr"]),
        )
        for year, row in df.iterrows()
    ]

    totals = Totals(
        paid_to_date=float(df["paid_to_date"].sum()),
        cl_ultimate=float(df["cl_ultimate"].sum()),
        cl_ibnr=float(df["cl_ibnr"].sum()),
        bf_ultimate=float(df["bf_ultimate"].sum()),
        bf_ibnr=float(df["bf_ibnr"].sum()),
        diff_ibnr=float(df["diff_ibnr"].sum()),
    )

    return ReserveResponse(
        results=results,
        totals=totals,
        assumptions=Assumptions(
            elr=elr,
            premiums={int(k): float(v) for k, v in bf.premiums.items()},
        ),
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health() -> dict:
    """Returns `{"status": "ok"}` when the server is running."""
    return {"status": "ok"}


@app.post(
    "/upload",
    response_model=ReserveResponse,
    status_code=status.HTTP_200_OK,
    summary="Run reserving models on an uploaded triangle",
)
async def upload_triangle(
    file: UploadFile = File(
        ...,
        description="Claims triangle CSV (accident_year index, dev_12…dev_72 columns).",
    ),
    elr: float = Form(
        default=0.65,
        description="A priori expected loss ratio for the BF method.",
    ),
    premiums: str | None = Form(
        default=None,
        description=(
            "Optional earned premiums as a JSON object keyed by accident year. "
            'Example: {"2018": 13500000, "2019": 14200000}. '
            "If omitted, defaults are used for AY 2018–2023; otherwise premiums "
            "are back-calculated from Chain Ladder ultimates."
        ),
    ),
) -> ReserveResponse:
    """
    Upload a claims development triangle and receive IBNR reserve estimates.

    The CSV must contain:
    - `accident_year` as the first column (used as the index)
    - Columns `dev_12`, `dev_24`, `dev_36`, `dev_48`, `dev_60`, `dev_72`
    - Cumulative paid claims in a lower-left triangle pattern (upper-right cells empty)

    Both the **Chain Ladder** (pure development) and **Bornhuetter-Ferguson**
    (credibility blend with a priori expected losses) methods are run.
    Results include per-accident-year IBNR and aggregate totals.
    """
    # ── Validate content type ────────────────────────────────────────────────
    is_csv_mime = (file.content_type or "").lower() in (
        "text/csv", "application/csv", "text/plain", "application/octet-stream",
    )
    is_csv_name = (file.filename or "").lower().endswith(".csv")
    if not (is_csv_mime or is_csv_name):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a .csv file, received content-type '{file.content_type}'.",
        )

    # ── Write upload to a temp file, then load via data_loader ───────────────
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            triangle = load_triangle(tmp_path)
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty or contains no parseable data.",
            )
        except pd.errors.ParserError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"CSV could not be parsed: {exc}. "
                    "Ensure the file uses comma separators and has a valid header row."
                ),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Missing required column: {exc}. "
                    "The CSV must include an 'accident_year' column and dev_12…dev_72."
                ),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    # ── Chain Ladder ─────────────────────────────────────────────────────────
    try:
        cl = ChainLadder(triangle)
        cl.run()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Chain Ladder failed: {exc}",
        )

    # ── Resolve premiums ─────────────────────────────────────────────────────
    resolved_premiums = _resolve_premiums(triangle, cl, elr, premiums)

    # ── Bornhuetter-Ferguson ─────────────────────────────────────────────────
    try:
        bf = BornhuetterFerguson(triangle, resolved_premiums, elr=elr)
        bf.run()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Bornhuetter-Ferguson failed: {exc}",
        )

    return _build_response(bf, elr)


# ------------------------------------------------------------------
# Dev server entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(_ROOT))
