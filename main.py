"""
main.py — precise-reserve

Entry point for the actuarial reserving engine. Loads a claims triangle,
fits the Chain Ladder and Bornhuetter-Ferguson models, prints a summary
table to the console, and saves a comparison chart to /output.

Usage
-----
    python main.py                          # uses default triangle and assumptions
    python main.py --triangle path/to.csv  # custom triangle file
    python main.py --elr 0.70              # override the a priori expected loss ratio
    python main.py --no-chart              # skip chart generation
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Path setup — add /engine and /output to the module search path so
# all imports resolve regardless of the working directory.
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "engine"))
sys.path.insert(0, str(_ROOT / "output"))

from data_loader import load_triangle                          # noqa: E402
from chain_ladder import ChainLadder                          # noqa: E402
from bornhuetter_ferguson import BornhuetterFerguson          # noqa: E402
from report_generator import (                                # noqa: E402
    build_summary,
    print_summary_table,
    plot_ibnr_comparison,
)

# ------------------------------------------------------------------
# Default assumptions
# ------------------------------------------------------------------
DEFAULT_TRIANGLE = _ROOT / "data" / "claims_triangle.csv"
DEFAULT_CHART_OUT = _ROOT / "output" / "ibnr_comparison.png"
DEFAULT_ELR = 0.65

# Earned premiums by accident year (small personal auto insurer).
# 2020 reflects reduced earned exposure from mid-year COVID refunds.
PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}

# ------------------------------------------------------------------
# Console helpers
# ------------------------------------------------------------------
_W = 70  # line width for the header banner


def _banner() -> None:
    print("=" * _W)
    print("  precise-reserve  |  Actuarial Reserving Engine")
    print("=" * _W)
    print()


def _step(n: int, total: int, msg: str) -> None:
    """Print a numbered progress step, leaving the cursor on the same line."""
    print(f"  [{n}/{total}] {msg}...", end="", flush=True)


def _ok(detail: str = "") -> None:
    """Print the success suffix for the current step."""
    suffix = f"  ({detail})" if detail else ""
    print(f"  done{suffix}")


def _abort(label: str, message: str, hint: str | None = None) -> None:
    """
    Print a structured error message and exit with a non-zero status code.

    Parameters
    ----------
    label   : short error category shown in the header (e.g. "FileNotFound")
    message : the underlying error text
    hint    : optional actionable suggestion printed on a second line
    """
    print()  # end the in-progress step line cleanly
    print()
    print(f"  ERROR  [{label}]")
    print(f"         {message}")
    if hint:
        print(f"    →  {hint}")
    print()
    sys.exit(1)


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precise-reserve",
        description="Actuarial reserving engine — Chain Ladder and Bornhuetter-Ferguson.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--triangle",
        type=Path,
        default=DEFAULT_TRIANGLE,
        metavar="PATH",
        help="Path to the claims triangle CSV file.",
    )
    parser.add_argument(
        "--elr",
        type=float,
        default=DEFAULT_ELR,
        metavar="RATIO",
        help="A priori expected loss ratio for the BF method (e.g. 0.65).",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation (print summary table only).",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run(
    triangle_path: Path,
    premiums: dict,
    elr: float,
    save_chart: bool = True,
) -> None:
    """
    Execute the full reserving pipeline with per-step error handling.

    Steps
    -----
    1. Load & validate the triangle CSV.
    2. Fit the Chain Ladder model.
    3. Fit the Bornhuetter-Ferguson model.
    4. Print the summary table and optionally save the chart.

    Errors at each step are caught, reported with a plain-English message
    and an actionable hint, and the process exits cleanly with code 1.

    Parameters
    ----------
    triangle_path : Path
        Location of the claims triangle CSV.
    premiums : dict
        Earned premium keyed by accident year integer.
    elr : float
        A priori expected loss ratio for BF.
    save_chart : bool
        If True, write ibnr_comparison.png to /output.
    """
    total_steps = 4 if save_chart else 3

    # ── Step 1: Load triangle ─────────────────────────────────────────────────
    _resolved = triangle_path.resolve()
    _display = _resolved.relative_to(_ROOT) if _resolved.is_relative_to(_ROOT) else _resolved
    _step(1, total_steps, f"Loading triangle from {_display}")
    try:
        triangle = load_triangle(triangle_path)
    except FileNotFoundError:
        _abort(
            "FileNotFound",
            f"No file found at: {triangle_path}",
            hint=(
                "Default location is data/claims_triangle.csv. "
                "Pass --triangle <path> to specify a different file."
            ),
        )
    except pd.errors.EmptyDataError:
        _abort(
            "EmptyFile",
            f"{triangle_path.name} is empty or contains no parseable data.",
            hint="Check that the file has content and is not a zero-byte placeholder.",
        )
    except pd.errors.ParserError as exc:
        _abort(
            "ParseError",
            f"CSV could not be parsed: {exc}",
            hint=(
                "Ensure the file is valid comma-separated text with a header row. "
                "Common causes: wrong delimiter, binary content, or encoding issues."
            ),
        )
    except KeyError as exc:
        _abort(
            "MissingColumn",
            f"Expected column not found in CSV: {exc}",
            hint=(
                "The triangle must have an 'accident_year' column plus "
                "dev_12, dev_24, dev_36, dev_48, dev_60, dev_72."
            ),
        )
    except ValueError as exc:
        _abort(
            "ValidationError",
            str(exc),
            hint=(
                "Check the triangle for: wrong number of rows/columns, "
                "non-decreasing cumulative values, or an incorrect NaN pattern."
            ),
        )

    rows, cols = triangle.shape
    _ok(f"{rows} accident years × {cols} development periods")

    # ── Step 2: Chain Ladder ──────────────────────────────────────────────────
    _step(2, total_steps, "Running Chain Ladder")
    try:
        cl = ChainLadder(triangle)
        cl.run()
    except ValueError as exc:
        _abort(
            "ChainLadderError",
            str(exc),
            hint="Verify that adjacent development columns share at least one overlapping row.",
        )
    except Exception as exc:
        _abort("UnexpectedError", f"{type(exc).__name__}: {exc}")

    n_ldfs = len(cl.ldfs)
    _ok(f"{n_ldfs} age-to-age factors, total CL IBNR ${cl.summary['ibnr'].sum():,.0f}")

    # ── Step 3: Bornhuetter-Ferguson ──────────────────────────────────────────
    _step(3, total_steps, f"Running Bornhuetter-Ferguson  (ELR: {elr:.0%})")
    try:
        bf = BornhuetterFerguson(triangle, premiums, elr=elr)
        bf.run()
    except ValueError as exc:
        _abort(
            "BFError",
            str(exc),
            hint=(
                "Check that PREMIUMS contains an entry for every accident year "
                "in the triangle, and that ELR is a positive value below 2.0."
            ),
        )
    except Exception as exc:
        _abort("UnexpectedError", f"{type(exc).__name__}: {exc}")

    _ok(f"total BF IBNR ${bf.bf_summary['bf_ibnr'].sum():,.0f}")

    # ── Step 4: Report ────────────────────────────────────────────────────────
    summary = build_summary(cl, bf)
    print_summary_table(summary)

    if save_chart:
        _step(total_steps, total_steps, "Saving chart")
        try:
            plot_ibnr_comparison(summary, DEFAULT_CHART_OUT)
        except PermissionError:
            _abort(
                "PermissionError",
                f"Cannot write to {DEFAULT_CHART_OUT}",
                hint="Check that the /output directory exists and is writable.",
            )
        except OSError as exc:
            _abort("OutputError", f"Failed to save chart: {exc}")
        _chart_display = DEFAULT_CHART_OUT.relative_to(_ROOT) if DEFAULT_CHART_OUT.is_relative_to(_ROOT) else DEFAULT_CHART_OUT
        _ok(str(_chart_display))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _banner()
    print(f"  Triangle : {args.triangle}")
    print(f"  ELR      : {args.elr:.0%}")
    print(f"  Chart    : {'yes' if not args.no_chart else 'no'}")
    print()

    run(
        triangle_path=args.triangle,
        premiums=PREMIUMS,
        elr=args.elr,
        save_chart=not args.no_chart,
    )

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
