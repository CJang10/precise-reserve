"""
report_generator.py

Generates a reserve comparison report combining Chain Ladder and
Bornhuetter-Ferguson results into:

  1. A printed summary table (stdout) — accident year, ultimates, IBNR, diff
  2. A two-panel matplotlib chart saved as PNG to /output:
       - Top panel  : grouped IBNR bars by accident year (CL vs BF)
       - Bottom panel: difference bars (BF − CL IBNR) showing where methods diverge

Usage
-----
    python output/report_generator.py              # from project root
    python report_generator.py                     # from /output directory

Both invocations resolve paths relative to this file's location.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — write to file, no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Path setup: make /engine importable regardless of invocation directory
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # /output
_ROOT = _HERE.parent                             # project root
_ENGINE = _ROOT / "engine"
_DATA = _ROOT / "data"

sys.path.insert(0, str(_ENGINE))

from data_loader import load_triangle            # noqa: E402
from chain_ladder import ChainLadder             # noqa: E402
from bornhuetter_ferguson import BornhuetterFerguson  # noqa: E402

# ------------------------------------------------------------------
# Default inputs — edit here to change assumptions
# ------------------------------------------------------------------
TRIANGLE_FILE = _DATA / "claims_triangle.csv"
OUTPUT_FILE = _HERE / "ibnr_comparison.png"

PREMIUMS = {
    2018: 13_500_000,
    2019: 14_200_000,
    2020: 12_800_000,
    2021: 15_000_000,
    2022: 15_800_000,
    2023: 16_500_000,
}
ELR = 0.65
EVAL_DATE = "December 2023"

# ------------------------------------------------------------------
# Palette
# ------------------------------------------------------------------
CL_COLOR = "#2563EB"   # blue
BF_COLOR = "#D97706"   # amber
DIFF_POS = "#16A34A"   # green  (BF > CL)
DIFF_NEG = "#DC2626"   # red    (BF < CL)
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#111827"


# ------------------------------------------------------------------
# Step 1: Fit models
# ------------------------------------------------------------------

def build_models() -> tuple[ChainLadder, BornhuetterFerguson]:
    """
    Load the triangle and fit both reserving models.

    Returns
    -------
    tuple of (ChainLadder, BornhuetterFerguson)
        Both models fully run and ready to query.
    """
    triangle = load_triangle(TRIANGLE_FILE)

    cl = ChainLadder(triangle)
    cl.run()

    bf = BornhuetterFerguson(triangle, PREMIUMS, elr=ELR)
    bf.run()

    return cl, bf


# ------------------------------------------------------------------
# Step 2: Unified summary DataFrame
# ------------------------------------------------------------------

def build_summary(cl: ChainLadder, bf: BornhuetterFerguson) -> pd.DataFrame:
    """
    Combine CL and BF results into a single comparison DataFrame.

    Columns
    -------
    paid_to_date   : latest observed cumulative paid
    cl_ultimate    : Chain Ladder projected ultimate
    cl_ibnr        : Chain Ladder IBNR
    bf_ultimate    : Bornhuetter-Ferguson projected ultimate
    bf_ibnr        : Bornhuetter-Ferguson IBNR
    ult_diff       : bf_ultimate − cl_ultimate
    ibnr_diff      : bf_ibnr − cl_ibnr

    Returns
    -------
    pd.DataFrame indexed by accident_year.
    """
    cl_df = cl.summary[["paid_to_date", "ultimate", "ibnr"]].rename(
        columns={"ultimate": "cl_ultimate", "ibnr": "cl_ibnr"}
    )
    bf_df = bf.bf_summary[["bf_ultimate", "bf_ibnr"]]

    summary = cl_df.join(bf_df)
    summary["ult_diff"] = summary["bf_ultimate"] - summary["cl_ultimate"]
    summary["ibnr_diff"] = summary["bf_ibnr"] - summary["cl_ibnr"]
    return summary


# ------------------------------------------------------------------
# Step 3: Printed summary table
# ------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a dollar amount compactly: $X.XXM or $XXXk."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:6.3f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:6.1f}K"
    return f"${value:8.0f}"


def print_summary_table(summary: pd.DataFrame) -> None:
    """
    Print a formatted reserve comparison table to stdout.

    Shows accident year, paid-to-date, CL ultimate, BF ultimate,
    ultimate difference, and both IBNR estimates with their difference.
    """
    col_w = 14
    head_fmt = f"{'Acc Year':>9}  {'Paid to Date':>{col_w}}  {'CL Ultimate':>{col_w}}  " \
               f"{'BF Ultimate':>{col_w}}  {'Ult Diff':>{col_w}}  " \
               f"{'CL IBNR':>{col_w}}  {'BF IBNR':>{col_w}}  {'IBNR Diff':>{col_w}}"
    sep = "-" * len(head_fmt)

    print()
    print("=" * len(head_fmt))
    print(f"  RESERVE COMPARISON REPORT  |  Eval: {EVAL_DATE}  |  A Priori ELR: {ELR:.0%}")
    print("=" * len(head_fmt))
    print(head_fmt)
    print(sep)

    for year, row in summary.iterrows():
        diff_tag = f"(+{_fmt(row.ibnr_diff)})" if row.ibnr_diff >= 0 else f"({_fmt(row.ibnr_diff)})"
        print(
            f"{year:>9}  "
            f"{_fmt(row.paid_to_date):>{col_w}}  "
            f"{_fmt(row.cl_ultimate):>{col_w}}  "
            f"{_fmt(row.bf_ultimate):>{col_w}}  "
            f"{_fmt(row.ult_diff):>{col_w}}  "
            f"{_fmt(row.cl_ibnr):>{col_w}}  "
            f"{_fmt(row.bf_ibnr):>{col_w}}  "
            f"{diff_tag:>{col_w}}"
        )

    print(sep)

    total_paid = summary["paid_to_date"].sum()
    total_cl_ult = summary["cl_ultimate"].sum()
    total_bf_ult = summary["bf_ultimate"].sum()
    total_cl_ibnr = summary["cl_ibnr"].sum()
    total_bf_ibnr = summary["bf_ibnr"].sum()
    total_ult_diff = total_bf_ult - total_cl_ult
    total_ibnr_diff = total_bf_ibnr - total_cl_ibnr

    print(
        f"{'TOTAL':>9}  "
        f"{_fmt(total_paid):>{col_w}}  "
        f"{_fmt(total_cl_ult):>{col_w}}  "
        f"{_fmt(total_bf_ult):>{col_w}}  "
        f"{_fmt(total_ult_diff):>{col_w}}  "
        f"{_fmt(total_cl_ibnr):>{col_w}}  "
        f"{_fmt(total_bf_ibnr):>{col_w}}  "
        f"{_fmt(total_ibnr_diff):>{col_w}}"
    )
    print("=" * len(head_fmt))
    print()


# ------------------------------------------------------------------
# Step 4: Chart
# ------------------------------------------------------------------

def _bar_label(ax: plt.Axes, bars, fmt_fn) -> None:
    """Annotate each bar with a dollar-formatted value label."""
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt_fn(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT_COLOR,
            fontweight="medium",
        )


def plot_ibnr_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Generate and save a two-panel chart comparing CL and BF reserves.

    Top panel
    ---------
    Grouped bar chart: CL IBNR (blue) and BF IBNR (amber) side by side for
    each accident year. Labels show the dollar amount on each bar.
    A text box in the lower-right shows aggregate totals for quick comparison.

    Bottom panel
    ------------
    Single bar chart showing BF − CL IBNR difference per accident year.
    Green bars indicate BF is more conservative; red bars indicate BF is lower.
    This highlights that the methods diverge most in immature accident years
    (where the a priori ELR has the most influence in BF).

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_summary() with cl_ibnr, bf_ibnr, and ibnr_diff columns.
    output_path : Path
        Destination path for the saved PNG file.
    """
    years = summary.index.astype(str).tolist()
    x = np.arange(len(years))
    bar_w = 0.35

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="white",
    )
    fig.subplots_adjust(hspace=0.45)

    # ── Top panel: grouped IBNR bars ─────────────────────────────────────────
    ax_main.set_facecolor("white")
    ax_main.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_main.set_axisbelow(True)

    bars_cl = ax_main.bar(
        x - bar_w / 2, summary["cl_ibnr"], bar_w,
        label="Chain Ladder", color=CL_COLOR, alpha=0.9, zorder=3,
    )
    bars_bf = ax_main.bar(
        x + bar_w / 2, summary["bf_ibnr"], bar_w,
        label="Bornhuetter-Ferguson", color=BF_COLOR, alpha=0.9, zorder=3,
    )

    _bar_label(ax_main, bars_cl, _fmt)
    _bar_label(ax_main, bars_bf, _fmt)

    ax_main.set_title(
        "IBNR Reserve Estimates by Accident Year\nChain Ladder vs Bornhuetter-Ferguson",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14,
    )
    ax_main.set_xlabel("Accident Year", fontsize=10, color=TEXT_COLOR, labelpad=8)
    ax_main.set_ylabel("IBNR Reserve ($)", fontsize=10, color=TEXT_COLOR, labelpad=8)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(years, fontsize=10)
    ax_main.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}K")
    )
    ax_main.tick_params(axis="both", colors=TEXT_COLOR, labelsize=9)

    # Extra headroom so bar labels aren't clipped at the top
    ax_main.set_ylim(0, max(summary["cl_ibnr"].max(), summary["bf_ibnr"].max()) * 1.18)

    # Remove spines except bottom
    for spine in ["top", "right", "left"]:
        ax_main.spines[spine].set_visible(False)
    ax_main.spines["bottom"].set_color(GRID_COLOR)

    # Legend
    ax_main.legend(
        handles=[
            mpatches.Patch(color=CL_COLOR, alpha=0.9, label="Chain Ladder"),
            mpatches.Patch(color=BF_COLOR, alpha=0.9, label="Bornhuetter-Ferguson"),
        ],
        loc="upper left", frameon=True, framealpha=0.9,
        edgecolor=GRID_COLOR, fontsize=9,
    )

    # Aggregate totals text box
    total_cl = summary["cl_ibnr"].sum()
    total_bf = summary["bf_ibnr"].sum()
    total_diff = total_bf - total_cl
    sign = "+" if total_diff >= 0 else ""
    totals_text = (
        f"Total IBNR\n"
        f"  Chain Ladder : {_fmt(total_cl)}\n"
        f"  BF           : {_fmt(total_bf)}\n"
        f"  Difference   : {sign}{_fmt(total_diff)}"
    )
    ax_main.text(
        0.98, 0.97, totals_text,
        transform=ax_main.transAxes,
        ha="right", va="top",
        fontsize=8.5, color=TEXT_COLOR,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=GRID_COLOR, linewidth=1),
    )

    # Evaluation date subtitle
    ax_main.text(
        0.5, -0.12,
        f"Evaluation Date: {EVAL_DATE}   |   A Priori ELR: {ELR:.0%}",
        transform=ax_main.transAxes,
        ha="center", va="top", fontsize=8.5,
        color="#6B7280", style="italic",
    )

    # ── Bottom panel: BF − CL difference bars ────────────────────────────────
    ax_diff.set_facecolor("white")
    ax_diff.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_diff.set_axisbelow(True)

    diffs = summary["ibnr_diff"].values
    diff_colors = [DIFF_POS if d >= 0 else DIFF_NEG for d in diffs]

    diff_bars = ax_diff.bar(x, diffs, bar_w * 1.5, color=diff_colors, alpha=0.85, zorder=3)

    # Annotate difference bars
    for bar, val in zip(diff_bars, diffs):
        if abs(val) < 1:
            continue
        sign = "+" if val >= 0 else ""
        ax_diff.annotate(
            f"{sign}{_fmt(val)}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4 if val >= 0 else -12),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color=TEXT_COLOR, fontweight="medium",
        )

    ax_diff.axhline(0, color="#9CA3AF", linewidth=0.8, zorder=2)
    ax_diff.set_title(
        "BF − Chain Ladder IBNR Difference",
        fontsize=10, fontweight="semibold", color=TEXT_COLOR, pad=8,
    )
    ax_diff.set_xlabel("Accident Year", fontsize=9, color=TEXT_COLOR)
    ax_diff.set_ylabel("Difference ($)", fontsize=9, color=TEXT_COLOR, labelpad=6)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(years, fontsize=9)
    ax_diff.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v / 1e3:.0f}K")
    )
    ax_diff.tick_params(axis="both", colors=TEXT_COLOR, labelsize=8.5)

    for spine in ["top", "right", "left"]:
        ax_diff.spines[spine].set_visible(False)
    ax_diff.spines["bottom"].set_color(GRID_COLOR)

    # Legend for diff panel
    ax_diff.legend(
        handles=[
            mpatches.Patch(color=DIFF_POS, alpha=0.85, label="BF > CL (BF more conservative)"),
            mpatches.Patch(color=DIFF_NEG, alpha=0.85, label="BF < CL (BF more optimistic)"),
        ],
        loc="upper left", frameon=True, framealpha=0.9,
        edgecolor=GRID_COLOR, fontsize=8,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart saved → {output_path}")


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def generate_report() -> pd.DataFrame:
    """
    Run the full report pipeline end-to-end.

    Steps
    -----
    1. Fit Chain Ladder and BF models from the claims triangle.
    2. Build a unified comparison DataFrame.
    3. Print a formatted summary table to stdout.
    4. Generate and save the IBNR comparison chart as a PNG.

    Returns
    -------
    pd.DataFrame
        The unified comparison summary (build_summary output).
    """
    print("  Loading triangle and fitting models...")
    cl, bf = build_models()

    summary = build_summary(cl, bf)

    print_summary_table(summary)

    print("  Generating chart...")
    plot_ibnr_comparison(summary, OUTPUT_FILE)

    return summary


if __name__ == "__main__":
    generate_report()
