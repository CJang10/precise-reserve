"""
data_loader.py
Loads and validates a loss development triangle from a CSV file.
"""

import pandas as pd
from pathlib import Path

MIN_ACCIDENT_YEARS = 3
DEV_COLUMNS = ["dev_12", "dev_24", "dev_36", "dev_48", "dev_60", "dev_72"]
OPTIONAL_COLUMNS = ["premium"]


def load_triangle(filepath: str | Path) -> pd.DataFrame:
    """
    Read a claims triangle CSV into a DataFrame and validate its structure.

    The CSV must have:
      - An 'accident_year' column used as the index
      - Development period columns: dev_12, dev_24, dev_36, dev_48, dev_60, dev_72
      - Cumulative paid values that are non-decreasing across columns within each row
      - A lower-left triangle pattern (upper-right entries are NaN)

    Parameters
    ----------
    filepath : str or Path
        Path to the claims triangle CSV file.

    Returns
    -------
    pd.DataFrame
        Triangle with accident_year as index and dev_* columns as floats.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the triangle fails any structural validation check.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Triangle file not found: {filepath}")

    df = pd.read_csv(filepath, index_col="accident_year")
    df.index = df.index.astype(int)
    df.columns = df.columns.str.strip()

    _validate(df)
    # Return only the development columns so callers always get a clean triangle.
    return df[DEV_COLUMNS].copy()


def _validate(df: pd.DataFrame) -> None:
    """Run all structural checks on the triangle."""
    _check_columns(df)           # required columns present (before anything else)
    _check_numeric_values(df)    # no strings/nulls masquerading as data
    _check_accident_year_count(df)  # at least MIN_ACCIDENT_YEARS rows
    _check_shape(df)             # correct number of dev-period columns
    _check_triangle_pattern(df)
    _check_non_decreasing(df)
    _check_positive_values(df)


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in DEV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required development columns: {missing}. "
            f"The triangle must have all of: {DEV_COLUMNS}"
        )
    extra = [c for c in df.columns if c not in DEV_COLUMNS and c not in OPTIONAL_COLUMNS]
    if extra:
        raise ValueError(
            f"Unexpected columns found: {extra}. "
            f"Only these columns are allowed: {DEV_COLUMNS} (plus optional: {OPTIONAL_COLUMNS})"
        )


def _check_numeric_values(df: pd.DataFrame) -> None:
    """All non-empty cells in development columns must be numeric."""
    for col in [c for c in DEV_COLUMNS if c in df.columns]:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # A cell is non-numeric if it wasn't originally NaN but became NaN after coercion
        bad_mask = coerced.isna() & df[col].notna()
        if bad_mask.any():
            bad = {int(yr): str(df.at[yr, col]) for yr in df.index[bad_mask]}
            raise ValueError(
                f"Non-numeric value(s) in column '{col}': {bad}. "
                "All claim amounts must be numbers — remove commas, currency symbols, "
                "or placeholder text (e.g. 'N/A', '-') before uploading."
            )


def _check_accident_year_count(df: pd.DataFrame) -> None:
    """Triangle must have at least MIN_ACCIDENT_YEARS rows for credible LDF calculation."""
    n = len(df)
    n_cols = len(DEV_COLUMNS)
    if n < MIN_ACCIDENT_YEARS:
        raise ValueError(
            f"Triangle contains only {n} accident year(s); "
            f"at least {MIN_ACCIDENT_YEARS} are required to calculate credible "
            "loss development factors."
        )
    if n > n_cols:
        raise ValueError(
            f"Triangle has {n} accident years but only {n_cols} development periods. "
            "The number of accident years cannot exceed the number of development periods "
            "in a standard lower-left triangle."
        )


def _check_shape(df: pd.DataFrame) -> None:
    dev_cols_present = [c for c in df.columns if c in DEV_COLUMNS]
    if len(dev_cols_present) != len(DEV_COLUMNS):
        raise ValueError(
            f"Expected {len(DEV_COLUMNS)} development period columns, got {len(dev_cols_present)}. "
            f"Required columns: {DEV_COLUMNS}"
        )


def _check_triangle_pattern(df: pd.DataFrame) -> None:
    """
    Validate the lower-left triangle pattern: for row i (0-indexed),
    only the first (n_cols - i) values should be present.
    """
    n_cols = len(DEV_COLUMNS)
    dev_df = df[[c for c in DEV_COLUMNS if c in df.columns]]
    for i, (year, row) in enumerate(dev_df.iterrows()):
        expected_filled = n_cols - i
        actual_filled = row.notna().sum()
        if actual_filled != expected_filled:
            raise ValueError(
                f"Accident year {year}: expected {expected_filled} non-null values "
                f"(triangle pattern), got {actual_filled}"
            )


def _check_non_decreasing(df: pd.DataFrame) -> None:
    """Cumulative paid claims must be non-decreasing across development periods."""
    dev_df = df[[c for c in DEV_COLUMNS if c in df.columns]]
    for year, row in dev_df.iterrows():
        values = row.dropna().values
        if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
            raise ValueError(
                f"Accident year {year}: cumulative paid claims are not non-decreasing "
                f"across development periods — check for data entry errors"
            )


def _check_positive_values(df: pd.DataFrame) -> None:
    dev_df = df[[c for c in DEV_COLUMNS if c in df.columns]]
    if (dev_df.dropna(how="all") <= 0).any().any():
        raise ValueError("Triangle contains zero or negative claim values")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "claims_triangle.csv"
    triangle = load_triangle(data_path)

    print("Triangle loaded successfully.\n")
    print(f"Shape: {triangle.shape[0]} accident years x {triangle.shape[1]} development periods\n")
    pd.set_option("display.float_format", "{:,.0f}".format)
    print(triangle.to_string())
