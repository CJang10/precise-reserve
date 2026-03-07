"""
data_loader.py
Loads and validates a loss development triangle from a CSV file.
"""

import pandas as pd
from pathlib import Path

EXPECTED_ACCIDENT_YEARS = 6
DEV_COLUMNS = ["dev_12", "dev_24", "dev_36", "dev_48", "dev_60", "dev_72"]


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
    return df


def _validate(df: pd.DataFrame) -> None:
    """Run all structural checks on the triangle."""
    _check_columns(df)
    _check_shape(df)
    _check_triangle_pattern(df)
    _check_non_decreasing(df)
    _check_positive_values(df)


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in DEV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected development columns: {missing}")
    extra = [c for c in df.columns if c not in DEV_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected columns found: {extra}")


def _check_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    if rows != EXPECTED_ACCIDENT_YEARS:
        raise ValueError(
            f"Expected {EXPECTED_ACCIDENT_YEARS} accident years, got {rows}"
        )
    if cols != len(DEV_COLUMNS):
        raise ValueError(
            f"Expected {len(DEV_COLUMNS)} development columns, got {cols}"
        )


def _check_triangle_pattern(df: pd.DataFrame) -> None:
    """
    Validate the lower-left triangle pattern: for row i (0-indexed),
    only the first (n_cols - i) values should be present.
    """
    n_cols = len(DEV_COLUMNS)
    for i, (year, row) in enumerate(df.iterrows()):
        expected_filled = n_cols - i
        actual_filled = row.notna().sum()
        if actual_filled != expected_filled:
            raise ValueError(
                f"Accident year {year}: expected {expected_filled} non-null values "
                f"(triangle pattern), got {actual_filled}"
            )


def _check_non_decreasing(df: pd.DataFrame) -> None:
    """Cumulative paid claims must be non-decreasing across development periods."""
    for year, row in df.iterrows():
        values = row.dropna().values
        if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
            raise ValueError(
                f"Accident year {year}: cumulative paid claims are not non-decreasing "
                f"across development periods — check for data entry errors"
            )


def _check_positive_values(df: pd.DataFrame) -> None:
    if (df.dropna(how="all") <= 0).any().any():
        raise ValueError("Triangle contains zero or negative claim values")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "claims_triangle.csv"
    triangle = load_triangle(data_path)

    print("Triangle loaded successfully.\n")
    print(f"Shape: {triangle.shape[0]} accident years x {triangle.shape[1]} development periods\n")
    pd.set_option("display.float_format", "{:,.0f}".format)
    print(triangle.to_string())
