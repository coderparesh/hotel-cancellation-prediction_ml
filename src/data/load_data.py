"""
Data loading and validation utilities.

Functions for reading raw CSV data, performing basic sanity checks,
and persisting DataFrames to disk.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.config import RAW_DATA_FILE, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(
    path: Optional[Union[str, Path]] = None,
    copy_to_raw: bool = True,
) -> pd.DataFrame:
    """
    Load the raw hotel bookings CSV and run basic validation.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file.  Defaults to ``RAW_DATA_FILE`` from config.
    copy_to_raw : bool
        If True, save a copy to ``data/raw/hotel_bookings.csv`` for
        reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as-is from the CSV.
    """
    path = Path(path) if path else RAW_DATA_FILE
    logger.info("Loading raw data from %s", path)

    df = pd.read_csv(path)

    # ── Basic validation ──
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning("Columns with nulls:\n%s", cols_with_nulls.to_string())
    else:
        logger.info("No null values found.")

    duplicates = df.duplicated().sum()
    logger.info("Duplicate rows: %d", duplicates)

    # ── Persist a copy to data/raw/ ──
    if copy_to_raw:
        raw_path = RAW_DATA_DIR / "hotel_bookings.csv"
        df.to_csv(raw_path, index=False)
        logger.info("Raw data copied to %s", raw_path)

    return df


def save_data(
    df: pd.DataFrame,
    path: Union[str, Path],
    file_format: str = "csv",
) -> None:
    """
    Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    path : str or Path
        Destination file path.
    file_format : str
        ``'csv'`` or ``'parquet'``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "csv":
        df.to_csv(path, index=False)
    elif file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    logger.info("Saved %s rows to %s", len(df), path)
