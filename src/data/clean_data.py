"""
Data cleaning pipeline.

Handles null imputation, column drops (leakage, high-null),
invalid-row removal, deduplication, and dtype corrections.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.utils.config import (
    COLUMNS_TO_DROP,
    INTERIM_DATA_DIR,
    ORDINAL_MONTH_MAP,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Full cleaning pipeline for the raw hotel bookings data.

    Steps
    -----
    1. Drop leakage / high-null columns.
    2. Replace "NULL" strings with ``NaN``.
    3. Impute remaining nulls.
    4. Remove invalid rows (zero-guest bookings, negative ADR).
    5. Convert ``arrival_date_month`` to numeric.
    6. Fix data types (``children``, ``agent`` → int).
    7. Deduplicate.
    8. Optionally save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    save_path : str or Path, optional
        If given, save the cleaned DataFrame here.  Defaults to
        ``data/interim/hotel_bookings_cleaned.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning  (input shape: %s)", df.shape)
    df = df.copy()

    # ── 1. Drop leakage / high-null columns ──
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    logger.info("Dropped columns: %s", cols_present)

    # ── 2. Replace "NULL" strings with NaN ──
    df.replace("NULL", np.nan, inplace=True)

    # ── 3. Impute nulls ──
    if "children" in df.columns:
        df["children"].fillna(0, inplace=True)

    if "country" in df.columns:
        df["country"].fillna("Unknown", inplace=True)

    if "agent" in df.columns:
        df["agent"] = pd.to_numeric(df["agent"], errors="coerce").fillna(0)

    logger.info("Null counts after imputation:\n%s",
                df.isnull().sum()[df.isnull().sum() > 0].to_string()
                or "  (none)")

    # ── 4. Remove invalid rows ──
    n_before = len(df)

    # Negative ADR
    df = df[df["adr"] >= 0]

    # Zero-guest bookings (no adults, children, or babies)
    guest_cols = ["adults", "children", "babies"]
    available = [c for c in guest_cols if c in df.columns]
    if available:
        df = df[df[available].sum(axis=1) > 0]

    n_removed = n_before - len(df)
    logger.info("Removed %d invalid rows (negative ADR / zero guests)", n_removed)

    # ── 5. Convert arrival_date_month to numeric ──
    if "arrival_date_month" in df.columns:
        df["arrival_date_month"] = (
            df["arrival_date_month"].map(ORDINAL_MONTH_MAP)
        )
        logger.info("Converted arrival_date_month to numeric (1–12)")

    # ── 6. Fix dtypes ──
    int_cast_cols = ["children", "agent"]
    for col in int_cast_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # ── 7. Deduplicate ──
    n_dups = df.duplicated().sum()
    if n_dups:
        df.drop_duplicates(inplace=True)
        logger.info("Dropped %d duplicate rows", n_dups)
    else:
        logger.info("No duplicates found")

    # ── 8. Save ──
    if save_path is not False:
        if save_path is None:
            save_path = INTERIM_DATA_DIR / "hotel_bookings_cleaned.csv"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Cleaned data saved to %s  (output shape: %s)", save_path, df.shape)

    return df
