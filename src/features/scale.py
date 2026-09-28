"""
Feature scaling utilities.

Provides StandardScaler and MinMaxScaler wrappers that operate on
selected columns and return both the transformed DataFrame and the
fitted scaler for later reuse (e.g. on test data or new predictions).
"""

from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def standard_scale(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise columns to zero mean and unit variance.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    (pd.DataFrame, StandardScaler)
    """
    df = df.copy()
    cols_present = [c for c in columns if c in df.columns]

    scaler = StandardScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])

    logger.info("StandardScaler applied to %d columns", len(cols_present))
    return df, scaler


def minmax_scale(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale columns to the [0, 1] range.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    (pd.DataFrame, MinMaxScaler)
    """
    df = df.copy()
    cols_present = [c for c in columns if c in df.columns]

    scaler = MinMaxScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])

    logger.info("MinMaxScaler applied to %d columns", len(cols_present))
    return df, scaler
