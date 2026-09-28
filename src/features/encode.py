"""
Encoding utilities for categorical features.

Provides label, one-hot, and frequency encoding strategies
with consistent interfaces.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)


def label_encode(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Apply LabelEncoder to specified columns (in-place copy).

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    (pd.DataFrame, dict)
        Transformed DataFrame and a dict mapping column → fitted encoder.
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipped", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Label-encoded '%s'  (%d classes)", col, len(le.classes_))

    return df, encoders


def one_hot_encode(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode specified columns using ``pd.get_dummies``.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
    drop_first : bool
        Drop the first category to avoid multicollinearity.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    cols_present = [c for c in columns if c in df.columns]

    df = pd.get_dummies(df, columns=cols_present, drop_first=drop_first, dtype=int)
    logger.info(
        "One-hot encoded %d columns → %d total columns",
        len(cols_present), df.shape[1],
    )
    return df


def frequency_encode(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Replace categories with their frequency (proportion) in the training set.

    Useful for high-cardinality columns like ``country`` and ``agent``.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    (pd.DataFrame, dict)
        Transformed DataFrame and dict mapping column → frequency Series.
    """
    df = df.copy()
    freq_maps: Dict[str, pd.Series] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not in DataFrame — skipped", col)
            continue
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq).fillna(0.0)
        freq_maps[col] = freq
        logger.info("Frequency-encoded '%s'  (%d categories)", col, len(freq))

    return df, freq_maps
