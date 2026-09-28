"""
Feature selection utilities.

Provides correlation filtering, recursive feature elimination (RFE),
and tree-based importance selection to reduce dimensionality and
remove noisy or redundant features.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE

from src.utils.logger import get_logger

logger = get_logger(__name__)


def correlation_filter(
    df: pd.DataFrame,
    threshold: float = 0.90,
    target: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop one of each pair of features whose absolute Pearson
    correlation exceeds *threshold*.

    When *target* is given, the feature with the **lower** correlation
    to the target is dropped; otherwise the second column in the pair
    is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric-only features (no target).
    threshold : float
    target : str, optional
        Name of the target column for tie-breaking.

    Returns
    -------
    (pd.DataFrame, list of str)
        Filtered DataFrame and list of dropped column names.
    """
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: List[str] = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            if target and target in df.columns:
                # keep the feature with higher target correlation
                correlated_cols = upper.index[upper[col] > threshold].tolist()
                for cc in correlated_cols:
                    if abs(df[col].corr(df[target])) >= abs(df[cc].corr(df[target])):
                        to_drop.append(cc)
                    else:
                        to_drop.append(col)
            else:
                to_drop.append(col)

    to_drop = list(set(to_drop))
    df_filtered = df.drop(columns=to_drop)
    logger.info(
        "Correlation filter (threshold=%.2f): dropped %d columns → %s",
        threshold, len(to_drop), to_drop,
    )
    return df_filtered, to_drop


def rfe_select(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    n_features: int = 15,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Recursive Feature Elimination.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    estimator
        A fitted or unfitted sklearn estimator with ``coef_`` or
        ``feature_importances_``.
    n_features : int
        Number of features to keep.

    Returns
    -------
    (pd.DataFrame, list of str)
        Reduced DataFrame and list of selected feature names.
    """
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X, y)

    selected = X.columns[selector.support_].tolist()
    logger.info("RFE selected %d features: %s", len(selected), selected)

    return X[selected], selected


def importance_select(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    top_n: int = 15,
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Select the *top_n* most important features from a tree-based
    model's ``feature_importances_``.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    estimator
        A tree-based sklearn estimator (e.g. RandomForest, XGBoost).
    top_n : int
        Number of top features to retain.

    Returns
    -------
    (pd.DataFrame, list of str, pd.Series)
        Reduced DataFrame, selected feature names, and full importance Series.
    """
    estimator.fit(X, y)
    importances = pd.Series(estimator.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    selected = importances.head(top_n).index.tolist()
    logger.info("Importance selection — top %d features: %s", top_n, selected)

    return X[selected], selected, importances
