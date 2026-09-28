"""
Hyperparameter tuning utilities.

Wraps GridSearchCV and RandomizedSearchCV with MLflow logging
so every search is tracked as an experiment.
"""

from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    TUNING_SCORING,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def grid_search_tune(
    model: Any,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring: str = TUNING_SCORING,
    use_mlflow: bool = True,
) -> Any:
    """
    Exhaustive grid search with cross-validation.

    Parameters
    ----------
    model : estimator
    param_grid : dict
    X, y : training data
    cv : int
    scoring : str
    use_mlflow : bool

    Returns
    -------
    GridSearchCV
        Fitted searcher (best model via ``.best_estimator_``).
    """
    logger.info("GridSearchCV on %s — %d param combos × %d folds",
                type(model).__name__, _count_combos(param_grid), cv)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X, y)

    logger.info("Best params: %s", gs.best_params_)
    logger.info("Best %s: %.4f", scoring, gs.best_score_)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"GridSearch_{type(model).__name__}"):
            mlflow.log_params(gs.best_params_)
            mlflow.log_metric(f"best_{scoring}", gs.best_score_)
            mlflow.sklearn.log_model(gs.best_estimator_, artifact_path="best_model")

    return gs


def random_search_tune(
    model: Any,
    param_distributions: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 50,
    cv: int = CV_FOLDS,
    scoring: str = TUNING_SCORING,
    use_mlflow: bool = True,
) -> Any:
    """
    Randomised search with cross-validation.

    Parameters
    ----------
    model : estimator
    param_distributions : dict
    X, y : training data
    n_iter : int
    cv : int
    scoring : str
    use_mlflow : bool

    Returns
    -------
    RandomizedSearchCV
        Fitted searcher.
    """
    logger.info("RandomizedSearchCV on %s — %d iterations × %d folds",
                type(model).__name__, n_iter, cv)

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    rs.fit(X, y)

    logger.info("Best params: %s", rs.best_params_)
    logger.info("Best %s: %.4f", scoring, rs.best_score_)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"RandomSearch_{type(model).__name__}"):
            mlflow.log_params(rs.best_params_)
            mlflow.log_metric(f"best_{scoring}", rs.best_score_)
            mlflow.sklearn.log_model(rs.best_estimator_, artifact_path="best_model")

    return rs


def _count_combos(param_grid: Dict) -> int:
    """Count total parameter combinations in a grid."""
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n
