"""
Multi-model training loop.

Trains several classifiers on the hotel cancellation dataset,
logs experiments with MLflow, and returns a results dictionary.
"""

from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import evaluate_classifier
from src.utils.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    SCALE_POS_WEIGHT,
    TEST_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Default model zoo
# ═══════════════════════════════════════════════════════════════════
def get_default_models() -> Dict[str, Any]:
    """Return a dict of name → unfitted estimator for baseline comparison."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
        ),
    }


def get_boosting_models() -> Dict[str, Any]:
    """Return XGBoost and LightGBM with regularization and class balancing."""
    models: Dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=SCALE_POS_WEIGHT,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBClassifier")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            is_unbalance=True,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed — skipping LGBMClassifier")

    return models


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Optional[Dict[str, Any]] = None,
    test_size: float = TEST_SIZE,
    use_mlflow: bool = True,
) -> Tuple[
    Dict[str, Tuple[Any, Dict[str, float]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Train multiple classifiers and evaluate each.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    models : dict, optional
        ``{name: estimator}``.  Defaults to all default + boosting models.
    test_size : float
    use_mlflow : bool
        Whether to log experiments to MLflow.

    Returns
    -------
    results : dict
        ``{name: (fitted_model, metrics_dict)}``.
    X_train, X_test, y_train, y_test
    """
    if models is None:
        models = {**get_default_models(), **get_boosting_models()}

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y,
    )
    logger.info(
        "Train/test split: train=%d, test=%d", len(X_train), len(X_test),
    )

    # MLflow setup
    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results: Dict[str, Tuple[Any, Dict[str, float]]] = {}

    # Create validation split for early stopping (15% of training data)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=RANDOM_SEED, stratify=y_train,
    )

    for name, model in models.items():
        logger.info("Training %s …", name)

        # Build fit kwargs — use eval_set for models that support early stopping
        fit_kwargs = {}
        model_type = type(model).__name__
        if model_type in ("XGBClassifier", "LGBMClassifier"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        if use_mlflow:
            with mlflow.start_run(run_name=name):
                model.fit(X_tr, y_tr, **fit_kwargs)
                metrics = evaluate_classifier(model, X_test, y_test)

                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")
        else:
            model.fit(X_tr, y_tr, **fit_kwargs)
            metrics = evaluate_classifier(model, X_test, y_test)

        results[name] = (model, metrics)
        logger.info("%s → %s", name, metrics)

    return results, X_train, X_test, y_train, y_test
