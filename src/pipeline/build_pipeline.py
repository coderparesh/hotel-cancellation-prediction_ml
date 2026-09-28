"""
End-to-end sklearn Pipeline builder.

Assembles a ``ColumnTransformer`` (numeric + categorical preprocessing)
followed by the best classifier, serialises the whole pipeline, and
writes metadata JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils.config import (
    CATEGORICAL_HIGH_CARD,
    CATEGORICAL_LOW_CARD,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_low_card: Optional[List[str]] = None,
    categorical_high_card: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` that handles imputation + encoding.

    Transformers
    ------------
    num   : SimpleImputer(median) → StandardScaler
    cat_low : SimpleImputer("Missing") → OneHotEncoder
    cat_high: SimpleImputer("Missing") → OrdinalEncoder

    Returns
    -------
    ColumnTransformer
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_low_card = categorical_low_card or CATEGORICAL_LOW_CARD
    categorical_high_card = categorical_high_card or CATEGORICAL_HIGH_CARD

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_low_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    categorical_high_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat_low", categorical_low_transformer, categorical_low_card),
            ("cat_high", categorical_high_transformer, categorical_high_card),
        ],
        remainder="drop",
    )

    logger.info(
        "Built preprocessor: %d numeric, %d cat_low, %d cat_high features",
        len(numeric_features), len(categorical_low_card), len(categorical_high_card),
    )
    return preprocessor


def build_pipeline(
    classifier: Any,
    numeric_features: Optional[List[str]] = None,
    categorical_low_card: Optional[List[str]] = None,
    categorical_high_card: Optional[List[str]] = None,
) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessor + classifier.

    Parameters
    ----------
    classifier : estimator
        A fitted or unfitted sklearn-compatible classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = build_preprocessor(
        numeric_features, categorical_low_card, categorical_high_card,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    logger.info(
        "Built pipeline with classifier: %s", type(classifier).__name__,
    )
    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Serialise pipeline + metadata to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    metrics : dict
        Evaluation metrics to store in metadata.
    model_path, metadata_path : Path, optional
        Override default locations from config.
    """
    model_path = Path(model_path) if model_path else MODEL_PATH
    metadata_path = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(pipeline, model_path)
    logger.info("Pipeline saved to %s", model_path)

    # Save metadata
    classifier = pipeline.named_steps["classifier"]
    metadata = {
        "model_type": type(classifier).__name__,
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "metrics": metrics,
        "features": {
            "numeric": list(
                pipeline.named_steps["preprocessor"]
                .transformers[0][2]
            ),
            "categorical_low": list(
                pipeline.named_steps["preprocessor"]
                .transformers[1][2]
            ),
            "categorical_high": list(
                pipeline.named_steps["preprocessor"]
                .transformers[2][2]
            ),
        },
        "random_seed": RANDOM_SEED,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)
