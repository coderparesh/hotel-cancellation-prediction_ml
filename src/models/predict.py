"""
Inference / prediction utilities.

Provides helpers to load a serialised model and make predictions
on single bookings or batch DataFrames.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODEL_METADATA_PATH, MODEL_PATH, OPTIMAL_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(
    path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Load a serialised sklearn pipeline / model from disk.

    Parameters
    ----------
    path : str or Path, optional
        Defaults to ``MODEL_PATH`` from config.

    Returns
    -------
    Fitted estimator or Pipeline.
    """
    path = Path(path) if path else MODEL_PATH
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def load_model_metadata(
    path: Optional[Union[str, Path]] = None,
) -> Dict:
    """Load the JSON metadata that accompanies the model."""
    path = Path(path) if path else MODEL_METADATA_PATH
    with open(path, "r") as f:
        meta = json.load(f)
    logger.info("Loaded metadata from %s", path)
    return meta


def predict_single(
    model: Any,
    input_dict: Dict[str, Any],
    threshold: float = OPTIMAL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict cancellation for a single booking.

    Parameters
    ----------
    model : fitted estimator / Pipeline
    input_dict : dict
        Feature-name → value mapping for one booking.
    threshold : float
        Probability cutoff for positive class (default from config).

    Returns
    -------
    dict
        ``{"prediction": 0|1, "probability": float}``.
    """
    df = pd.DataFrame([input_dict])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])
        prediction = int(probability >= threshold)
    else:
        prediction = int(model.predict(df)[0])

    result = {"prediction": prediction, "probability": probability}
    logger.info("Single prediction (threshold=%.2f): %s", threshold, result)
    return result


def predict_batch(
    model: Any,
    df: pd.DataFrame,
    threshold: float = OPTIMAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Predict cancellation for a batch of bookings.

    Appends ``predicted_cancellation`` and (optionally)
    ``cancellation_probability`` columns.

    Parameters
    ----------
    model : fitted estimator / Pipeline
    df : pd.DataFrame
    threshold : float
        Probability cutoff for positive class (default from config).

    Returns
    -------
    pd.DataFrame
        Original data plus prediction columns.
    """
    df = df.copy()
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[:, 1]
        df["cancellation_probability"] = probabilities
        df["predicted_cancellation"] = (probabilities >= threshold).astype(int)
    else:
        df["predicted_cancellation"] = model.predict(df)

    logger.info("Batch prediction (threshold=%.2f): %d rows", threshold, len(df))
    return df
