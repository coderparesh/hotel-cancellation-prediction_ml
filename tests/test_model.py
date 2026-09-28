"""
Tests for model training, evaluation, and prediction modules.

Uses a small synthetic dataset to verify that models train, predict,
and return metrics in the expected format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import evaluate_classifier, classification_report_df
from src.models.predict import predict_single, predict_batch
from src.utils.config import RANDOM_SEED


# ── Fixtures ──
@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a small synthetic binary classification dataset."""
    np.random.seed(RANDOM_SEED)
    n = 500
    X = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "feature_c": np.random.randint(0, 5, n),
    })
    y = pd.Series((X["feature_a"] + X["feature_b"] > 0).astype(int), name="target")
    return X, y


@pytest.fixture(scope="module")
def trained_model(synthetic_data):
    """Fit a simple RandomForest on synthetic data."""
    X, y = synthetic_data
    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    model.fit(X, y)
    return model


# ── Tests ──
class TestEvaluateClassifier:
    def test_returns_dict(self, trained_model, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_classifier(trained_model, X, y)
        assert isinstance(metrics, dict)

    def test_contains_expected_keys(self, trained_model, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_classifier(trained_model, X, y)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, trained_model, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_classifier(trained_model, X, y)
        for key, val in metrics.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_accuracy_above_baseline(self, trained_model, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_classifier(trained_model, X, y)
        # Random baseline ~0.5 for balanced data
        assert metrics["accuracy"] > 0.60


class TestClassificationReport:
    def test_returns_dataframe(self, trained_model, synthetic_data):
        X, y = synthetic_data
        report = classification_report_df(trained_model, X, y)
        assert isinstance(report, pd.DataFrame)
        assert "precision" in report.columns


class TestPrediction:
    def test_predict_single_returns_dict(self, trained_model):
        input_dict = {"feature_a": 0.5, "feature_b": 1.0, "feature_c": 2}
        result = predict_single(trained_model, input_dict)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert result["prediction"] in [0, 1]

    def test_predict_single_has_probability(self, trained_model):
        input_dict = {"feature_a": 0.5, "feature_b": 1.0, "feature_c": 2}
        result = predict_single(trained_model, input_dict)
        assert result["probability"] is not None
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_batch_returns_correct_shape(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        result = predict_batch(trained_model, X.head(10))
        assert len(result) == 10
        assert "predicted_cancellation" in result.columns

    def test_predict_batch_has_probability_column(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        result = predict_batch(trained_model, X.head(10))
        assert "cancellation_probability" in result.columns
