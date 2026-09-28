"""
Model evaluation utilities.

Computes classification metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
and generates standard diagnostic plots (confusion matrix, ROC curve).
Includes optimal threshold tuning to balance precision and recall.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Core metrics
# ═══════════════════════════════════════════════════════════════════
def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a fitted classifier on test data.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    # AUC requires predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    else:
        metrics["roc_auc"] = None

    return metrics


def classification_report_df(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Return sklearn's classification report as a tidy DataFrame.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).T


def find_optimal_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the classification threshold that maximizes F1 score.

    Instead of the default 0.5 cutoff, sweeps all thresholds along
    the precision-recall curve and picks the one that best balances
    precision and recall (via their harmonic mean, F1).

    Parameters
    ----------
    model : fitted estimator with ``predict_proba``
    X_test, y_test : test data

    Returns
    -------
    (best_threshold, metrics_at_best_threshold)
        The optimal threshold and a dict of all metrics evaluated at it.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Compute F1 at each threshold
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1])
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    # Evaluate at optimal threshold
    metrics = evaluate_at_threshold(model, X_test, y_test, best_threshold)

    logger.info(
        "Optimal threshold: %.4f → P=%.4f, R=%.4f, F1=%.4f",
        best_threshold, metrics["precision"], metrics["recall"], metrics["f1"],
    )
    return best_threshold, metrics


def evaluate_at_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a classifier at a custom probability threshold.

    Parameters
    ----------
    model : fitted estimator with ``predict_proba``
    X_test, y_test : test data
    threshold : float
        Probability cutoff for the positive class.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "threshold": round(threshold, 4),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot and optionally save a confusion-matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
    else:
        save_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    models_dict : dict
        ``{model_name: fitted_model}``.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            logger.warning("%s has no predict_proba — skipped", name)
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
    else:
        save_path = FIGURES_DIR / "roc_curves_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)
