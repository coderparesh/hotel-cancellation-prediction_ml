"""
Central configuration for the Hotel Booking Cancellation project.

All paths, constants, feature lists, and hyperparameter grids are
defined here so every module draws from a single source of truth.
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model artefacts
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Experiments
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MLRUNS_DIR = EXPERIMENTS_DIR / "mlruns"

# Raw data file
RAW_DATA_FILE = PROJECT_ROOT / "hotel_bookings.csv"

# ═══════════════════════════════════════════════════════════════════
# Create directories if they don't exist
# ═══════════════════════════════════════════════════════════════════
for _dir in [
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, FIGURES_DIR, MLRUNS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ═══════════════════════════════════════════════════════════════════
# Target variable
# ═══════════════════════════════════════════════════════════════════
TARGET = "is_canceled"

# ═══════════════════════════════════════════════════════════════════
# Columns to drop (leakage / too many nulls / not useful)
# ═══════════════════════════════════════════════════════════════════
COLUMNS_TO_DROP = [
    "reservation_status",       # directly encodes the target → leakage
    "reservation_status_date",  # derived from reservation_status → leakage
    "company",                  # 94 % null
]

# ═══════════════════════════════════════════════════════════════════
# Feature groups (post-cleaning, pre-engineering)
# ═══════════════════════════════════════════════════════════════════
NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_LOW_CARD = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

CATEGORICAL_HIGH_CARD = [
    "country",
]

ORDINAL_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# ═══════════════════════════════════════════════════════════════════
# MLflow
# ═══════════════════════════════════════════════════════════════════
MLFLOW_TRACKING_URI = str(MLRUNS_DIR)
MLFLOW_EXPERIMENT_NAME = "hotel_cancellation_prediction"

# ═══════════════════════════════════════════════════════════════════
# Class imbalance (63% not-canceled / 37% canceled → ratio ≈ 1.7)
# ═══════════════════════════════════════════════════════════════════
SCALE_POS_WEIGHT = 1.7

# ═══════════════════════════════════════════════════════════════════
# Optimal threshold (updated after threshold tuning; default 0.5)
# ═══════════════════════════════════════════════════════════════════
OPTIMAL_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════════
# Hyperparameter Grids (balanced precision-recall focus)
# ═══════════════════════════════════════════════════════════════════
XGBOOST_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [1.5, 1.7, 2.0],
    "min_child_weight": [3, 5, 7],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
}

LIGHTGBM_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [1.5, 1.7, 2.0],
    "min_child_weight": [3, 5, 7],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
}

# Use F1 as scoring (balances precision & recall equally)
# NOT accuracy (which favors the majority class)
TUNING_SCORING = "f1"

