"""
Tests for feature engineering and encoding modules.

Verifies that engineered features produce correct values and that
encoding does not introduce unexpected nulls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineer import engineer_features
from src.features.encode import label_encode, one_hot_encode, frequency_encode
from src.features.scale import standard_scale


# ── Fixtures ──
@pytest.fixture
def sample_df():
    """Create a small sample DataFrame mimicking cleaned hotel data."""
    return pd.DataFrame({
        "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
        "lead_time": [100, 50, 200],
        "arrival_date_month": [7, 12, 1],
        "stays_in_weekend_nights": [2, 0, 1],
        "stays_in_week_nights": [5, 3, 2],
        "adults": [2, 1, 2],
        "children": [1, 0, 0],
        "babies": [0, 0, 1],
        "country": ["PRT", "USA", "GBR"],
        "reserved_room_type": ["A", "B", "A"],
        "assigned_room_type": ["A", "B", "C"],
        "booking_changes": [3, 0, 1],
        "previous_cancellations": [1, 0, 2],
        "previous_bookings_not_canceled": [2, 5, 0],
        "adr": [100.0, 200.0, 150.0],
        "is_canceled": [0, 1, 0],
        "meal": ["BB", "HB", "SC"],
        "market_segment": ["Direct", "Online TA", "Corporate"],
        "distribution_channel": ["Direct", "TA/TO", "Corporate"],
        "deposit_type": ["No Deposit", "Non Refund", "No Deposit"],
        "customer_type": ["Transient", "Contract", "Transient"],
        "is_repeated_guest": [0, 1, 0],
        "arrival_date_year": [2016, 2017, 2016],
        "arrival_date_week_number": [27, 50, 1],
        "arrival_date_day_of_month": [1, 15, 28],
        "agent": [10, 0, 250],
        "days_in_waiting_list": [0, 5, 0],
        "required_car_parking_spaces": [1, 0, 0],
        "total_of_special_requests": [2, 0, 1],
    })


# ── Feature Engineering Tests ──
class TestEngineerFeatures:
    def test_total_stay(self, sample_df):
        result = engineer_features(sample_df)
        expected = sample_df["stays_in_weekend_nights"] + sample_df["stays_in_week_nights"]
        pd.testing.assert_series_equal(
            result["total_stay"], expected, check_names=False,
        )

    def test_total_guests(self, sample_df):
        result = engineer_features(sample_df)
        expected = sample_df["adults"] + sample_df["children"] + sample_df["babies"]
        pd.testing.assert_series_equal(
            result["total_guests"], expected, check_names=False,
        )

    def test_is_local(self, sample_df):
        result = engineer_features(sample_df)
        assert result["is_local"].tolist() == [1, 0, 0]

    def test_adr_per_person(self, sample_df):
        result = engineer_features(sample_df)
        # Row 0: adr=100, guests=3 → 33.33
        assert abs(result["adr_per_person"].iloc[0] - 100 / 3) < 0.01

    def test_is_same_room(self, sample_df):
        result = engineer_features(sample_df)
        assert result["is_same_room"].tolist() == [1, 1, 0]

    def test_booking_changes_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert result["booking_changes_flag"].tolist() == [1, 0, 1]

    def test_is_family(self, sample_df):
        result = engineer_features(sample_df)
        assert result["is_family"].tolist() == [1, 0, 1]

    def test_cancellation_ratio(self, sample_df):
        result = engineer_features(sample_df)
        # Row 0: prev_cancel=1, total_prev=3 → 0.333
        assert abs(result["cancellation_ratio"].iloc[0] - 1 / 3) < 0.01
        # Row 1: prev_cancel=0, total_prev=5 → 0.0
        assert result["cancellation_ratio"].iloc[1] == 0.0

    def test_no_new_nulls(self, sample_df):
        result = engineer_features(sample_df)
        new_cols = [
            "total_stay", "total_guests", "is_local", "adr_per_person",
            "is_same_room", "booking_changes_flag", "is_family",
            "total_previous", "cancellation_ratio",
        ]
        for col in new_cols:
            assert result[col].isnull().sum() == 0, f"Null in {col}"


# ── Encoding Tests ──
class TestEncoding:
    def test_label_encode_returns_integers(self, sample_df):
        result, encoders = label_encode(sample_df, ["hotel"])
        assert result["hotel"].dtype in [np.int32, np.int64]
        assert "hotel" in encoders

    def test_one_hot_encode_increases_columns(self, sample_df):
        result = one_hot_encode(sample_df, ["meal"])
        assert result.shape[1] > sample_df.shape[1]
        assert "meal" not in result.columns  # original dropped

    def test_frequency_encode_values_in_0_1(self, sample_df):
        result, freq_maps = frequency_encode(sample_df, ["country"])
        assert result["country"].min() >= 0.0
        assert result["country"].max() <= 1.0

    def test_encoding_no_new_nulls(self, sample_df):
        null_before = sample_df.isnull().sum().sum()
        result = one_hot_encode(sample_df, ["hotel", "meal"])
        null_after = result.isnull().sum().sum()
        assert null_after <= null_before


# ── Scaling Tests ──
class TestScaling:
    def test_standard_scale_zero_mean(self, sample_df):
        result, scaler = standard_scale(sample_df, ["lead_time", "adr"])
        assert abs(result["lead_time"].mean()) < 0.01
        assert abs(result["adr"].mean()) < 0.01

    def test_standard_scale_unit_variance(self, sample_df):
        result, scaler = standard_scale(sample_df, ["lead_time", "adr"])
        assert abs(result["lead_time"].std(ddof=0) - 1.0) < 0.01
