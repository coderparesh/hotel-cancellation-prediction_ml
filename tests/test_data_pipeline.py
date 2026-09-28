"""
Tests for the data loading and cleaning pipeline.

Verifies that raw data loads correctly, cleaning removes expected
columns and nulls, and processed output meets shape/type expectations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.utils.config import RAW_DATA_FILE


# ── Fixtures ──
@pytest.fixture(scope="module")
def raw_df():
    """Load raw data once for all tests in this module."""
    if not RAW_DATA_FILE.exists():
        pytest.skip("Raw data file not found")
    return load_raw_data(copy_to_raw=False)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    """Clean data once for all tests."""
    return clean_data(raw_df, save_path=False)


# ── Tests ──
class TestLoadData:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_load_has_expected_columns(self, raw_df):
        expected = ["hotel", "is_canceled", "lead_time", "adr", "country"]
        for col in expected:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_load_not_empty(self, raw_df):
        assert len(raw_df) > 0

    def test_load_expected_shape(self, raw_df):
        assert raw_df.shape[0] > 100000  # ~119K rows
        assert raw_df.shape[1] == 32


class TestCleanData:
    def test_leakage_columns_removed(self, clean_df):
        assert "reservation_status" not in clean_df.columns
        assert "reservation_status_date" not in clean_df.columns
        assert "company" not in clean_df.columns

    def test_no_null_children(self, clean_df):
        assert clean_df["children"].isnull().sum() == 0

    def test_no_null_country(self, clean_df):
        assert clean_df["country"].isnull().sum() == 0

    def test_no_negative_adr(self, clean_df):
        assert clean_df["adr"].min() >= 0

    def test_no_zero_guest_rows(self, clean_df):
        guest_sum = clean_df[["adults", "children", "babies"]].sum(axis=1)
        assert guest_sum.min() > 0

    def test_arrival_month_numeric(self, clean_df):
        assert clean_df["arrival_date_month"].dtype in [
            np.int64, np.int32, np.float64,
        ]

    def test_output_shape_reasonable(self, clean_df):
        # Should lose very few rows (~a few hundred)
        assert len(clean_df) > 100000
        # Should have 29 columns (32 - 3 dropped)
        assert clean_df.shape[1] == 29

    def test_target_present(self, clean_df):
        assert "is_canceled" in clean_df.columns
        assert set(clean_df["is_canceled"].unique()).issubset({0, 1})
