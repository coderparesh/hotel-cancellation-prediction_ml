"""
Feature engineering — domain-specific transformations.

Creates new features that capture hotel-booking domain knowledge,
such as total stay duration, guest counts, price per person, and
binary indicators for families, room changes, etc.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the DataFrame.

    New columns
    -----------
    total_stay              stays_in_weekend_nights + stays_in_week_nights
    total_guests            adults + children + babies
    is_local                1 if country == 'PRT' (Portugal) else 0
    adr_per_person          adr / total_guests (0 when total_guests == 0)
    is_same_room            1 if reserved_room_type == assigned_room_type
    lead_time_category      short / medium / long / very_long
    booking_changes_flag    1 if booking_changes > 0
    is_family               1 if children > 0 or babies > 0
    arrival_month_sin       cyclical sine of month
    arrival_month_cos       cyclical cosine of month
    total_previous          previous_cancellations + previous_bookings_not_canceled
    cancellation_ratio      previous_cancellations / total_previous (0 if none)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (post ``clean_data``).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns.
    """
    df = df.copy()
    n_before = df.shape[1]

    # ── Total stay duration ──
    df["total_stay"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )

    # ── Total guests ──
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # ── Is local (Portuguese) ──
    if "country" in df.columns:
        df["is_local"] = (df["country"] == "PRT").astype(int)

    # ── ADR per person ──
    df["adr_per_person"] = np.where(
        df["total_guests"] > 0,
        df["adr"] / df["total_guests"],
        0.0,
    )

    # ── Room type match ──
    if "reserved_room_type" in df.columns and "assigned_room_type" in df.columns:
        df["is_same_room"] = (
            df["reserved_room_type"] == df["assigned_room_type"]
        ).astype(int)

    # ── Lead-time category ──
    bins = [0, 7, 30, 120, 800]
    labels = ["short", "medium", "long", "very_long"]
    df["lead_time_category"] = pd.cut(
        df["lead_time"], bins=bins, labels=labels, include_lowest=True,
    )

    # ── Booking changes flag ──
    df["booking_changes_flag"] = (df["booking_changes"] > 0).astype(int)

    # ── Family indicator ──
    df["is_family"] = (
        (df["children"] > 0) | (df["babies"] > 0)
    ).astype(int)

    # ── Cyclical month encoding ──
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"]
        df["arrival_month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * month_num / 12)

    # ── Previous booking aggregates ──
    df["total_previous"] = (
        df["previous_cancellations"]
        + df["previous_bookings_not_canceled"]
    )
    df["cancellation_ratio"] = np.where(
        df["total_previous"] > 0,
        df["previous_cancellations"] / df["total_previous"],
        0.0,
    )

    n_new = df.shape[1] - n_before
    logger.info("Engineered %d new features → %d total columns", n_new, df.shape[1])

    return df
