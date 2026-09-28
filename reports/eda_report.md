# EDA Report — Hotel Booking Cancellation Dataset

**Date:** 2026-08-25  
**Dataset:** hotel_bookings.csv (119,390 rows × 32 columns)

---

## 1. Dataset Overview

- **Source:** Hotel booking demand dataset (two hotels: Resort Hotel & City Hotel)
- **Period:** July 2015 – August 2017
- **Target:** `is_canceled` (binary: 0 = Not Canceled, 1 = Canceled)

## 2. Target Distribution

| Class | Count | Percentage |
|---|---|---|
| Not Canceled (0) | 75,166 | 63.0% |
| Canceled (1) | 44,224 | 37.0% |

Moderate class imbalance — stratified splitting is required.

## 3. Missing Values

| Column | Null Count | % Missing | Strategy |
|---|---|---|---|
| `company` | 112,593 | 94.3% | **Drop column** |
| `agent` | 16,340 | 13.7% | Fill with 0 (no agent) |
| `country` | 488 | 0.4% | Fill with "Unknown" |
| `children` | 4 | 0.003% | Fill with 0 |

Additionally, `agent` and `company` contain string `"NULL"` values that must be converted to actual NaN before imputation.

## 4. Data Leakage

> **CRITICAL:** `reservation_status` has values "Check-Out", "Canceled", "No-Show" — this perfectly encodes the target variable. Both `reservation_status` and `reservation_status_date` **must be dropped** before modeling.

## 5. Key Findings

### Cancellation Drivers
- **Deposit Type:** Non-refundable deposits have ~99% cancellation rate
- **Lead Time:** Higher lead time strongly correlates with cancellation
- **Hotel Type:** City Hotels (~42% cancel rate) vs Resort Hotels (~28%)
- **Market Segment:** Online TA bookings cancel more frequently
- **Customer Type:** Transient guests cancel most often

### Numerical Feature Insights
- `adr` (Average Daily Rate): range [-6.38, 5400] — negative values are invalid
- `lead_time`: range [0, 737] — heavily right-skewed
- `stays_in_week_nights`: max=50 (potential outlier)
- Most bookings have 0 `children` and 0 `babies`

### Outliers
- `adr`: extreme outlier at 5400, negative values exist
- `lead_time`: bookings up to 2 years in advance
- `adults`: max=55 (likely data entry error)

## 6. Recommendations for Preprocessing

1. Drop `reservation_status`, `reservation_status_date`, `company`
2. Remove rows with `adr < 0` and zero-guest bookings
3. Convert `arrival_date_month` from string to numeric (1–12)
4. Apply frequency encoding for high-cardinality `country` (177 unique)
5. One-hot encode low-cardinality categoricals (< 15 classes)
6. Engineer domain features: total_stay, adr_per_person, is_family, etc.
