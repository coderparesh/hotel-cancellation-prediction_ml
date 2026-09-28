# 🏨 Hotel Booking Cancellation Prediction

An end-to-end machine learning project that predicts hotel booking cancellations using the **Hotel Booking Demand** dataset. The project follows a 4-sprint development methodology with modular source code, interactive notebooks, a Streamlit web app, CI/CD, and Docker deployment.

---

## 📊 Problem Statement

Hotels face significant revenue loss from booking cancellations. This project builds a binary classification model to predict whether a booking will be canceled, enabling hotels to:
- Proactively manage overbooking strategies
- Identify high-risk bookings for targeted interventions
- Optimize revenue management decisions

## 📁 Project Structure

```
hotel_cancellation/
│
├── data/
│   ├── raw/                          # Original untouched data
│   ├── interim/                      # Partially cleaned data
│   ├── processed/                    # Final model-ready dataset
│   └── external/                     # Reference/lookup data
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # EDA & data inspection
│   ├── 02_preprocessing.ipynb        # Cleaning, encoding, scaling
│   ├── 03_baseline_models.ipynb      # 6-model baseline comparison
│   ├── 04_feature_engineering.ipynb   # Feature creation & selection
│   └── 05_hyperparameter_tuning.ipynb# GridSearch/RandomSearch + final model
│
├── src/
│   ├── data/                         # Data loading & cleaning
│   ├── features/                     # Encoding, scaling, engineering, selection
│   ├── models/                       # Training, evaluation, tuning, prediction
│   ├── pipeline/                     # sklearn Pipeline builder
│   └── utils/                        # Config & logging
│
├── models/                           # Serialized model + metadata
├── app/                              # Streamlit web app
├── reports/                          # EDA report, model comparison, figures
├── tests/                            # pytest test suite
├── deployment/                       # Dockerfile + prod requirements
├── experiments/                      # MLflow tracking
└── .github/workflows/                # CI/CD pipeline
```

## 🏗️ Dataset

| Property | Value |
|---|---|
| **Source** | Hotel Booking Demand |
| **Rows** | 119,390 |
| **Columns** | 32 |
| **Target** | `is_canceled` (0 = Not Canceled, 1 = Canceled) |
| **Class Balance** | 63% Not Canceled / 37% Canceled |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Using pip
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate hotel-cancellation
```

### 2. Run Notebooks (Sequential)

```bash
cd notebooks/
jupyter notebook
```

Run in order: `01 → 02 → 03 → 04 → 05`

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Launch Streamlit App

```bash
streamlit run app/app.py
```

### 5. Docker Deployment

```bash
docker build -f deployment/Dockerfile -t hotel-predictor .
docker run -p 8501:8501 hotel-predictor
```

## 🤖 Models Compared

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |
| XGBoost | Ensemble (Boosting) |
| LightGBM | Ensemble (Boosting) |

## 📈 Engineered Features

- `total_stay` — total nights (weekend + weekday)
- `total_guests` — adults + children + babies
- `adr_per_person` — price per guest per night
- `is_local` — Portuguese guest flag
- `is_same_room` — reserved vs assigned room match
- `is_family` — has children or babies
- `cancellation_ratio` — historical cancellation rate
- Cyclical month encoding (sin/cos)

## 📋 Sprint Roadmap

| Sprint | Focus | Key Deliverables |
|---|---|---|
| **1** | Data Understanding & Preprocessing | EDA notebook, cleaning pipeline, EDA report |
| **2** | Baseline Models | 6-model comparison, metrics table, ROC curves |
| **3** | Feature Engineering & Tuning | 12 new features, hyperparameter search, best model |
| **4** | Pipeline, App & Deployment | sklearn Pipeline, Streamlit app, Docker, CI/CD |

## 📄 License

MIT License — see [LICENSE](LICENSE)
