# Citi Bike Demand Forecasting

A time-series forecasting project that predicts hourly Citi Bike pickup demand for high-traffic stations using public trip-history data.

The project currently focuses on the 25 busiest Citi Bike stations from January through April 2026. It builds an hourly station-level demand dataset, evaluates seasonal baselines, trains a gradient-boosting model, and validates performance with expanding-window backtests.

## Project Goals

- Convert raw Citi Bike trip records into hourly station-level pickup demand
- Explore demand patterns by hour, day, station, borough, and neighborhood
- Build simple seasonal forecasting baselines
- Engineer leakage-safe time-series features
- Train a machine-learning model to predict hourly pickups
- Compare model performance against honest baseline forecasts
- Prepare the project for a future dashboard and operational risk layer

## Current Status

The project currently includes:

- Data download pipeline for January through April 2026
- Hourly pickup dataset for the 25 busiest stations
- Station metadata and neighborhood enrichment
- Exploratory data analysis notebook
- Seasonal baseline forecasting notebook and script
- Gradient-boosting forecasting notebook and script
- Expanding-window backtesting
- Permutation-based feature importance

## Data

Raw trip-history data comes from public Citi Bike trip archives.

The raw ZIP files and generated Parquet files are not committed to GitHub. They are generated locally by running the project scripts.

### Processed Dataset

The main processed dataset has the structure:

```text
timestamp | station_id | station_name | pickups
```

Current processed range:

```text
2026-01-01 00:00:00 through 2026-04-30 23:00:00
```

Current dataset size:

```text
120 days × 24 hours × 25 stations = 72,000 rows
```

## Modeling Approach

### Baselines

Two seasonal baselines are evaluated:

- Same hour yesterday
- Same hour last week

The strongest baseline is the same-hour-last-week forecast.

### Machine-Learning Model

The main model is a global gradient-boosting regressor trained across all selected stations.

Features include:

- Station identity
- Hour of day
- Day of week
- Month
- Weekend flag
- Cyclical hour and day-of-week encodings
- Daily and weekly lag values
- Rolling weekly mean and standard deviation

The model uses only historical information available before the forecast period to avoid future-data leakage.

## Results

### Expanding-Window Validation

Average validation performance across three April backtest folds:

| Model | Mean MAE | Std MAE | Mean RMSE | Std RMSE |
|---|---:|---:|---:|---:|
| Gradient boosting | 5.02 | 0.24 | 8.08 | 0.52 |
| Last week baseline | 6.28 | 0.37 | 9.99 | 0.48 |
| Yesterday baseline | 6.63 | 0.56 | 10.64 | 0.88 |

Gradient boosting improved average validation MAE by approximately **20.1%** compared with the strongest baseline.

### Frozen Final Test Week

Final test period:

```text
2026-04-24 through 2026-04-30
```

| Model | MAE | RMSE |
|---|---:|---:|
| Gradient boosting | 6.14 | 10.11 |
| Last week baseline | 6.98 | 11.33 |
| Yesterday baseline | 7.87 | 12.49 |

The final test week was more difficult than the validation weeks, largely because April 25 showed unusually low demand across the system. The gradient-boosting model still outperformed both baselines.

## Key Findings

- Citi Bike pickup demand follows strong daily and weekly cycles.
- Weekday demand shows commute-like morning and evening peaks.
- Weekend demand rises later and is more evenly distributed throughout the day.
- Most selected high-demand stations are concentrated in Manhattan.
- Recent demand, weekly demand, day of week, and hour of day are the most important predictive signals.
- Weather or event data may be needed to explain sudden systemwide demand shocks.

## Project Structure

```text
citi-bike-forecast/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_ml_model.ipynb
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   ├── make_dataset.py
│   │   └── make_station_metadata.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── baseline.py
│       └── gradient_boosting.py
├── requirements.txt
└── README.md
```

## How to Reproduce

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the raw Citi Bike data:

```powershell
python src/data/download_data.py
```

Build the hourly pickup dataset:

```powershell
python src/data/make_dataset.py
```

Build the station metadata:

```powershell
python src/data/make_station_metadata.py
```

Build the model features:

```powershell
python src/features/build_features.py
```

Run the baseline evaluation:

```powershell
python src/models/baseline.py
```

Run the gradient-boosting model evaluation:

```powershell
python src/models/gradient_boosting.py
```

## Next Steps

Planned improvements include:

- Add weather features
- Add holiday and event indicators
- Tune model hyperparameters using time-based validation
- Add dropoff and net-flow forecasting
- Use live Citi Bike station-status data
- Build a Streamlit dashboard
- Add station shortage and overflow risk scoring
