# Citi Bike Demand Forecasting

A time-series machine-learning project that predicts hourly Citi Bike pickup demand for high-traffic New York City stations.

The project builds a reproducible pipeline from public trip-history data through feature engineering, expanding-window backtesting, archived weather forecasts, and final holdout evaluation.

## Project Goals

- Convert raw Citi Bike trips into hourly station-level demand
- Explore demand patterns by time, station, borough, and neighborhood
- Establish honest seasonal forecasting baselines
- Engineer leakage-safe time-series features
- Measure the value of weather information
- Evaluate weather using forecasts available 24 hours in advance
- Prepare the model for an interactive forecasting application

## Current Status

The project currently includes:

- Citi Bike data ingestion from January through May 2026
- A complete hourly dataset for 25 high-demand stations
- Station selection using January–April development data only
- Station metadata and neighborhood enrichment
- Exploratory data analysis
- Seasonal baseline forecasting
- Leakage-safe lag and rolling features
- Gradient-boosting demand forecasting
- Expanding-window backtesting
- Observed-weather analysis
- Archived 24-hour-ahead weather forecasts
- An untouched final test covering May 25–31

## Data

### Citi Bike Trips

Raw trip-history data comes from the public [Citi Bike system data archive](https://citibikenyc.com/system-data).

Individual trips are aggregated into:

```text
timestamp | station_id | station_name | pickups
```

The current processed dataset contains:

```text
151 days × 24 hours × 25 stations = 90,600 rows
```

It covers:

```text
2026-01-01 00:00:00 through 2026-05-31 23:00:00
```

The 25 stations are selected using demand before May 1. This prevents the May holdout from influencing which stations enter the dataset.

### Weather

Weather data comes from Open-Meteo.

Two weather sources are maintained:

- Observed historical weather for exploratory analysis
- Archived forecasts issued 24 hours before each valid timestamp for modeling

Using archived day-ahead forecasts prevents the model from receiving actual future weather that would not be known during deployment.

Weather variables include:

- Temperature
- Apparent temperature
- Relative humidity
- Precipitation
- Wind speed
- Weather condition code
- Precipitation indicator

Raw ZIP, JSON, and generated Parquet files are excluded from Git because they can be regenerated using the project scripts.

## Feature Engineering

The model uses:

- Station identity
- Hour of day
- Day of week
- Month
- Weekend status
- Cyclical hour and day-of-week encodings
- Demand 24, 48, 168, and 336 hours earlier
- Rolling 168-hour demand mean and standard deviation
- Day-ahead weather forecasts

Lagged and rolling features use only prior pickup observations. Rolling windows are shifted before calculation to prevent target leakage.

## Models

### Seasonal Baselines

Two seasonal baselines are evaluated:

- Same hour yesterday
- Same hour last week

These provide simple benchmarks that a machine-learning model must outperform.

### Gradient Boosting

The primary model is a global `HistGradientBoostingRegressor` trained across all selected stations using Poisson loss for nonnegative count data.

Two configurations are compared:

- Gradient boosting without weather
- Gradient boosting with archived 24-hour weather forecasts

Both configurations use identical training periods, model parameters, and evaluation periods. The weather features are the only difference.

## Evaluation Strategy

### Expanding-Window Backtesting

The models are trained on all information available before each validation week and tested on the following seven days.

Validation weeks begin on:

```text
2026-04-03
2026-04-10
2026-04-17
```

### Locked Final Test

After weather-feature development, a new holdout was locked before inspecting its outcomes:

```text
2026-05-25 through 2026-05-31
```

Station selection uses only January–April demand, and weather inputs are forecasts issued 24 hours earlier.

## Results

### Expanding-Window Validation

| Model | Mean MAE | Std MAE | Mean RMSE | Std RMSE |
|---|---:|---:|---:|---:|
| Gradient boosting + 24-hour forecast | 4.51 | 0.04 | 7.13 | 0.11 |
| Gradient boosting | 5.02 | 0.24 | 8.08 | 0.52 |
| Same hour last week | 6.28 | 0.37 | 9.99 | 0.48 |
| Same hour yesterday | 6.63 | 0.56 | 10.64 | 0.88 |

Adding day-ahead weather forecasts improved mean MAE by approximately **10.2%** over gradient boosting without weather and **28.2%** over the strongest seasonal baseline.

### Untouched Final Test

| Model | MAE | RMSE |
|---|---:|---:|
| Gradient boosting + 24-hour forecast | 4.85 | 7.39 |
| Gradient boosting | 5.08 | 8.06 |
| Same hour yesterday | 7.00 | 11.39 |
| Same hour last week | 8.33 | 12.95 |

On unseen data, forecasted weather improved MAE by approximately **4.5%** and RMSE by **8.3%** over gradient boosting without weather.

The weather-enhanced model reduced MAE by approximately **30.7%** compared with the strongest final-test baseline.

## Key Findings

- Citi Bike pickup demand follows strong daily and weekly cycles.
- Weekdays exhibit commute-like morning and evening peaks.
- Weekend demand rises later and is more evenly distributed.
- Daily and weekly lag features are strong predictive signals.
- Heavy precipitation can suppress demand well below normal seasonal patterns.
- Observed weather provides useful explanatory evidence but is inappropriate as future model input.
- Archived day-ahead forecasts improve performance without assuming perfect knowledge of future weather.
- The strongest seasonal baseline can change between evaluation periods.
- Weather improves the model consistently, although its impact varies between weeks.

## Project Structure

```text
citi-bike-forecast/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_ml_model.ipynb
│   └── 04_weather.ipynb
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   ├── download_weather.py
│   │   ├── make_dataset.py
│   │   ├── make_station_metadata.py
│   │   └── make_weather_dataset.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── baseline.py
│       └── gradient_boosting.py
├── requirements.txt
└── README.md
```

## How to Reproduce

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the data pipeline:

```powershell
python src/data/download_data.py
python src/data/download_weather.py
python src/data/make_dataset.py
python src/data/make_station_metadata.py
python src/data/make_weather_dataset.py
```

Build model features:

```powershell
python src/features/build_features.py
```

Run model evaluations:

```powershell
python src/models/baseline.py
python src/models/gradient_boosting.py
```

The notebooks contain exploratory analysis, visualizations, baseline development, model interpretation, and weather analysis.

## Next Steps

Planned improvements include:

- Add holiday and major-event indicators
- Tune hyperparameters using time-based validation
- Add forecast uncertainty or precipitation probability
- Predict station dropoffs and net flow
- Incorporate live Citi Bike station status
- Build an interactive Streamlit dashboard
- Estimate station shortage and overflow risk