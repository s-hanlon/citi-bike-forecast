# Citi Bike Demand Forecasting

A time-series machine-learning project that predicts hourly Citi Bike pickup demand for high-traffic New York City stations.

The project builds a reproducible pipeline from public trip-history data through feature engineering, expanding-window backtesting, archived weather forecasts, holiday features, and model diagnostics.

## Project Goals

* Convert raw Citi Bike trips into hourly station-level demand
* Explore demand patterns by time, station, borough, and neighborhood
* Establish honest seasonal forecasting baselines
* Engineer leakage-safe time-series features
* Measure the value of weather information
* Evaluate weather using forecasts available 24 hours in advance
* Add calendar features for holidays and holiday-adjacent demand
* Prepare the model for an interactive forecasting application

## Current Status

The project currently includes:

* Citi Bike data ingestion from January 2025 through May 2026
* A complete hourly dataset for 25 high-demand stations
* Station selection using data before May 2026
* Station metadata and neighborhood enrichment
* Exploratory data analysis
* Seasonal baseline forecasting
* Leakage-safe lag and rolling features
* US federal holiday features
* Gradient-boosting demand forecasting
* Seasonal expanding-window backtesting
* Observed-weather analysis
* Archived 24-hour-ahead weather forecasts
* Final May 2026 benchmark evaluation
* Final v1 model diagnostics

A completed v1 model is preserved in Git under the tag:

```text
v1.0-weather-forecast
```

The active v2 branch expands the training history, adds holiday features, and evaluates performance across multiple seasons.

## Data

### Citi Bike Trips

Raw trip-history data comes from the public [Citi Bike system data archive](https://citibikenyc.com/system-data).

Individual trips are aggregated into:

```text
timestamp | station_id | station_name | pickups
```

The current processed pickup dataset contains:

```text
516 days x 24 hours x 25 stations = 309,600 rows
```

It covers:

```text
2025-01-01 00:00:00 through 2026-05-31 23:00:00
```

The 25 stations are selected using demand before May 1, 2026. This prevents the May benchmark period from influencing which stations enter the dataset.

### Weather

Weather data comes from Open-Meteo.

Two weather sources are maintained:

* Observed historical weather for exploratory analysis
* Archived forecasts issued 24 hours before each valid timestamp for modeling

Using archived day-ahead forecasts prevents the model from receiving actual future weather that would not be known during deployment.

The current weather dataset contains:

```text
516 days x 24 hours = 12,384 hourly weather records
```

Weather variables include:

* Temperature
* Apparent temperature
* Relative humidity
* Precipitation
* Wind speed
* Weather condition code
* Precipitation indicator

Raw ZIP, JSON, and generated Parquet files are excluded from Git because they can be regenerated using the project scripts.

## Feature Engineering

The model uses:

* Station identity
* Hour of day
* Day of week
* Month
* Weekend status
* Cyclical hour and day-of-week encodings
* Demand 24, 48, 168, and 336 hours earlier
* Rolling 168-hour demand mean and standard deviation
* US federal holiday indicators
* Day before and day after holiday indicators
* Holiday-window indicator
* Days to nearest holiday
* Day-ahead weather forecasts

The current model feature table contains:

```text
301,200 rows
30 columns
```

Lagged and rolling features use only prior pickup observations. Rolling windows are shifted before calculation to prevent target leakage.

## Models

### Seasonal Baselines

Two seasonal baselines are evaluated:

* Same hour yesterday
* Same hour last week

These provide simple benchmarks that a machine-learning model must outperform.

### Gradient Boosting

The primary model is a global `HistGradientBoostingRegressor` trained across all selected stations using Poisson loss for nonnegative count data.

Two configurations are compared:

* Gradient boosting with calendar, lag, rolling, and station features
* Gradient boosting with calendar, lag, rolling, station, and archived 24-hour weather forecast features

Both configurations use identical training periods, model parameters, and evaluation periods. The weather features are the only difference.

## Evaluation Strategy

### Seasonal Expanding-Window Backtesting

The models are trained on all information available before each validation week and tested on the following seven days.

Validation weeks begin on:

```text
2025-07-11
2025-08-08
2025-09-12
2025-10-10
2025-11-07
2025-12-05
2026-01-09
2026-02-06
2026-03-06
2026-04-03
2026-04-10
2026-04-17
```

This evaluates the model across summer, fall, winter, and spring rather than only one narrow time period.

### May 2026 Benchmark

The project also evaluates a May 25-31, 2026 benchmark period.

This period includes Memorial Day, which makes it useful for testing whether holiday features help the model handle calendar-driven demand changes.

The original v1 model treated May 25-31 as an untouched holdout. In v2, this period is best understood as a consistent benchmark because it has now been inspected during iterative model development.

## Results

### Seasonal Backtesting

| Model                                | Mean MAE | Std MAE | Mean RMSE | Std RMSE |
| ------------------------------------ | -------: | ------: | --------: | -------: |
| Gradient boosting + 24-hour forecast |     4.29 |    1.02 |      6.83 |     1.80 |
| Gradient boosting                    |     4.78 |    1.21 |      7.66 |     2.06 |
| Same hour last week                  |     5.96 |    1.58 |      9.48 |     2.56 |
| Same hour yesterday                  |     6.39 |    1.65 |     10.32 |     2.80 |

Across 12 seasonal validation weeks, the weather-enhanced model improved mean MAE by approximately:

```text
10.3% vs gradient boosting without weather
28.0% vs same-hour-last-week baseline
32.9% vs same-hour-yesterday baseline
```

The weather-enhanced model beat the non-weather gradient-boosting model in 10 of 12 validation weeks. The two weeks where it underperformed were small losses of about 1-2%.

### Fold-by-Fold Weather Impact

| Validation Week | Weather Model MAE | No-Weather Model MAE | Weather Improvement |
| --------------- | ----------------: | -------------------: | ------------------: |
| 2025-07-11      |              5.64 |                 6.09 |               7.40% |
| 2025-08-08      |              5.32 |                 5.26 |              -1.16% |
| 2025-09-12      |              5.33 |                 5.86 |               9.02% |
| 2025-10-10      |              4.91 |                 6.47 |              24.08% |
| 2025-11-07      |              4.66 |                 4.81 |               3.15% |
| 2025-12-05      |              3.48 |                 3.43 |              -1.53% |
| 2026-01-09      |              3.41 |                 3.57 |               4.58% |
| 2026-02-06      |              1.99 |                 2.23 |              10.52% |
| 2026-03-06      |              4.51 |                 5.03 |              10.44% |
| 2026-04-03      |              4.01 |                 4.58 |              12.48% |
| 2026-04-10      |              3.97 |                 4.84 |              18.03% |
| 2026-04-17      |              4.24 |                 5.18 |              18.05% |

### May 2026 Benchmark

| Model                                |  MAE |  RMSE |
| ------------------------------------ | ---: | ----: |
| Gradient boosting + 24-hour forecast | 4.28 |  6.66 |
| Gradient boosting                    | 4.62 |  7.40 |
| Same hour yesterday                  | 6.70 | 11.18 |
| Same hour last week                  | 7.99 | 12.67 |

The v2 model improves substantially over the original v1 benchmark:

| Model Version                             |  MAE | RMSE |
| ----------------------------------------- | ---: | ---: |
| v1 weather model                          | 4.85 | 7.39 |
| v2 long-history + holiday + weather model | 4.28 | 6.66 |

This is an approximate improvement of:

```text
11.8% MAE reduction
9.9% RMSE reduction
```

## Key Findings

* Citi Bike pickup demand follows strong daily and weekly cycles.
* Weekdays exhibit commute-like morning and evening peaks.
* Weekend demand rises later and is more evenly distributed.
* Daily and weekly lag features are strong predictive signals.
* Longer history improves seasonal generalization.
* Holiday features help the model handle calendar anomalies such as Memorial Day.
* Heavy precipitation can suppress demand well below normal seasonal patterns.
* Observed weather is useful for analysis but inappropriate as future model input.
* Archived day-ahead forecasts improve performance without assuming perfect knowledge of future weather.
* Weather improves the model in most seasonal validation weeks, although its impact varies by week.
* The model is strongest at predicting overall demand and weaker at predicting sharp station-level spikes during peak periods.

## Project Structure

```text
citi-bike-forecast/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baseline.ipynb
|   |-- 03_ml_model.ipynb
|   |-- 04_weather.ipynb
|   `-- 05_error_analysis.ipynb
|-- src/
|   |-- data/
|   |   |-- download_data.py
|   |   |-- download_weather.py
|   |   |-- make_dataset.py
|   |   |-- make_station_metadata.py
|   |   `-- make_weather_dataset.py
|   |-- features/
|   |   `-- build_features.py
|   `-- models/
|       |-- baseline.py
|       `-- gradient_boosting.py
|-- requirements.txt
`-- README.md
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

The notebooks contain exploratory analysis, visualizations, baseline development, model interpretation, weather analysis, and final error analysis.

## Next Steps

Planned improvements include:

* Add station availability filtering
* Add NYC major-event indicators
* Explore sports, concerts, street events, and large public gatherings
* Tune hyperparameters using time-based validation
* Add forecast uncertainty or precipitation probability
* Predict station dropoffs and net flow
* Incorporate live Citi Bike station status
* Build an interactive Streamlit dashboard
* Estimate station shortage and overflow risk

```
```
