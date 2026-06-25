# CitiFlow Intelligence

A machine-learning and data-engineering project for forecasting Citi Bike station demand, pickup and return flows, and live station availability risk across New York City.

The project started as an hourly pickup-demand forecasting model for high-traffic Citi Bike stations. It has now expanded into a broader **bike-share availability intelligence system** that combines historical trip data, weather forecasts, holiday features, live GBFS station inventory, and station-flow models.

## Project Overview

Bike-share systems are not only demand-forecasting problems. They are also availability problems.

A station can have high demand, but if it has zero bikes, no additional pickups can occur. A station can also receive many returns, but if it has no open docks, riders cannot end trips there. This means observed trip history only shows completed trips, not all latent demand.

This project models three related operational questions:

```text
How many pickups will happen at this station?
How many returns will happen at this station?
Will this station have enough bikes and docks in the near future?
```

The current system includes:

* Historical station-level pickup demand forecasting
* Weather-aware demand modeling using archived 24-hour-ahead forecasts
* Holiday and calendar feature engineering
* Live Citi Bike GBFS station availability integration
* Pickup, return, and net-flow forecasting
* Prototype availability-risk projection
* Interactive Streamlit dashboard

## Current Status

The active branch is:

```text
v3-availability-intelligence
```

A completed v1 model is preserved under the Git tag:

```text
v1.0-weather-forecast
```

The project currently supports:

* Citi Bike trip ingestion from January 2025 through May 2026
* Hourly station-level pickup dataset for 25 selected high-demand stations
* Station metadata generation
* Observed historical weather ingestion
* Archived 24-hour-ahead weather forecast ingestion
* Weather-safe model features
* Federal holiday features
* Expanding-window seasonal backtesting
* Gradient-boosting demand forecasting
* Live GBFS station status fetching
* Live station availability network dashboard
* GBFS station status snapshot collection
* Hourly availability dataset generation
* Pickup, return, and net-flow dataset generation
* Flow feature engineering
* Pickup, return, and net-flow model evaluation
* Final flow model training scripts
* Prototype station availability projection

## Why Availability Matters

Trip history data records completed rides. It does not directly show riders who wanted a bike but could not get one because a station was empty.

That creates a key modeling issue:

```text
Low pickups could mean low demand.
Low pickups could also mean no bikes were available.
```

Live station inventory helps separate these two cases. By combining current station inventory with predicted pickup and return pressure, the project can estimate whether a station is trending toward empty or full.

The core projection logic is:

```text
projected_bikes = current_bikes + predicted_returns - predicted_pickups
projected_docks = current_docks + predicted_pickups - predicted_returns
```

This creates a path from demand forecasting toward operational station-risk forecasting.

## Data Sources

### Citi Bike Trip History

Raw trip-history data comes from the public Citi Bike system data archive:

```text
https://citibikenyc.com/system-data
```

Trips are aggregated into hourly station-level pickup records:

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

The 25 stations are selected using data before May 1, 2026, which prevents the May benchmark period from influencing station selection.

### Weather Data

Weather data comes from Open-Meteo.

The project maintains two weather sources:

* Observed historical weather for exploratory analysis
* Archived forecasts issued 24 hours before each valid timestamp for modeling

The model uses archived day-ahead forecasts rather than actual future weather. This avoids giving the model information that would not be available in a real deployment.

The current hourly weather dataset contains:

```text
516 days x 24 hours = 12,384 weather records
```

Weather variables include:

* Temperature
* Apparent temperature
* Relative humidity
* Precipitation
* Wind speed
* Weather condition code
* Precipitation indicator

### Live Citi Bike GBFS Data

Citi Bike publishes live station information and station status through GBFS feeds.

The v3 branch fetches live station inventory, including:

* Bikes available
* E-bikes available
* Open docks
* Station capacity
* Renting status
* Returning status
* Installed status
* Last reported timestamp

This data powers the live station availability dashboard and the prototype availability projection.

## Feature Engineering

### Pickup Demand Features

The pickup demand model uses:

* Station identity
* Hour of day
* Day of week
* Month
* Weekend status
* Cyclical hour encoding
* Cyclical day-of-week encoding
* Pickup demand 24, 48, 168, and 336 hours earlier
* Rolling 168-hour pickup mean
* Rolling 168-hour pickup standard deviation
* US federal holiday indicators
* Day-before-holiday and day-after-holiday indicators
* Holiday-window indicator
* Days to nearest holiday
* Day-ahead weather forecast features

The pickup model feature table contains:

```text
301,200 rows
30 columns
```

Lagged and rolling features use only prior observations. Rolling windows are shifted before calculation to avoid target leakage.

### Flow Features

The v3 branch adds station-flow modeling. For each selected station and hour, the flow dataset includes:

```text
pickups
returns
net_outflow = pickups - returns
net_inflow = returns - pickups
```

The flow feature table contains:

```text
301,200 rows
45 columns
0 missing values
```

Flow features include:

* Pickup lags
* Return lags
* Net-outflow lags
* Pickup rolling statistics
* Return rolling statistics
* Net-outflow rolling statistics
* Calendar features
* Holiday features
* Weather forecast features
* Station identity

## Models

### Baselines

The project evaluates simple seasonal baselines:

* Same hour yesterday
* Same hour last week

These provide interpretable benchmarks that machine-learning models must outperform.

### Pickup Demand Model

The primary pickup-demand model is a global `HistGradientBoostingRegressor` trained across all selected stations.

The strongest configuration uses:

* Station identity
* Calendar features
* Holiday features
* Lagged pickup demand
* Rolling pickup demand
* Archived 24-hour-ahead weather forecast features
* Poisson loss for nonnegative count prediction

### Flow Models

The v3 branch trains separate models for:

* Pickups
* Returns
* Net outflow

Pickup and return models use Poisson loss because their targets are nonnegative count values.

Net outflow can be negative, so it uses squared-error loss.

These models support future availability projection:

```text
future bikes = current bikes + predicted returns - predicted pickups
```

## Evaluation Strategy

### Seasonal Expanding-Window Backtesting

Models are trained only on data available before each validation week and tested on the following seven days.

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

This evaluates performance across summer, fall, winter, and spring.

### May 2026 Benchmark

The project also evaluates a May 25-31, 2026 benchmark period.

This period includes Memorial Day, which makes it useful for testing whether holiday features help the model handle calendar-driven anomalies.

The original v1 model treated May 25-31 as an untouched holdout. In later versions, this period is better described as a consistent benchmark because it has been inspected during iterative development.

## Results

### Pickup Demand Backtesting

| Model                                | Mean MAE | Std MAE | Mean RMSE | Std RMSE |
| ------------------------------------ | -------: | ------: | --------: | -------: |
| Gradient boosting + 24-hour forecast |     4.29 |    1.02 |      6.83 |     1.80 |
| Gradient boosting                    |     4.78 |    1.21 |      7.66 |     2.06 |
| Same hour last week                  |     5.96 |    1.58 |      9.48 |     2.56 |
| Same hour yesterday                  |     6.39 |    1.65 |     10.32 |     2.80 |

Across 12 seasonal validation weeks, the weather-enhanced pickup model improved mean MAE by approximately:

```text
10.3% vs gradient boosting without weather
28.0% vs same-hour-last-week baseline
32.9% vs same-hour-yesterday baseline
```

The weather-enhanced model beat the non-weather model in 10 of 12 validation weeks.

### May 2026 Pickup Benchmark

| Model                                |  MAE |  RMSE |
| ------------------------------------ | ---: | ----: |
| Gradient boosting + 24-hour forecast | 4.28 |  6.66 |
| Gradient boosting                    | 4.62 |  7.40 |
| Same hour yesterday                  | 6.70 | 11.18 |
| Same hour last week                  | 7.99 | 12.67 |

The v2 long-history, holiday, and weather model improved over the original v1 weather model:

| Model Version                             |  MAE | RMSE |
| ----------------------------------------- | ---: | ---: |
| v1 weather model                          | 4.85 | 7.39 |
| v2 long-history + holiday + weather model | 4.28 | 6.66 |

Approximate improvement:

```text
11.8% MAE reduction
9.9% RMSE reduction
```

### Flow Model Backtesting

| Target      | Mean Model MAE | Mean Model RMSE | Mean Last-Week MAE | Mean Last-Week RMSE | Mean MAE Improvement |
| ----------- | -------------: | --------------: | -----------------: | ------------------: | -------------------: |
| Pickups     |           4.28 |            6.82 |               5.96 |                9.49 |               27.05% |
| Returns     |           4.12 |            6.56 |               5.72 |                9.07 |               26.89% |
| Net outflow |           3.90 |            5.94 |               5.15 |                7.84 |               23.68% |

### May 2026 Flow Benchmark

| Target      | Model MAE | Model RMSE | Last-Week MAE | Last-Week RMSE | MAE Improvement |
| ----------- | --------: | ---------: | ------------: | -------------: | --------------: |
| Pickups     |      4.29 |       6.71 |          7.99 |          12.67 |          46.24% |
| Returns     |      4.01 |       6.40 |          7.40 |          11.83 |          45.84% |
| Net outflow |      4.05 |       6.25 |          5.46 |           8.69 |          25.81% |

## Dashboard

The Streamlit dashboard includes:

* Executive overview
* Forecast explorer
* Live station availability panel
* Prototype availability projection
* Live availability network map
* Historical station map
* Model performance summaries
* Demand pattern charts
* Methodology and metadata

Run the dashboard with:

```powershell
streamlit run src/app/app.py
```

The app currently contains two different operating modes:

```text
Historical forecast explorer:
Uses processed historical feature rows and model predictions.

Live availability network:
Uses current GBFS station inventory from Citi Bike.
```

The prototype availability projection combines live inventory with predicted pickup and return pressure for a selected station-hour pattern.

## Current Limitation

The current app is not yet a fully live future forecasting system.

The reason is that true future prediction requires live feature rows for the next 1 to 6 hours, including:

* Recent pickup history
* Recent return history
* Current station inventory
* Calendar and holiday features
* Live weather forecasts
* Continuously collected GBFS station status snapshots

The current prototype proves the projection logic, but a future version should generate live future feature rows directly.

## Project Structure

```text
citi-bike-forecast/
|-- .streamlit/
|   `-- config.toml
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|   |-- citi_bike_demand_model.joblib
|   |-- citi_bike_pickup_flow_model.joblib
|   |-- citi_bike_return_flow_model.joblib
|   |-- citi_bike_net_outflow_model.joblib
|   `-- *.json
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baseline.ipynb
|   |-- 03_ml_model.ipynb
|   |-- 04_weather.ipynb
|   `-- 05_error_analysis.ipynb
|-- src/
|   |-- app/
|   |   `-- app.py
|   |-- data/
|   |   |-- collect_station_status.py
|   |   |-- download_data.py
|   |   |-- download_weather.py
|   |   |-- make_availability_dataset.py
|   |   |-- make_dataset.py
|   |   |-- make_station_flows.py
|   |   |-- make_station_metadata.py
|   |   `-- make_weather_dataset.py
|   |-- features/
|   |   |-- build_features.py
|   |   `-- build_flow_features.py
|   |-- live/
|   |   |-- __init__.py
|   |   `-- gbfs.py
|   `-- models/
|       |-- availability_projection.py
|       |-- baseline.py
|       |-- flow_forecasting.py
|       |-- gradient_boosting.py
|       |-- train_final_flow_models.py
|       `-- train_final_model.py
|-- requirements.txt
`-- README.md
```

Generated data files and trained model artifacts are excluded from Git because they can be regenerated from the pipeline scripts.

## Reproduction

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download source data

```powershell
python src/data/download_data.py
python src/data/download_weather.py
```

### 4. Build processed datasets

```powershell
python src/data/make_dataset.py
python src/data/make_station_metadata.py
python src/data/make_weather_dataset.py
python src/data/make_station_flows.py
```

### 5. Build model features

```powershell
python src/features/build_features.py
python src/features/build_flow_features.py
```

### 6. Evaluate models

```powershell
python src/models/baseline.py
python src/models/gradient_boosting.py
python src/models/flow_forecasting.py
```

### 7. Train final model artifacts

```powershell
python src/models/train_final_model.py
python src/models/train_final_flow_models.py
```

### 8. Run the dashboard

```powershell
streamlit run src/app/app.py
```

## Availability Data Pipeline

Collect one live GBFS station status snapshot:

```powershell
python src/data/collect_station_status.py --max-snapshots 1
```

Collect snapshots every 15 minutes until manually stopped:

```powershell
python src/data/collect_station_status.py --interval-minutes 15 --max-snapshots 0
```

Build the hourly availability dataset:

```powershell
python src/data/make_availability_dataset.py
```

The availability dataset is generated from raw GBFS station status snapshots and summarizes station inventory by station-hour.

## Key Findings

* Citi Bike pickup demand follows strong daily and weekly cycles.
* Weekdays exhibit commute-like morning and evening peaks.
* Weekend demand rises later and is more evenly distributed.
* Daily and weekly lag features are strong predictive signals.
* Longer training history improves seasonal generalization.
* Holiday features help the model handle calendar anomalies.
* Archived day-ahead weather forecasts improve performance without assuming perfect future weather knowledge.
* Weather improves the pickup model in most seasonal validation weeks.
* Pickup and return flows can be modeled separately with similar accuracy.
* Net outflow forecasting provides a direct bridge to availability-risk estimation.
* Live station inventory changes the project from demand forecasting into operational station-risk analysis.

## Next Steps

Planned improvements include:

* Build a true live future feature generator
* Generate next 1 to 6 hour pickup and return forecasts
* Use live weather forecasts instead of historical forecast rows
* Use recently collected GBFS snapshots as operational context
* Add availability-risk forecasts to the Live Network tab
* Add station rebalancing recommendations
* Add uncertainty bands around pickup, return, and availability forecasts
* Explore station capacity constraints and censored demand
* Add NYC major event indicators
* Improve Streamlit UI polish and product-style dashboard layout
* Deploy the dashboard publicly
