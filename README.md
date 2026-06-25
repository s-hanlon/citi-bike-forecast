# CitiFlow Intelligence

CitiFlow Intelligence is an end-to-end bike-share operations intelligence project built around Citi Bike data in New York City.

The project began as a station-level pickup demand forecasting system and has expanded into a live availability intelligence platform. It now combines historical trip data, weather forecasts, holiday features, live GBFS station availability, Postgres snapshot storage, Dockerized data collection, flow forecasting, and a Streamlit dashboard for operational monitoring.

The long-term goal is to help a user answer a practical question:

> Will there be bikes or docks available at the station I care about in the next few hours?

## Current Project Status

The project currently supports:

* Historical station-level pickup demand forecasting
* Pickup, return, and net outflow modeling
* Live Citi Bike station availability collection from GBFS
* A Postgres database for storing station availability snapshots
* A Dockerized collector that writes new snapshots every 5 minutes
* A Streamlit dashboard with live network status, historical availability trends, station-level history, and operational risk rankings
* A prototype future availability projection that combines current inventory with predicted station flow

The future availability prediction model is still a prototype. The project is now collecting the live availability history needed to eventually train a supervised model that directly predicts future bike and dock availability.

## Why Availability Matters

Predicting pickup demand is useful, but it does not fully answer whether a rider can actually get a bike.

A station can show low pickups for two different reasons:

1. Few people wanted bikes there.
2. The station was empty, so demand was censored.

The second case is operationally important. If a station is empty, observed pickups understate true demand. Similarly, if a station is full, observed returns may understate true return demand.

That is why this project now tracks both demand and availability.

## System Architecture

```text
Historical Citi Bike trip data
        |
        v
Hourly pickup and return datasets
        |
        v
Weather, holiday, calendar, and lag features
        |
        v
Demand and flow forecasting models
        |
        v
Streamlit dashboard


Live Citi Bike GBFS feed
        |
        v
Dockerized availability collector
        |
        v
Postgres station availability database
        |
        v
Live Network and Historical Availability dashboard tabs
```

## Main Components

### 1. Historical Demand Forecasting

The original modeling pipeline uses historical Citi Bike trip data to predict hourly station pickups.

Pipeline:

```text
raw Citi Bike trip ZIPs
→ hourly station pickup dataset
→ weather and holiday features
→ lag and rolling demand features
→ global machine learning model
→ station-level hourly demand forecasts
```

The model uses:

* Station ID
* Hour of day
* Day of week
* Month
* Weekend indicator
* Cyclical time encodings
* Federal holiday features
* Observed and forecast weather
* Lagged demand
* Rolling demand statistics

### 2. Flow Forecasting

The project also builds hourly pickup, return, and net outflow datasets.

```text
net outflow = pickups - returns
```

This matters because station inventory changes according to both bikes leaving and bikes arriving.

The flow models predict:

* Pickups
* Returns
* Net outflow

These are used as the foundation for the prototype availability projection.

### 3. Live Availability Collection

The project collects live station availability data from Citi Bike’s GBFS feed.

The collector stores snapshots in Postgres every 5 minutes.

Each snapshot includes:

* Station ID
* Station name
* Latitude and longitude
* Capacity
* Bikes available
* E-bikes available
* Docks available
* Percent bikes available
* Percent docks available
* Availability status
* Station installed/renting/returning flags
* Last reported timestamp
* Snapshot timestamp

Availability statuses include:

* healthy
* nearly_empty
* nearly_full
* low_bikes
* low_docks
* empty
* full
* station_offline
* station_not_installed

### 4. Postgres Availability Database

The live availability system stores data in two Postgres tables:

```text
station_information
station_status_snapshots
```

The database supports:

* Latest station availability lookup
* Network-wide availability history
* Station-level availability history
* Top empty/full risk stations
* Collector freshness checks

The Postgres database runs locally through Docker Compose.

### 5. Streamlit Dashboard

The dashboard includes:

* Overview
* Forecast Explorer
* Live Network
* Historical Availability
* Station Map
* Model Performance
* Demand Patterns
* About

The dashboard reads the latest availability from Postgres first. If Postgres is unavailable or empty, it falls back to a direct GBFS fetch.

## Dashboard Features

### Live Network

The Live Network tab shows the current Citi Bike system state using the latest stored Postgres snapshot.

It includes:

* Live station map
* Healthy station count
* Bike shortage risk count
* Dock shortage risk count
* Risk tables
* Availability status distribution

### Historical Availability

The Historical Availability tab uses stored Postgres snapshots to show availability trends over time.

It includes:

* Network risk over time
* Average bike and dock fill levels
* Selected station inventory history
* Top recent problem stations
* Operational risk rankings

Operational rankings include:

* Most bike shortage risk
* Most dock shortage risk
* Most often empty
* Most often full

### Collector Freshness

The sidebar includes collector status so the dashboard is honest about whether the latest availability data is fresh.

It shows:

* Fresh / slightly stale / stale status
* Snapshot age
* Stored snapshot count

### Prototype Availability Projection

The app includes a prototype projection that combines current station inventory with predicted pickups and returns:

```text
projected bikes = current bikes + predicted returns - predicted pickups
projected docks = current docks + predicted pickups - predicted returns
```

This is currently a prototype, not the final future availability model.

## Data Sources

### Citi Bike Trip Data

Historical Citi Bike trip data is used to build pickup, return, and net outflow datasets.

Raw fields include:

* ride_id
* rideable_type
* started_at
* ended_at
* start_station_id
* start_station_name
* end_station_id
* end_station_name
* start_lat
* start_lng
* end_lat
* end_lng
* member_casual

### Citi Bike GBFS

Live station availability is collected from Citi Bike GBFS endpoints:

* Station information
* Station status

### Weather Data

Weather features are used for historical and forecast-aware modeling.

Features include:

* Temperature
* Precipitation
* Wind speed
* Relative humidity
* Weather code features
* Archived day-ahead forecasts

### Holiday Features

Federal holiday features are included to capture demand shifts around holidays.

Features include:

* Is federal holiday
* Day before holiday
* Day after holiday
* Holiday window
* Days to nearest holiday

## Model Results

The project evaluates models against simple baselines such as yesterday and last week.

Current flow model results show meaningful improvements over last-week baselines for pickup, return, and net outflow forecasting.

Recent benchmark results:

| Target      | Model MAE | Model RMSE | Last Week MAE | Last Week RMSE | MAE Improvement |
| ----------- | --------: | ---------: | ------------: | -------------: | --------------: |
| Pickups     |      4.29 |       6.71 |          7.99 |          12.67 |          46.24% |
| Returns     |      4.01 |       6.40 |          7.40 |          11.83 |          45.84% |
| Net Outflow |      4.05 |       6.25 |          5.46 |           8.69 |          25.81% |

The demand model also improved over earlier versions after adding longer historical data, weather forecasts, and holiday features.

## Running Locally

### 1. Start Postgres and the collector

```powershell
docker compose up -d
```

This starts:

```text
citibike-postgres
citibike-collector
```

The collector writes new GBFS station availability snapshots into Postgres every 5 minutes.

### 2. Check the collector

```powershell
docker ps
```

```powershell
docker logs citibike-collector --tail 30
```

### 3. Check stored snapshot count

```powershell
docker exec -it citibike-postgres psql -U citibike -d citibike -c "SELECT COUNT(DISTINCT snapshot_utc) AS snapshots, COUNT(*) AS rows, MAX(snapshot_utc) AS latest_snapshot FROM station_status_snapshots;"
```

### 4. Run the Streamlit app

```powershell
streamlit run src/app/app.py
```

The app is configured to run on port 8502.

```text
http://localhost:8502
```

## Project Structure

```text
citi-bike-forecast/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── app/
│   │   └── app.py
│   ├── data/
│   │   ├── collect_station_status.py
│   │   ├── collect_station_status_db.py
│   │   ├── download_data.py
│   │   ├── download_weather.py
│   │   ├── make_availability_dataset.py
│   │   ├── make_dataset.py
│   │   ├── make_station_flows.py
│   │   ├── make_station_metadata.py
│   │   └── make_weather_dataset.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres.py
│   │   ├── queries.py
│   │   └── schema.sql
│   ├── features/
│   │   ├── build_features.py
│   │   └── build_flow_features.py
│   ├── live/
│   │   ├── __init__.py
│   │   └── gbfs.py
│   └── models/
│       ├── availability_projection.py
│       ├── baseline.py
│       ├── flow_forecasting.py
│       ├── gradient_boosting.py
│       ├── train_final_flow_models.py
│       └── train_final_model.py
├── Dockerfile.collector
├── docker-compose.yml
├── requirements.txt
├── requirements-collector.txt
└── README.md
```

## Current Limitations

### Future Availability Prediction Is Still a Prototype

The current projection combines live inventory with predicted pickup and return flow. It is not yet a model trained directly on future availability outcomes.

The future goal is to train models that predict:

* Bikes available 30 minutes later
* Bikes available 1 hour later
* Bikes available 2 hours later
* Docks available at future horizons
* Empty risk
* Full risk

### More Availability History Is Needed

The live collector is now creating the data needed for direct future availability modeling.

A few snapshots are enough to test the system, but more data is needed for stronger modeling:

```text
1 day: useful for smoke tests and early patterns
1 week: useful for early supervised modeling
2 to 4 weeks: stronger short-term availability models
2 to 3 months: much better station-level reliability
```

### Rebalancing Is Hard to Predict

Citi Bike operators rebalance stations by moving bikes with trucks. This can cause sudden inventory jumps that are not explained by rider trips alone.

Future work could detect likely rebalancing events from snapshot changes and use them as features.

## Roadmap

### Near Term

* Continue collecting GBFS availability snapshots
* Let the Postgres availability history grow
* Improve README and project documentation
* Keep dashboard labels clear about prototype versus production-ready forecasting

### Next Modeling Step

Create a supervised future availability dataset.

Example target rows:

```text
station_id
current timestamp
current bikes
current docks
current e-bikes
capacity
recent bike trend
recent dock trend
recent pickup trend
recent return trend
weather
calendar features
target horizon
bikes available at T + horizon
docks available at T + horizon
empty risk at T + horizon
full risk at T + horizon
```

Potential horizons:

* 30 minutes
* 1 hour
* 2 hours
* 3 hours

### Long-Term Dream Version

The ideal end version of the app would let a user select a station and future time, then answer:

```text
Will there be bikes available when I need one?
Will there be open docks when I arrive?
What nearby stations are safer alternatives?
```

Future dashboard features could include:

* Predicted bikes available by horizon
* Predicted docks available by horizon
* Empty/full risk probability
* Confidence level
* Nearby backup station recommendations
* Rebalancing candidate recommendations

### Cloud Deployment

The current system runs locally. A production-style deployment would move the architecture to the cloud:

```text
cloud Postgres
→ always-on collector container
→ hosted dashboard
```

This would allow the availability database to keep growing 24/7 without relying on a local laptop.
