from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "citi_bike_demand_model.joblib"
METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "citi_bike_demand_model_metadata.json"
)
FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)


@st.cache_resource
def load_model():
    """Load the trained forecasting model."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata() -> dict:
    """Load model metadata."""
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_features() -> pd.DataFrame:
    """Load model feature table."""
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = features["timestamp"].dt.date
    return features


def main() -> None:
    st.set_page_config(
        page_title="Citi Bike Demand Forecasting",
        page_icon="🚲",
        layout="wide",
    )

    st.title("🚲 Citi Bike Demand Forecasting")
    st.write(
        "Explore hourly station-level pickup forecasts from the "
        "v2 long-history weather and holiday model."
    )

    model = load_model()
    metadata = load_metadata()
    features = load_features()

    feature_columns = metadata["feature_columns"]

    st.sidebar.header("Forecast Selection")

    stations = (
        features[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_name")
    )

    station_label_to_id = {
        f"{row.station_name} ({row.station_id})": row.station_id
        for row in stations.itertuples(index=False)
    }

    selected_station_label = st.sidebar.selectbox(
        "Station",
        list(station_label_to_id.keys()),
    )

    selected_station_id = station_label_to_id[selected_station_label]

    station_data = features[
        features["station_id"] == selected_station_id
    ].copy()

    min_date = station_data["date"].min()
    max_date = station_data["date"].max()

    selected_date = st.sidebar.date_input(
        "Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    selected_hour = st.sidebar.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=17,
    )

    selected_rows = station_data[
        (station_data["date"] == selected_date)
        & (station_data["hour"] == selected_hour)
    ]

    if selected_rows.empty:
        st.warning(
            "No feature row exists for that station, date, and hour."
        )
        return

    selected_row = selected_rows.iloc[[0]]

    prediction = model.predict(
        selected_row[feature_columns]
    )[0]

    actual_pickups = selected_row["pickups"].iloc[0]
    absolute_error = abs(prediction - actual_pickups)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted pickups",
        f"{prediction:.1f}",
    )

    col2.metric(
        "Actual pickups",
        f"{actual_pickups:.0f}",
    )

    col3.metric(
        "Absolute error",
        f"{absolute_error:.1f}",
    )

    st.subheader("Selected Forecast Context")

    context_columns = [
        "timestamp",
        "station_name",
        "pickups",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "is_us_federal_holiday",
        "is_holiday_window",
        "lag_24",
        "lag_168",
        "rolling_mean_168",
    ]

    st.dataframe(
        selected_row[context_columns].T.rename(columns={selected_row.index[0]: "value"}),
        use_container_width=True,
    )

    st.subheader("Selected Station: Actual vs Predicted for the Day")

    day_data = station_data[
        station_data["date"] == selected_date
    ].copy()

    day_data["prediction"] = model.predict(
        day_data[feature_columns]
    )

    chart_data = day_data[
        [
            "hour",
            "pickups",
            "prediction",
        ]
    ].set_index("hour")

    st.line_chart(chart_data)

    st.subheader("Model Details")

    st.write(
        f"Model version: `{metadata['model_version']}`"
    )
    st.write(
        f"Training rows: `{metadata['training_rows']:,}`"
    )
    st.write(
        f"Training range: `{metadata['training_start']}` through "
        f"`{metadata['training_end']}`"
    )
    st.write(
        f"Stations: `{metadata['station_count']}`"
    )


if __name__ == "__main__":
    main()