from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
STATION_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "station_metadata.parquet"
)


BACKTEST_RESULTS = pd.DataFrame(
    [
        ["2025-07-11", 5.64, 6.09, 7.10, 7.59, 7.40],
        ["2025-08-08", 5.32, 5.26, 5.85, 8.64, -1.16],
        ["2025-09-12", 5.33, 5.86, 6.84, 7.79, 9.02],
        ["2025-10-10", 4.91, 6.47, 8.80, 8.16, 24.08],
        ["2025-11-07", 4.66, 4.81, 6.60, 6.93, 3.15],
        ["2025-12-05", 3.48, 3.43, 4.93, 4.70, -1.53],
        ["2026-01-09", 3.41, 3.57, 4.42, 4.97, 4.58],
        ["2026-02-06", 1.99, 2.23, 2.40, 2.83, 10.52],
        ["2026-03-06", 4.51, 5.03, 6.39, 6.20, 10.44],
        ["2026-04-03", 4.01, 4.58, 5.70, 6.07, 12.48],
        ["2026-04-10", 3.97, 4.84, 6.67, 5.94, 18.03],
        ["2026-04-17", 4.24, 5.18, 5.86, 6.93, 18.05],
    ],
    columns=[
        "validation_week",
        "weather_model_mae",
        "no_weather_model_mae",
        "last_week_mae",
        "yesterday_mae",
        "weather_improvement_pct",
    ],
)

MODEL_SUMMARY = pd.DataFrame(
    [
        ["Gradient boosting + 24-hour forecast", 4.29, 6.83],
        ["Gradient boosting", 4.78, 7.66],
        ["Same hour last week", 5.96, 9.48],
        ["Same hour yesterday", 6.39, 10.32],
    ],
    columns=["model", "mean_mae", "mean_rmse"],
)

MAY_BENCHMARK = pd.DataFrame(
    [
        ["Gradient boosting + 24-hour forecast", 4.28, 6.66],
        ["Gradient boosting", 4.62, 7.40],
        ["Same hour yesterday", 6.70, 11.18],
        ["Same hour last week", 7.99, 12.67],
    ],
    columns=["model", "mae", "rmse"],
)

pio.templates.default = "plotly_dark"

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


@st.cache_data
def load_station_metadata() -> pd.DataFrame:
    """Load station latitude and longitude data if available."""
    if not STATION_METADATA_FILE.exists():
        return pd.DataFrame()

    return pd.read_parquet(STATION_METADATA_FILE)


@st.cache_data
def add_predictions(
    features: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Add model predictions to the full feature table."""
    model = load_model()
    feature_columns = metadata["feature_columns"]

    predictions = features.copy()
    predictions["prediction"] = model.predict(
        predictions[feature_columns]
    )
    predictions["error"] = (
        predictions["prediction"] - predictions["pickups"]
    )
    predictions["absolute_error"] = predictions["error"].abs()

    return predictions


def apply_custom_css() -> None:
    """Apply custom dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #07111F;
            --bg-panel: #0E1B2E;
            --bg-panel-soft: #10243A;
            --border-soft: rgba(148, 163, 184, 0.22);
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --accent-cyan: #00D4FF;
            --accent-blue: #3B82F6;
            --accent-green: #22C55E;
            --accent-orange: #F59E0B;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0, 212, 255, 0.14), transparent 32rem),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 28rem),
                var(--bg-main);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081425 0%, #0B1728 100%);
            border-right: 1px solid var(--border-soft);
        }

        section[data-testid="stSidebar"] label {
            color: var(--text-muted) !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stDateInput"] input {
            background-color: #111F33 !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 0.75rem !important;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem;
            border-radius: 1.35rem;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(14, 39, 65, 0.96)),
                radial-gradient(circle at 85% 20%, rgba(0, 212, 255, 0.25), transparent 18rem);
            border: 1px solid rgba(125, 211, 252, 0.28);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
            margin-bottom: 1.25rem;
        }

        .hero-eyebrow {
            color: var(--accent-cyan);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            color: var(--text-main);
            font-size: 2.75rem;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: -0.045em;
            margin-bottom: 0.75rem;
        }

        .hero-subtitle {
            color: #CBD5E1;
            font-size: 1.03rem;
            line-height: 1.65;
            max-width: 860px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.4rem;
        }

        .hero-stat {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
        }

        .hero-stat-label {
            color: var(--text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }

        .hero-stat-value {
            color: var(--text-main);
            font-size: 1.3rem;
            font-weight: 850;
            margin-top: 0.18rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 36, 58, 0.96), rgba(10, 24, 41, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 1rem 1.1rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-main);
            font-weight: 850;
        }

        .badge {
            display: inline-block;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
            background: rgba(0, 212, 255, 0.12);
            border: 1px solid rgba(0, 212, 255, 0.28);
            color: #A5F3FC;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        div[data-testid="stTabs"] button {
            color: #CBD5E1;
            font-weight: 800;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent-cyan);
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        h2, h3 {
            color: #E2E8F0;
        }

        p, li {
            color: #CBD5E1;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 1rem;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .small-note {
            color: var(--text-muted);
            font-size: 0.92rem;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .hero-title {
                font-size: 2.1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero(metadata: dict) -> None:
    """Render page hero."""
    hero_html = (
        '<div class="hero-card">'
        '<div class="hero-eyebrow">Urban mobility forecasting dashboard</div>'
        '<div class="hero-title">Citi Bike Demand Intelligence</div>'
        '<div class="hero-subtitle">'
        "Station-level hourly pickup forecasts powered by long-history trip data, "
        "holiday-aware features, and archived 24-hour-ahead weather forecasts."
        "</div>"
        '<div class="hero-grid">'
        '<div class="hero-stat">'
        '<div class="hero-stat-label">Stations</div>'
        f'<div class="hero-stat-value">{metadata["station_count"]}</div>'
        "</div>"
        '<div class="hero-stat">'
        '<div class="hero-stat-label">Training rows</div>'
        f'<div class="hero-stat-value">{metadata["training_rows"]:,}</div>'
        "</div>"
        '<div class="hero-stat">'
        '<div class="hero-stat-label">Backtest MAE</div>'
        '<div class="hero-stat-value">4.29</div>'
        "</div>"
        '<div class="hero-stat">'
        '<div class="hero-stat-label">Benchmark MAE</div>'
        '<div class="hero-stat-value">4.28</div>'
        "</div>"
        "</div>"
        "</div>"
    )

    st.markdown(
        hero_html,
        unsafe_allow_html=True,
    )

def build_sidebar_controls(
    features: pd.DataFrame,
) -> tuple[str, str, object, int]:
    """Build shared sidebar controls."""
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

    return (
        selected_station_label,
        selected_station_id,
        selected_date,
        selected_hour,
    )


def get_selected_row(
    predictions: pd.DataFrame,
    selected_station_id: str,
    selected_date,
    selected_hour: int,
) -> pd.DataFrame:
    """Return the selected station-date-hour row."""
    station_data = predictions[
        predictions["station_id"] == selected_station_id
    ].copy()

    return station_data[
        (station_data["date"] == selected_date)
        & (station_data["hour"] == selected_hour)
    ]


def render_overview_tab(
    predictions: pd.DataFrame,
    metadata: dict,
) -> None:
    """Render project overview."""
    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Stations", f"{metadata['station_count']}")
    col2.metric("Training rows", f"{metadata['training_rows']:,}")
    col3.metric("Best backtest MAE", "4.29")
    col4.metric("May benchmark MAE", "4.28")

    st.markdown(
        """
        <span class="badge">Long-history data</span>
        <span class="badge">Holiday features</span>
        <span class="badge">Day-ahead weather forecasts</span>
        <span class="badge">Station-level predictions</span>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "The v2 model predicts hourly pickup demand for 25 high-traffic "
        "Citi Bike stations. It uses lagged demand, rolling demand, station "
        "identity, calendar features, federal holiday indicators, and archived "
        "24-hour-ahead weather forecasts."
    )

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Seasonal Backtest Summary")
        st.dataframe(
            MODEL_SUMMARY,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        fig = px.bar(
            MODEL_SUMMARY.sort_values("mean_mae", ascending=True),
            x="mean_mae",
            y="model",
            orientation="h",
            title="Mean MAE by Model",
            labels={
                "mean_mae": "Mean MAE",
                "model": "Model",
            },
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Coverage")

    coverage = pd.DataFrame(
        [
            ["Feature table start", metadata["training_start"]],
            ["Feature table end", metadata["training_end"]],
            ["Feature rows", f"{metadata['training_rows']:,}"],
            ["Stations", metadata["station_count"]],
            ["Prediction rows loaded", f"{len(predictions):,}"],
        ],
        columns=["item", "value"],
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )


def render_forecast_explorer_tab(
    predictions: pd.DataFrame,
    selected_station_label: str,
    selected_station_id: str,
    selected_date,
    selected_hour: int,
) -> None:
    """Render station/date/hour forecast explorer."""
    st.subheader("Forecast Explorer")

    selected_rows = get_selected_row(
        predictions,
        selected_station_id,
        selected_date,
        selected_hour,
    )

    if selected_rows.empty:
        st.warning(
            "No feature row exists for that station, date, and hour."
        )
        return

    selected_row = selected_rows.iloc[0]

    prediction = selected_row["prediction"]
    actual_pickups = selected_row["pickups"]
    absolute_error = selected_row["absolute_error"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Predicted pickups", f"{prediction:.1f}")
    col2.metric("Actual pickups", f"{actual_pickups:.0f}")
    col3.metric("Absolute error", f"{absolute_error:.1f}")
    col4.metric("Hour", f"{selected_hour}:00")

    badge_html = ""

    if selected_row["is_us_federal_holiday"] == 1:
        badge_html += '<span class="badge">Federal holiday</span>'

    if selected_row["is_holiday_window"] == 1:
        badge_html += '<span class="badge">Holiday window</span>'

    if selected_row["is_precipitating"] == 1:
        badge_html += '<span class="badge">Precipitation forecast</span>'

    if badge_html:
        st.markdown(badge_html, unsafe_allow_html=True)

    st.write(f"**Selected station:** {selected_station_label}")

    station_day = predictions[
        (predictions["station_id"] == selected_station_id)
        & (predictions["date"] == selected_date)
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=station_day["hour"],
            y=station_day["pickups"],
            mode="lines+markers",
            name="Actual pickups",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=station_day["hour"],
            y=station_day["prediction"],
            mode="lines+markers",
            name="Predicted pickups",
        )
    )

    fig.update_layout(
        title="Selected Station: Actual vs Predicted Demand",
        xaxis_title="Hour of day",
        yaxis_title="Pickups",
        hovermode="x unified",
        height=460,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Summary")

    daily_actual = station_day["pickups"].sum()
    daily_predicted = station_day["prediction"].sum()
    daily_mae = station_day["absolute_error"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Daily actual pickups", f"{daily_actual:.0f}")
    col2.metric("Daily predicted pickups", f"{daily_predicted:.1f}")
    col3.metric("Daily MAE", f"{daily_mae:.2f}")

    st.subheader("Forecast Context")

    context_columns = [
        "timestamp",
        "station_name",
        "pickups",
        "prediction",
        "absolute_error",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "is_us_federal_holiday",
        "is_holiday_window",
        "lag_24",
        "lag_168",
        "lag_336",
        "rolling_mean_168",
    ]

    context = (
        selected_rows[context_columns]
        .T
        .rename(columns={selected_rows.index[0]: "value"})
    )

    st.dataframe(context, use_container_width=True)


def render_station_map_tab(
    predictions: pd.DataFrame,
    station_metadata: pd.DataFrame,
    selected_date,
    selected_hour: int,
) -> None:
    """Render selected station map."""
    st.subheader("Station Map")

    if station_metadata.empty:
        st.info(
            "Station metadata was not found. Run "
            "`python src/data/make_station_metadata.py` to generate it."
        )
        return

    hour_data = predictions[
        (predictions["date"] == selected_date)
        & (predictions["hour"] == selected_hour)
    ].copy()

    map_data = hour_data.merge(
        station_metadata,
        on=["station_id", "station_name"],
        how="left",
    ).dropna(subset=["latitude", "longitude"])

    fig = px.scatter_mapbox(
        map_data,
        lat="latitude",
        lon="longitude",
        size="prediction",
        color="prediction",
        hover_name="station_name",
        hover_data={
            "station_id": True,
            "pickups": True,
            "prediction": ":.1f",
            "absolute_error": ":.1f",
            "latitude": False,
            "longitude": False,
        },
        zoom=11,
        height=620,
        title=f"Predicted Pickup Demand at {selected_hour}:00 on {selected_date}",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        map_data[
            [
                "station_name",
                "pickups",
                "prediction",
                "absolute_error",
            ]
        ]
        .sort_values("prediction", ascending=False)
        .round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_model_performance_tab() -> None:
    """Render model performance diagnostics."""
    st.subheader("Model Performance")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Seasonal backtest summary**")
        st.dataframe(
            MODEL_SUMMARY,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.write("**May 2026 benchmark**")
        st.dataframe(
            MAY_BENCHMARK,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Fold-by-Fold Weather Improvement")

    fig = px.bar(
        BACKTEST_RESULTS,
        x="validation_week",
        y="weather_improvement_pct",
        title="Weather Model Improvement vs No-Weather Model",
        labels={
            "validation_week": "Validation week",
            "weather_improvement_pct": "MAE improvement (%)",
        },
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        BACKTEST_RESULTS.round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_demand_patterns_tab(
    predictions: pd.DataFrame,
) -> None:
    """Render demand pattern charts."""
    st.subheader("Demand Patterns")

    hourly_pattern = (
        predictions.groupby("hour")
        .agg(
            actual_pickups=("pickups", "mean"),
            predicted_pickups=("prediction", "mean"),
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly_pattern["hour"],
            y=hourly_pattern["actual_pickups"],
            mode="lines+markers",
            name="Average actual pickups",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_pattern["hour"],
            y=hourly_pattern["predicted_pickups"],
            mode="lines+markers",
            name="Average predicted pickups",
        )
    )

    fig.update_layout(
        title="Average Hourly Demand Pattern",
        xaxis_title="Hour of day",
        yaxis_title="Average pickups per station-hour",
        hovermode="x unified",
        height=450,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    day_type_pattern = (
        predictions.assign(
            day_type=lambda df: df["is_weekend"].map(
                {
                    0: "Weekday",
                    1: "Weekend",
                }
            )
        )
        .groupby(["day_type", "hour"])
        .agg(actual_pickups=("pickups", "mean"))
        .reset_index()
    )

    fig = px.line(
        day_type_pattern,
        x="hour",
        y="actual_pickups",
        color="day_type",
        markers=True,
        title="Weekday vs Weekend Demand Pattern",
        labels={
            "hour": "Hour of day",
            "actual_pickups": "Average pickups per station-hour",
            "day_type": "Day type",
        },
    )

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    holiday_summary = (
        predictions.assign(
            holiday_window=lambda df: df["is_holiday_window"].map(
                {
                    0: "Non-holiday window",
                    1: "Holiday window",
                }
            )
        )
        .groupby("holiday_window")
        .agg(
            avg_pickups=("pickups", "mean"),
            avg_prediction=("prediction", "mean"),
            avg_error=("absolute_error", "mean"),
        )
        .reset_index()
    )

    st.subheader("Holiday Window Summary")
    st.dataframe(
        holiday_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_about_tab(metadata: dict) -> None:
    """Render about page."""
    st.subheader("About This Project")

    st.write(
        "This project forecasts station-level hourly Citi Bike pickup demand. "
        "It is designed as an applied time-series and machine-learning project, "
        "with an emphasis on realistic evaluation and operational inputs."
    )

    st.markdown(
        """
        **Core modeling choices**

        - Global gradient-boosting model across selected stations
        - Poisson loss for nonnegative pickup counts
        - Lagged demand features for daily and weekly seasonality
        - Rolling demand features for recent station behavior
        - US federal holiday features
        - Archived 24-hour-ahead weather forecasts
        - Expanding-window validation across multiple seasons
        """
    )

    st.subheader("Saved Model Metadata")

    metadata_display = {
        key: value
        for key, value in metadata.items()
        if key != "stations"
    }

    st.json(metadata_display)


def main() -> None:
    st.set_page_config(
        page_title="Citi Bike Demand Forecasting",
        page_icon="🚲",
        layout="wide",
    )

    apply_custom_css()

    metadata = load_metadata()
    render_hero(metadata)
    features = load_features()
    station_metadata = load_station_metadata()
    predictions = add_predictions(features, metadata)

    (
        selected_station_label,
        selected_station_id,
        selected_date,
        selected_hour,
    ) = build_sidebar_controls(predictions)

    tabs = st.tabs(
        [
            "Overview",
            "Forecast Explorer",
            "Station Map",
            "Model Performance",
            "Demand Patterns",
            "About",
        ]
    )

    with tabs[0]:
        render_overview_tab(predictions, metadata)

    with tabs[1]:
        render_forecast_explorer_tab(
            predictions,
            selected_station_label,
            selected_station_id,
            selected_date,
            selected_hour,
        )

    with tabs[2]:
        render_station_map_tab(
            predictions,
            station_metadata,
            selected_date,
            selected_hour,
        )

    with tabs[3]:
        render_model_performance_tab()

    with tabs[4]:
        render_demand_patterns_tab(predictions)

    with tabs[5]:
        render_about_tab(metadata)


if __name__ == "__main__":
    main()