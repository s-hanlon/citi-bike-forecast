from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.live.gbfs import fetch_live_station_availability
from src.models.availability_projection import project_station_availability
from src.models.flow_forecasting import MODEL_FEATURES as FLOW_MODEL_FEATURES
from src.database.queries import get_latest_station_availability

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

FLOW_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flow_model_features.parquet"
)

PICKUP_FLOW_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "citi_bike_pickup_flow_model.joblib"
)

RETURN_FLOW_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "citi_bike_return_flow_model.joblib"
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

@st.cache_data(ttl=60)
def load_live_availability() -> pd.DataFrame:
    """Load latest station availability from Postgres, falling back to GBFS."""
    try:
        availability = get_latest_station_availability()

        if not availability.empty:
            availability["station_id"] = availability["station_id"].astype(str)
            availability["availability_source"] = "postgres_latest_snapshot"
            return availability

    except Exception as error:
        print(f"Postgres availability load failed, falling back to GBFS: {error}")

    availability = fetch_live_station_availability()
    availability["station_id"] = availability["station_id"].astype(str)
    availability["availability_source"] = "direct_gbfs"

    return availability

@st.cache_data
def load_flow_features() -> pd.DataFrame:
    """Load pickup/return flow model feature table."""
    flow_features = pd.read_parquet(FLOW_FEATURE_FILE)
    flow_features["station_id"] = flow_features["station_id"].astype(str)
    flow_features["timestamp"] = pd.to_datetime(flow_features["timestamp"])

    return flow_features


@st.cache_resource
def load_flow_models():
    """Load final pickup and return flow models."""
    pickup_model = joblib.load(PICKUP_FLOW_MODEL_FILE)
    return_model = joblib.load(RETURN_FLOW_MODEL_FILE)

    return pickup_model, return_model

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

def apply_product_polish_css() -> None:
    """Apply additional product-style dashboard polish."""
    st.markdown(
        """
<style>
.product-shell {
    display: grid;
    grid-template-columns: 1.25fr 0.75fr;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.product-panel {
    background: linear-gradient(180deg, rgba(15, 32, 54, 0.96), rgba(8, 18, 32, 0.96));
    border: 1px solid rgba(125, 211, 252, 0.18);
    border-radius: 1.25rem;
    padding: 1.2rem;
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.25);
}

.product-kicker {
    color: #22D3EE;
    font-size: 0.74rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 900;
    margin-bottom: 0.45rem;
}

.product-title {
    color: #F8FAFC;
    font-size: 2.85rem;
    line-height: 0.98;
    letter-spacing: -0.06em;
    font-weight: 950;
    margin-bottom: 0.75rem;
}

.product-copy {
    color: #CBD5E1;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 760px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.15rem;
}

.product-metric {
    background: rgba(2, 8, 23, 0.52);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 1rem;
    padding: 1rem;
}

.product-metric-label {
    color: #94A3B8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 850;
}

.product-metric-value {
    color: #F8FAFC;
    font-size: 1.75rem;
    line-height: 1.1;
    font-weight: 950;
    margin-top: 0.35rem;
}

.product-metric-caption {
    color: #94A3B8;
    font-size: 0.78rem;
    margin-top: 0.25rem;
}

.signal-card {
    background: rgba(2, 8, 23, 0.42);
    border: 1px solid rgba(34, 211, 238, 0.16);
    border-radius: 1rem;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

.signal-label {
    color: #94A3B8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 850;
}

.signal-value {
    color: #A5F3FC;
    font-size: 1.15rem;
    font-weight: 900;
    margin-top: 0.25rem;
}

.signal-copy {
    color: #CBD5E1;
    font-size: 0.85rem;
    line-height: 1.45;
    margin-top: 0.35rem;
}

.executive-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-top: 0.9rem;
}

.executive-card {
    background: linear-gradient(180deg, rgba(16, 36, 58, 0.92), rgba(8, 18, 32, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 1.05rem;
    padding: 1rem;
}

.executive-card-title {
    color: #F8FAFC;
    font-weight: 900;
    font-size: 1rem;
    margin-bottom: 0.4rem;
}

.executive-card-copy {
    color: #CBD5E1;
    line-height: 1.5;
    font-size: 0.9rem;
}

.section-label {
    color: #F8FAFC;
    font-size: 1.3rem;
    font-weight: 950;
    letter-spacing: -0.035em;
    margin-top: 1.2rem;
    margin-bottom: 0.25rem;
}

.section-subcopy {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-bottom: 0.85rem;
}

@media (max-width: 1050px) {
    .product-shell {
        grid-template-columns: 1fr;
    }

    .metric-grid,
    .executive-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 650px) {
    .metric-grid,
    .executive-grid {
        grid-template-columns: 1fr;
    }

    .product-title {
        font-size: 2.25rem;
    }
}

.availability-panel {
    background: linear-gradient(180deg, rgba(10, 24, 41, 0.96), rgba(3, 10, 22, 0.96));
    border: 1px solid rgba(34, 211, 238, 0.18);
    border-radius: 1.15rem;
    padding: 1.05rem;
    margin: 1rem 0 1.1rem 0;
}

.availability-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.9rem;
}

.availability-title {
    color: #F8FAFC;
    font-size: 1.05rem;
    font-weight: 950;
}

.availability-subtitle {
    color: #94A3B8;
    font-size: 0.82rem;
    margin-top: 0.18rem;
}

.status-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 900;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}

.status-good {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #86EFAC;
}

.status-warning {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #FCD34D;
}

.status-danger {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #FCA5A5;
}

.status-neutral {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: #CBD5E1;
}

.availability-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.75rem;
}

.availability-card {
    background: rgba(2, 8, 23, 0.48);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 0.95rem;
    padding: 0.85rem;
}

.availability-label {
    color: #94A3B8;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 850;
}

.availability-value {
    color: #F8FAFC;
    font-size: 1.45rem;
    font-weight: 950;
    margin-top: 0.25rem;
}

.availability-note {
    color: #94A3B8;
    font-size: 0.8rem;
    line-height: 1.45;
    margin-top: 0.85rem;
}

@media (max-width: 950px) {
    .availability-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .availability-header {
        flex-direction: column;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )

def render_hero(metadata: dict) -> None:
    """Render product-style hero."""
    hero_html = (
        '<div class="product-shell">'
        '<div class="product-panel">'
        '<div class="product-kicker">NYC mobility forecasting system</div>'
        '<div class="product-title">CitiFlow Intelligence</div>'
        '<div class="product-copy">'
        "Station-level Citi Bike demand forecasting built with long-history trip data, "
        "holiday-aware calendar features, and archived 24-hour-ahead weather forecasts. "
        "Designed to estimate pickup demand patterns across the urban bike-share network."
        "</div>"
        '<div class="metric-grid">'
        '<div class="product-metric">'
        '<div class="product-metric-label">Stations</div>'
        f'<div class="product-metric-value">{metadata["station_count"]}</div>'
        '<div class="product-metric-caption">high-demand locations</div>'
        "</div>"
        '<div class="product-metric">'
        '<div class="product-metric-label">Training rows</div>'
        f'<div class="product-metric-value">{metadata["training_rows"]:,}</div>'
        '<div class="product-metric-caption">station-hour examples</div>'
        "</div>"
        '<div class="product-metric">'
        '<div class="product-metric-label">Backtest MAE</div>'
        '<div class="product-metric-value">4.29</div>'
        '<div class="product-metric-caption">seasonal validation</div>'
        "</div>"
        '<div class="product-metric">'
        '<div class="product-metric-label">Weather lift</div>'
        '<div class="product-metric-value">10.3%</div>'
        '<div class="product-metric-caption">vs non-weather ML</div>'
        "</div>"
        "</div>"
        "</div>"
        '<div class="product-panel">'
        '<div class="signal-card">'
        '<div class="signal-label">Best model</div>'
        '<div class="signal-value">Gradient boosting + weather + holidays</div>'
        '<div class="signal-copy">Global station model trained on lagged demand, rolling demand, calendar effects, and day-ahead weather forecasts.</div>'
        "</div>"
        '<div class="signal-card">'
        '<div class="signal-label">Evaluation</div>'
        '<div class="signal-value">12 seasonal validation weeks</div>'
        '<div class="signal-copy">Backtests span summer, fall, winter, and spring instead of relying on one narrow time window.</div>'
        "</div>"
        '<div class="signal-card">'
        '<div class="signal-label">Operational caveat</div>'
        '<div class="signal-value">Historical explorer</div>'
        '<div class="signal-copy">The current app explores historical forecast rows. A future version can connect live station status and real-time weather forecasts.</div>'
        "</div>"
        "</div>"
        "</div>"
    )

    st.markdown(hero_html, unsafe_allow_html=True)

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

def find_live_station_availability(
    live_availability: pd.DataFrame,
    selected_station_id: str,
    selected_station_name: str,
) -> pd.DataFrame:
    """Find a live GBFS row using station ID first, then station name."""
    if live_availability.empty:
        return pd.DataFrame()

    live_data = live_availability.copy()
    live_data["station_id"] = live_data["station_id"].astype(str)
    live_data["station_name_normalized"] = live_data["station_name"].apply(
        normalize_station_name
    )

    selected_station_id = str(selected_station_id)
    selected_station_name_normalized = normalize_station_name(
        selected_station_name
    )

    selected_live = live_data[
        live_data["station_id"] == selected_station_id
    ]

    if selected_live.empty:
        selected_live = live_data[
            live_data["station_name_normalized"]
            == selected_station_name_normalized
        ]

    return selected_live

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

def render_availability_source_note(live_availability: pd.DataFrame) -> None:
    """Render a note describing where availability data came from."""
    if live_availability.empty:
        return

    source = live_availability.get(
        "availability_source",
        pd.Series(["unknown"]),
    ).iloc[0]

    if source == "postgres_latest_snapshot" and "snapshot_utc" in live_availability.columns:
        latest_snapshot = pd.to_datetime(
            live_availability["snapshot_utc"]
        ).max()

        source_label = (
            "Source: Postgres latest stored GBFS snapshot "
            f"({format_live_timestamp(latest_snapshot)})."
        )
    elif source == "direct_gbfs":
        source_label = "Source: Direct live GBFS fetch."
    else:
        source_label = "Source: Availability data loaded."

    st.markdown(
        f'<div class="section-subcopy">{source_label}</div>',
        unsafe_allow_html=True,
    )

def render_overview_tab(
    predictions: pd.DataFrame,
    metadata: dict,
) -> None:
    """Render product-style project overview."""
    st.markdown(
        '<div class="section-label">Executive Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "A compact summary of the forecasting system, evaluation design, "
            "and current model performance."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="executive-grid">
    <div class="executive-card">
        <div class="executive-card-title">Forecasting task</div>
        <div class="executive-card-copy">
            Predict hourly pickup demand for high-volume Citi Bike stations using historical demand, holidays, and weather forecasts.
        </div>
    </div>
    <div class="executive-card">
        <div class="executive-card-title">Why it matters</div>
        <div class="executive-card-copy">
            Accurate demand forecasts can support rebalancing, staffing, station monitoring, and shortage-risk analysis.
        </div>
    </div>
    <div class="executive-card">
        <div class="executive-card-title">Current limitation</div>
        <div class="executive-card-copy">
            The app currently explores historical forecast rows. Live deployment would require real-time station and weather feeds.
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Model Snapshot</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "The strongest model combines demand history, station identity, "
            "holiday signals, and archived day-ahead weather forecasts."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.95, 1.05])

    with col1:
        fig = px.bar(
            MODEL_SUMMARY.sort_values("mean_mae", ascending=True),
            x="mean_mae",
            y="model",
            orientation="h",
            title="Seasonal Backtest MAE by Model",
            labels={
                "mean_mae": "Mean MAE",
                "model": "",
            },
            text="mean_mae",
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            height=390,
            margin=dict(l=10, r=35, t=55, b=10),
            xaxis_title="Mean absolute error",
            yaxis_title="",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            BACKTEST_RESULTS,
            x="validation_week",
            y="weather_improvement_pct",
            title="Weather Forecast Lift by Validation Week",
            labels={
                "validation_week": "Validation week",
                "weather_improvement_pct": "MAE improvement vs no-weather model (%)",
            },
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
        )

        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=55, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-label">System Coverage</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "The current dashboard loads the trained v2 model and evaluates "
            "historical station-hour forecast rows from the processed feature table."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    coverage_html = (
        '<div class="executive-grid">'
        '<div class="executive-card">'
        '<div class="executive-card-title">Training window</div>'
        f'<div class="executive-card-copy">{metadata["training_start"]} through {metadata["training_end"]}</div>'
        "</div>"
        '<div class="executive-card">'
        '<div class="executive-card-title">Feature table</div>'
        f'<div class="executive-card-copy">{metadata["training_rows"]:,} station-hour rows across {metadata["station_count"]} selected stations.</div>'
        "</div>"
        '<div class="executive-card">'
        '<div class="executive-card-title">Loaded predictions</div>'
        f'<div class="executive-card-copy">{len(predictions):,} rows scored in the dashboard from the saved model artifact.</div>'
        "</div>"
        "</div>"
    )

    st.markdown(
        coverage_html,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Current Results</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Seasonal validation summary**")
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

def format_percent(value) -> str:
    """Format a decimal ratio as a percentage string."""
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.0f}%"

def normalize_station_name(value) -> str:
    """Normalize station names for matching historical data to live GBFS data."""
    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .strip()
        .replace("&", "and")
        .replace(".", "")
        .replace("  ", " ")
    )

def format_live_timestamp(value) -> str:
    """Format a live GBFS timestamp for display."""
    if pd.isna(value):
        return "Unknown"

    timestamp = pd.to_datetime(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    return timestamp.tz_convert("America/New_York").strftime(
        "%b %d, %Y %I:%M %p ET"
    )


def format_availability_status(status: str) -> str:
    """Format an availability status label."""
    return str(status).replace("_", " ").title()


def availability_status_class(status: str) -> str:
    """Map availability status to a CSS class."""
    status = str(status)

    if status in {"healthy"}:
        return "status-good"

    if status in {
        "nearly_empty",
        "nearly_full",
        "low_bikes",
        "low_docks",
        "nearly_empty_risk",
        "nearly_full_risk",
    }:
        return "status-warning"

    if status in {
        "empty",
        "full",
        "station_offline",
        "station_not_installed",
        "empty_risk",
        "full_risk",
    }:
        return "status-danger"

    return "status-neutral"

def render_live_availability_panel(
    live_availability: pd.DataFrame,
    selected_station_id: str,
    selected_station_name: str,
) -> None:
    """Render live availability for the selected station."""
    st.markdown(
        '<div class="section-label">Live Station Availability</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "Real-time Citi Bike inventory from the GBFS station status feed. "
            "This is current station context, not yet an input to the trained "
            "historical demand model."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    render_availability_source_note(live_availability)

    if live_availability.empty:
        st.info("Live availability data is not available right now.")
        return

    live_data = live_availability.copy()
    live_data["station_id"] = live_data["station_id"].astype(str)
    live_data["station_name_normalized"] = live_data["station_name"].apply(
        normalize_station_name
    )

    selected_station_id = str(selected_station_id)
    selected_station_name_normalized = normalize_station_name(
        selected_station_name
    )

    selected_live = live_data[
        live_data["station_id"] == selected_station_id
    ]

    if selected_live.empty:
        selected_live = live_data[
            live_data["station_name_normalized"]
            == selected_station_name_normalized
        ]

    if selected_live.empty:
        st.info(
            "This selected station was not found in the live GBFS availability feed."
        )

        with st.expander("Debug matching details"):
            st.write("Historical station ID:", selected_station_id)
            st.write("Historical station name:", selected_station_name)
            st.write(
                "Live stations with similar names:",
                live_data[
                    live_data["station_name"]
                    .str.lower()
                    .str.contains(
                        str(selected_station_name).lower().split(" ")[0],
                        na=False,
                    )
                ][["station_id", "station_name"]].head(20),
            )

        return

    row = selected_live.iloc[0]

    status = row["availability_status"]
    status_label = format_availability_status(status)
    status_class = availability_status_class(status)

    station_name = row["station_name"]
    capacity = row["capacity"]
    bikes = row["num_bikes_available"]
    ebikes = row["num_ebikes_available"]
    docks = row["num_docks_available"]
    pct_bikes = format_percent(row["pct_bikes_available"])
    pct_docks = format_percent(row["pct_docks_available"])
    last_reported = format_live_timestamp(row["last_reported_utc"])

    panel_html = (
        '<div class="availability-panel">'
        '<div class="availability-header">'
        "<div>"
        '<div class="availability-title">'
        f"{station_name}"
        "</div>"
        '<div class="availability-subtitle">'
        f"Last reported: {last_reported}"
        "</div>"
        "</div>"
        f'<div class="status-pill {status_class}">{status_label}</div>'
        "</div>"
        '<div class="availability-grid">'
        '<div class="availability-card">'
        '<div class="availability-label">Bikes</div>'
        f'<div class="availability-value">{bikes:.0f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">E-bikes</div>'
        f'<div class="availability-value">{ebikes:.0f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Open docks</div>'
        f'<div class="availability-value">{docks:.0f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Bike fill</div>'
        f'<div class="availability-value">{pct_bikes}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Dock fill</div>'
        f'<div class="availability-value">{pct_docks}</div>'
        "</div>"
        "</div>"
        '<div class="availability-note">'
        "Interpretation: high predicted pickup demand combined with low bike "
        "availability can indicate empty-station risk. High return pressure "
        "combined with low dock availability can indicate full-station risk. "
        "We will model those risks directly in Level 3."
        "</div>"
        "</div>"
    )

    st.markdown(panel_html, unsafe_allow_html=True)

def render_availability_projection_panel(
    live_availability: pd.DataFrame,
    flow_features: pd.DataFrame,
    pickup_flow_model,
    return_flow_model,
    selected_station_id: str,
    selected_station_name: str,
    selected_timestamp,
) -> None:
    """Render prototype future availability projection."""
    st.markdown(
        '<div class="section-label">Prototype Availability Projection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "This combines live station inventory with model-predicted pickup "
            "and return pressure for the selected station-hour pattern. "
            "The current version is a prototype because the demand features "
            "come from a historical row, while the station inventory is live."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    selected_live = find_live_station_availability(
        live_availability=live_availability,
        selected_station_id=selected_station_id,
        selected_station_name=selected_station_name,
    )

    if selected_live.empty:
        st.info("Live station inventory was not found for this station.")
        return

    selected_flow_row = flow_features[
        (flow_features["station_id"].astype(str) == str(selected_station_id))
        & (flow_features["timestamp"] == pd.to_datetime(selected_timestamp))
    ]

    if selected_flow_row.empty:
        st.info(
            "No pickup/return flow feature row exists for this selected station-hour."
        )
        return

    live_row = selected_live.iloc[0]
    flow_row = selected_flow_row.iloc[[0]]

    predicted_pickups = float(
        pickup_flow_model.predict(flow_row[FLOW_MODEL_FEATURES])[0]
    )
    predicted_returns = float(
        return_flow_model.predict(flow_row[FLOW_MODEL_FEATURES])[0]
    )

    predicted_pickups = max(0, predicted_pickups)
    predicted_returns = max(0, predicted_returns)

    current_bikes = float(live_row["num_bikes_available"])
    current_docks = float(live_row["num_docks_available"])
    capacity = float(live_row["capacity"])

    projection = project_station_availability(
        current_bikes=current_bikes,
        current_docks=current_docks,
        capacity=capacity,
        predicted_pickups=predicted_pickups,
        predicted_returns=predicted_returns,
    )

    risk_label = format_availability_status(projection.risk_status)
    risk_class = availability_status_class(projection.risk_status)

    projection_html = (
        '<div class="availability-panel">'
        '<div class="availability-header">'
        "<div>"
        '<div class="availability-title">One-Hour Availability Projection</div>'
        '<div class="availability-subtitle">'
        "Projected station state after applying predicted pickups and returns."
        "</div>"
        "</div>"
        f'<div class="status-pill {risk_class}">{risk_label}</div>'
        "</div>"
        '<div class="availability-grid">'
        '<div class="availability-card">'
        '<div class="availability-label">Current bikes</div>'
        f'<div class="availability-value">{projection.current_bikes:.0f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Predicted pickups</div>'
        f'<div class="availability-value">{projection.predicted_pickups:.1f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Predicted returns</div>'
        f'<div class="availability-value">{projection.predicted_returns:.1f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Projected bikes</div>'
        f'<div class="availability-value">{projection.projected_bikes:.1f}</div>'
        "</div>"
        '<div class="availability-card">'
        '<div class="availability-label">Projected docks</div>'
        f'<div class="availability-value">{projection.projected_docks:.1f}</div>'
        "</div>"
        "</div>"
        '<div class="availability-note">'
        "Formula: projected bikes = current bikes + predicted returns - predicted pickups. "
        "Projected docks move in the opposite direction. This is the core logic "
        "for the future live availability-risk model."
        "</div>"
        "</div>"
    )

    st.markdown(projection_html, unsafe_allow_html=True)

def render_forecast_explorer_tab(
    predictions: pd.DataFrame,
    live_availability: pd.DataFrame,
    flow_features: pd.DataFrame,
    pickup_flow_model,
    return_flow_model,
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

    render_live_availability_panel(
        live_availability,
        selected_station_id,
        selected_row["station_name"],
    )
    
    render_availability_projection_panel(
        live_availability=live_availability,
        flow_features=flow_features,
        pickup_flow_model=pickup_flow_model,
        return_flow_model=return_flow_model,
        selected_station_id=selected_station_id,
        selected_station_name=selected_row["station_name"],
        selected_timestamp=selected_row["timestamp"],
    )

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

def render_live_network_tab(live_availability: pd.DataFrame) -> None:
    """Render live Citi Bike availability across the full station network."""
    st.markdown(
        '<div class="section-label">Live Availability Network</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="section-subcopy">'
            "A real-time view of station inventory across the Citi Bike network. "
            "This tab shows current bike and dock availability, separate from the "
            "historical forecast explorer."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if live_availability.empty:
        st.info("Live availability data is not available right now.")
        return
    
    render_availability_source_note(live_availability)

    live_data = live_availability.copy()

    live_data = live_data.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    status_counts = (
        live_data["availability_status"]
        .value_counts()
        .reset_index()
    )
    status_counts.columns = ["availability_status", "station_count"]
    status_counts["status_label"] = status_counts[
        "availability_status"
    ].apply(format_availability_status)

    total_stations = len(live_data)
    empty_or_nearly_empty = live_data[
        live_data["availability_status"].isin(
            ["empty", "nearly_empty", "low_bikes"]
        )
    ]
    full_or_nearly_full = live_data[
        live_data["availability_status"].isin(
            ["full", "nearly_full", "low_docks"]
        )
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Live stations", f"{total_stations:,}")
    col2.metric("Healthy", f"{(live_data['availability_status'] == 'healthy').sum():,}")
    col3.metric("Bike shortage risk", f"{len(empty_or_nearly_empty):,}")
    col4.metric("Dock shortage risk", f"{len(full_or_nearly_full):,}")

    st.markdown(
        '<div class="section-label">Network Status Map</div>',
        unsafe_allow_html=True,
    )

    status_color_map = {
        "healthy": "#22C55E",
        "nearly_empty": "#F59E0B",
        "low_bikes": "#F59E0B",
        "empty": "#EF4444",
        "nearly_full": "#A855F7",
        "low_docks": "#A855F7",
        "full": "#EC4899",
        "station_offline": "#64748B",
        "station_not_installed": "#334155",
    }

    live_data["status_label"] = live_data[
        "availability_status"
    ].apply(format_availability_status)

    fig = px.scatter_mapbox(
        live_data,
        lat="latitude",
        lon="longitude",
        color="availability_status",
        size="capacity",
        hover_name="station_name",
        hover_data={
            "capacity": True,
            "num_bikes_available": True,
            "num_ebikes_available": True,
            "num_docks_available": True,
            "pct_bikes_available": ":.0%",
            "pct_docks_available": ":.0%",
            "availability_status": True,
            "latitude": False,
            "longitude": False,
        },
        color_discrete_map=status_color_map,
        zoom=10.5,
        height=650,
        title="Current Citi Bike Station Availability",
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=55, b=0),
        legend_title_text="Availability status",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-label">Network Risk Tables</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Bike shortage risk**")
        bike_risk = (
            empty_or_nearly_empty[
                [
                    "station_name",
                    "capacity",
                    "num_bikes_available",
                    "num_ebikes_available",
                    "num_docks_available",
                    "pct_bikes_available",
                    "availability_status",
                ]
            ]
            .sort_values(
                ["num_bikes_available", "pct_bikes_available"],
                ascending=True,
            )
            .head(20)
            .copy()
        )
        bike_risk["availability_status"] = bike_risk[
            "availability_status"
        ].apply(format_availability_status)

        st.dataframe(
            bike_risk,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.write("**Dock shortage risk**")
        dock_risk = (
            full_or_nearly_full[
                [
                    "station_name",
                    "capacity",
                    "num_bikes_available",
                    "num_ebikes_available",
                    "num_docks_available",
                    "pct_docks_available",
                    "availability_status",
                ]
            ]
            .sort_values(
                ["num_docks_available", "pct_docks_available"],
                ascending=True,
            )
            .head(20)
            .copy()
        )
        dock_risk["availability_status"] = dock_risk[
            "availability_status"
        ].apply(format_availability_status)

        st.dataframe(
            dock_risk,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        '<div class="section-label">Status Distribution</div>',
        unsafe_allow_html=True,
    )

    fig = px.bar(
        status_counts.sort_values("station_count", ascending=True),
        x="station_count",
        y="status_label",
        orientation="h",
        title="Live Station Count by Availability Status",
        labels={
            "station_count": "Station count",
            "status_label": "",
        },
        text="station_count",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=40, t=55, b=10),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

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
    apply_product_polish_css()

    metadata = load_metadata()
    render_hero(metadata)
    features = load_features()
    station_metadata = load_station_metadata()
    predictions = add_predictions(features, metadata)
    flow_features = load_flow_features()
    pickup_flow_model, return_flow_model = load_flow_models()

    if st.sidebar.button("Refresh live availability"):
        load_live_availability.clear()

    try:
        live_availability = load_live_availability()
    except Exception as error:
        st.sidebar.warning(f"Live availability unavailable: {error}")
        live_availability = pd.DataFrame()

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
            "Live Network",
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
            live_availability,
            flow_features,
            pickup_flow_model,
            return_flow_model,
            selected_station_label,
            selected_station_id,
            selected_date,
            selected_hour,
        )

    with tabs[2]:
        render_live_network_tab(live_availability)

    with tabs[3]:
        render_station_map_tab(
            predictions,
            station_metadata,
            selected_date,
            selected_hour,
        )

    with tabs[4]:
        render_model_performance_tab()

    with tabs[5]:
        render_demand_patterns_tab(predictions)

    with tabs[6]:
        render_about_tab(metadata)


if __name__ == "__main__":
    main()