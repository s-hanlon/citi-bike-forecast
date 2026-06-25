from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLOW_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_station_flows.parquet"
MODEL_FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "flow_model_features.parquet"


EXOGENOUS_COLUMNS = [
    "timestamp",
    "station_id",
    "station_name",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_code",
    "is_precipitating",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_us_federal_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_window",
    "days_to_nearest_holiday",
]


def add_lag_features(
    flows: pd.DataFrame,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Add lag and rolling features for a station-hour flow column."""
    flows = flows.sort_values(["station_id", "timestamp"]).copy()

    group = flows.groupby("station_id")[value_column]

    flows[f"{prefix}_lag_24"] = group.shift(24)
    flows[f"{prefix}_lag_48"] = group.shift(48)
    flows[f"{prefix}_lag_168"] = group.shift(168)
    flows[f"{prefix}_lag_336"] = group.shift(336)

    flows[f"{prefix}_rolling_mean_168"] = group.transform(
        lambda series: series.shift(1).rolling(168).mean()
    )
    flows[f"{prefix}_rolling_std_168"] = group.transform(
        lambda series: series.shift(1).rolling(168).std()
    )

    return flows


def build_flow_features() -> pd.DataFrame:
    """Build model-ready features for pickup and return forecasting."""
    flows = pd.read_parquet(FLOW_FILE)
    model_features = pd.read_parquet(MODEL_FEATURE_FILE)

    flows["station_id"] = flows["station_id"].astype(str)
    model_features["station_id"] = model_features["station_id"].astype(str)

    flows["timestamp"] = pd.to_datetime(flows["timestamp"])
    model_features["timestamp"] = pd.to_datetime(model_features["timestamp"])

    flows = add_lag_features(
        flows,
        value_column="pickups",
        prefix="pickup",
    )
    flows = add_lag_features(
        flows,
        value_column="returns",
        prefix="return",
    )
    flows = add_lag_features(
        flows,
        value_column="net_outflow",
        prefix="net_outflow",
    )

    exogenous_features = model_features[EXOGENOUS_COLUMNS].copy()

    flow_features = flows.merge(
        exogenous_features,
        on=["timestamp", "station_id", "station_name"],
        how="inner",
    )

    flow_features = flow_features.dropna().reset_index(drop=True)

    return flow_features


def main() -> None:
    """Build and save flow model features."""
    flow_features = build_flow_features()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    flow_features.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(flow_features):,} flow feature rows to {OUTPUT_FILE}")
    print()
    print("Date range:")
    print(
        flow_features["timestamp"].min(),
        "through",
        flow_features["timestamp"].max(),
    )
    print()
    print("Columns:")
    print(len(flow_features.columns))
    print()
    print("Missing values:")
    print(flow_features.isna().sum().sum())
    print()
    print("Targets:")
    print(f"Pickups total: {flow_features['pickups'].sum():,}")
    print(f"Returns total: {flow_features['returns'].sum():,}")
    print(f"Net outflow total: {flow_features['net_outflow'].sum():,}")
    print()
    print("Sample:")
    print(flow_features.head(10).to_string(index=False))


if __name__ == "__main__":
    main()