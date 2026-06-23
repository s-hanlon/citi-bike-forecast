from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PICKUPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_pickups.parquet"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_weather_forecast_day1.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

LAGS = [
    24,
    48,
    168,
    336,
]

WEATHER_COLUMNS = [
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_code",
]


def add_weather(
    pickups: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Join one citywide weather observation to each station-hour."""

    features = pickups.merge(
        weather,
        on="timestamp",
        how="left",
        validate="many_to_one",
    )

    missing_weather = features[
        WEATHER_COLUMNS
    ].isna().sum()

    if missing_weather.any():
        raise ValueError(
            "Missing weather values after timestamp join:\n"
            f"{missing_weather[missing_weather > 0]}"
        )

    features["is_precipitating"] = (
        features["precipitation_mm"] > 0
    ).astype("int8")

    return features


def add_holiday_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add US federal holiday calendar features."""

    features = features.copy()

    calendar = USFederalHolidayCalendar()

    start_date = (
        features["timestamp"].min().normalize()
        - pd.Timedelta(days=7)
    )

    end_date = (
        features["timestamp"].max().normalize()
        + pd.Timedelta(days=7)
    )

    holidays = calendar.holidays(
        start=start_date,
        end=end_date,
    )

    normalized_dates = features["timestamp"].dt.normalize()

    features["is_us_federal_holiday"] = (
        normalized_dates.isin(holidays)
    ).astype("int8")

    features["is_day_before_holiday"] = (
        (normalized_dates + pd.Timedelta(days=1))
        .isin(holidays)
    ).astype("int8")

    features["is_day_after_holiday"] = (
        (normalized_dates - pd.Timedelta(days=1))
        .isin(holidays)
    ).astype("int8")

    features["is_holiday_window"] = (
        (
            features["is_us_federal_holiday"]
            | features["is_day_before_holiday"]
            | features["is_day_after_holiday"]
        )
        .astype("int8")
    )

    holiday_values = holidays.values.astype("datetime64[D]")
    date_values = normalized_dates.values.astype("datetime64[D]")

    insertion_points = np.searchsorted(
        holiday_values,
        date_values,
    )

    previous_indices = np.clip(
        insertion_points - 1,
        0,
        len(holiday_values) - 1,
    )

    next_indices = np.clip(
        insertion_points,
        0,
        len(holiday_values) - 1,
    )

    days_since_previous_holiday = (
        date_values - holiday_values[previous_indices]
    ).astype("timedelta64[D]").astype(int)

    days_until_next_holiday = (
        holiday_values[next_indices] - date_values
    ).astype("timedelta64[D]").astype(int)

    features["days_to_nearest_holiday"] = np.minimum(
        np.abs(days_since_previous_holiday),
        np.abs(days_until_next_holiday),
    )

    return features

def build_features(
    pickups: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Create weather, calendar, and historical-demand features."""

    features = add_weather(
        pickups,
        weather,
    )

    features = (
        features.sort_values(
            ["station_id", "timestamp"]
        )
        .reset_index(drop=True)
        .copy()
    )

    features["hour"] = features["timestamp"].dt.hour
    features["day_of_week"] = (
        features["timestamp"].dt.dayofweek
    )
    features["month"] = features["timestamp"].dt.month
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype("int8")

    features = add_holiday_features(features)

    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour"] / 24
    )
    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour"] / 24
    )

    features["day_of_week_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )
    features["day_of_week_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    pickup_groups = features.groupby(
        "station_id",
        sort=False,
    )["pickups"]

    for lag in LAGS:
        features[f"lag_{lag}"] = (
            pickup_groups.shift(lag)
        )

    features["rolling_mean_168"] = (
        pickup_groups.transform(
            lambda station: (
                station.shift(24)
                .rolling(
                    window=168,
                    min_periods=168,
                )
                .mean()
            )
        )
    )

    features["rolling_std_168"] = (
        pickup_groups.transform(
            lambda station: (
                station.shift(24)
                .rolling(
                    window=168,
                    min_periods=168,
                )
                .std()
            )
        )
    )

    required_history = [
        f"lag_{lag}"
        for lag in LAGS
    ] + [
        "rolling_mean_168",
        "rolling_std_168",
    ]

    features = (
        features.dropna(
            subset=required_history
        )
        .reset_index(drop=True)
    )

    return features


def main() -> None:
    """Build and save the model feature table."""

    if not PICKUPS_FILE.exists():
        raise FileNotFoundError(
            f"Pickup data not found: {PICKUPS_FILE}\n"
            "Run src/data/make_dataset.py first."
        )

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Weather data not found: {WEATHER_FILE}\n"
            "Run src/data/make_weather_dataset.py first."
        )

    pickups = pd.read_parquet(
        PICKUPS_FILE,
    )

    weather = pd.read_parquet(
        WEATHER_FILE,
    )

    features = build_features(
        pickups,
        weather,
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved {len(features):,} feature rows "
        f"to {OUTPUT_FILE}"
    )
    print(
        f"Feature range: "
        f"{features['timestamp'].min()} through "
        f"{features['timestamp'].max()}"
    )
    print(
        f"Columns: {len(features.columns)}"
    )
    print(
        f"Missing values: "
        f"{features.isna().sum().sum()}"
    )


if __name__ == "__main__":
    main()