import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVED_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_hourly_weather_observed_202501_202605.json"
)

FORECAST_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_hourly_weather_forecast_day1_202501_202605.json"
)

OBSERVED_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_weather.parquet"
)

FORECAST_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_weather_forecast_day1.parquet"
)

OBSERVED_COLUMN_NAMES = {
    "time": "timestamp",
    "temperature_2m": "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_kmh",
    "weather_code": "weather_code",
}

FORECAST_COLUMN_NAMES = {
    "time": "timestamp",
    "temperature_2m_previous_day1": "temperature_c",
    "apparent_temperature_previous_day1": (
        "apparent_temperature_c"
    ),
    "relative_humidity_2m_previous_day1": (
        "relative_humidity_pct"
    ),
    "precipitation_previous_day1": "precipitation_mm",
    "wind_speed_10m_previous_day1": "wind_speed_kmh",
    "weather_code_previous_day1": "weather_code",
}

EXPECTED_START = pd.Timestamp(
    "2025-01-01 00:00:00"
)

EXPECTED_END = pd.Timestamp(
    "2026-05-31 23:00:00"
)


def load_raw_weather(
    input_file: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load hourly values and units from a raw weather response."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Weather file not found: {input_file}\n"
            "Run src/data/download_weather.py first."
        )

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "hourly" not in data:
        raise ValueError(
            f"{input_file.name} does not contain hourly data."
        )

    weather = pd.DataFrame(
        data["hourly"]
    )

    units = data.get(
        "hourly_units",
        {},
    )

    return weather, units


def clean_weather(
    weather: pd.DataFrame,
    column_names: dict[str, str],
) -> pd.DataFrame:
    """Select, rename, and validate weather columns."""

    required_columns = set(
        column_names
    )

    missing_columns = (
        required_columns
        - set(weather.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing weather columns: "
            f"{sorted(missing_columns)}"
        )

    weather = weather[
        list(column_names)
    ].copy()

    weather = weather.rename(
        columns=column_names,
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="raise",
    )

    weather = (
        weather.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if weather["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found in weather data."
        )

    missing_values = weather.isna().sum()

    if missing_values.any():
        raise ValueError(
            "Missing values found in weather data:\n"
            f"{missing_values[missing_values > 0]}"
        )

    return weather


def validate_weather(
    weather: pd.DataFrame,
) -> None:
    """Confirm that every expected timestamp is present."""

    expected_timestamps = pd.date_range(
        start=EXPECTED_START,
        end=EXPECTED_END,
        freq="h",
    )

    actual_timestamps = pd.DatetimeIndex(
        weather["timestamp"]
    )

    missing_timestamps = (
        expected_timestamps.difference(
            actual_timestamps
        )
    )

    unexpected_timestamps = (
        actual_timestamps.difference(
            expected_timestamps
        )
    )

    if len(missing_timestamps) > 0:
        raise ValueError(
            f"Missing {len(missing_timestamps)} "
            "expected timestamps."
        )

    if len(unexpected_timestamps) > 0:
        raise ValueError(
            f"Found {len(unexpected_timestamps)} "
            "unexpected timestamps."
        )

    if len(weather) != len(expected_timestamps):
        raise ValueError(
            "Weather row count does not match "
            "the expected hourly range."
        )


def save_weather(
    weather: pd.DataFrame,
    output_file: Path,
    label: str,
) -> None:
    """Save one cleaned weather dataset."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_parquet(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(weather):,} {label} rows to:"
    )
    print(output_file)
    print(
        f"Range: {weather['timestamp'].min()} through "
        f"{weather['timestamp'].max()}"
    )
    print(
        f"Missing values: "
        f"{weather.isna().sum().sum()}\n"
    )


def process_weather_source(
    input_file: Path,
    output_file: Path,
    column_names: dict[str, str],
    label: str,
) -> None:
    """Run the complete cleaning pipeline for one source."""

    raw_weather, _ = load_raw_weather(
        input_file
    )

    weather = clean_weather(
        raw_weather,
        column_names,
    )

    validate_weather(
        weather
    )

    save_weather(
        weather,
        output_file,
        label,
    )


def main() -> None:
    process_weather_source(
        OBSERVED_INPUT_FILE,
        OBSERVED_OUTPUT_FILE,
        OBSERVED_COLUMN_NAMES,
        "observed-weather",
    )

    process_weather_source(
        FORECAST_INPUT_FILE,
        FORECAST_OUTPUT_FILE,
        FORECAST_COLUMN_NAMES,
        "day-ahead forecast",
    )


if __name__ == "__main__":
    main()