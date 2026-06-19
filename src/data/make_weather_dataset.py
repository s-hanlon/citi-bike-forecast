import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_hourly_weather_202601_202604.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_weather.parquet"
)

COLUMN_NAMES = {
    "time": "timestamp",
    "temperature_2m": "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_kmh",
}


def load_raw_weather() -> tuple[pd.DataFrame, dict]:
    """Load hourly weather values and their units from raw JSON."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Weather file not found: {INPUT_FILE}\n"
            "Run src/data/download_weather.py first."
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "hourly" not in data:
        raise ValueError(
            "Raw weather data does not contain an 'hourly' section."
        )

    weather = pd.DataFrame(data["hourly"])
    units = data.get("hourly_units", {})

    return weather, units


def clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the hourly weather table."""

    required_columns = set(COLUMN_NAMES) | {"weather_code"}

    missing_columns = required_columns - set(weather.columns)

    if missing_columns:
        raise ValueError(
            f"Missing weather columns: {sorted(missing_columns)}"
        )

    weather = weather[
        list(COLUMN_NAMES) + ["weather_code"]
    ].copy()

    weather = weather.rename(
        columns=COLUMN_NAMES,
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="raise",
    )

    weather = weather.sort_values(
        "timestamp",
    ).reset_index(drop=True)

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


def validate_weather(weather: pd.DataFrame) -> None:
    """Confirm that every expected hourly timestamp is present."""

    expected_timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        end="2026-04-30 23:00:00",
        freq="h",
    )

    actual_timestamps = pd.DatetimeIndex(
        weather["timestamp"],
    )

    missing_timestamps = expected_timestamps.difference(
        actual_timestamps,
    )

    unexpected_timestamps = actual_timestamps.difference(
        expected_timestamps,
    )

    if len(missing_timestamps) > 0:
        raise ValueError(
            f"Missing {len(missing_timestamps)} expected timestamps."
        )

    if len(unexpected_timestamps) > 0:
        raise ValueError(
            f"Found {len(unexpected_timestamps)} unexpected timestamps."
        )

    if len(weather) != len(expected_timestamps):
        raise ValueError(
            "Weather row count does not match the expected hourly range."
        )


def save_weather(
    weather: pd.DataFrame,
    units: dict,
) -> None:
    """Save the cleaned weather dataset."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved {len(weather):,} weather rows to:")
    print(OUTPUT_FILE)
    print(
        f"Weather range: "
        f"{weather['timestamp'].min()} through "
        f"{weather['timestamp'].max()}"
    )
    print(f"Missing values: {weather.isna().sum().sum()}")

    print("\nSource units:")

    for source_column, output_column in COLUMN_NAMES.items():
        if source_column == "time":
            continue

        print(
            f"  {output_column}: "
            f"{units.get(source_column, 'unknown')}"
        )


def main() -> None:
    weather, units = load_raw_weather()
    weather = clean_weather(weather)
    validate_weather(weather)
    save_weather(weather, units)


if __name__ == "__main__":
    main()