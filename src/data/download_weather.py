import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OBSERVED_OUTPUT_FILE = (
    RAW_DATA_DIR
    / "nyc_hourly_weather_observed_202601_202605.json"
)

FORECAST_OUTPUT_FILE = (
    RAW_DATA_DIR
    / "nyc_hourly_weather_forecast_day1_202601_202605.json"
)

OBSERVED_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

PREVIOUS_RUNS_API_URL = (
    "https://previous-runs-api.open-meteo.com/v1/forecast"
)

BASE_PARAMS = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "start_date": "2026-01-01",
    "end_date": "2026-05-31",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    "timezone": "America/New_York",
}

OBSERVED_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
]

DAY_AHEAD_FORECAST_VARIABLES = [
    "temperature_2m_previous_day1",
    "apparent_temperature_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
    "weather_code_previous_day1",
]


def download_weather(
    api_url: str,
    variables: list[str],
    label: str,
) -> dict:
    """Download one hourly weather dataset."""

    params = {
        **BASE_PARAMS,
        "hourly": ",".join(variables),
    }

    print(f"Requesting {label}...")

    response = requests.get(
        api_url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"The {label} response does not contain hourly data."
        )

    return data


def save_weather(
    data: dict,
    output_file: Path,
    label: str,
) -> None:
    """Save an unmodified weather API response."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    times = data["hourly"]["time"]

    print(f"Saved {len(times):,} {label} records to:")
    print(output_file)
    print(
        f"Range: {times[0]} through {times[-1]}\n"
    )


def main() -> None:
    observed_weather = download_weather(
        OBSERVED_API_URL,
        OBSERVED_VARIABLES,
        "observed weather",
    )

    save_weather(
        observed_weather,
        OBSERVED_OUTPUT_FILE,
        "observed weather",
    )

    forecast_weather = download_weather(
        PREVIOUS_RUNS_API_URL,
        DAY_AHEAD_FORECAST_VARIABLES,
        "24-hour-ahead weather forecasts",
    )

    save_weather(
        forecast_weather,
        FORECAST_OUTPUT_FILE,
        "24-hour-ahead forecast",
    )


if __name__ == "__main__":
    main()