from __future__ import annotations

import json
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OBSERVED_OUTPUT_FILE = (
    RAW_DATA_DIR / "nyc_hourly_weather_observed_202501_202605.json"
)

FORECAST_OUTPUT_FILE = (
    RAW_DATA_DIR / "nyc_hourly_weather_forecast_day1_202501_202605.json"
)

OBSERVED_API_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

LATITUDE = 40.7128
LONGITUDE = -74.0060
TIMEZONE = "America/New_York"

START_MONTH = "202501"
END_MONTH = "202605"

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


def generate_months(start_month: str, end_month: str) -> list[str]:
    """Generate YYYYMM strings from start_month through end_month, inclusive."""
    start_year = int(start_month[:4])
    start_month_number = int(start_month[4:])

    end_year = int(end_month[:4])
    end_month_number = int(end_month[4:])

    months = []

    year = start_year
    month = start_month_number

    while (year, month) <= (end_year, end_month_number):
        months.append(f"{year}{month:02d}")

        month += 1

        if month == 13:
            month = 1
            year += 1

    return months


def month_date_range(month_string: str) -> tuple[str, str]:
    """Return start and end dates for a YYYYMM month."""
    year = int(month_string[:4])
    month = int(month_string[4:])

    start_date = f"{year}-{month:02d}-01"

    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    # Use pandas-free date math to get the final day of the current month.
    from datetime import date, timedelta

    next_month_start = date(next_year, next_month, 1)
    end_date = next_month_start - timedelta(days=1)

    return start_date, end_date.isoformat()


def download_weather_chunk(
    api_url: str,
    variables: list[str],
    start_date: str,
    end_date: str,
    description: str,
) -> dict:
    """Download one weather chunk from Open-Meteo."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    weather = response.json()

    if "hourly" not in weather:
        raise ValueError(
            f"Missing hourly weather data for {description} "
            f"from {start_date} through {end_date}."
        )

    return weather


def merge_weather_chunks(chunks: list[dict]) -> dict:
    """Merge monthly Open-Meteo responses into one response-shaped dictionary."""
    if not chunks:
        raise ValueError("No weather chunks were downloaded.")

    merged = {
        key: value
        for key, value in chunks[0].items()
        if key != "hourly"
    }

    merged["hourly"] = {}

    hourly_keys = chunks[0]["hourly"].keys()

    for key in hourly_keys:
        merged["hourly"][key] = []

    for chunk in chunks:
        for key in hourly_keys:
            merged["hourly"][key].extend(chunk["hourly"][key])

    return merged


def download_weather_monthly(
    api_url: str,
    variables: list[str],
    description: str,
) -> dict:
    """Download weather month by month and merge the results."""
    chunks = []

    for month_string in generate_months(START_MONTH, END_MONTH):
        start_date, end_date = month_date_range(month_string)

        print(
            f"Requesting {description} for "
            f"{start_date} through {end_date}..."
        )

        chunk = download_weather_chunk(
            api_url=api_url,
            variables=variables,
            start_date=start_date,
            end_date=end_date,
            description=description,
        )

        chunks.append(chunk)

        # Be polite to the API and reduce timeout/rate-limit risk.
        time.sleep(0.5)

    return merge_weather_chunks(chunks)


def save_weather(weather: dict, output_file: Path, description: str) -> None:
    """Save raw weather JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(weather, file)

    times = weather["hourly"]["time"]

    print(f"Saved {len(times):,} {description} records to:")
    print(output_file)
    print(f"Range: {times[0]} through {times[-1]}")
    print()


def main() -> None:
    observed_weather = download_weather_monthly(
        OBSERVED_API_URL,
        OBSERVED_VARIABLES,
        "observed weather",
    )

    save_weather(
        observed_weather,
        OBSERVED_OUTPUT_FILE,
        "observed weather",
    )

    forecast_weather = download_weather_monthly(
        PREVIOUS_RUNS_API_URL,
        DAY_AHEAD_FORECAST_VARIABLES,
        "24-hour-ahead weather forecasts",
    )

    save_weather(
        forecast_weather,
        FORECAST_OUTPUT_FILE,
        "24-hour-ahead weather forecast",
    )


if __name__ == "__main__":
    main()