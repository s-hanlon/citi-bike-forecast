import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nyc_hourly_weather_202601_202604.json"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "start_date": "2026-01-01",
    "end_date": "2026-04-30",
    "hourly": [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    ],
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    "timezone": "America/New_York",
}


def download_weather() -> dict:
    """Download hourly historical weather for New York City."""

    print("Requesting hourly NYC weather from Open-Meteo...")

    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "The API response does not contain hourly weather data."
        )

    return data


def save_weather(data: dict) -> None:
    """Save the unmodified API response as a raw JSON file."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    hourly_rows = len(data["hourly"]["time"])

    print(f"Saved {hourly_rows:,} hourly records to:")
    print(OUTPUT_FILE)
    print(
        f"Weather range: "
        f"{data['hourly']['time'][0]} through "
        f"{data['hourly']['time'][-1]}"
    )


def main() -> None:
    data = download_weather()
    save_weather(data)


if __name__ == "__main__":
    main()