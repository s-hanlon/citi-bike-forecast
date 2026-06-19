from pathlib import Path

import requests


DATA_URL = "https://s3.amazonaws.com/tripdata/202604-citibike-tripdata.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DATA_DIR / "202604-citibike-tripdata.zip"

def download_trip_data():
    """Download one month of Citi Bike trip data into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"File already exists: {OUTPUT_PATH}")
        return

    print(f"Downloading data from {DATA_URL}")

    with requests.get(DATA_URL, stream=True, timeout=60) as response:
        response.raise_for_status()

        with OUTPUT_PATH.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"Download complete: {OUTPUT_PATH}")


if __name__ == "__main__":
    download_trip_data()