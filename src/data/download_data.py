from pathlib import Path

import requests


BASE_URL = "https://s3.amazonaws.com/tripdata"

MONTHS = (
    "202601",
    "202602",
    "202603",
    "202604",
    "202605",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def download_month(month: str) -> None:
    """Download one month of Citi Bike trip data."""
    filename = f"{month}-citibike-tripdata.zip"
    url = f"{BASE_URL}/{filename}"

    output_path = RAW_DATA_DIR / filename
    temporary_path = output_path.with_suffix(
        ".zip.part"
    )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Already downloaded: {filename}")
        return

    print(f"Downloading: {filename}")

    with requests.get(
        url,
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        total_bytes = int(
            response.headers.get("content-length", 0)
        )
        downloaded_bytes = 0
        next_report = 10

        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded_bytes += len(chunk)

                if total_bytes:
                    percent = int(
                        downloaded_bytes
                        / total_bytes
                        * 100
                    )

                    if percent >= next_report:
                        print(f"{month}: {percent}%")
                        next_report += 10

    temporary_path.replace(output_path)
    print(f"Completed: {filename}")


def main() -> None:
    """Download every configured month."""
    for month in MONTHS:
        download_month(month)


if __name__ == "__main__":
    main()