from pathlib import Path

import requests


BASE_URL = "https://s3.amazonaws.com/tripdata"

START_MONTH = "202501"
END_MONTH = "202605"


def generate_months(start_month: str, end_month: str) -> tuple[str, ...]:
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

    return tuple(months)


MONTHS = generate_months(START_MONTH, END_MONTH)

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