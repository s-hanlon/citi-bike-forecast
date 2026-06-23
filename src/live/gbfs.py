from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests


STATION_INFORMATION_URL = (
    "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
)
STATION_STATUS_URL = (
    "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
)

REQUEST_TIMEOUT_SECONDS = 20


def fetch_json(url: str) -> dict:
    """Fetch a JSON response from a GBFS endpoint."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_station_information() -> pd.DataFrame:
    """Fetch Citi Bike station information from the GBFS feed."""
    payload = fetch_json(STATION_INFORMATION_URL)
    stations = payload["data"]["stations"]

    station_info = pd.DataFrame(stations)

    expected_columns = [
        "station_id",
        "name",
        "lat",
        "lon",
        "capacity",
    ]

    for column in expected_columns:
        if column not in station_info.columns:
            station_info[column] = pd.NA

    station_info = station_info[expected_columns].copy()

    station_info = station_info.rename(
        columns={
            "name": "station_name",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    return station_info


def extract_ebike_count(row: pd.Series) -> int:
    """Extract e-bike count from GBFS station status when available."""
    if "num_ebikes_available" in row and pd.notna(row["num_ebikes_available"]):
        return int(row["num_ebikes_available"])

    vehicle_types = row.get("vehicle_types_available")

    if not isinstance(vehicle_types, list):
        return 0

    ebike_count = 0

    for vehicle_type in vehicle_types:
        vehicle_type_id = str(vehicle_type.get("vehicle_type_id", "")).lower()
        count = int(vehicle_type.get("count", 0))

        if "ebike" in vehicle_type_id or "electric" in vehicle_type_id:
            ebike_count += count

    return ebike_count


def convert_epoch_seconds(value) -> pd.Timestamp:
    """Convert GBFS epoch seconds to a UTC timestamp."""
    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(value, unit="s", utc=True)


def fetch_station_status() -> pd.DataFrame:
    """Fetch Citi Bike station status from the GBFS feed."""
    payload = fetch_json(STATION_STATUS_URL)
    stations = payload["data"]["stations"]

    station_status = pd.DataFrame(stations)

    expected_columns = [
        "station_id",
        "num_bikes_available",
        "num_docks_available",
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported",
        "vehicle_types_available",
    ]

    for column in expected_columns:
        if column not in station_status.columns:
            station_status[column] = pd.NA

    station_status["num_ebikes_available"] = station_status.apply(
        extract_ebike_count,
        axis=1,
    )

    station_status["last_reported_utc"] = station_status[
        "last_reported"
    ].apply(convert_epoch_seconds)

    fetched_at_utc = datetime.now(timezone.utc)
    station_status["status_fetched_at_utc"] = fetched_at_utc

    keep_columns = [
        "station_id",
        "num_bikes_available",
        "num_ebikes_available",
        "num_docks_available",
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported_utc",
        "status_fetched_at_utc",
    ]

    return station_status[keep_columns].copy()


def classify_availability(row: pd.Series) -> str:
    """Classify station availability risk from current inventory."""
    is_installed = row.get("is_installed", 1)
    is_renting = row.get("is_renting", 1)
    is_returning = row.get("is_returning", 1)

    if is_installed != 1:
        return "station_not_installed"

    if is_renting != 1 and is_returning != 1:
        return "station_offline"

    bikes = row.get("num_bikes_available", 0)
    docks = row.get("num_docks_available", 0)
    capacity = row.get("capacity", pd.NA)

    if pd.isna(bikes):
        bikes = 0

    if pd.isna(docks):
        docks = 0

    if bikes <= 0 and is_renting == 1:
        return "empty"

    if docks <= 0 and is_returning == 1:
        return "full"

    if pd.notna(capacity) and capacity > 0:
        pct_bikes_available = bikes / capacity
        pct_docks_available = docks / capacity

        if pct_bikes_available <= 0.15:
            return "nearly_empty"

        if pct_docks_available <= 0.15:
            return "nearly_full"

    if bikes <= 3:
        return "low_bikes"

    if docks <= 3:
        return "low_docks"

    return "healthy"


def fetch_live_station_availability() -> pd.DataFrame:
    """Fetch and combine live station information and live station status."""
    station_info = fetch_station_information()
    station_status = fetch_station_status()

    availability = station_status.merge(
        station_info,
        on="station_id",
        how="left",
    )

    availability["capacity"] = pd.to_numeric(
        availability["capacity"],
        errors="coerce",
    )

    availability["num_bikes_available"] = pd.to_numeric(
        availability["num_bikes_available"],
        errors="coerce",
    ).fillna(0)

    availability["num_ebikes_available"] = pd.to_numeric(
        availability["num_ebikes_available"],
        errors="coerce",
    ).fillna(0)

    availability["num_docks_available"] = pd.to_numeric(
        availability["num_docks_available"],
        errors="coerce",
    ).fillna(0)

    availability["pct_bikes_available"] = (
        availability["num_bikes_available"] / availability["capacity"]
    )

    availability["pct_docks_available"] = (
        availability["num_docks_available"] / availability["capacity"]
    )

    availability["availability_status"] = availability.apply(
        classify_availability,
        axis=1,
    )

    ordered_columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "capacity",
        "num_bikes_available",
        "num_ebikes_available",
        "num_docks_available",
        "pct_bikes_available",
        "pct_docks_available",
        "availability_status",
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported_utc",
        "status_fetched_at_utc",
    ]

    return availability[ordered_columns].copy()


def main() -> None:
    """Smoke test the GBFS availability fetcher."""
    availability = fetch_live_station_availability()

    print(f"Fetched {len(availability):,} live station records")
    print()
    print(
        availability[
            [
                "station_name",
                "capacity",
                "num_bikes_available",
                "num_ebikes_available",
                "num_docks_available",
                "availability_status",
            ]
        ]
        .sort_values("station_name")
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Availability status counts:")
    print(availability["availability_status"].value_counts().to_string())


if __name__ == "__main__":
    main()