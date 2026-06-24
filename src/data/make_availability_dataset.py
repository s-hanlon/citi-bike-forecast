from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gbfs_station_status"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_station_availability.parquet"


def load_snapshots() -> pd.DataFrame:
    """Load all raw GBFS station status snapshots."""
    snapshot_files = sorted(INPUT_DIR.glob("date=*/station_status_*.parquet"))

    if not snapshot_files:
        raise FileNotFoundError(
            f"No GBFS station status snapshots found under {INPUT_DIR}"
        )

    snapshots = []

    for file_path in snapshot_files:
        snapshot = pd.read_parquet(file_path)
        snapshots.append(snapshot)

    availability = pd.concat(snapshots, ignore_index=True)

    return availability


def build_hourly_availability(availability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw station status snapshots to station-hour availability."""
    availability = availability.copy()

    availability["snapshot_utc"] = pd.to_datetime(
        availability["snapshot_utc"],
        utc=True,
    )

    availability["snapshot_hour_utc"] = availability[
        "snapshot_utc"
    ].dt.floor("h")

    numeric_columns = [
        "capacity",
        "num_bikes_available",
        "num_ebikes_available",
        "num_docks_available",
        "pct_bikes_available",
        "pct_docks_available",
        "is_installed",
        "is_renting",
        "is_returning",
    ]

    for column in numeric_columns:
        if column in availability.columns:
            availability[column] = pd.to_numeric(
                availability[column],
                errors="coerce",
            )

    hourly = (
        availability.groupby(
            [
                "snapshot_hour_utc",
                "station_id",
                "station_name",
                "latitude",
                "longitude",
            ],
            dropna=False,
        )
        .agg(
            capacity=("capacity", "max"),
            avg_bikes_available=("num_bikes_available", "mean"),
            min_bikes_available=("num_bikes_available", "min"),
            max_bikes_available=("num_bikes_available", "max"),
            avg_ebikes_available=("num_ebikes_available", "mean"),
            avg_docks_available=("num_docks_available", "mean"),
            min_docks_available=("num_docks_available", "min"),
            max_docks_available=("num_docks_available", "max"),
            avg_pct_bikes_available=("pct_bikes_available", "mean"),
            avg_pct_docks_available=("pct_docks_available", "mean"),
            snapshots_in_hour=("snapshot_utc", "count"),
            is_installed=("is_installed", "max"),
            is_renting=("is_renting", "max"),
            is_returning=("is_returning", "max"),
        )
        .reset_index()
    )

    hourly["is_empty"] = hourly["min_bikes_available"] <= 0
    hourly["is_full"] = hourly["min_docks_available"] <= 0
    hourly["is_nearly_empty"] = hourly["avg_pct_bikes_available"] <= 0.15
    hourly["is_nearly_full"] = hourly["avg_pct_docks_available"] <= 0.15

    hourly["availability_risk"] = "healthy"

    hourly.loc[hourly["is_nearly_empty"], "availability_risk"] = "nearly_empty"
    hourly.loc[hourly["is_nearly_full"], "availability_risk"] = "nearly_full"
    hourly.loc[hourly["is_empty"], "availability_risk"] = "empty"
    hourly.loc[hourly["is_full"], "availability_risk"] = "full"

    return hourly


def main() -> None:
    """Build an hourly availability dataset from raw GBFS snapshots."""
    availability = load_snapshots()
    hourly = build_hourly_availability(availability)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(OUTPUT_FILE, index=False)

    print(f"Loaded {len(availability):,} raw station-snapshot rows")
    print(f"Saved {len(hourly):,} hourly station availability rows to {OUTPUT_FILE}")
    print()
    print("Snapshot hour range:")
    print(hourly["snapshot_hour_utc"].min(), "through", hourly["snapshot_hour_utc"].max())
    print()
    print("Availability risk counts:")
    print(hourly["availability_risk"].value_counts().to_string())


if __name__ == "__main__":
    main()