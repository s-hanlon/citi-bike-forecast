from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
HOURLY_PICKUPS_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_pickups.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_station_flows.parquet"

TRIP_ZIP_PATTERN = "*citibike-tripdata.zip"


def load_selected_stations() -> pd.DataFrame:
    """Load the selected stations from the existing hourly pickup dataset."""
    pickups = pd.read_parquet(HOURLY_PICKUPS_FILE)

    stations = (
        pickups[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_name")
        .reset_index(drop=True)
    )

    stations["station_id"] = stations["station_id"].astype(str)

    return stations


def list_trip_zip_files() -> list[Path]:
    """List Citi Bike monthly trip ZIP files."""
    zip_files = sorted(RAW_DATA_DIR.glob(TRIP_ZIP_PATTERN))

    if not zip_files:
        raise FileNotFoundError(f"No trip ZIP files found in {RAW_DATA_DIR}")

    return zip_files


def read_trip_csv_from_zip(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """Read only the columns needed for station flow aggregation."""
    columns = [
        "started_at",
        "ended_at",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
    ]

    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(csv_name) as csv_file:
            trips = pd.read_csv(
                csv_file,
                usecols=columns,
                dtype={
                    "start_station_id": "string",
                    "start_station_name": "string",
                    "end_station_id": "string",
                    "end_station_name": "string",
                },
            )

    return trips


def aggregate_trip_file(
    trips: pd.DataFrame,
    selected_station_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate pickups and returns for one trip file."""
    trips = trips.copy()

    trips["started_at"] = pd.to_datetime(
        trips["started_at"],
        errors="coerce",
    )
    trips["ended_at"] = pd.to_datetime(
        trips["ended_at"],
        errors="coerce",
    )

    trips["start_station_id"] = trips["start_station_id"].astype(str)
    trips["end_station_id"] = trips["end_station_id"].astype(str)

    pickup_trips = trips[
        trips["start_station_id"].isin(selected_station_ids)
        & trips["started_at"].notna()
    ].copy()

    return_trips = trips[
        trips["end_station_id"].isin(selected_station_ids)
        & trips["ended_at"].notna()
    ].copy()

    pickup_trips["timestamp"] = pickup_trips["started_at"].dt.floor("h")
    return_trips["timestamp"] = return_trips["ended_at"].dt.floor("h")

    pickups = (
        pickup_trips.groupby(["timestamp", "start_station_id"])
        .size()
        .reset_index(name="pickups")
        .rename(columns={"start_station_id": "station_id"})
    )

    returns = (
        return_trips.groupby(["timestamp", "end_station_id"])
        .size()
        .reset_index(name="returns")
        .rename(columns={"end_station_id": "station_id"})
    )

    return pickups, returns


def build_raw_flow_counts(
    selected_station_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw pickup and return counts from all monthly trip ZIP files."""
    pickup_parts = []
    return_parts = []

    zip_files = list_trip_zip_files()

    for zip_path in zip_files:
        print(f"Processing {zip_path.name}")

        with zipfile.ZipFile(zip_path) as zip_file:
            csv_names = [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".csv")
            ]

        for csv_name in csv_names:
            print(f"  Reading {csv_name}")
            trips = read_trip_csv_from_zip(zip_path, csv_name)

            pickups, returns = aggregate_trip_file(
                trips,
                selected_station_ids,
            )

            pickup_parts.append(pickups)
            return_parts.append(returns)

    all_pickups = pd.concat(pickup_parts, ignore_index=True)
    all_returns = pd.concat(return_parts, ignore_index=True)

    all_pickups = (
        all_pickups.groupby(["timestamp", "station_id"], as_index=False)
        .agg(pickups=("pickups", "sum"))
    )

    all_returns = (
        all_returns.groupby(["timestamp", "station_id"], as_index=False)
        .agg(returns=("returns", "sum"))
    )

    return all_pickups, all_returns


def build_complete_station_hour_grid(
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Build a complete station-hour grid matching the pickup dataset range."""
    hourly_pickups = pd.read_parquet(HOURLY_PICKUPS_FILE)

    start_time = hourly_pickups["timestamp"].min()
    end_time = hourly_pickups["timestamp"].max()

    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq="h",
    )

    grid = (
        pd.MultiIndex.from_product(
            [timestamps, stations["station_id"]],
            names=["timestamp", "station_id"],
        )
        .to_frame(index=False)
        .merge(stations, on="station_id", how="left")
    )

    return grid


def build_station_flows() -> pd.DataFrame:
    """Build hourly station pickup and return flow dataset."""
    stations = load_selected_stations()
    selected_station_ids = set(stations["station_id"])

    pickups, returns = build_raw_flow_counts(selected_station_ids)
    grid = build_complete_station_hour_grid(stations)

    flows = (
        grid.merge(pickups, on=["timestamp", "station_id"], how="left")
        .merge(returns, on=["timestamp", "station_id"], how="left")
    )

    flows["pickups"] = flows["pickups"].fillna(0).astype(int)
    flows["returns"] = flows["returns"].fillna(0).astype(int)

    flows["net_outflow"] = flows["pickups"] - flows["returns"]
    flows["net_inflow"] = flows["returns"] - flows["pickups"]

    return flows


def main() -> None:
    """Build hourly station flow dataset."""
    flows = build_station_flows()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(flows):,} station-hour flow rows to {OUTPUT_FILE}")
    print()
    print("Date range:")
    print(flows["timestamp"].min(), "through", flows["timestamp"].max())
    print()
    print("Totals:")
    print(f"Pickups: {flows['pickups'].sum():,}")
    print(f"Returns: {flows['returns'].sum():,}")
    print(f"Net outflow sum: {flows['net_outflow'].sum():,}")
    print()
    print("Sample:")
    print(flows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()