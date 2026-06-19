from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "202604-citibike-tripdata.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_pickups.parquet"
)

COLUMNS = [
    "started_at",
    "start_station_id",
    "start_station_name",
]

def process_csv_part(
    zip_file: ZipFile,
    member_name: str,
) -> pd.DataFrame:
    """Convert one trip CSV into hourly station pickup counts."""
    print(f"Processing {member_name}")

    with zip_file.open(member_name) as csv_file:
        rides = pd.read_csv(
            csv_file,
            usecols=COLUMNS,
            dtype={"start_station_id": "string"},
            parse_dates=["started_at"],
        )

    rides = rides.dropna(
        subset=[
            "started_at",
            "start_station_id",
            "start_station_name",
        ]
    )

    rides["timestamp"] = rides["started_at"].dt.floor("h")

    hourly_pickups = (
        rides.groupby(
            [
                "timestamp",
                "start_station_id",
                "start_station_name",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "pickups"})
    )

    return hourly_pickups

def load_hourly_pickups() -> pd.DataFrame:
    """Load every CSV part and combine duplicate station-hour groups."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_FILE}\n"
            "Run src/data/download_data.py first."
        )

    with ZipFile(RAW_FILE) as zip_file:
        csv_members = [
            name
            for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        ]

        hourly_parts = [
            process_csv_part(zip_file, member_name)
            for member_name in csv_members
        ]

    hourly_pickups = pd.concat(
        hourly_parts,
        ignore_index=True,
    )

    hourly_pickups = (
        hourly_pickups.groupby(
            [
                "timestamp",
                "start_station_id",
                "start_station_name",
            ],
            as_index=False,
        )["pickups"]
        .sum()
    )

    return hourly_pickups

def build_dataset(top_n: int = 25) -> None:
    """Build and save a complete hourly dataset for the busiest stations."""
    hourly_pickups = load_hourly_pickups()

    station_names = (
        hourly_pickups.groupby(
            [
                "start_station_id",
                "start_station_name",
            ],
            as_index=False,
        )["pickups"]
        .sum()
        .sort_values(
            ["start_station_id", "pickups"],
            ascending=[True, False],
        )
        .drop_duplicates("start_station_id")
        .set_index("start_station_id")["start_station_name"]
    )

    station_hourly = (
        hourly_pickups.groupby(
            [
                "timestamp",
                "start_station_id",
            ],
            as_index=False,
        )["pickups"]
        .sum()
    )

    top_station_ids = (
        station_hourly.groupby("start_station_id")["pickups"]
        .sum()
        .nlargest(top_n)
        .index
        .tolist()
    )

    station_hourly = station_hourly[
        station_hourly["start_station_id"].isin(top_station_ids)
    ].rename(columns={"start_station_id": "station_id"})

    all_hours = pd.date_range(
        start=station_hourly["timestamp"].min(),
        end=station_hourly["timestamp"].max(),
        freq="h",
    )

    complete_index = pd.MultiIndex.from_product(
        [all_hours, top_station_ids],
        names=["timestamp", "station_id"],
    )

    dataset = (
        station_hourly.set_index(
            ["timestamp", "station_id"]
        )[["pickups"]]
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    dataset["station_name"] = dataset["station_id"].map(
        station_names
    )

    dataset = dataset[
        [
            "timestamp",
            "station_id",
            "station_name",
            "pickups",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(dataset):,} rows to {OUTPUT_FILE}")
    print(
        f"Dataset covers {dataset['timestamp'].min()} "
        f"through {dataset['timestamp'].max()}"
    )


if __name__ == "__main__":
    build_dataset()