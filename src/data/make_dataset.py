from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

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
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
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

    rides = rides[
        (rides["started_at"] >= month_start)
        & (rides["started_at"] < month_end)
    ].copy()

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
    """Load every monthly archive and combine station-hour counts."""
    raw_files = sorted(
        RAW_DATA_DIR.glob(
            "20????-citibike-tripdata.zip"
        )
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No monthly archives found in {RAW_DATA_DIR}\n"
            "Run src/data/download_data.py first."
        )

    hourly_parts = []

    for raw_file in raw_files:
        print(f"\nOpening {raw_file.name}")

        archive_month = raw_file.name[:6]

        month_start = pd.to_datetime(
            archive_month,
            format="%Y%m",
        )

        month_end = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        with ZipFile(raw_file) as zip_file:
            csv_members = [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".csv")
            ]

            for member_name in csv_members:
                hourly_parts.append(
                    process_csv_part(
                        zip_file,
                        member_name,
                        month_start,
                        month_end,
                    )
                )

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
        .set_index("start_station_id")[
            "start_station_name"
        ]
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
        station_hourly.groupby(
            "start_station_id"
        )["pickups"]
        .sum()
        .nlargest(top_n)
        .index
        .tolist()
    )

    station_hourly = station_hourly[
        station_hourly[
            "start_station_id"
        ].isin(top_station_ids)
    ].rename(
        columns={
            "start_station_id": "station_id"
        }
    )

    all_hours = pd.date_range(
        start=month_floor(
            station_hourly["timestamp"].min()
        ),
        end=month_ceiling(
            station_hourly["timestamp"].max()
        ),
        freq="h",
    )

    complete_index = pd.MultiIndex.from_product(
        [
            all_hours,
            top_station_ids,
        ],
        names=[
            "timestamp",
            "station_id",
        ],
    )

    dataset = (
        station_hourly.set_index(
            [
                "timestamp",
                "station_id",
            ]
        )[["pickups"]]
        .reindex(
            complete_index,
            fill_value=0,
        )
        .reset_index()
    )

    dataset["station_name"] = (
        dataset["station_id"].map(
            station_names
        )
    )

    dataset["pickups"] = (
        dataset["pickups"].astype("int32")
    )

    dataset = dataset[
        [
            "timestamp",
            "station_id",
            "station_name",
            "pickups",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved {len(dataset):,} rows "
        f"to {OUTPUT_FILE}"
    )
    print(
        f"Dataset covers "
        f"{dataset['timestamp'].min()} through "
        f"{dataset['timestamp'].max()}"
    )


def month_floor(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the first hour of a timestamp's month."""
    return timestamp.to_period("M").start_time


def month_ceiling(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the final hour of a timestamp's month."""
    return (
        timestamp.to_period("M").end_time.floor("h")
    )


if __name__ == "__main__":
    build_dataset()