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
    / "station_metadata.parquet"
)

COLUMNS = [
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lng",
]

def build_station_metadata() -> None:
    """Create one coordinate record for each station."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_FILE}\n"
            "Run src/data/download_data.py first."
        )

    metadata_parts = []

    with ZipFile(RAW_FILE) as zip_file:
        csv_members = [
            name
            for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        ]

        for member_name in csv_members:
            print(f"Processing {member_name}")

            with zip_file.open(member_name) as csv_file:
                rides = pd.read_csv(
                    csv_file,
                    usecols=COLUMNS,
                    dtype={"start_station_id": "string"},
                )

            rides = rides.dropna(subset=COLUMNS)

            station_part = (
                rides.groupby(
                    [
                        "start_station_id",
                        "start_station_name",
                    ],
                    as_index=False,
                )
                .agg(
                    latitude_sum=("start_lat", "sum"),
                    longitude_sum=("start_lng", "sum"),
                    observations=("start_lat", "size"),
                )
            )

            metadata_parts.append(station_part)

    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    metadata = (
        metadata.groupby(
            [
                "start_station_id",
                "start_station_name",
            ],
            as_index=False,
        )
        .agg(
            latitude_sum=("latitude_sum", "sum"),
            longitude_sum=("longitude_sum", "sum"),
            observations=("observations", "sum"),
        )
    )

    metadata["latitude"] = (
        metadata["latitude_sum"]
        / metadata["observations"]
    )
    metadata["longitude"] = (
        metadata["longitude_sum"]
        / metadata["observations"]
    )

    metadata = (
        metadata.sort_values(
            ["start_station_id", "observations"],
            ascending=[True, False],
        )
        .drop_duplicates("start_station_id")
        .rename(
            columns={
                "start_station_id": "station_id",
                "start_station_name": "station_name",
            }
        )
    )

    metadata = metadata[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(OUTPUT_FILE, index=False)

    print(
        f"Saved metadata for {len(metadata):,} stations "
        f"to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_station_metadata()