from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.database.postgres import connect
from src.live.gbfs import fetch_live_station_availability


def clean_value(value):
    """Convert pandas/numpy missing values and timestamps for database insertion."""
    if pd.isna(value):
        return None

    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()

    if hasattr(value, "item"):
        return value.item()

    return value


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def upsert_station_information(connection, availability: pd.DataFrame) -> None:
    """Upsert station metadata into Postgres."""
    records = []

    for row in availability.itertuples(index=False):
        records.append(
            (
                clean_value(row.station_id),
                clean_value(row.station_name),
                clean_value(row.latitude),
                clean_value(row.longitude),
                clean_value(row.capacity),
            )
        )

    sql = """
        INSERT INTO station_information (
            station_id,
            station_name,
            latitude,
            longitude,
            capacity
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (station_id)
        DO UPDATE SET
            station_name = EXCLUDED.station_name,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            capacity = EXCLUDED.capacity,
            last_seen_at_utc = NOW();
    """

    with connection.cursor() as cursor:
        cursor.executemany(sql, records)


def insert_station_status_snapshots(
    connection,
    availability: pd.DataFrame,
    snapshot_utc: datetime,
) -> int:
    """Insert station status snapshot rows into Postgres."""
    records = []

    for row in availability.itertuples(index=False):
        records.append(
            (
                snapshot_utc,
                clean_value(row.station_id),
                clean_value(row.num_bikes_available),
                clean_value(row.num_ebikes_available),
                clean_value(row.num_docks_available),
                clean_value(row.pct_bikes_available),
                clean_value(row.pct_docks_available),
                clean_value(row.availability_status),
                clean_value(row.is_installed),
                clean_value(row.is_renting),
                clean_value(row.is_returning),
                clean_value(row.last_reported_utc),
                clean_value(row.status_fetched_at_utc),
            )
        )

    sql = """
        INSERT INTO station_status_snapshots (
            snapshot_utc,
            station_id,
            num_bikes_available,
            num_ebikes_available,
            num_docks_available,
            pct_bikes_available,
            pct_docks_available,
            availability_status,
            is_installed,
            is_renting,
            is_returning,
            last_reported_utc,
            status_fetched_at_utc
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (snapshot_utc, station_id)
        DO NOTHING;
    """

    with connection.cursor() as cursor:
        cursor.executemany(sql, records)
        inserted_count = cursor.rowcount

    return inserted_count


def save_snapshot_to_database(database_url: str | None = None) -> None:
    """Fetch one GBFS station availability snapshot and save it to Postgres."""
    snapshot_utc = utc_now()
    availability = fetch_live_station_availability()

    with connect(database_url) as connection:
        upsert_station_information(connection, availability)
        inserted_count = insert_station_status_snapshots(
            connection,
            availability,
            snapshot_utc,
        )
        connection.commit()

    print(
        f"[{snapshot_utc.isoformat()}] "
        f"Inserted {inserted_count:,} station status rows into Postgres"
    )


def collect_snapshots(
    interval_minutes: float,
    max_snapshots: int | None,
    database_url: str | None,
) -> None:
    """Collect station status snapshots repeatedly."""
    snapshot_count = 0

    while True:
        try:
            save_snapshot_to_database(database_url)
            snapshot_count += 1
        except Exception as error:
            print(f"Database snapshot failed: {error}")

        if max_snapshots is not None and snapshot_count >= max_snapshots:
            break

        print(f"Sleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect Citi Bike GBFS station status snapshots into Postgres."
    )

    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=15,
        help="Minutes between snapshots.",
    )

    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=1,
        help=(
            "Maximum number of snapshots to collect. "
            "Use 0 to collect forever until interrupted."
        ),
    )

    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional Postgres database URL. Defaults to DATABASE_URL or local dev DB.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the Postgres station status collector."""
    args = parse_args()

    max_snapshots = args.max_snapshots

    if max_snapshots == 0:
        max_snapshots = None

    collect_snapshots(
        interval_minutes=args.interval_minutes,
        max_snapshots=max_snapshots,
        database_url=args.database_url,
    )


if __name__ == "__main__":
    main()