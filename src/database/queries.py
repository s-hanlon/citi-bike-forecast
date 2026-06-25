from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.database.postgres import connect


def query_dataframe(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a SQL query and return a pandas DataFrame."""
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)

            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def get_latest_station_availability() -> pd.DataFrame:
    """Fetch the latest station availability snapshot from Postgres."""
    sql = """
        WITH latest_snapshot AS (
            SELECT MAX(snapshot_utc) AS snapshot_utc
            FROM station_status_snapshots
        )
        SELECT
            s.snapshot_utc,
            s.station_id,
            i.station_name,
            i.latitude,
            i.longitude,
            i.capacity,
            s.num_bikes_available,
            s.num_ebikes_available,
            s.num_docks_available,
            s.pct_bikes_available,
            s.pct_docks_available,
            s.availability_status,
            s.is_installed,
            s.is_renting,
            s.is_returning,
            s.last_reported_utc,
            s.status_fetched_at_utc
        FROM station_status_snapshots AS s
        INNER JOIN latest_snapshot AS latest
            ON s.snapshot_utc = latest.snapshot_utc
        LEFT JOIN station_information AS i
            ON s.station_id = i.station_id
        ORDER BY i.station_name;
    """

    return query_dataframe(sql)


def get_snapshot_summary() -> pd.DataFrame:
    """Summarize stored GBFS snapshots."""
    sql = """
        SELECT
            COUNT(DISTINCT snapshot_utc) AS snapshot_count,
            COUNT(*) AS row_count,
            MIN(snapshot_utc) AS first_snapshot_utc,
            MAX(snapshot_utc) AS latest_snapshot_utc
        FROM station_status_snapshots;
    """

    return query_dataframe(sql)


def main() -> None:
    """Smoke test Postgres availability queries."""
    summary = get_snapshot_summary()
    latest = get_latest_station_availability()

    print("Snapshot summary:")
    print(summary.to_string(index=False))
    print()

    print("Latest availability rows:")
    print(len(latest))
    print()

    print("Availability status counts:")
    print(latest["availability_status"].value_counts().to_string())
    print()

    print("Sample:")
    print(
        latest[
            [
                "station_name",
                "capacity",
                "num_bikes_available",
                "num_ebikes_available",
                "num_docks_available",
                "availability_status",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()