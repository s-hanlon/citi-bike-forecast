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

def get_network_availability_history(hours: int = 24) -> pd.DataFrame:
    """Fetch network-level availability metrics over recent snapshots."""
    sql = """
        SELECT
            snapshot_utc,
            COUNT(*) AS station_count,

            SUM(CASE WHEN availability_status = 'healthy' THEN 1 ELSE 0 END)
                AS healthy_stations,

            SUM(CASE
                WHEN availability_status IN ('empty', 'nearly_empty', 'low_bikes')
                THEN 1 ELSE 0
            END) AS bike_shortage_risk_stations,

            SUM(CASE
                WHEN availability_status IN ('full', 'nearly_full', 'low_docks')
                THEN 1 ELSE 0
            END) AS dock_shortage_risk_stations,

            SUM(CASE WHEN availability_status = 'empty' THEN 1 ELSE 0 END)
                AS empty_stations,

            SUM(CASE WHEN availability_status = 'full' THEN 1 ELSE 0 END)
                AS full_stations,

            AVG(pct_bikes_available) AS avg_pct_bikes_available,
            AVG(pct_docks_available) AS avg_pct_docks_available

        FROM station_status_snapshots
        WHERE snapshot_utc >= NOW() - (%s * INTERVAL '1 hour')
        GROUP BY snapshot_utc
        ORDER BY snapshot_utc;
    """

    return query_dataframe(sql, (hours,))


def get_top_availability_risk_stations(hours: int = 24, limit: int = 15) -> pd.DataFrame:
    """Fetch stations with the highest recent empty/full risk."""
    sql = """
        SELECT
            s.station_id,
            i.station_name,
            i.latitude,
            i.longitude,
            i.capacity,

            COUNT(*) AS snapshots,

            SUM(CASE
                WHEN s.availability_status IN ('empty', 'nearly_empty', 'low_bikes')
                THEN 1 ELSE 0
            END) AS bike_shortage_risk_snapshots,

            SUM(CASE
                WHEN s.availability_status IN ('full', 'nearly_full', 'low_docks')
                THEN 1 ELSE 0
            END) AS dock_shortage_risk_snapshots,

            SUM(CASE WHEN s.availability_status = 'empty' THEN 1 ELSE 0 END)
                AS empty_snapshots,

            SUM(CASE WHEN s.availability_status = 'full' THEN 1 ELSE 0 END)
                AS full_snapshots,

            ROUND(
                100.0 * SUM(CASE
                    WHEN s.availability_status IN ('empty', 'nearly_empty', 'low_bikes')
                    THEN 1 ELSE 0
                END) / COUNT(*),
                1
            ) AS bike_shortage_risk_pct,

            ROUND(
                100.0 * SUM(CASE
                    WHEN s.availability_status IN ('full', 'nearly_full', 'low_docks')
                    THEN 1 ELSE 0
                END) / COUNT(*),
                1
            ) AS dock_shortage_risk_pct,

            AVG(s.num_bikes_available) AS avg_bikes_available,
            AVG(s.num_docks_available) AS avg_docks_available

        FROM station_status_snapshots AS s
        LEFT JOIN station_information AS i
            ON s.station_id = i.station_id
        WHERE s.snapshot_utc >= NOW() - (%s * INTERVAL '1 hour')
        GROUP BY
            s.station_id,
            i.station_name,
            i.latitude,
            i.longitude,
            i.capacity
        ORDER BY
            (
                SUM(CASE
                    WHEN s.availability_status IN (
                        'empty',
                        'nearly_empty',
                        'low_bikes',
                        'full',
                        'nearly_full',
                        'low_docks'
                    )
                    THEN 1 ELSE 0
                END)
            ) DESC,
            s.station_id
        LIMIT %s;
    """

    return query_dataframe(sql, (hours, limit))


def get_station_availability_history(station_id: str, hours: int = 24) -> pd.DataFrame:
    """Fetch recent availability history for one station."""
    sql = """
        SELECT
            s.snapshot_utc,
            s.station_id,
            i.station_name,
            i.capacity,
            s.num_bikes_available,
            s.num_ebikes_available,
            s.num_docks_available,
            s.pct_bikes_available,
            s.pct_docks_available,
            s.availability_status
        FROM station_status_snapshots AS s
        LEFT JOIN station_information AS i
            ON s.station_id = i.station_id
        WHERE s.station_id = %s
          AND s.snapshot_utc >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY s.snapshot_utc;
    """

    return query_dataframe(sql, (station_id, hours))

def main() -> None:
    """Smoke test Postgres availability queries."""
    summary = get_snapshot_summary()
    latest = get_latest_station_availability()
    network_history = get_network_availability_history(hours=24)
    top_risk = get_top_availability_risk_stations(hours=24, limit=10)

    print("Snapshot summary:")
    print(summary.to_string(index=False))
    print()

    print("Latest availability rows:")
    print(len(latest))
    print()

    print("Availability status counts:")
    print(latest["availability_status"].value_counts().to_string())
    print()

    print("Network availability history:")
    print(network_history.tail(10).to_string(index=False))
    print()

    print("Top recent risk stations:")
    print(
        top_risk[
            [
                "station_name",
                "snapshots",
                "bike_shortage_risk_snapshots",
                "dock_shortage_risk_snapshots",
                "empty_snapshots",
                "full_snapshots",
                "bike_shortage_risk_pct",
                "dock_shortage_risk_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()