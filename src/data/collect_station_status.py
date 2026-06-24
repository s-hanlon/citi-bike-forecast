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

from src.live.gbfs import fetch_live_station_availability


OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gbfs_station_status"


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def build_snapshot_path(snapshot_time: datetime) -> Path:
    """Build a partitioned snapshot path."""
    date_folder = snapshot_time.strftime("%Y-%m-%d")
    timestamp_label = snapshot_time.strftime("%Y%m%dT%H%M%SZ")

    return (
        OUTPUT_DIR
        / f"date={date_folder}"
        / f"station_status_{timestamp_label}.parquet"
    )


def save_snapshot() -> Path:
    """Fetch live station availability and save one snapshot."""
    snapshot_time = utc_now()

    availability = fetch_live_station_availability()
    availability["snapshot_utc"] = pd.Timestamp(snapshot_time)

    output_path = build_snapshot_path(snapshot_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    availability.to_parquet(output_path, index=False)

    print(
        f"[{snapshot_time.isoformat()}] "
        f"Saved {len(availability):,} station records to {output_path}"
    )

    return output_path


def collect_snapshots(
    interval_minutes: float,
    max_snapshots: int | None,
) -> None:
    """Collect snapshots repeatedly."""
    snapshot_count = 0

    while True:
        try:
            save_snapshot()
            snapshot_count += 1
        except Exception as error:
            print(f"Snapshot failed: {error}")

        if max_snapshots is not None and snapshot_count >= max_snapshots:
            break

        sleep_seconds = interval_minutes * 60
        print(f"Sleeping for {interval_minutes} minutes...")
        time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect live Citi Bike GBFS station status snapshots."
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

    return parser.parse_args()


def main() -> None:
    """Run the station status collector."""
    args = parse_args()

    max_snapshots = args.max_snapshots

    if max_snapshots == 0:
        max_snapshots = None

    collect_snapshots(
        interval_minutes=args.interval_minutes,
        max_snapshots=max_snapshots,
    )


if __name__ == "__main__":
    main()