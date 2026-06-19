from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_pickups.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

LAGS = [
    24,
    48,
    168,
    336,
]


def build_features(
    pickups: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar and historical-demand features."""
    features = (
        pickups.sort_values(
            ["station_id", "timestamp"]
        )
        .reset_index(drop=True)
        .copy()
    )

    features["hour"] = features["timestamp"].dt.hour
    features["day_of_week"] = (
        features["timestamp"].dt.dayofweek
    )
    features["month"] = features["timestamp"].dt.month
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype("int8")

    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour"] / 24
    )
    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour"] / 24
    )

    features["day_of_week_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )
    features["day_of_week_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    pickup_groups = features.groupby(
        "station_id",
        sort=False,
    )["pickups"]

    for lag in LAGS:
        features[f"lag_{lag}"] = (
            pickup_groups.shift(lag)
        )

    features["rolling_mean_168"] = (
        pickup_groups.transform(
            lambda station: (
                station.shift(24)
                .rolling(
                    window=168,
                    min_periods=168,
                )
                .mean()
            )
        )
    )

    features["rolling_std_168"] = (
        pickup_groups.transform(
            lambda station: (
                station.shift(24)
                .rolling(
                    window=168,
                    min_periods=168,
                )
                .std()
            )
        )
    )

    required_history = [
        f"lag_{lag}"
        for lag in LAGS
    ] + [
        "rolling_mean_168",
        "rolling_std_168",
    ]

    features = (
        features.dropna(
            subset=required_history
        )
        .reset_index(drop=True)
    )

    return features


def main() -> None:
    """Build and save the model feature table."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found: {INPUT_FILE}\n"
            "Run src/data/make_dataset.py first."
        )

    pickups = pd.read_parquet(INPUT_FILE)
    features = build_features(pickups)

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved {len(features):,} feature rows "
        f"to {OUTPUT_FILE}"
    )
    print(
        f"Feature range: "
        f"{features['timestamp'].min()} through "
        f"{features['timestamp'].max()}"
    )
    print(
        f"Missing values: "
        f"{features.isna().sum().sum()}"
    )


if __name__ == "__main__":
    main()