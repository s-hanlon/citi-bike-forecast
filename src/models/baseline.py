from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_pickups.parquet"
)

TEST_DAYS = 7

def add_baseline_predictions(
    pickups: pd.DataFrame,
) -> pd.DataFrame:
    """Add previous-day and previous-week predictions."""
    predictions = (
        pickups.sort_values(
            ["station_id", "timestamp"]
        )
        .reset_index(drop=True)
        .copy()
    )

    station_groups = predictions.groupby(
        "station_id",
        sort=False,
    )

    predictions["lag_24"] = (
        station_groups["pickups"].shift(24)
    )

    predictions["lag_168"] = (
        station_groups["pickups"].shift(168)
    )

    return predictions

def evaluate_baselines(
    predictions: pd.DataFrame,
    test_days: int = TEST_DAYS,
) -> pd.DataFrame:
    """Evaluate seasonal baselines on the final test days."""
    test_start = (
        predictions["timestamp"].max().normalize()
        - pd.Timedelta(days=test_days - 1)
    )

    test = predictions[
        predictions["timestamp"] >= test_start
    ].dropna(
        subset=[
            "lag_24",
            "lag_168",
        ]
    )

    baseline_columns = {
        "Same hour yesterday": "lag_24",
        "Same hour last week": "lag_168",
    }

    results = []

    for model_name, prediction_column in baseline_columns.items():
        results.append(
            {
                "model": model_name,
                "MAE": mean_absolute_error(
                    test["pickups"],
                    test[prediction_column],
                ),
                "RMSE": root_mean_squared_error(
                    test["pickups"],
                    test[prediction_column],
                ),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

def main() -> None:
    """Load the processed data and report baseline metrics."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_FILE}\n"
            "Run src/data/make_dataset.py first."
        )

    pickups = pd.read_parquet(DATA_FILE)

    predictions = add_baseline_predictions(pickups)
    results = evaluate_baselines(predictions)

    print(
        results.to_string(
            index=False,
            formatters={
                "MAE": "{:.2f}".format,
                "RMSE": "{:.2f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()