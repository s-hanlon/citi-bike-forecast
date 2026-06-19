from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

CATEGORICAL_FEATURES = [
    "station_id",
]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "lag_24",
    "lag_48",
    "lag_168",
    "lag_336",
    "rolling_mean_168",
    "rolling_std_168",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

VALIDATION_STARTS = pd.to_datetime(
    [
        "2026-04-03",
        "2026-04-10",
        "2026-04-17",
    ]
)

TEST_DAYS = 7


def create_model() -> Pipeline:
    """Create a new, unfitted forecasting pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "station",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )

    regressor = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

def evaluate_models(
    data: pd.DataFrame,
    ml_predictions,
) -> pd.DataFrame:
    """Compare ML predictions with both seasonal baselines."""
    prediction_sources = {
        "Yesterday": data["lag_24"],
        "Last week": data["lag_168"],
        "Gradient boosting": ml_predictions,
    }

    records = []

    for model_name, predictions in prediction_sources.items():
        records.append(
            {
                "model": model_name,
                "MAE": mean_absolute_error(
                    data["pickups"],
                    predictions,
                ),
                "RMSE": root_mean_squared_error(
                    data["pickups"],
                    predictions,
                ),
            }
        )

    return pd.DataFrame(records)

def run_backtest(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Run expanding-window validation over configured weeks."""
    fold_results = []

    for validation_start in VALIDATION_STARTS:
        validation_end = (
            validation_start
            + pd.Timedelta(days=7)
        )

        training_data = features[
            features["timestamp"]
            < validation_start
        ]

        validation_data = features[
            (
                features["timestamp"]
                >= validation_start
            )
            & (
                features["timestamp"]
                < validation_end
            )
        ]

        model = create_model()

        model.fit(
            training_data[MODEL_FEATURES],
            training_data["pickups"],
        )

        ml_predictions = model.predict(
            validation_data[MODEL_FEATURES]
        )

        results = evaluate_models(
            validation_data,
            ml_predictions,
        )

        results.insert(
            0,
            "validation_week",
            validation_start.date(),
        )

        fold_results.append(results)

        print(
            f"Completed validation week "
            f"{validation_start.date()}"
        )

    return pd.concat(
        fold_results,
        ignore_index=True,
    )

def run_final_test(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, Pipeline]:
    """Train through April 23 and evaluate the frozen final week."""
    test_start = (
        features["timestamp"].max().normalize()
        - pd.Timedelta(days=TEST_DAYS - 1)
    )

    training_data = features[
        features["timestamp"] < test_start
    ]

    test_data = features[
        features["timestamp"] >= test_start
    ]

    model = create_model()

    model.fit(
        training_data[MODEL_FEATURES],
        training_data["pickups"],
    )

    ml_predictions = model.predict(
        test_data[MODEL_FEATURES]
    )

    results = evaluate_models(
        test_data,
        ml_predictions,
    )

    return results, model

def main() -> None:
    """Run rolling validation and final testing."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {FEATURE_FILE}\n"
            "Run src/features/build_features.py first."
        )

    features = pd.read_parquet(FEATURE_FILE)

    print("Running expanding-window backtest...\n")
    backtest_results = run_backtest(features)

    backtest_summary = (
        backtest_results.groupby(
            "model",
            as_index=False,
        )
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
        )
        .sort_values("mean_MAE")
    )

    print("\nBacktest summary:")
    print(
        backtest_summary.to_string(
            index=False,
            formatters={
                "mean_MAE": "{:.2f}".format,
                "std_MAE": "{:.2f}".format,
                "mean_RMSE": "{:.2f}".format,
                "std_RMSE": "{:.2f}".format,
            },
        )
    )

    print("\nRunning frozen final test...\n")
    final_results, _ = run_final_test(features)

    final_results = final_results.sort_values("MAE")

    print("Final test results:")
    print(
        final_results.to_string(
            index=False,
            formatters={
                "MAE": "{:.2f}".format,
                "RMSE": "{:.2f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()