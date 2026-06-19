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

BASE_NUMERIC_FEATURES = [
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

WEATHER_FEATURES = [
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_code",
    "is_precipitating",
]

BASE_MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + BASE_NUMERIC_FEATURES
)

FORECAST_MODEL_FEATURES = (
    BASE_MODEL_FEATURES
    + WEATHER_FEATURES
)

MODEL_CONFIGURATIONS = {
    "Gradient boosting": BASE_MODEL_FEATURES,
    "Gradient boosting + 24h forecast": (
        FORECAST_MODEL_FEATURES
    ),
}

VALIDATION_STARTS = pd.to_datetime(
    [
        "2026-04-03",
        "2026-04-10",
        "2026-04-17",
    ]
)

FINAL_TEST_START = pd.Timestamp(
    "2026-05-25"
)

FINAL_TEST_END = pd.Timestamp(
    "2026-06-01"
)

EXPECTED_TEST_ROWS = (
    7
    * 24
    * 25
)


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


def train_ml_models(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> dict[str, object]:
    """Train each ML configuration and return predictions."""

    predictions = {}

    for model_name, model_features in (
        MODEL_CONFIGURATIONS.items()
    ):
        model = create_model()

        model.fit(
            training_data[model_features],
            training_data["pickups"],
        )

        predictions[model_name] = model.predict(
            evaluation_data[model_features]
        )

    return predictions


def evaluate_predictions(
    data: pd.DataFrame,
    ml_predictions: dict[str, object],
) -> pd.DataFrame:
    """Compare ML predictions with seasonal baselines."""

    prediction_sources = {
        "Yesterday": data["lag_24"],
        "Last week": data["lag_168"],
        **ml_predictions,
    }

    records = []

    for model_name, predictions in (
        prediction_sources.items()
    ):
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

        ml_predictions = train_ml_models(
            training_data,
            validation_data,
        )

        results = evaluate_predictions(
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
) -> pd.DataFrame:
    """Evaluate the locked May 25–31 holdout exactly once."""

    training_data = features[
        features["timestamp"]
        < FINAL_TEST_START
    ]

    test_data = features[
        (
            features["timestamp"]
            >= FINAL_TEST_START
        )
        & (
            features["timestamp"]
            < FINAL_TEST_END
        )
    ]

    if len(test_data) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} final test rows, "
            f"but found {len(test_data):,}."
        )

    ml_predictions = train_ml_models(
        training_data,
        test_data,
    )

    return evaluate_predictions(
        test_data,
        ml_predictions,
    )


def summarize_backtest(
    backtest_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average and variation across validation folds."""

    return (
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


def main() -> None:
    """Run validation followed by the locked final test."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {FEATURE_FILE}\n"
            "Run src/features/build_features.py first."
        )

    features = pd.read_parquet(
        FEATURE_FILE,
    )

    print("Running expanding-window backtest...\n")

    backtest_results = run_backtest(
        features
    )

    backtest_summary = summarize_backtest(
        backtest_results
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

    print(
        "\nRunning locked final test "
        "(May 25–31)...\n"
    )

    final_results = (
        run_final_test(features)
        .sort_values("MAE")
    )

    print("Final untouched test results:")

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