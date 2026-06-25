from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "flow_model_features.parquet"

CATEGORICAL_FEATURES = ["station_id"]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_us_federal_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_window",
    "days_to_nearest_holiday",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_code",
    "is_precipitating",
    "pickup_lag_24",
    "pickup_lag_48",
    "pickup_lag_168",
    "pickup_lag_336",
    "pickup_rolling_mean_168",
    "pickup_rolling_std_168",
    "return_lag_24",
    "return_lag_48",
    "return_lag_168",
    "return_lag_336",
    "return_rolling_mean_168",
    "return_rolling_std_168",
    "net_outflow_lag_24",
    "net_outflow_lag_48",
    "net_outflow_lag_168",
    "net_outflow_lag_336",
    "net_outflow_rolling_mean_168",
    "net_outflow_rolling_std_168",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

VALIDATION_STARTS = [
    "2025-07-11",
    "2025-08-08",
    "2025-09-12",
    "2025-10-10",
    "2025-11-07",
    "2025-12-05",
    "2026-01-09",
    "2026-02-06",
    "2026-03-06",
    "2026-04-03",
    "2026-04-10",
    "2026-04-17",
]

FINAL_TEST_START = pd.Timestamp("2026-05-25")
FINAL_TEST_END = pd.Timestamp("2026-06-01")
EXPECTED_TEST_ROWS = 7 * 24 * 25


def create_model(loss: str = "poisson") -> Pipeline:
    """Create a gradient boosting model pipeline for station flow prediction."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    model = HistGradientBoostingRegressor(
        loss=loss,
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
            ("model", model),
        ]
    )


def evaluate_predictions(
    actual: pd.Series,
    predicted,
) -> dict:
    """Calculate regression metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
    }


def evaluate_baseline(
    validation_data: pd.DataFrame,
    target_column: str,
    lag_column: str,
) -> dict:
    """Evaluate same-hour-last-week baseline."""
    return evaluate_predictions(
        validation_data[target_column],
        validation_data[lag_column],
    )


def run_backtest_for_target(
    features: pd.DataFrame,
    target_column: str,
    baseline_lag_column: str,
    loss: str,
) -> pd.DataFrame:
    """Run seasonal backtests for one target column."""
    results = []

    for validation_start_text in VALIDATION_STARTS:
        validation_start = pd.Timestamp(validation_start_text)
        validation_end = validation_start + pd.Timedelta(days=7)

        train_data = features[features["timestamp"] < validation_start].copy()
        validation_data = features[
            (features["timestamp"] >= validation_start)
            & (features["timestamp"] < validation_end)
        ].copy()

        if len(validation_data) != EXPECTED_TEST_ROWS:
            print(
                f"Warning: {validation_start.date()} has "
                f"{len(validation_data):,} rows, expected {EXPECTED_TEST_ROWS:,}"
            )

        model = create_model(loss=loss)

        model.fit(
            train_data[MODEL_FEATURES],
            train_data[target_column],
        )

        predictions = model.predict(validation_data[MODEL_FEATURES])

        model_metrics = evaluate_predictions(
            validation_data[target_column],
            predictions,
        )

        baseline_metrics = evaluate_baseline(
            validation_data,
            target_column,
            baseline_lag_column,
        )

        results.append(
            {
                "target": target_column,
                "validation_week": validation_start.date(),
                "model_mae": model_metrics["mae"],
                "model_rmse": model_metrics["rmse"],
                "last_week_mae": baseline_metrics["mae"],
                "last_week_rmse": baseline_metrics["rmse"],
                "mae_improvement_pct": (
                    (
                        baseline_metrics["mae"] - model_metrics["mae"]
                    )
                    / baseline_metrics["mae"]
                    * 100
                ),
            }
        )

        print(
            f"{target_column} | {validation_start.date()} | "
            f"model MAE {model_metrics['mae']:.2f} | "
            f"last week MAE {baseline_metrics['mae']:.2f}"
        )

    return pd.DataFrame(results)


def run_final_may_benchmark(
    features: pd.DataFrame,
    target_column: str,
    baseline_lag_column: str,
    loss: str,
) -> dict:
    """Evaluate one target on the May 2026 benchmark week."""
    train_data = features[features["timestamp"] < FINAL_TEST_START].copy()
    test_data = features[
        (features["timestamp"] >= FINAL_TEST_START)
        & (features["timestamp"] < FINAL_TEST_END)
    ].copy()

    if len(test_data) != EXPECTED_TEST_ROWS:
        print(
            f"Warning: May benchmark has {len(test_data):,} rows, "
            f"expected {EXPECTED_TEST_ROWS:,}"
        )

    model = create_model(loss=loss)

    model.fit(
        train_data[MODEL_FEATURES],
        train_data[target_column],
    )

    predictions = model.predict(test_data[MODEL_FEATURES])

    model_metrics = evaluate_predictions(
        test_data[target_column],
        predictions,
    )

    baseline_metrics = evaluate_baseline(
        test_data,
        target_column,
        baseline_lag_column,
    )

    return {
        "target": target_column,
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "last_week_mae": baseline_metrics["mae"],
        "last_week_rmse": baseline_metrics["rmse"],
        "mae_improvement_pct": (
            (baseline_metrics["mae"] - model_metrics["mae"])
            / baseline_metrics["mae"]
            * 100
        ),
    }


def main() -> None:
    """Train and evaluate station flow forecasting models."""
    features = pd.read_parquet(FEATURE_FILE)
    features["station_id"] = features["station_id"].astype(str)
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    print(f"Loaded {len(features):,} flow feature rows")
    print(
        features["timestamp"].min(),
        "through",
        features["timestamp"].max(),
    )
    print()

    targets = [
        ("pickups", "pickup_lag_168", "poisson"),
        ("returns", "return_lag_168", "poisson"),
        ("net_outflow", "net_outflow_lag_168", "squared_error"),
    ]

    all_backtests = []

    for target_column, baseline_lag_column, loss in targets:
        print("=" * 80)
        print(f"Backtesting target: {target_column}")
        print("=" * 80)

        target_results = run_backtest_for_target(
            features,
            target_column,
            baseline_lag_column,
            loss,
        )

        all_backtests.append(target_results)
        print()

    backtest_results = pd.concat(all_backtests, ignore_index=True)

    print("=" * 80)
    print("Backtest summary")
    print("=" * 80)

    summary = (
        backtest_results.groupby("target")
        .agg(
            mean_model_mae=("model_mae", "mean"),
            mean_model_rmse=("model_rmse", "mean"),
            mean_last_week_mae=("last_week_mae", "mean"),
            mean_last_week_rmse=("last_week_rmse", "mean"),
            mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
        )
        .reset_index()
    )

    print(summary.round(2).to_string(index=False))
    print()

    print("=" * 80)
    print("May 2026 benchmark")
    print("=" * 80)

    final_results = []

    for target_column, baseline_lag_column, loss in targets:
        final_results.append(
            run_final_may_benchmark(
                features,
                target_column,
                baseline_lag_column,
                loss,
            )
        )

    final_results = pd.DataFrame(final_results)

    print(final_results.round(2).to_string(index=False))


if __name__ == "__main__":
    main()