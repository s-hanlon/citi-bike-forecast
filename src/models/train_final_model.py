from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.gradient_boosting import (
    FEATURE_FILE,
    FORECAST_MODEL_FEATURES,
    create_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "citi_bike_demand_model.joblib"
METADATA_FILE = MODEL_DIR / "citi_bike_demand_model_metadata.json"

TARGET_COLUMN = "pickups"


def main() -> None:
    """Train and save the final v2 forecasting model."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}\n"
            "Run src/features/build_features.py first."
        )

    features = pd.read_parquet(FEATURE_FILE)

    X = features[FORECAST_MODEL_FEATURES]
    y = features[TARGET_COLUMN]

    model = create_model()

    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    station_table = (
        features[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_id")
        .to_dict(orient="records")
    )

    metadata = {
        "model_name": "citi_bike_demand_model",
        "model_version": "v2-long-history-holiday-weather",
        "target_column": TARGET_COLUMN,
        "feature_columns": FORECAST_MODEL_FEATURES,
        "training_rows": int(len(features)),
        "training_start": str(features["timestamp"].min()),
        "training_end": str(features["timestamp"].max()),
        "station_count": int(features["station_id"].nunique()),
        "stations": station_table,
        "notes": (
            "Final v2 model trained on long-history station-hour demand "
            "features with lagged demand, rolling demand, holiday features, "
            "station identity, and archived 24-hour-ahead weather forecasts."
        ),
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
    print(f"Training rows: {len(features):,}")
    print(
        "Training range: "
        f"{features['timestamp'].min()} through "
        f"{features['timestamp'].max()}"
    )
    print(f"Features: {len(FORECAST_MODEL_FEATURES)}")
    print(f"Stations: {features['station_id'].nunique()}")


if __name__ == "__main__":
    main()