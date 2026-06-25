from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.flow_forecasting import (
    FEATURE_FILE,
    MODEL_FEATURES,
    create_model,
)


MODEL_DIR = PROJECT_ROOT / "models"

PICKUP_MODEL_FILE = MODEL_DIR / "citi_bike_pickup_flow_model.joblib"
RETURN_MODEL_FILE = MODEL_DIR / "citi_bike_return_flow_model.joblib"
NET_OUTFLOW_MODEL_FILE = MODEL_DIR / "citi_bike_net_outflow_model.joblib"
METADATA_FILE = MODEL_DIR / "citi_bike_flow_model_metadata.json"


def train_and_save_model(
    features: pd.DataFrame,
    target_column: str,
    loss: str,
    output_file: Path,
) -> None:
    """Train and save one final flow model."""
    model = create_model(loss=loss)

    model.fit(
        features[MODEL_FEATURES],
        features[target_column],
    )

    joblib.dump(model, output_file)

    print(f"Saved {target_column} model to {output_file}")


def main() -> None:
    """Train and save final pickup, return, and net-flow models."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Flow feature file not found: {FEATURE_FILE}\n"
            "Run src/features/build_flow_features.py first."
        )

    features = pd.read_parquet(FEATURE_FILE)
    features["station_id"] = features["station_id"].astype(str)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_and_save_model(
        features=features,
        target_column="pickups",
        loss="poisson",
        output_file=PICKUP_MODEL_FILE,
    )

    train_and_save_model(
        features=features,
        target_column="returns",
        loss="poisson",
        output_file=RETURN_MODEL_FILE,
    )

    train_and_save_model(
        features=features,
        target_column="net_outflow",
        loss="squared_error",
        output_file=NET_OUTFLOW_MODEL_FILE,
    )

    metadata = {
        "model_version": "v3-availability-intelligence-flow-models",
        "feature_columns": MODEL_FEATURES,
        "training_rows": int(len(features)),
        "training_start": str(features["timestamp"].min()),
        "training_end": str(features["timestamp"].max()),
        "station_count": int(features["station_id"].nunique()),
        "targets": {
            "pickups": {
                "loss": "poisson",
                "artifact": str(PICKUP_MODEL_FILE.relative_to(PROJECT_ROOT)),
            },
            "returns": {
                "loss": "poisson",
                "artifact": str(RETURN_MODEL_FILE.relative_to(PROJECT_ROOT)),
            },
            "net_outflow": {
                "loss": "squared_error",
                "artifact": str(NET_OUTFLOW_MODEL_FILE.relative_to(PROJECT_ROOT)),
            },
        },
        "availability_projection_formula": (
            "future_bikes = current_bikes + predicted_returns - predicted_pickups"
        ),
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved metadata to {METADATA_FILE}")
    print(f"Training rows: {len(features):,}")
    print(
        "Training range: "
        f"{features['timestamp'].min()} through {features['timestamp'].max()}"
    )
    print(f"Features: {len(MODEL_FEATURES)}")
    print(f"Stations: {features['station_id'].nunique()}")


if __name__ == "__main__":
    main()