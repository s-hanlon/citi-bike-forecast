from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AvailabilityProjection:
    """Projected future station availability."""

    current_bikes: float
    current_docks: float
    capacity: float
    predicted_pickups: float
    predicted_returns: float
    projected_bikes: float
    projected_docks: float
    projected_bike_fill_pct: float
    projected_dock_fill_pct: float
    risk_status: str


def classify_projected_availability(
    projected_bikes: float,
    projected_docks: float,
    capacity: float,
) -> str:
    """Classify projected station availability risk."""
    if capacity <= 0:
        return "unknown"

    projected_bike_fill_pct = projected_bikes / capacity
    projected_dock_fill_pct = projected_docks / capacity

    if projected_bikes <= 0:
        return "empty_risk"

    if projected_docks <= 0:
        return "full_risk"

    if projected_bike_fill_pct <= 0.15:
        return "nearly_empty_risk"

    if projected_dock_fill_pct <= 0.15:
        return "nearly_full_risk"

    return "healthy"


def project_station_availability(
    current_bikes: float,
    current_docks: float,
    capacity: float,
    predicted_pickups: float,
    predicted_returns: float,
) -> AvailabilityProjection:
    """Project future bike and dock availability from current inventory and predicted flows."""
    projected_bikes = current_bikes + predicted_returns - predicted_pickups
    projected_docks = current_docks + predicted_pickups - predicted_returns

    projected_bikes = max(0, min(projected_bikes, capacity))
    projected_docks = max(0, min(projected_docks, capacity))

    if capacity > 0:
        projected_bike_fill_pct = projected_bikes / capacity
        projected_dock_fill_pct = projected_docks / capacity
    else:
        projected_bike_fill_pct = 0
        projected_dock_fill_pct = 0

    risk_status = classify_projected_availability(
        projected_bikes=projected_bikes,
        projected_docks=projected_docks,
        capacity=capacity,
    )

    return AvailabilityProjection(
        current_bikes=current_bikes,
        current_docks=current_docks,
        capacity=capacity,
        predicted_pickups=predicted_pickups,
        predicted_returns=predicted_returns,
        projected_bikes=projected_bikes,
        projected_docks=projected_docks,
        projected_bike_fill_pct=projected_bike_fill_pct,
        projected_dock_fill_pct=projected_dock_fill_pct,
        risk_status=risk_status,
    )


def main() -> None:
    """Smoke test availability projection logic."""
    projection = project_station_availability(
        current_bikes=8,
        current_docks=30,
        capacity=40,
        predicted_pickups=25,
        predicted_returns=5,
    )

    print(projection)


if __name__ == "__main__":
    main()