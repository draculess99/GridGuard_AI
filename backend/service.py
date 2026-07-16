from __future__ import annotations

from backend.modeling import ModelBundle, forecast_recursive
from backend.risk import assess_grid_risk


def build_forecast_package(
    bundle: ModelBundle,
    horizon: int,
    capacity_mw: float,
    scenario: dict | None = None,
) -> dict:
    scenario = scenario or {}
    forecast = forecast_recursive(
        bundle,
        horizon=horizon,
        temperature_delta=float(scenario.get("temperature_delta", 0.0)),
        demand_shock_pct=float(scenario.get("demand_shock_pct", 0.0)),
    )
    risk = assess_grid_risk(
        forecast,
        capacity_mw=capacity_mw,
        outage_mw=float(scenario.get("outage_mw", 0.0)),
    )
    return {"forecast": forecast, "risk": risk, "scenario": scenario}
