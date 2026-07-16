from backend.data import generate_synthetic_demand
from backend.modeling import train_forecaster
from backend.service import build_forecast_package


def test_forecast_package_includes_risk_and_scenario():
    bundle = train_forecaster(generate_synthetic_demand(days=45, seed=3))
    package = build_forecast_package(
        bundle=bundle,
        horizon=12,
        capacity_mw=26000,
        scenario={"temperature_delta": 8, "outage_mw": 1000, "demand_shock_pct": 4},
    )
    assert len(package["forecast"]) == 12
    assert "level" in package["risk"]
    assert package["scenario"]["outage_mw"] == 1000
