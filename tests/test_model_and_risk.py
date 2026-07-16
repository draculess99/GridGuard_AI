from backend.data import generate_synthetic_demand
from backend.modeling import forecast_recursive, train_forecaster
from backend.risk import assess_grid_risk


def test_training_and_forecast():
    history = generate_synthetic_demand(days=45, seed=11)
    bundle = train_forecaster(history)
    forecast = forecast_recursive(bundle, horizon=24)
    assert len(forecast) == 24
    assert forecast["forecast_mw"].min() > 0
    assert bundle.metrics["xgb_mae"] > 0
    assert bundle.metrics["naive_mae"] > 0


def test_risk_escalates_when_capacity_is_low():
    history = generate_synthetic_demand(days=45, seed=12)
    bundle = train_forecaster(history)
    forecast = forecast_recursive(bundle, horizon=24)
    risk = assess_grid_risk(forecast, capacity_mw=float(forecast["forecast_mw"].max() * 0.90))
    assert risk["level"] == "CRITICAL"
    assert risk["reserve_margin_pct"] < 0
