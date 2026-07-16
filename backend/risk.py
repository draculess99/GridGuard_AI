from __future__ import annotations

import pandas as pd


def assess_grid_risk(
    forecast: pd.DataFrame,
    capacity_mw: float,
    outage_mw: float = 0.0,
) -> dict:
    if forecast.empty:
        raise ValueError("Forecast cannot be empty.")
    effective_capacity = max(float(capacity_mw) - float(outage_mw), 1.0)
    peak_index = forecast["forecast_mw"].idxmax()
    peak = float(forecast.loc[peak_index, "forecast_mw"])
    peak_time = pd.Timestamp(forecast.loc[peak_index, "timestamp"])
    reserve_margin = (effective_capacity - peak) / peak * 100
    utilization = forecast["forecast_mw"] / effective_capacity
    high_risk_hours = int((utilization >= 0.92).sum())

    if reserve_margin < 0 or high_risk_hours >= 4:
        level = "CRITICAL"
        headline = "Forecast demand challenges effective capacity."
        recommendation = (
            "Escalate to the grid-operations lead. Prepare emergency demand-response measures, "
            "validate transfer capability, and confirm contingency resources before the peak window."
        )
    elif reserve_margin < 5 or high_risk_hours >= 2:
        level = "ELEVATED"
        headline = "Reserve margin is thin during the forecast peak."
        recommendation = (
            "Activate targeted demand response, confirm fast-start resources, and schedule operator "
            "reviews before each high-risk hour."
        )
    elif reserve_margin < 12 or high_risk_hours >= 1:
        level = "WATCH"
        headline = "Conditions merit enhanced monitoring."
        recommendation = (
            "Prepare voluntary demand response, verify resource availability, and reforecast when "
            "new demand or weather data arrives."
        )
    else:
        level = "NORMAL"
        headline = "Forecast demand remains within the assumed operating margin."
        recommendation = "Continue routine monitoring and refresh the forecast as new observations arrive."

    return {
        "level": level,
        "headline": headline,
        "recommendation": recommendation,
        "peak_mw": peak,
        "peak_time": peak_time.isoformat(),
        "effective_capacity_mw": effective_capacity,
        "reserve_margin_pct": float(reserve_margin),
        "high_risk_hours": high_risk_hours,
    }
