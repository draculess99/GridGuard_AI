import random

def run_operational_benchmark(
    scenario_type: str = "Extreme Grid Stress",
    cost_per_unserved_mwh: float = None
) -> dict:
    """
    Simulated operational benchmark across different scenarios comparing
    various grid responses.
    """
    scenarios = {
        "Summer Heatwave": {"peak_demand": 28000, "capacity": 27000, "duration_hours": 8},
        "Winter Freeze": {"peak_demand": 29000, "capacity": 26500, "duration_hours": 12},
        "Major Plant Trip": {"peak_demand": 25000, "capacity": 23000, "duration_hours": 4},
        "Extreme Grid Stress": {"peak_demand": 30000, "capacity": 27000, "duration_hours": 10},
    }

    base = scenarios.get(scenario_type, scenarios["Extreme Grid Stress"])
    
    # Calculate baseline (No intervention)
    baseline_deficit_mwh = max(0, (base["peak_demand"] - base["capacity"]) * base["duration_hours"] * 0.8)
    baseline_high_risk_hours = base["duration_hours"]

    # Simulating different responses
    responses = {
        "No intervention": {"reduction_mw": 0, "reliability": 1.0},
        "Voluntary demand response": {"reduction_mw": 1000, "reliability": 0.6},
        "Targeted demand reduction": {"reduction_mw": 2500, "reliability": 0.85},
        "Emergency load shedding": {"reduction_mw": 4000, "reliability": 1.0},
        "GridGuard-selected response": {"reduction_mw": 3200, "reliability": 0.95},
    }

    results = []
    for r_name, r_stats in responses.items():
        effective_demand = base["peak_demand"] - (r_stats["reduction_mw"] * r_stats["reliability"])
        deficit_mw = max(0, effective_demand - base["capacity"])
        deficit_mwh = deficit_mw * base["duration_hours"] * 0.8
        
        avoided_deficit = baseline_deficit_mwh - deficit_mwh
        
        # Calculate high-risk hours based on deficit
        if deficit_mw <= 0:
            high_risk_hours = 0
        else:
            ratio = deficit_mw / max(1, (base["peak_demand"] - base["capacity"]))
            high_risk_hours = max(1, int(base["duration_hours"] * ratio))
        
        reserve_margin_pct = ((base["capacity"] - effective_demand) / base["capacity"]) * 100
        baseline_margin_pct = ((base["capacity"] - base["peak_demand"]) / base["capacity"]) * 100
        
        financial_impact = None
        if cost_per_unserved_mwh is not None:
            financial_impact = avoided_deficit * cost_per_unserved_mwh

        results.append({
            "response": r_name,
            "capacity_deficit_mwh": deficit_mwh,
            "avoided_capacity_deficit_mwh": avoided_deficit,
            "high_risk_hours": high_risk_hours,
            "reserve_margin_improvement_pct": reserve_margin_pct - baseline_margin_pct,
            "financial_impact_modeled": financial_impact
        })

    return {
        "label": "Simulated operational benchmark",
        "scenario": scenario_type,
        "baseline_high_risk_hours": baseline_high_risk_hours,
        "percentage_of_scenarios_stabilized": 85.0,
        "response_selection_distribution": {
            "Voluntary demand response": 20,
            "Targeted demand reduction": 50,
            "Emergency load shedding": 5,
            "GridGuard-selected response": 25
        },
        "comparisons": results
    }
