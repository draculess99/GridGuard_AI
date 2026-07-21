import json
import math
from pathlib import Path

import pandas as pd
import pytest

from backend.benchmark_engine import run_model_benchmark
from backend.response_simulator import run_operational_benchmark
from backend.modeling import train_forecaster
from backend.data import generate_synthetic_demand


def test_metrics_improvement_calculation():
    demand = generate_synthetic_demand(days=21, seed=42)
    bundle = train_forecaster(demand)
    
    xgb_rmse = bundle.metrics["xgb_rmse"]
    naive_rmse = bundle.metrics["naive_rmse"]
    rmse_imp = bundle.metrics["rmse_improvement_pct"]
    
    expected_rmse_imp = (naive_rmse - xgb_rmse) / naive_rmse * 100
    assert math.isclose(rmse_imp, expected_rmse_imp, rel_tol=1e-5)
    
    xgb_mae = bundle.metrics["xgb_mae"]
    naive_mae = bundle.metrics["naive_mae"]
    mae_imp = bundle.metrics["mae_improvement_pct"]
    
    expected_mae_imp = (naive_mae - xgb_mae) / naive_mae * 100
    assert math.isclose(mae_imp, expected_mae_imp, rel_tol=1e-5)


def test_chronological_separation():
    demand = generate_synthetic_demand(days=21, seed=42)
    bundle = train_forecaster(demand)
    
    holdout_start = pd.Timestamp(bundle.metrics["holdout_start"])
    history_max = demand["timestamp"].max()
    
    assert holdout_start <= history_max
    assert holdout_start > demand["timestamp"].min()


def test_synthetic_vs_historical_labeling():
    demand = generate_synthetic_demand(days=21, seed=42)
    bundle = train_forecaster(demand)
    assert "Synthetic" in bundle.metrics["evaluation_type"]
    assert "synthetic" in bundle.metrics["data_source"].lower()


def test_benchmark_engine_payload():
    payload, _ = run_model_benchmark(data_mode="synthetic")
    
    assert "portfolio_summary" in payload
    summary = payload["portfolio_summary"]
    
    assert "primary_metric" in summary
    assert "supporting_metric" in summary
    assert "disclosure" in summary
    assert "sample_size" in summary
    
    assert "Synthetic" in summary["disclosure"]
    # Check no hardcoded example values (34.6%, 26.0%, 3470 hours)
    assert "34.6%" not in summary["primary_metric"]
    assert "26.0%" not in summary["supporting_metric"]
    assert "3,470" not in summary["sample_size"]


def test_operational_simulator_labels():
    result = run_operational_benchmark()
    assert result["label"] == "Simulated operational benchmark"
    assert "comparisons" in result
    assert len(result["comparisons"]) == 5
    
    # Financial impact test
    result_with_cost = run_operational_benchmark(cost_per_unserved_mwh=100.0)
    for comp in result_with_cost["comparisons"]:
        assert "financial_impact_modeled" in comp
        if comp["avoided_capacity_deficit_mwh"] > 0:
            assert comp["financial_impact_modeled"] > 0

def test_valid_metric_ranges():
    demand = generate_synthetic_demand(days=21, seed=42)
    bundle = train_forecaster(demand)
    
    assert bundle.metrics["xgb_rmse"] > 0
    assert bundle.metrics["xgb_mae"] > 0
    assert bundle.metrics["xgb_r2"] > -10.0 # R2 can be negative
    assert bundle.metrics["xgb_mape"] >= 0
