from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

from backend.data import load_demand_data, build_data_profile
from backend.modeling import train_forecaster


def generate_portfolio_summary(metrics: dict, profile: dict) -> dict:
    is_synthetic = "synthetic" in profile.get("source", "").lower()
    disclosure = "Synthetic chronological holdout" if is_synthetic else "Historical chronological holdout"

    rmse_imp = metrics.get("rmse_improvement_pct", 0.0)
    mae_imp = metrics.get("mae_improvement_pct", 0.0)
    
    primary_metric = f"{rmse_imp:.1f}% lower forecasting RMSE"
    supporting_metric = f"{mae_imp:.1f}% lower MAE · XGBoost vs. seasonal naive"
    sample_size = f"{metrics.get('test_rows', 0):,} hours"
    
    capability_summary = "12–48 hour forecasting · 4 grid-stress scenarios · Human-approved response"

    return {
        "primary_metric": primary_metric,
        "supporting_metric": supporting_metric,
        "disclosure": disclosure,
        "sample_size": sample_size,
        "capability_summary": capability_summary,
    }


def run_model_benchmark(data_mode: str = "kaggle_historical") -> tuple[dict, "pd.DataFrame"]:
    """
    Executes a model benchmark: loads data, runs strict chronological holdout, 
    calculates metrics, and generates a portfolio summary.
    """
    print(f"Loading data for benchmark in '{data_mode}' mode...")
    demand = load_demand_data(mode=data_mode)
    
    print("Training XGBoost and calculating metrics (chronological holdout)...")
    bundle = train_forecaster(demand)
    
    profile = build_data_profile(demand)
    metrics = bundle.metrics
    portfolio_summary = generate_portfolio_summary(metrics, profile)
    
    benchmark_payload = {
        "metadata": {
            "source": profile.get("source"),
            "region": profile.get("region"),
            "training_rows": metrics["training_rows"],
            "test_rows": metrics["test_rows"],
            "total_rows": metrics["total_rows"],
            "evaluation_type": metrics["evaluation_type"],
            "model_version": metrics["model_version"],
            "timestamp": metrics["evaluation_timestamp"],
            "holdout_start": metrics["holdout_start"],
            "holdout_end": metrics["holdout_end"],
        },
        "metrics": metrics,
        "portfolio_summary": portfolio_summary
    }

    return benchmark_payload, bundle.test_predictions
