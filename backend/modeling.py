from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from backend.features import FEATURE_COLUMNS, build_training_frame, next_feature_row


@dataclass
class ModelBundle:
    model: XGBRegressor
    history: pd.DataFrame
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame
    model_version: str


def train_forecaster(history: pd.DataFrame) -> ModelBundle:
    clean_history = history.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    training = build_training_frame(clean_history)
    if len(training) < 240:
        raise ValueError("Insufficient feature rows for a chronological train/test split.")

    test_size = max(72, int(len(training) * 0.2))
    train = training.iloc[:-test_size]
    test = training.iloc[-test_size:]

    model = XGBRegressor(
        n_estimators=420,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(train[FEATURE_COLUMNS], train["demand_mw"])

    xgb_prediction = model.predict(test[FEATURE_COLUMNS])
    naive_prediction = test["lag_168"].to_numpy()
    actual = test["demand_mw"].to_numpy()

    xgb_mae = float(mean_absolute_error(actual, xgb_prediction))
    naive_mae = float(mean_absolute_error(actual, naive_prediction))
    metrics = {
        "xgb_mae": xgb_mae,
        "naive_mae": naive_mae,
        "xgb_rmse": float(mean_squared_error(actual, xgb_prediction) ** 0.5),
        "naive_rmse": float(mean_squared_error(actual, naive_prediction) ** 0.5),
        "mae_improvement_pct": float((naive_mae - xgb_mae) / naive_mae * 100) if naive_mae else 0.0,
    }

    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_.astype(float)}
    ).sort_values("importance", ascending=False)

    comparison = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual_mw": actual,
            "xgb_mw": xgb_prediction,
            "naive_mw": naive_prediction,
        }
    )

    version = datetime.now(timezone.utc).strftime("gridguard-xgb-%Y%m%d%H%M%S")
    return ModelBundle(
        model=model,
        history=clean_history,
        metrics=metrics,
        feature_importance=importance,
        test_predictions=comparison,
        model_version=version,
    )


def forecast_recursive(
    bundle: ModelBundle,
    horizon: int = 24,
    temperature_delta: float = 0.0,
    demand_shock_pct: float = 0.0,
) -> pd.DataFrame:
    history = bundle.history[["timestamp", "demand_mw", "temperature_f", "is_holiday", "source"]].copy()
    rows: list[dict] = []
    last_timestamp = pd.Timestamp(history["timestamp"].max())

    recent_temp = history["temperature_f"].tail(168)
    for step in range(1, horizon + 1):
        timestamp = last_timestamp + pd.Timedelta(hours=step)
        seasonal_temp = float(recent_temp.iloc[(step - 1) % len(recent_temp)] + temperature_delta)
        features = next_feature_row(history, timestamp=timestamp, temperature_f=seasonal_temp)
        prediction = float(bundle.model.predict(features)[0])
        prediction *= 1 + demand_shock_pct / 100.0
        prediction = max(prediction, 0.0)

        row = {
            "timestamp": timestamp,
            "forecast_mw": prediction,
            "temperature_f": seasonal_temp,
        }
        rows.append(row)
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "timestamp": timestamp,
                            "demand_mw": prediction,
                            "temperature_f": seasonal_temp,
                            "is_holiday": 0,
                            "source": "forecast",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(rows)
