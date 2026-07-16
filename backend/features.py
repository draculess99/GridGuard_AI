from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "temperature_f",
    "temperature_sq",
    "is_holiday",
]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"], utc=True)
    result["hour"] = timestamp.dt.hour
    result["dayofweek"] = timestamp.dt.dayofweek
    result["month"] = timestamp.dt.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    result["temperature_sq"] = result["temperature_f"] ** 2
    return result


def build_training_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.sort_values("timestamp").copy()
    demand = frame["demand_mw"]
    for lag in (1, 2, 24, 48, 168):
        frame[f"lag_{lag}"] = demand.shift(lag)
    frame["rolling_mean_24"] = demand.shift(1).rolling(24).mean()
    frame["rolling_std_24"] = demand.shift(1).rolling(24).std()
    frame["rolling_mean_168"] = demand.shift(1).rolling(168).mean()
    frame = add_time_features(frame)
    return frame.dropna(subset=FEATURE_COLUMNS + ["demand_mw"]).reset_index(drop=True)


def next_feature_row(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    temperature_f: float,
    is_holiday: int = 0,
) -> pd.DataFrame:
    values = history["demand_mw"].astype(float).tolist()
    if len(values) < 168:
        raise ValueError("At least 168 hourly demand values are required for forecasting.")

    row = {
        "timestamp": pd.Timestamp(timestamp),
        "temperature_f": float(temperature_f),
        "is_holiday": int(is_holiday),
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_24": values[-24],
        "lag_48": values[-48],
        "lag_168": values[-168],
        "rolling_mean_24": float(np.mean(values[-24:])),
        "rolling_std_24": float(np.std(values[-24:], ddof=1)),
        "rolling_mean_168": float(np.mean(values[-168:])),
    }
    return add_time_features(pd.DataFrame([row]))[FEATURE_COLUMNS]
