# Forecast and Model Governance

GridGuard compares XGBoost against a seasonal-naive forecast based on the value from one week earlier. The chronological holdout test must be used instead of a random split. If XGBoost does not outperform the seasonal-naive baseline, the application must display the limitation and should not treat XGBoost as independently sufficient evidence.

Recursive multi-hour forecasts can accumulate error. Temperature data in live EIA mode is currently a proxy until a weather connector is implemented. Point forecasts do not represent calibrated uncertainty intervals.

Feature importance explains model behavior globally but is not proof of causality. Operational recommendations should combine model output, deterministic rules, retrieved policy context, and human judgment.
