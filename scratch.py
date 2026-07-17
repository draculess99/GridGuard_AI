import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
from backend.data import generate_synthetic_demand
from backend.features import build_training_frame, FEATURE_COLUMNS

history = generate_synthetic_demand(days=90, seed=42)
clean_history = history.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
training = build_training_frame(clean_history)
test_size = max(72, int(len(training) * 0.2))
train = training.iloc[:-test_size]
test = training.iloc[-test_size:]

model = XGBRegressor(n_estimators=420, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_alpha=0.05, reg_lambda=1.2, objective='reg:squarederror', eval_metric='mae', random_state=42, n_jobs=4, tree_method='hist')
model.fit(train[FEATURE_COLUMNS], train['demand_mw'])

gain_imp = pd.DataFrame({'feature': FEATURE_COLUMNS, 'importance': model.feature_importances_.astype(float)}).sort_values('importance', ascending=False)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test[FEATURE_COLUMNS])
mean_shap_values = np.abs(shap_values).mean(axis=0)
shap_imp = pd.DataFrame({'feature': FEATURE_COLUMNS, 'importance': mean_shap_values.astype(float)}).sort_values('importance', ascending=False)

print('--- OLD GAIN IMPORTANCE ---')
print(gain_imp.head(8).to_string(index=False))
print('\n--- NEW SHAP IMPORTANCE ---')
print(shap_imp.head(8).to_string(index=False))
