import pandas as pd

from backend.data import generate_synthetic_demand
from backend.features import FEATURE_COLUMNS, build_training_frame


def test_synthetic_data_shape():
    frame = generate_synthetic_demand(days=30, seed=7)
    assert len(frame) == 720
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["demand_mw"].min() > 0


def test_feature_frame_contains_expected_columns():
    frame = generate_synthetic_demand(days=30, seed=7)
    features = build_training_frame(frame)
    assert not features.empty
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0
