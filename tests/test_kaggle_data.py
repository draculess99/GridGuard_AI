from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from backend.data import (
    CANONICAL_COLUMNS,
    DataLoadError,
    data_source_catalog,
    inspect_kaggle_source,
    list_kaggle_tables,
    load_kaggle_demand,
)


def _pjm_csv(hours: int = 500, include_gap: bool = False) -> bytes:
    timestamp = pd.date_range("2024-01-01", periods=hours, freq="h")
    frame = pd.DataFrame(
        {
            "Datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "PJME_MW": 25000 + 1500 * np.sin(np.arange(hours) * 2 * np.pi / 24),
            "AEP_MW": 14000 + 900 * np.cos(np.arange(hours) * 2 * np.pi / 24),
        }
    )
    if include_gap:
        frame = frame.drop(index=[100, 101]).reset_index(drop=True)
    return frame.to_csv(index=False).encode("utf-8")


def test_kaggle_inspection_detects_pjm_columns():
    inspection = inspect_kaggle_source(_pjm_csv(), filename="pjm.csv")
    assert "Datetime" in inspection["timestamp_candidates"]
    assert {"PJME_MW", "AEP_MW"}.issubset(inspection["demand_candidates"])


def test_kaggle_loader_normalizes_common_schema_and_interpolates_gap():
    result = load_kaggle_demand(
        _pjm_csv(include_gap=True),
        filename="pjm.csv",
        timestamp_column="Datetime",
        demand_column="PJME_MW",
        region="PJM East",
        timezone_name="UTC",
        missing_policy="interpolate",
    )
    assert list(result.columns) == CANONICAL_COLUMNS
    assert len(result) == 500
    assert result["demand_mw"].isna().sum() == 0
    assert (result["data_quality_status"] == "interpolated").sum() == 2
    assert result["region"].iloc[0] == "PJM East"
    assert result.attrs["profile"]["temperature_method"] == "seasonal_proxy"


def test_kaggle_loader_error_policy_reports_gap():
    with pytest.raises(DataLoadError, match="missing hourly observations"):
        load_kaggle_demand(
            _pjm_csv(include_gap=True),
            filename="pjm.csv",
            timestamp_column="Datetime",
            demand_column="PJME_MW",
            timezone_name="UTC",
            missing_policy="error",
        )


def test_kaggle_zip_lists_and_loads_selected_csv():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("PJME_hourly.csv", _pjm_csv())
        archive.writestr("notes.txt", "not data")
    raw = buffer.getvalue()
    assert list_kaggle_tables(raw, filename="hourly-energy.zip") == ["PJME_hourly.csv"]
    result = load_kaggle_demand(
        raw,
        filename="hourly-energy.zip",
        table_name="PJME_hourly.csv",
        timestamp_column="Datetime",
        demand_column="AEP_MW",
        timezone_name="UTC",
    )
    assert len(result) == 500
    assert result["source"].iloc[0].startswith("kaggle:PJME_hourly.csv:AEP_MW")


def test_catalog_exposes_all_three_sources(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    catalog = data_source_catalog(kaggle_available=True)
    assert [item["id"] for item in catalog] == ["synthetic", "kaggle_historical", "eia_live"]
    assert catalog[1]["ready"] is True
    assert catalog[2]["ready"] is False


def test_eia_adapter_normalizes_mocked_live_rows(monkeypatch):
    from backend import data as data_module

    timestamps = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    rows = [
        {"period": timestamp.strftime("%Y-%m-%dT%H"), "value": str(18000 + index)}
        for index, timestamp in enumerate(timestamps)
    ]

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": {"data": rows}}

    monkeypatch.setattr(data_module.requests, "get", lambda *args, **kwargs: MockResponse())
    result = data_module.load_eia_demand("test-key", respondent="ISNE", history_hours=400)
    assert len(result) == 400
    assert list(result.columns) == CANONICAL_COLUMNS
    assert result["region"].iloc[0] == "ISNE"
    assert result["source"].iloc[0] == "eia:ISNE"
    assert result["temperature_f"].notna().all()


def test_full_kaggle_mode_can_feed_xgboost():
    from backend.data import load_demand_data
    from backend.modeling import train_forecaster

    data = load_demand_data(
        mode="kaggle_historical",
        history_days=30,
        kaggle_source=_pjm_csv(hours=800),
        kaggle_filename="pjm.csv",
        kaggle_timestamp_column="Datetime",
        kaggle_demand_column="PJME_MW",
        kaggle_region="PJM East",
        kaggle_timezone="UTC",
    )
    bundle = train_forecaster(data)
    assert bundle.metrics["xgb_mae"] > 0
    assert bundle.history["source"].iloc[0].startswith("kaggle:")
