from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import requests


class DataLoadError(RuntimeError):
    """Raised when a selected data source cannot be loaded or normalized."""


CANONICAL_COLUMNS = [
    "timestamp",
    "demand_mw",
    "temperature_f",
    "region",
    "is_holiday",
    "source",
    "data_quality_status",
]

TIMESTAMP_HINTS = (
    "datetime",
    "date_time",
    "timestamp",
    "period",
    "date",
    "time",
)
DEMAND_HINTS = (
    "_mw",
    "demand",
    "load",
    "consumption",
    "energy",
    "usage",
)
TEMPERATURE_HINTS = ("temperature", "temp_f", "temp", "drybulb")


def _temperature_proxy(timestamp: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    values = pd.DatetimeIndex(timestamp)
    hour = values.hour.to_numpy()
    day = values.dayofyear.to_numpy()
    return (
        62
        + 18 * np.sin(2 * np.pi * (day - 172) / 365.25)
        + 7 * np.sin(2 * np.pi * (hour - 14) / 24)
    ).astype(float)


def _parse_timestamps(values: pd.Series, timezone_name: str = "UTC") -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        try:
            localized = parsed.dt.tz_localize(
                timezone_name,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        except (TypeError, ValueError):
            localized = parsed.dt.tz_localize(
                timezone_name,
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
        return localized.dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def _normalize_hourly_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    region: str,
    missing_policy: str = "interpolate",
    temperature_is_proxy: bool = False,
) -> pd.DataFrame:
    required = {"timestamp", "demand_mw"}
    if not required.issubset(frame.columns):
        raise DataLoadError(f"Data adapter is missing required fields: {sorted(required - set(frame.columns))}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["demand_mw"] = pd.to_numeric(normalized["demand_mw"], errors="coerce")
    original_rows = len(normalized)
    invalid_rows = int(normalized[["timestamp", "demand_mw"]].isna().any(axis=1).sum())
    normalized = normalized.dropna(subset=["timestamp", "demand_mw"])
    duplicate_rows = int(normalized.duplicated(subset=["timestamp"], keep="last").sum())
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    if normalized.empty:
        raise DataLoadError("No valid hourly demand rows remained after timestamp and demand validation.")

    normalized = normalized.set_index("timestamp")
    full_index = pd.date_range(normalized.index.min(), normalized.index.max(), freq="h", tz="UTC")
    normalized = normalized.reindex(full_index)
    observed_mask = normalized["demand_mw"].notna()
    missing_hours = int((~observed_mask).sum())

    policy = missing_policy.strip().lower()
    if missing_hours:
        if policy == "error":
            raise DataLoadError(
                f"The dataset contains {missing_hours:,} missing hourly observations. "
                "Select interpolation or drop mode to continue."
            )
        if policy == "interpolate":
            normalized["demand_mw"] = normalized["demand_mw"].interpolate(
                method="time", limit_direction="both"
            )
        elif policy == "drop":
            normalized = normalized.dropna(subset=["demand_mw"])
            observed_mask = pd.Series(True, index=normalized.index)
        else:
            raise DataLoadError("Missing-data policy must be 'interpolate', 'drop', or 'error'.")

    if "temperature_f" not in normalized.columns or normalized["temperature_f"].isna().all():
        normalized["temperature_f"] = _temperature_proxy(normalized.index)
        temperature_is_proxy = True
    else:
        normalized["temperature_f"] = pd.to_numeric(normalized["temperature_f"], errors="coerce")
        missing_temperature = normalized["temperature_f"].isna()
        if missing_temperature.any():
            proxy = pd.Series(_temperature_proxy(normalized.index), index=normalized.index)
            normalized.loc[missing_temperature, "temperature_f"] = proxy.loc[missing_temperature]
            temperature_is_proxy = True

    if "is_holiday" not in normalized.columns:
        normalized["is_holiday"] = 0
    normalized["is_holiday"] = pd.to_numeric(normalized["is_holiday"], errors="coerce").fillna(0).astype(int)
    normalized["region"] = region
    normalized["source"] = source
    if policy == "interpolate":
        observed_aligned = observed_mask.reindex(normalized.index, fill_value=False)
        normalized["data_quality_status"] = np.where(observed_aligned, "observed", "interpolated")
    else:
        normalized["data_quality_status"] = "observed"

    normalized = normalized.reset_index(names="timestamp")
    normalized = normalized[CANONICAL_COLUMNS]
    normalized.attrs["profile"] = {
        "source": source,
        "region": region,
        "rows_loaded": int(len(normalized)),
        "original_rows": int(original_rows),
        "invalid_rows_removed": invalid_rows,
        "duplicate_timestamps_removed": duplicate_rows,
        "missing_hours_detected": missing_hours,
        "interpolated_hours": missing_hours if policy == "interpolate" else 0,
        "missing_policy": policy,
        "temperature_method": "seasonal_proxy" if temperature_is_proxy else "dataset",
        "start": normalized["timestamp"].min().isoformat(),
        "end": normalized["timestamp"].max().isoformat(),
    }
    return normalized


def build_data_profile(frame: pd.DataFrame) -> dict[str, Any]:
    stored = dict(frame.attrs.get("profile", {}))
    stored.update(
        {
            "rows_loaded": int(len(frame)),
            "start": pd.Timestamp(frame["timestamp"].min()).isoformat(),
            "end": pd.Timestamp(frame["timestamp"].max()).isoformat(),
            "source": stored.get("source", str(frame["source"].iloc[-1])),
            "region": stored.get("region", str(frame["region"].iloc[-1])),
            "quality_counts": {
                str(key): int(value)
                for key, value in frame["data_quality_status"].value_counts().to_dict().items()
            },
        }
    )
    return stored


def generate_synthetic_demand(days: int = 90, seed: int = 42) -> pd.DataFrame:
    if days < 14:
        raise ValueError("At least 14 days are required.")
    rng = np.random.default_rng(seed)
    periods = days * 24
    end = pd.Timestamp.now(tz="UTC").floor("h")
    timestamp = pd.date_range(end=end, periods=periods, freq="h")
    hour = timestamp.hour.to_numpy()
    dow = timestamp.dayofweek.to_numpy()
    day_index = np.arange(periods) / 24.0

    seasonal_temp = 62 + 18 * np.sin(2 * np.pi * (timestamp.dayofyear.to_numpy() - 172) / 365.25)
    daily_temp = 7 * np.sin(2 * np.pi * (hour - 14) / 24)
    temperature = seasonal_temp + daily_temp + rng.normal(0, 2.2, periods)

    morning = 1700 * np.exp(-((hour - 8) / 3.2) ** 2)
    evening = 2800 * np.exp(-((hour - 18) / 4.2) ** 2)
    overnight = -1300 * np.exp(-((hour - 3) / 3.5) ** 2)
    weekend = np.where(dow >= 5, -1200, 0)
    temp_effect = 28 * np.maximum(temperature - 72, 0) ** 1.35 + 19 * np.maximum(45 - temperature, 0) ** 1.25
    slow_trend = day_index * 3.0
    stress = np.zeros(periods)
    for _ in range(max(2, days // 30)):
        start = int(rng.integers(168, max(169, periods - 36)))
        width = int(rng.integers(8, 30))
        stress[start : min(periods, start + width)] += rng.uniform(900, 2400)

    demand = (
        14500
        + morning
        + evening
        + overnight
        + weekend
        + temp_effect
        + slow_trend
        + stress
        + rng.normal(0, 320, periods)
    )
    demand = np.maximum(demand, 7000)

    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            "demand_mw": demand.astype(float),
            "temperature_f": temperature.astype(float),
            "is_holiday": 0,
        }
    )
    result = _normalize_hourly_frame(
        result,
        source="synthetic:iso-ne-style",
        region="ISNE-SYNTHETIC",
        missing_policy="error",
    )
    result["data_quality_status"] = "synthetic"
    result.attrs["profile"]["quality_counts"] = {"synthetic": len(result)}
    result.attrs["profile"]["synthetic_seed"] = seed
    return result


def _eia_url() -> str:
    return "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def load_eia_demand(api_key: str, respondent: str = "ISNE", history_hours: int = 2160) -> pd.DataFrame:
    if not api_key:
        raise DataLoadError("EIA_API_KEY is required for live EIA mode.")

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=max(history_hours + 168, 720))
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", respondent),
        ("facets[type][]", "D"),
        ("start", start.strftime("%Y-%m-%dT%H")),
        ("end", end.strftime("%Y-%m-%dT%H")),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", "0"),
        ("length", str(min(max(history_hours, 720), 5000))),
    ]
    try:
        response = requests.get(_eia_url(), params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataLoadError(f"EIA live-data request failed: {exc}") from exc

    rows = payload.get("response", {}).get("data", [])
    if not rows:
        api_error = payload.get("error") or payload.get("response", {}).get("warnings")
        raise DataLoadError(f"EIA returned no demand rows. Details: {api_error or 'none'}")

    raw = pd.DataFrame(rows)
    required = {"period", "value"}
    if not required.issubset(raw.columns):
        raise DataLoadError(f"EIA response is missing required columns: {required - set(raw.columns)}")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["period"], utc=True, errors="coerce"),
            "demand_mw": pd.to_numeric(raw["value"], errors="coerce"),
        }
    )
    normalized = _normalize_hourly_frame(
        frame,
        source=f"eia:{respondent}",
        region=respondent,
        missing_policy="interpolate",
        temperature_is_proxy=True,
    ).tail(history_hours)
    if len(normalized) < 336:
        raise DataLoadError(f"EIA returned only {len(normalized)} usable hourly rows; at least 336 are required.")
    normalized.attrs["profile"] = build_data_profile(normalized)
    normalized.attrs["profile"]["temperature_method"] = "seasonal_proxy"
    return normalized


def _read_source_bytes(source: bytes | bytearray | str | Path | BinaryIO) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "uploaded.csv"
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"Kaggle data file was not found: {path}")
        return path.read_bytes(), path.name
    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return raw, getattr(source, "name", "uploaded.csv")
    raise DataLoadError("Unsupported Kaggle source type.")


def list_kaggle_tables(source: bytes | bytearray | str | Path | BinaryIO, filename: str | None = None) -> list[str]:
    raw, inferred_name = _read_source_bytes(source)
    name = filename or inferred_name
    if name.lower().endswith(".zip") or zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return sorted(
                item for item in archive.namelist() if item.lower().endswith(".csv") and not item.endswith("/")
            )
    return [name]


def _read_kaggle_table(
    source: bytes | bytearray | str | Path | BinaryIO,
    *,
    filename: str | None = None,
    table_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    raw, inferred_name = _read_source_bytes(source)
    name = filename or inferred_name
    if name.lower().endswith(".zip") or zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            tables = sorted(item for item in archive.namelist() if item.lower().endswith(".csv"))
            if not tables:
                raise DataLoadError("The uploaded ZIP does not contain a CSV file.")
            selected = table_name or tables[0]
            if selected not in tables:
                raise DataLoadError(f"CSV '{selected}' was not found inside the uploaded ZIP.")
            with archive.open(selected) as handle:
                return pd.read_csv(handle), selected
    return pd.read_csv(io.BytesIO(raw)), name


def _candidate_columns(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    lowered = {column: column.strip().lower() for column in columns}
    return [
        column
        for column, lower in lowered.items()
        if any(hint == lower or hint in lower for hint in hints)
    ]


def inspect_kaggle_source(
    source: bytes | bytearray | str | Path | BinaryIO,
    *,
    filename: str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    frame, selected_table = _read_kaggle_table(source, filename=filename, table_name=table_name)
    columns = [str(column) for column in frame.columns]
    timestamp_candidates = _candidate_columns(columns, TIMESTAMP_HINTS)
    temperature_candidates = _candidate_columns(columns, TEMPERATURE_HINTS)
    demand_candidates = _candidate_columns(columns, DEMAND_HINTS)

    # Retain only columns that contain enough numeric observations for demand candidates.
    numeric_demand_candidates = []
    for column in demand_candidates:
        numeric_share = pd.to_numeric(frame[column], errors="coerce").notna().mean()
        if numeric_share >= 0.5:
            numeric_demand_candidates.append(column)

    if not timestamp_candidates and columns:
        for column in columns:
            parsed_share = pd.to_datetime(frame[column], errors="coerce").notna().mean()
            if parsed_share >= 0.8:
                timestamp_candidates.append(column)
                break

    if not numeric_demand_candidates:
        excluded = set(timestamp_candidates + temperature_candidates)
        for column in columns:
            if column in excluded:
                continue
            numeric_share = pd.to_numeric(frame[column], errors="coerce").notna().mean()
            if numeric_share >= 0.8:
                numeric_demand_candidates.append(column)

    return {
        "table": selected_table,
        "rows": int(len(frame)),
        "columns": columns,
        "timestamp_candidates": timestamp_candidates,
        "demand_candidates": numeric_demand_candidates,
        "temperature_candidates": temperature_candidates,
    }


def load_kaggle_demand(
    source: bytes | bytearray | str | Path | BinaryIO,
    *,
    filename: str | None = None,
    table_name: str | None = None,
    timestamp_column: str | None = None,
    demand_column: str | None = None,
    temperature_column: str | None = None,
    region: str | None = None,
    timezone_name: str = "America/New_York",
    missing_policy: str = "interpolate",
    history_days: int | None = None,
) -> pd.DataFrame:
    raw, selected_table = _read_kaggle_table(source, filename=filename, table_name=table_name)
    inspection = inspect_kaggle_source(source, filename=filename, table_name=selected_table)

    timestamp_column = timestamp_column or next(iter(inspection["timestamp_candidates"]), None)
    if not timestamp_column or timestamp_column not in raw.columns:
        raise DataLoadError(
            "A timestamp column could not be detected. Select a timestamp column from the Kaggle controls."
        )

    demand_candidates = inspection["demand_candidates"]
    demand_column = demand_column or (demand_candidates[0] if len(demand_candidates) == 1 else None)
    if not demand_column or demand_column not in raw.columns:
        raise DataLoadError(
            "Select one demand series from the Kaggle file. Detected candidates: "
            + ", ".join(demand_candidates[:20])
        )

    if temperature_column and temperature_column not in raw.columns:
        raise DataLoadError(f"Temperature column '{temperature_column}' is not present in the Kaggle file.")

    frame = pd.DataFrame(
        {
            "timestamp": _parse_timestamps(raw[timestamp_column], timezone_name=timezone_name),
            "demand_mw": pd.to_numeric(raw[demand_column], errors="coerce"),
        }
    )
    if temperature_column:
        frame["temperature_f"] = pd.to_numeric(raw[temperature_column], errors="coerce")

    inferred_region = region or demand_column.removesuffix("_MW").removesuffix(" MW") or "KAGGLE"
    normalized = _normalize_hourly_frame(
        frame,
        source=f"kaggle:{selected_table}:{demand_column}",
        region=inferred_region,
        missing_policy=missing_policy,
        temperature_is_proxy=not bool(temperature_column),
    )
    if history_days:
        cutoff = normalized["timestamp"].max() - pd.Timedelta(days=int(history_days))
        normalized = normalized.loc[normalized["timestamp"] >= cutoff].reset_index(drop=True)
    if len(normalized) < 336:
        raise DataLoadError(
            f"The selected Kaggle series produced only {len(normalized)} usable hourly rows; at least 336 are required."
        )
    normalized.attrs["profile"] = build_data_profile(normalized)
    normalized.attrs["profile"].update(
        {
            "table": selected_table,
            "timestamp_column": timestamp_column,
            "demand_column": demand_column,
            "temperature_column": temperature_column or "seasonal proxy",
            "timezone_assumption": timezone_name,
        }
    )
    return normalized


def configured_kaggle_path() -> str:
    return os.getenv("GRIDGUARD_KAGGLE_DATA_PATH", "data/kaggle/hourly_energy_consumption.csv").strip()


def data_source_catalog(kaggle_available: bool | None = None) -> list[dict[str, Any]]:
    if kaggle_available is None:
        kaggle_available = Path(configured_kaggle_path()).exists()
    return [
        {
            "id": "synthetic",
            "label": "Synthetic Demo",
            "ready": True,
            "real_data": False,
            "purpose": "Offline demonstrations, stress testing, and reproducible scenarios.",
        },
        {
            "id": "kaggle_historical",
            "label": "Kaggle Historical",
            "ready": bool(kaggle_available),
            "real_data": True,
            "purpose": "Multiyear historical model training and chronological backtesting.",
        },
        {
            "id": "eia_live",
            "label": "EIA Live",
            "ready": bool(os.getenv("EIA_API_KEY", "").strip()),
            "real_data": True,
            "purpose": "Recent balancing-authority demand for deployed operational demonstrations.",
        },
    ]


def load_demand_data(
    mode: str = "synthetic",
    history_days: int = 90,
    *,
    synthetic_seed: int = 42,
    kaggle_source: bytes | bytearray | str | Path | BinaryIO | None = None,
    kaggle_filename: str | None = None,
    kaggle_table_name: str | None = None,
    kaggle_timestamp_column: str | None = None,
    kaggle_demand_column: str | None = None,
    kaggle_temperature_column: str | None = None,
    kaggle_region: str | None = None,
    kaggle_timezone: str = "America/New_York",
    kaggle_missing_policy: str = "interpolate",
) -> pd.DataFrame:
    normalized = mode.strip().lower()
    if normalized == "synthetic":
        return generate_synthetic_demand(days=history_days, seed=synthetic_seed)
    if normalized == "kaggle_historical":
        source = kaggle_source or configured_kaggle_path()
        return load_kaggle_demand(
            source,
            filename=kaggle_filename,
            table_name=kaggle_table_name,
            timestamp_column=kaggle_timestamp_column,
            demand_column=kaggle_demand_column,
            temperature_column=kaggle_temperature_column,
            region=kaggle_region,
            timezone_name=kaggle_timezone,
            missing_policy=kaggle_missing_policy,
            history_days=history_days,
        )
    if normalized == "eia_live":
        return load_eia_demand(
            api_key=os.getenv("EIA_API_KEY", "").strip(),
            respondent=os.getenv("GRIDGUARD_EIA_RESPONDENT", "ISNE").strip(),
            history_hours=int(history_days * 24),
        )
    raise DataLoadError(f"Unsupported data mode: {mode}")
