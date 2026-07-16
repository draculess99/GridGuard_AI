from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_mode: str = os.getenv("GRIDGUARD_DATA_MODE", "synthetic").strip().lower()
    eia_api_key: str = os.getenv("EIA_API_KEY", "").strip()
    eia_respondent: str = os.getenv("GRIDGUARD_EIA_RESPONDENT", "ISNE").strip()
    eia_history_hours: int = int(os.getenv("GRIDGUARD_EIA_HISTORY_HOURS", "2160"))
    kaggle_data_path: str = os.getenv("GRIDGUARD_KAGGLE_DATA_PATH", "data/kaggle/hourly_energy_consumption.csv").strip()
    kaggle_timezone: str = os.getenv("GRIDGUARD_KAGGLE_TIMEZONE", "America/New_York").strip()
    kaggle_missing_policy: str = os.getenv("GRIDGUARD_KAGGLE_MISSING_POLICY", "interpolate").strip().lower()
    forecast_hours: int = int(os.getenv("GRIDGUARD_FORECAST_HOURS", "24"))
    persistence_mode: str = os.getenv("GRIDGUARD_PERSISTENCE_MODE", "json").strip().lower()
    database_url: str = os.getenv("DATABASE_URL", "").strip()
    decision_provider: str = os.getenv("GRIDGUARD_DECISION_PROVIDER", "internal_expert_system").strip().lower()
    memory_path: str = os.getenv("GRIDGUARD_MEMORY_PATH", "data/runtime/decision_memory.json").strip()
    memory_max_records: int = int(os.getenv("GRIDGUARD_MEMORY_MAX_RECORDS", "200"))
    token_ledger_path: str = os.getenv("GRIDGUARD_TOKEN_LEDGER_PATH", "data/runtime/token_usage.json").strip()
    rag_docs_dir: str = os.getenv("GRIDGUARD_RAG_DOCS_DIR", "docs/rag").strip()
    grok_model: str = os.getenv("GRIDGUARD_GROK_MODEL", "grok-4.5").strip()
    groq_model: str = os.getenv("GRIDGUARD_GROQ_MODEL", "openai/gpt-oss-120b").strip()
    gemini_model: str = os.getenv("GRIDGUARD_GEMINI_MODEL", "gemini-2.5-flash").strip()


def get_settings() -> Settings:
    return Settings()
